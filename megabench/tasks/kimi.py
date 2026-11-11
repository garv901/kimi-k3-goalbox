"""Kimi K3 text decode step for a batch of B sequences, at tensor parallel TP.

Model: 93 layers. Layer 0 has a dense FFN. Layers 3, 7, ..., 91 and 92 use
gated MLA (no rotary transform). All other layers use KDA (Kimi delta
attention) with a V-first FP32 recurrent state and a 4-tap short convolution.
Layers 1..92 use a latent MoE: 896 routed experts (top 16, sigmoid scores with
a correction bias), MXFP4 expert weights in a 3584-wide latent space, and two
shared experts. Block attention residuals mix the layer inputs.

Weights come from one of two sources (case.params['checkpoint'] set or not):
  synthetic  - fresh random weights for each seed (anti-caching trials);
  checkpoint - the real moonshotai/Kimi-K3 weights, converted per TP rank by
               setup/convert_checkpoint.py (see kimi_checkpoint.py).
The per-sequence state (tokens, reset flags, MLA latent caches, KDA conv and
recurrent states) is always fresh random data for each seed.

reference(..., fp32=False) is the BF16 serving oracle. reference(..., fp32=True)
runs the same math with FP32 activations and FP32-decoded weights. compare()
grades a candidate with the oracle band: its distance to the FP32 reference
must stay within BAND times the BF16 oracle's own distance.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import seed_for
from .parallel import initialize
from .quantization import unpack_mxfp4

BAND = 2.0            # candidate error may be at most BAND x the BF16 oracle error
FLOOR = 2e-3          # absolute allowance on relative L2, for outputs where the oracle error is ~0
ROUTE_FLOOR = 1e-3   # router-score units: extra near-tie room for a chosen expert below the k-th best
WEIGHTS_ENV = 'KIMI_K3_WEIGHTS'


def full_layer(case: Case, layer: int) -> bool:
    return (layer+1) % 4 == 0 or layer == case.params['layers']-1


def kda_layers(case: Case) -> list[int]:
    return [layer for layer in range(case.params['layers']) if not full_layer(case, layer)]


def mla_layers(case: Case) -> list[int]:
    return [layer for layer in range(case.params['layers']) if full_layer(case, layer)]


def _randn(seed, name, shape, scale, device, dtype=torch.bfloat16, shift=0.0):
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, name))
    return (torch.randn(shape, generator=generator, device=device)*scale+shift).to(dtype)


def _mxfp4(seed, name, shape, fan_in, device):
    """Random E2M1 codes and E8M0 scales. RMS of the decoded values ~ fan_in**-0.5."""
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, name))
    *lead, cols = shape
    blocks = torch.randint(0, 256, (*lead, cols//32, 16), generator=generator, device=device, dtype=torch.uint8)
    # uniform E2M1 codes have RMS ~2.93: choose 2**e ~ 1/(2.93*sqrt(fan_in)), +-1 per block
    exponent = round(-torch.log2(torch.tensor(2.93*fan_in**0.5)).item())
    scales = (127+exponent-1+torch.randint(0, 3, (*lead, cols//32), generator=generator, device=device)).to(torch.uint8)
    return blocks, scales


def weight_shapes(case: Case) -> dict[str, tuple[tuple[int, ...], torch.dtype, int, bool]]:
    """Per-rank weight tensors: name -> (shape, dtype, fan_in, split by TP rank)."""
    p, tp = case.params, case.tp
    h, lh, d, heads = p['hidden'], p['latent_hidden'], p['head_dim'], p['q_heads']
    L, V, qdim = p['layers'], p['vocab'], p['q_heads']*p['head_dim']
    for n in (heads, V, p['intermediate'], p['dense_intermediate']):
        if n % tp:
            raise ValueError('Kimi attention heads, vocabulary and FFN intermediate must divide TP')
    if (p['intermediate']//tp) % 32:
        raise ValueError('MXFP4 down-projection shards must align to groups of 32')
    bf, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    out = {'embed': ((V//tp, h), bf, h, True), 'lm_head': ((V//tp, h), bf, h, True),
           'fnorm': ((h,), bf, 0, False), 'output_res_norm': ((h,), bf, 0, False),
           'output_res_proj': ((h,), bf, h, False)}
    for name in ('ln1', 'ln2', 'attention_res_norm', 'mlp_res_norm'):
        out[name] = ((L, h), bf, 0, False)
    for name in ('attention_res_proj', 'mlp_res_proj'):
        out[name] = ((L, h), bf, h, False)
    for layer in range(L):
        x = f'l{layer}_'
        if full_layer(case, layer):
            qr, kr, nr, rd, vd = p['q_lora_rank'], p['kv_lora_rank'], p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
            out |= {x+'qa': ((qr, h), bf, h, False), x+'qb': ((heads//tp*(nr+rd), qr), bf, qr, True),
                    x+'ka': ((kr+rd, h), bf, h, False), x+'kb': ((heads//tp*(nr+vd), kr), bf, kr, True),
                    x+'o': ((h, heads//tp*vd), bf, heads*vd, True), x+'output_gate': ((heads//tp*vd, h), bf, h, True),
                    x+'qn': ((qr,), bf, 0, False), x+'kn': ((kr,), bf, 0, False)}
        else:
            for name in ('q', 'k', 'v', 'output_gate'):
                out[x+name] = ((qdim//tp, h), bf, h, True)
            out |= {x+'o': ((h, qdim//tp), bf, qdim, True), x+'fa': ((d, h), bf, h, False),
                    x+'fb': ((qdim//tp, d), bf, d, True), x+'beta': ((heads//tp, h), bf, h, True),
                    x+'onorm': ((d,), f32, 0, False), x+'A_log': ((heads//tp,), f32, 0, True),
                    x+'dt_bias': ((qdim//tp,), f32, 0, True),
                    x+'conv_weight': ((3, qdim//tp, p['conv_kernel']), f32, p['conv_kernel'], True)}
        if layer < p['first_dense']:
            inter = p['dense_intermediate']//tp
            out |= {x+'gate': ((inter, h), bf, h, True), x+'up': ((inter, h), bf, h, True),
                    x+'down': ((h, inter), bf, p['dense_intermediate'], True)}
        else:
            E, inter = p['experts'], p['intermediate']//tp
            shared = p['intermediate']*p['shared_experts']//tp
            out |= {x+'latent_down': ((lh, h), bf, h, False), x+'latent_up': ((h, lh), bf, lh, False),
                    x+'latent_norm': ((lh,), bf, 0, False), x+'router': ((E, h), bf, h, False),
                    x+'router_bias': ((E,), f32, 0, False),
                    x+'gate_blocks': ((E, inter, lh//32, 16), u8, lh, True), x+'gate_scales': ((E, inter, lh//32), u8, 0, True),
                    x+'up_blocks': ((E, inter, lh//32, 16), u8, lh, True), x+'up_scales': ((E, inter, lh//32), u8, 0, True),
                    x+'down_blocks': ((E, lh, inter//32, 16), u8, p['intermediate'], True),
                    x+'down_scales': ((E, lh, inter//32), u8, 0, True),
                    x+'shared_gate': ((shared, h), bf, h, True), x+'shared_up': ((shared, h), bf, h, True),
                    x+'shared_down': ((h, shared), bf, p['intermediate']*p['shared_experts'], True)}
    return out


def synthetic_weights(case: Case, seed: int, device: str, rank: int) -> dict[str, torch.Tensor]:
    """Fresh random weights. TP-split tensors use a per-rank stream; replicated ones a shared stream."""
    values = {}
    for name, (shape, dtype, fan_in, split) in weight_shapes(case).items():
        stream = f'{name}@{rank}' if split else name
        if name.endswith('_scales'):
            continue
        if name.endswith('_blocks'):
            values[name], values[name[:-7]+'_scales'] = _mxfp4(seed, stream, shape[:-2]+(shape[-2]*32,), fan_in, device)
        elif name.endswith('A_log'):
            values[name] = _randn(seed, stream, shape, 0.5, device, dtype)
        elif name.endswith('dt_bias'):
            values[name] = _randn(seed, stream, shape, 1.0, device, dtype, shift=-4.0)
        elif name.endswith('router_bias'):
            values[name] = _randn(seed, stream, shape, 0.01, device, dtype)
        elif fan_in == 0:  # norm weights
            values[name] = _randn(seed, stream, shape, 0.1, device, dtype, shift=1.0)
        else:
            values[name] = _randn(seed, stream, shape, fan_in**-0.5, device, dtype)
    return values


_CHECKPOINT_CACHE: dict = {}


def checkpoint_weights(case: Case, device: str, rank: int) -> dict[str, torch.Tensor]:
    """Real weights, loaded once per process from the converted per-rank files."""
    key = (case.params['layers'], case.tp, rank, str(device))
    if key not in _CHECKPOINT_CACHE:
        from .kimi_checkpoint import load_weights
        root = os.environ.get(WEIGHTS_ENV)
        if not root:
            raise RuntimeError(f'{case.id} needs real weights: set {WEIGHTS_ENV} to the converted weight directory '
                               '(setup/convert_checkpoint.py)')
        _CHECKPOINT_CACHE.clear()
        _CHECKPOINT_CACHE[key] = load_weights(case, root, rank, device)
    return dict(_CHECKPOINT_CACHE[key])


def make_state(case: Case, seed: int, device: str, rank: int) -> dict[str, torch.Tensor]:
    p, tp, B = case.params, case.tp, case.params['batch']
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, 'token'))
    values = {'token': torch.randint(0, p['vocab'], (B,), generator=generator, device=device),
              # row 0 always continues; about one row in four restarts its sequence
              'reset': torch.rand((B,), generator=generator, device=device) < 0.25}
    values['reset'][0] = False
    if B == 1:
        values['reset'][0] = seed % 3 == 0
    qdim, heads, d = p['q_heads']*p['head_dim']//tp, p['q_heads']//tp, p['head_dim']
    for layer in range(p['layers']):
        x = f'l{layer}_'
        if full_layer(case, layer):
            values[x+'kv_cache'] = _randn(seed, x+'kv_cache', (B, p['context'], p['kv_lora_rank']), 1.0, device)
            values[x+'pe_cache'] = _randn(seed, x+'pe_cache', (B, p['context'], p['qk_rope_dim']), 1.0, device)
        else:
            values[x+'conv_state'] = _randn(seed, f'{x}conv_state@{rank}', (B, 3, qdim, p['conv_kernel']), 1.0, device)
            values[x+'recurrent_state'] = _randn(seed, f'{x}recurrent@{rank}', (B, heads, d, d), 0.05, device, torch.float32)
    return values


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None):
    rank = (dist.get_rank() if dist.is_initialized() else 0) if rank is None else rank
    if case.ep != 1 or case.gpus != case.tp or not 0 <= rank < case.gpus:
        raise ValueError('Kimi requires one TP group spanning all ranks')
    source = 'checkpoint' if case.params.get('checkpoint') else 'synthetic'
    weights = checkpoint_weights(case, device, rank) if source == 'checkpoint' else synthetic_weights(case, seed, device, rank)
    return weights | make_state(case, seed, device, rank)


def situ(gate, up):
    gate, up = gate.float(), up.float()
    return 4*torch.tanh(gate/4)*gate.sigmoid()*(25*torch.tanh(up/25))


def attention_residual(prefix, blocks, norm, projection, cast):
    """HF _apply_attn_res: softmax over [blocks..., prefix] of rms(v)·(norm*proj), all in FP32."""
    v = torch.stack([*blocks, prefix]).float()                       # [n, B, h]
    k = v*torch.rsqrt(v.square().mean(-1, keepdim=True)+1e-5)
    probability = (k*(norm.float()*projection.float())).sum(-1).softmax(0)
    return cast((probability[..., None]*v).sum(0))


def kda_step(query, key, value, gate, beta, A_log, dt_bias, state):
    """One KDA token per row. q,k,v,gate [..., H, D]; beta [..., H]; state [..., H, V, K] (V-first)."""
    query = query.float()*torch.rsqrt(query.float().square().sum(-1, keepdim=True)+1e-6)/query.shape[-1]**0.5
    key = key.float()*torch.rsqrt(key.float().square().sum(-1, keepdim=True)+1e-6)
    decay = (-5*torch.sigmoid(A_log.float().exp()[:, None]*(gate.float()+dt_bias.float()))).exp()
    updated = state.float()*decay[..., None, :]
    error = value.float()-torch.einsum('...hvk,...hk->...hv', updated, key)
    updated = updated+torch.einsum('...hv,...hk->...hvk', error*beta.float().sigmoid()[..., None], key)
    return torch.einsum('...hvk,...hk->...hv', updated, query), updated


def _greedy(logits, offset, group, serial):
    """Global argmax per row of vocabulary-sharded logits; lowest index wins ties."""
    if serial:
        return logits.argmax(-1)
    maximum = logits.max(-1).values.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    index = torch.arange(logits.shape[-1], device=logits.device)+offset
    token = torch.where(logits == maximum[:, None], index, torch.iinfo(torch.int64).max).min(-1).values
    dist.all_reduce(token, op=dist.ReduceOp.MIN, group=group)
    return token


def reference(case: Case, values, *, serial=None, fp32=False, routes=None):
    """One decode step. `routes` [L, B, topk] forces the expert sets of the MoE layers (the grader uses
    it to compare continuous outputs on the same discrete routes) and adds 'route_shortfall' [L, B]:
    how far below the k-th best biased FP32 router score the weakest forced expert is (0 = a valid top-k)."""
    serial = case.gpus == 1 if serial is None else serial
    if serial and (case.gpus, case.tp, case.ep) != (1, 1, 1):
        raise ValueError('serial validation requires unsharded geometry')
    ctx = SimpleNamespace(tp_rank=0, tp_group=None) if serial else initialize(case)
    sum_value = (lambda value: value) if serial else (lambda value: _reduce(value, ctx.tp_group))
    act = torch.float32 if fp32 else torch.bfloat16
    cast = lambda t: t.to(act)
    p, tp = case.params, case.tp
    h, d, heads, B = p['hidden'], p['head_dim'], p['q_heads']//tp, p['batch']
    chunk, offset = p['vocab']//tp, ctx.tp_rank*(p['vocab']//tp)
    device = values['token'].device

    def project(key, x):
        return cast(x)@cast(values[key]).T

    token, reset = values['token'], values['reset']
    local = (token >= offset) & (token < offset+chunk)
    prefix = torch.where(local[:, None], values['embed'][(token-offset).clamp(0, chunk-1)], 0)
    prefix = cast(sum_value(prefix))
    blocks, latent_writes, pe_writes, recurrent, conv_writes, chosen, shortfall = [], [], [], [], [], [], []
    keep = (~reset)
    for layer in range(p['layers']):
        x = f'l{layer}_'
        hidden = attention_residual(prefix, blocks, values['attention_res_norm'][layer],
                                    values['attention_res_proj'][layer], cast) if blocks else prefix
        if layer % p['attn_res_block_size'] == 0:
            blocks.append(prefix)
            prefix = None
        normalized = rms(hidden, values['ln1'][layer], eps=1e-5)
        if full_layer(case, layer):
            kr, nr, rd, vd = p['kv_lora_rank'], p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
            query = project(x+'qb', rms(project(x+'qa', normalized), values[x+'qn'], eps=1e-6)).view(B, heads, nr+rd)
            raw = project(x+'ka', normalized)
            latent = rms(raw[:, :kr], values[x+'kn'], eps=1e-6)
            pe = raw[:, kr:]
            latent_writes.append(latent)
            pe_writes.append(pe)
            weight = cast(values[x+'kb']).view(heads, nr+vd, kr)
            absorbed = torch.einsum('bhd,hdc->bhc', query[..., :nr], weight[:, :nr])
            outputs = []
            for b in range(B):
                all_latent, all_pe = latent[b:b+1], pe[b:b+1]
                if keep[b]:
                    all_latent = torch.cat((cast(values[x+'kv_cache'][b]), all_latent))
                    all_pe = torch.cat((cast(values[x+'pe_cache'][b]), all_pe))
                scores = (absorbed[b].float()@all_latent.float().T+query[b, :, nr:].float()@all_pe.float().T)/(nr+rd)**0.5
                outputs.append(cast(scores.softmax(-1))@all_latent)
            output = torch.einsum('bhc,hdc->bhd', torch.stack(outputs), weight[:, nr:]).flatten(1)
            output = output*project(x+'output_gate', normalized).sigmoid()
            attention = cast(sum_value(project(x+'o', output)))
        else:
            raw = torch.stack([project(x+key, normalized) for key in ('q', 'k', 'v')], 1)   # [B, 3, D]
            old = values[x+'conv_state']*keep[:, None, None, None]
            conv = torch.cat((cast(old[..., 1:]), raw[..., None]), -1)
            conv_writes.append(conv)
            q, k, v = cast(F.silu((conv.float()*values[x+'conv_weight'].float()).sum(-1))).view(B, 3, heads, d).unbind(1)
            gate = project(x+'fb', project(x+'fa', normalized)).view(B, heads, d)
            beta = project(x+'beta', normalized)
            state = values[x+'recurrent_state']*keep[:, None, None, None]
            output, new_state = kda_step(q, k, v, gate, beta, values[x+'A_log'],
                                         values[x+'dt_bias'].view(heads, d), state)
            recurrent.append(new_state)
            output = cast(output).float()
            output = output*torch.rsqrt(output.square().mean(-1, keepdim=True)+1e-5)*values[x+'onorm'].float()
            output = cast(output*project(x+'output_gate', normalized).view(B, heads, d).float().sigmoid())
            attention = cast(sum_value(project(x+'o', output.flatten(1))))
        prefix = attention if prefix is None else prefix+attention
        hidden = attention_residual(prefix, blocks, values['mlp_res_norm'][layer], values['mlp_res_proj'][layer], cast)
        normalized = rms(hidden, values['ln2'][layer], eps=1e-5)
        if layer < p['first_dense']:
            output = cast(sum_value(project(x+'down', cast(situ(project(x+'gate', normalized), project(x+'up', normalized))))))
            chosen.append(torch.full((B, p['topk']), -1, device=device, dtype=torch.int64))
            shortfall.append(torch.zeros(B, device=device))
        else:
            scores = (normalized.float()@values[x+'router'].float().T).sigmoid()
            biased = scores+values[x+'router_bias'].float()
            indices = biased.topk(p['topk'], -1).indices
            if routes is not None:
                indices = routes[layer].to(device=device, dtype=torch.int64)
                shortfall.append((biased.topk(p['topk'], -1).values[:, -1]-biased.gather(1, indices).min(-1).values).clamp_min(0))
            weights = scores.gather(1, indices)
            weights = weights/(weights.sum(-1, keepdim=True)+1e-20)
            chosen.append(indices.sort(-1).values)
            latent = project(x+'latent_down', normalized)
            expert_sum = torch.zeros(B, p['latent_hidden'], device=device)
            for expert in indices.unique().tolist():
                rows, slots = (indices == expert).nonzero(as_tuple=True)
                weight = lambda key: unpack_mxfp4(values[f'{x}{key}_blocks'][expert], values[f'{x}{key}_scales'][expert]).to(act)
                out = cast(cast(situ(latent[rows]@weight('gate').T, latent[rows]@weight('up').T))@weight('down').T)
                expert_sum.index_add_(0, rows, out.float()*weights[rows, slots, None])
            expert_sum = sum_value(expert_sum)
            routed = project(x+'latent_up', rms(cast(expert_sum), values[x+'latent_norm'], eps=1e-5))
            shared = sum_value(project(x+'shared_down', cast(situ(project(x+'shared_gate', normalized),
                                                                     project(x+'shared_up', normalized)))))
            output = routed+cast(shared)
        prefix = prefix+output
    hidden = attention_residual(prefix, blocks, values['output_res_norm'], values['output_res_proj'], cast)
    logits = project('lm_head', rms(hidden, values['fnorm'], eps=1e-5)).float()
    length = torch.where(reset, 1, p['context']+1).to(torch.int64)
    stack = lambda items: torch.stack(items) if items else torch.zeros(0, device=device)
    out = {'logits': logits, 'next_token': _greedy(logits, offset, ctx.tp_group, serial),
           'kv_write': stack(latent_writes), 'pe_write': stack(pe_writes),
           'recurrent_state': stack(recurrent), 'conv_state': stack(conv_writes),
           'expert_ids': torch.stack(chosen), 'cache_length': length}
    if routes is not None:
        out['route_shortfall'] = torch.stack(shortfall)
    return out


def _reduce(value, group):
    dtype = value.dtype
    accumulated = value.float().clone()
    dist.all_reduce(accumulated, group=group)
    return accumulated.to(dtype)


# ---------------------------------------------------------------- grading --

def _all_sum(value, group, serial):
    if not serial:
        dist.all_reduce(value, group=group)
    return value


def _valid_routes(case: Case, routes, expected):
    """Expert ids in range and distinct in each row; dense layers must be -1 as in the oracle."""
    p, dense = case.params, case.params['first_dense']
    if not torch.equal(routes[:dense], expected[:dense]):
        return False
    moe = routes[dense:]
    if bool(((moe < 0) | (moe >= p['experts'])).any()):
        return False
    ordered = moe.sort(-1).values
    return not bool((ordered[..., 1:] == ordered[..., :-1]).any())


def compare(case: Case, values, expected, actual, *, serial=None):
    """Oracle-band grading. Returns per-output details; raises AssertionError on failure.

    Routing is discrete, so the continuous outputs are graded on fixed routes. The grader runs
    the FP32 reference twice: on the oracle's expert sets (truth_o) and on the candidate's
    expert sets (truth_c).
    Float outputs: rel-L2(candidate, truth_c) <= BAND*rel-L2(BF16 oracle, truth_o) + FLOOR,
    for every batch row of the logits and for each other output as a whole.
    next_token: the candidate token's truth_c logit must be within the near-tie margin of the
    truth_c maximum (margin = 2x the oracle's max logit error to truth_o in that row).
    expert_ids: sets (order free), identical on all ranks, distinct, in range. Each candidate set
    must be a near-top-k set under the FP32 router scores of truth_c: the weakest chosen expert may
    be at most BAND*(the oracle's worst shortfall under truth_o) + ROUTE_FLOOR below the k-th best.
    cache_length: exact.
    """
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AssertionError(f'output keys: expected {sorted(expected)}, got '
                             f'{sorted(actual) if isinstance(actual, dict) else type(actual)}')
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f'{name}: output must be a Tensor')
        if got.shape != want.shape or got.dtype != want.dtype or got.device != want.device:
            raise AssertionError(f'{name}: expected {tuple(want.shape)} {want.dtype} on {want.device}, '
                                 f'got {tuple(got.shape)} {got.dtype} on {got.device}')
    serial = case.gpus == 1 if serial is None else serial
    group = None if serial else initialize(case).tp_group
    routes = actual['expert_ids'].long()
    # one global decision on the routes before any collective in the forced references
    ordered = routes.sort(-1).values
    checksum = (ordered*torch.arange(1, ordered.numel()+1, device=ordered.device).view(ordered.shape)).sum()
    flags = torch.tensor([0 if _valid_routes(case, routes, expected['expert_ids']) else 1,
                          int(checksum), -int(checksum), 0 if torch.equal(routes.sort(-1).values,
                                                                                    expected['expert_ids']) else 1],
                         device=routes.device, dtype=torch.int64)
    if not serial:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    if flags[0] or int(flags[1]) != -int(flags[2]):
        raise AssertionError('expert_ids: ids out of range, repeated in a row, wrong in a dense layer, '
                             'or different between ranks')
    truth_o = reference(case, values, serial=serial, fp32=True, routes=expected['expert_ids'])
    truth_c = truth_o if not flags[3] else reference(case, values, serial=serial, fp32=True, routes=routes)
    details, failures = {}, []
    for name in ('logits', 'kv_write', 'pe_write', 'recurrent_state', 'conv_state'):
        ref_o, ref_c = truth_o[name].float(), truth_c[name].float()
        oracle, got = expected[name].float(), actual[name].float()
        rows = ref_o.shape[0] if name == 'logits' else 1
        ref_o, ref_c, oracle, got = (t.reshape(rows, -1) for t in (ref_o, ref_c, oracle, got))
        if not torch.isfinite(got).all():
            failures.append(f'{name}: non-finite values')
        sums = torch.stack([(oracle-ref_o).square().sum(-1), ref_o.square().sum(-1),
                            (got-ref_c).square().sum(-1), ref_c.square().sum(-1)])
        sums = _all_sum(sums.double(), group, serial)
        oracle_err = (sums[0]/sums[1].clamp_min(1e-30)).sqrt()
        cand_err = (sums[2]/sums[3].clamp_min(1e-30)).sqrt()
        limit = BAND*oracle_err+FLOOR
        details[name] = {'candidate_rel_l2_max': float(cand_err.max()), 'oracle_rel_l2_max': float(oracle_err.max()),
                         'worst_ratio': float((cand_err/limit).max())}
        if bool((cand_err > limit).any()) or not torch.isfinite(cand_err).all():
            failures.append(f'{name}: {details[name]}')
    # next_token: near-tie rule against the FP32 logits on the candidate routes
    ref = truth_c['logits'].float()
    chunk = ref.shape[-1]
    offset = 0 if serial else initialize(case).tp_rank*chunk
    stats = torch.stack([ref.max(-1).values, (expected['logits'].float()-truth_o['logits'].float()).abs().max(-1).values])
    if not serial:
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
    token = actual['next_token'].long()
    mine = (token >= offset) & (token < offset+chunk)
    picked = torch.where(mine, ref.gather(1, (token-offset).clamp(0, chunk-1)[:, None])[:, 0], -torch.inf)
    if not serial:
        dist.all_reduce(picked, op=dist.ReduceOp.MAX, group=group)
    bad = picked < stats[0]-2*stats[1]
    details['next_token'] = {'rows_outside_tie_margin': int(bad.sum()),
                             'rows_differing_from_fp32_argmax': int((token != truth_c['next_token']).sum())}
    if bad.any():
        failures.append(f'next_token: {details["next_token"]}')
    # expert routes: near-top-k under the FP32 router scores
    oracle_short = float(truth_o['route_shortfall'].max()) if truth_o['route_shortfall'].numel() else 0.0
    cand_short = truth_c['route_shortfall']
    allowed = BAND*oracle_short+ROUTE_FLOOR
    details['expert_ids'] = {'candidate_max_shortfall': float(cand_short.max()) if cand_short.numel() else 0.0,
                             'oracle_max_shortfall': oracle_short, 'allowed': allowed,
                             'sets_differing_from_oracle': int((routes.sort(-1).values != expected['expert_ids']).any(-1).sum())}
    if cand_short.numel() and bool((cand_short > allowed).any()):
        failures.append(f'expert_ids: {details["expert_ids"]}')
    mismatch = int((actual['cache_length'] != expected['cache_length']).sum())
    details['cache_length'] = {'mismatched_elements': mismatch}
    if mismatch:
        failures.append(f'cache_length: {mismatch} mismatches')
    if failures:
        raise AssertionError('; '.join(failures))
    return details
