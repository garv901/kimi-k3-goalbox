"""Check the MegaBench Kimi K3 oracle against the official Hugging Face modeling code.

Part 1 (kernels): compare the oracle's KDA step, short convolution and gated output norm
with the flash-linear-attention (fla) kernels that modeling_kimi_linear.py calls.
Part 2 (model): load the first 4 real layers (dense layer 0, KDA 1-2, MLA 3) through the
converted-weight loader, run 2 decode steps for B sequences with the oracle, and run the
same tokens through modeling_kimi_linear.KimiLinearForCausalLM truncated to 4 layers.
Report logits rel-L2 and top-1 agreement for each step.

Needs: transformers==4.56.2, fla-core, einops (a private venv), the HF snapshot
(--hf) with shards 1-4 and 94, the HF source files (--hf-src), and the converted TP1
weights of 4 layers (--weights, from kimi_checkpoint --tp 1 --layers 4). One B200/B300.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from megabench.cases import select_cases  # noqa: E402
from megabench.tasks import kimi  # noqa: E402
from megabench.tasks.quantization import unpack_mxfp4  # noqa: E402


def rel(a, b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30))


def part1(dev):
    from fla.ops.kda import fused_recurrent_kda
    from fla.modules import ShortConvolution, FusedRMSNormGated
    torch.manual_seed(0)
    B, H, D = 3, 4, 128
    q, k, v, g = (torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16) for _ in range(4))
    beta = torch.randn(B, 1, H, device=dev)
    A_log, dt = torch.randn(H, device=dev)*0.5, torch.randn(H*D, device=dev)-4
    state = torch.randn(B, H, D, D, device=dev)*0.05
    o, s = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt, initial_state=state.clone(),
                               output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                               use_beta_sigmoid_in_kernel=True, lower_bound=-5.0, transpose_state_layout=True)
    ro, rs = kimi.kda_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], A_log, dt.view(H, D), state)
    out = {'kda_output_rel': rel(o[:, 0], ro), 'kda_state_rel': rel(s, rs)}
    # short convolution, decode step with a cache
    Dm, K = 256, 4
    conv = ShortConvolution(hidden_size=Dm, kernel_size=K, activation='silu').to(dev)
    x = torch.randn(B, 1, Dm, device=dev, dtype=torch.bfloat16)
    cache = torch.randn(B, Dm, K, device=dev, dtype=torch.bfloat16)
    y, new_cache = conv(x=x, cache=cache.clone(), output_final_state=True)
    window = torch.cat((cache[..., 1:], x[:, 0, :, None]), -1)
    ry = torch.nn.functional.silu((window.float()*conv.weight[:, 0, :].float()).sum(-1))
    out |= {'conv_output_rel': rel(y[:, 0], ry), 'conv_state_equal': bool(torch.equal(new_cache, window))}
    # gated output norm, sigmoid activation
    norm = FusedRMSNormGated(D, eps=1e-5, activation='sigmoid').to(dev)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(D, device=dev))
    oo, gg = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16), torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    ref = oo.float()*torch.rsqrt(oo.float().square().mean(-1, keepdim=True)+1e-5)*norm.weight.float()*gg.float().sigmoid()
    out['onorm_rel'] = rel(norm(oo, gg), ref)
    return out


def hf_model(src: Path, hf: Path, layers: int, experts_needed: dict[int, set[int]], dev):
    """modeling_kimi_linear with `layers` layers; only the needed experts get weights."""
    pkg = 'kimi_hf_src'
    spec = importlib.util.spec_from_file_location(pkg, src/'__init__.py', submodule_search_locations=[str(src)])
    (src/'__init__.py').touch()
    sys.modules[pkg] = importlib.util.module_from_spec(spec)
    cfgmod = importlib.import_module(f'{pkg}.configuration_kimi_k3')
    model_mod = importlib.import_module(f'{pkg}.modeling_kimi_linear')
    text = json.loads((hf/'config.json').read_text())['text_config']
    text = {k: v for k, v in text.items() if k not in ('auto_map', 'architectures', 'quantization_config')}
    config = cfgmod.KimiLinearConfig(**text | {'num_hidden_layers': layers})
    with torch.device('meta'):
        model = model_mod.KimiLinearForCausalLM(config)
    model.config._attn_implementation = 'eager'
    from safetensors import safe_open
    index = json.loads((hf/'model.safetensors.index.json').read_text())['weight_map']
    handles = {}

    def read(name):
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(str(hf/f), 'pt', device='cpu')
        return handles[f].get_tensor(name)

    state = {}
    for name, param in list(model.named_parameters()):
        if '.experts.' in name:
            layer, expert = int(name.split('.layers.')[1].split('.')[0]), int(name.split('.experts.')[1].split('.')[0])
            if expert not in experts_needed.get(layer, ()):
                continue
            base = 'language_model.'+name.rsplit('.weight', 1)[0]
            blocks = read(base+'.weight_packed')
            weight = unpack_mxfp4(blocks.view(blocks.shape[0], -1, 16), read(base+'.weight_scale'))
            state[name] = weight.to(torch.bfloat16)
            continue
        tensor = read('language_model.'+name)
        if name.endswith('A_log'):
            tensor = tensor[:param.shape[0]]
        state[name] = tensor.reshape(param.shape).to(param.dtype if param.dtype != torch.float32 else tensor.dtype)
    for name, tensor in state.items():
        module_name, attr = name.rsplit('.', 1)
        module = model.get_submodule(module_name)
        setattr(module, attr, torch.nn.Parameter(tensor.to(dev), requires_grad=False))
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f'meta buffer {name}')
    return model.eval(), model_mod


def part2(args, dev):
    case = select_cases('all', ['kimi-k3-dev-l4-tp1-b8-s8192'])[0]
    B = args.batch
    case = replace(case, params=case.params | {'checkpoint': 1, 'batch': B, 'context': 1})
    import os
    os.environ[kimi.WEIGHTS_ENV] = args.weights
    weights = kimi.checkpoint_weights(case, dev, 0)
    # compressed-tensors decode of one expert vs the fixture decode
    report = {}
    try:
        from compressed_tensors.compressors.quantized_compressors.fp4_quantized import unpack_fp4_from_uint8
        from safetensors import safe_open
        index = json.loads((Path(args.hf)/'model.safetensors.index.json').read_text())['weight_map']
        n = 'language_model.model.layers.1.block_sparse_moe.experts.5.w2.weight_packed'
        packed = safe_open(str(Path(args.hf)/index[n]), 'pt').get_tensor(n)
        codes = unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1]*2, dtype=torch.float32)
        mine = unpack_mxfp4(packed.view(packed.shape[0], -1, 16), torch.full((packed.shape[0], packed.shape[1]//16), 127, dtype=torch.uint8))
        report['compressed_tensors_fp4_decode_equal'] = bool(torch.equal(codes, mine))
    except Exception as exc:  # noqa: BLE001
        report['compressed_tensors_fp4_decode_equal'] = f'not checked: {type(exc).__name__}: {exc}'
    generator = torch.Generator(device=dev).manual_seed(1)
    tokens = torch.randint(0, case.params['vocab'], (2, B), generator=generator, device=dev)
    # step 1: fresh sequences
    state = kimi.make_state(case, 11, dev, 0)
    state['token'], state['reset'] = tokens[0], torch.ones(B, dtype=torch.bool, device=dev)
    out1 = kimi.reference(case, weights | state, serial=True)
    f1 = kimi.reference(case, weights | state, serial=True, fp32=True)
    # step 2: carry the state written in step 1
    state2 = dict(state, token=tokens[1], reset=torch.zeros(B, dtype=torch.bool, device=dev))
    for i, layer in enumerate(kimi.mla_layers(case)):
        state2[f'l{layer}_kv_cache'] = out1['kv_write'][i][:, None]
        state2[f'l{layer}_pe_cache'] = out1['pe_write'][i][:, None]
    for i, layer in enumerate(kimi.kda_layers(case)):
        state2[f'l{layer}_conv_state'] = out1['conv_state'][i]
        state2[f'l{layer}_recurrent_state'] = out1['recurrent_state'][i]
    out2 = kimi.reference(case, weights | state2, serial=True)
    needed = {}
    for out in (out1, out2):
        for layer in range(1, 4):
            needed.setdefault(layer, set()).update(out['expert_ids'][layer].flatten().tolist())
    del weights
    torch.cuda.empty_cache()
    model, model_mod = hf_model(Path(args.hf_src), Path(args.hf), 4, needed, dev)
    with torch.no_grad():
        cache = model_mod.KimiDynamicCache(model.config)
        h1 = model(input_ids=tokens[0][:, None], past_key_values=cache, use_cache=True).logits[:, -1].float()
        h2 = model(input_ids=tokens[1][:, None], past_key_values=cache, use_cache=True).logits[:, -1].float()
    report |= {
        'step1_logits_rel_l2_oracle_vs_hf': rel(out1['logits'], h1),
        'step1_logits_rel_l2_fp32_vs_hf': rel(f1['logits'], h1),
        'step1_top1_agree': int((out1['logits'].argmax(-1) == h1.argmax(-1)).sum()), 'batch': B,
        'step2_logits_rel_l2_oracle_vs_hf': rel(out2['logits'], h2),
        'step2_top1_agree': int((out2['logits'].argmax(-1) == h2.argmax(-1)).sum()),
        'experts_materialized_for_hf': {k: len(v) for k, v in needed.items()},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--hf', required=True, help='HF snapshot directory (config, index, shards 1-4 and 94)')
    parser.add_argument('--hf-src', required=True, help='directory with modeling_kimi_linear.py and configuration_kimi_k3.py')
    parser.add_argument('--weights', required=True, help='converted weights root (tp1, 4 layers)')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    dev = 'cuda'
    print(json.dumps({'part1_kernels': part1(dev)}, indent=1), flush=True)
    with torch.inference_mode(False), torch.no_grad():
        print(json.dumps({'part2_model': part2(args, dev)}, indent=1), flush=True)


if __name__ == '__main__':
    main()
