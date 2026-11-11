"""Real Kimi K3 weights in the fixture layout.

convert(): read the Hugging Face checkpoint moonshotai/Kimi-K3 (compressed-tensors,
MXFP4 routed experts) and write one directory of safetensors files per TP rank:
    <out>/tp<TP>/rank<R>/global.safetensors
    <out>/tp<TP>/rank<R>/layer<NNN>.safetensors
load_weights(): read those files for one rank and return exactly the weight keys of
kimi.weight_shapes(case) (the first case.params['layers'] layers).

Name map (HF prefix language_model.model., per layer layers.N.):
  embed_tokens.weight                         embed            row split (vocab)
  language_model.lm_head.weight               lm_head          row split (vocab)
  norm.weight                                 fnorm
  output_attn_res_norm / _proj.weight         output_res_norm / output_res_proj ([1,h] -> [h])
  input_layernorm / post_attention_layernorm  ln1 / ln2        (stacked over layers)
  self_attention_res_norm / _proj             attention_res_norm / attention_res_proj
  mlp_res_norm / mlp_res_proj                 mlp_res_norm / mlp_res_proj
  MLA (layers 3,7,...,91,92):
  self_attn.q_a_proj, q_a_layernorm           qa, qn
  self_attn.q_b_proj                          qb               row split (heads x 192)
  self_attn.kv_a_proj_with_mqa, kv_a_layernorm ka, kn
  self_attn.kv_b_proj                         kb               row split (heads x 256: k_nope | v)
  self_attn.g_proj                            output_gate      row split
  self_attn.o_proj                            o                column split
  KDA (all other layers):
  self_attn.{q,k,v}_proj, g_proj              q, k, v, output_gate   row split
  self_attn.o_proj                            o                column split
  self_attn.f_a_proj / f_b_proj               fa / fb (fb row split)
  self_attn.b_proj                            beta             row split (heads)
  self_attn.o_norm.weight (F32)               onorm
  self_attn.A_log (F32 [128], first 96 used)  A_log            split (heads)
  self_attn.dt_bias (F32)                     dt_bias          split
  self_attn.{q,k,v}_conv1d.weight [D,1,4] F32 conv_weight [3, D/TP, 4]
  Dense FFN (layer 0): mlp.{gate,up}_proj (row split), mlp.down_proj (column split)
  Latent MoE (layers >= 1):
  block_sparse_moe.gate.weight / .e_score_correction_bias   router / router_bias
  block_sparse_moe.routed_expert_down_proj / _up_proj / _norm  latent_down / latent_up / latent_norm
  block_sparse_moe.experts.E.w1 (gate) / w3 (up)   {gate,up}_blocks/_scales  row split
  block_sparse_moe.experts.E.w2 (down)             down_blocks/_scales       column split
      weight_packed U8 [rows, cols/2] (E2M1, low nibble = even column) -> blocks [rows, cols/32, 16]
      weight_scale  U8 [rows, cols/32] (E8M0, bias 127)                 -> scales
  block_sparse_moe.shared_experts.{gate,up}_proj (row split), down_proj (column split)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

REPO = 'moonshotai/Kimi-K3'
REVISION = 'f831ab66814297da540d832a5235f8e904f29d06'
P = 'language_model.model.'


class _Reader:
    def __init__(self, directory: str):
        from safetensors import safe_open
        self.dir = Path(directory)
        self.map = json.loads((self.dir/'model.safetensors.index.json').read_text())['weight_map']
        self._open, self._safe_open = {}, safe_open

    def __call__(self, name: str) -> torch.Tensor:
        file = self.map[name]
        if file not in self._open:
            self._open[file] = self._safe_open(str(self.dir/file), 'pt')
        return self._open[file].get_tensor(name)


def _split(t: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    n = t.shape[dim]//tp
    return t.narrow(dim, rank*n, n).contiguous()


def _layer_tensors(read, case, layer: int, tp: int) -> list[dict[str, torch.Tensor]]:
    """All fixture tensors of one layer, for every rank: list indexed by rank."""
    from .kimi import full_layer
    p = case.params
    heads = p['q_heads']
    L = f'{P}layers.{layer}.'
    out = [{} for _ in range(tp)]

    def put(key, tensor, dim=None, dtype=None):
        if dtype is not None:
            tensor = tensor.to(dtype)
        for r in range(tp):
            out[r][key] = tensor.contiguous() if dim is None else _split(tensor, dim, r, tp)

    for key, name in (('ln1', 'input_layernorm'), ('ln2', 'post_attention_layernorm'),
                      ('attention_res_norm', 'self_attention_res_norm'), ('mlp_res_norm', 'mlp_res_norm')):
        put(key, read(L+name+'.weight'))
    put('attention_res_proj', read(L+'self_attention_res_proj.weight')[0])
    put('mlp_res_proj', read(L+'mlp_res_proj.weight')[0])
    a = L+'self_attn.'
    if full_layer(case, layer):
        put('qa', read(a+'q_a_proj.weight'))
        put('qn', read(a+'q_a_layernorm.weight'))
        put('qb', read(a+'q_b_proj.weight'), 0)
        put('ka', read(a+'kv_a_proj_with_mqa.weight'))
        put('kn', read(a+'kv_a_layernorm.weight'))
        put('kb', read(a+'kv_b_proj.weight'), 0)
        put('output_gate', read(a+'g_proj.weight'), 0)
        put('o', read(a+'o_proj.weight'), 1)
    else:
        for key in ('q', 'k', 'v'):
            put(key, read(a+f'{key}_proj.weight'), 0)
        put('output_gate', read(a+'g_proj.weight'), 0)
        put('o', read(a+'o_proj.weight'), 1)
        put('fa', read(a+'f_a_proj.weight'))
        put('fb', read(a+'f_b_proj.weight'), 0)
        put('beta', read(a+'b_proj.weight'), 0)
        put('onorm', read(a+'o_norm.weight'), dtype=torch.float32)
        put('A_log', read(a+'A_log')[:heads], 0, torch.float32)
        put('dt_bias', read(a+'dt_bias'), 0, torch.float32)
        conv = torch.stack([read(a+f'{key}_conv1d.weight')[:, 0, :] for key in ('q', 'k', 'v')])
        put('conv_weight', conv, 1, torch.float32)
    if layer < p['first_dense']:
        put('gate', read(L+'mlp.gate_proj.weight'), 0)
        put('up', read(L+'mlp.up_proj.weight'), 0)
        put('down', read(L+'mlp.down_proj.weight'), 1)
    else:
        m = L+'block_sparse_moe.'
        put('router', read(m+'gate.weight'))
        put('router_bias', read(m+'gate.e_score_correction_bias'), dtype=torch.float32)
        put('latent_down', read(m+'routed_expert_down_proj.weight'))
        put('latent_up', read(m+'routed_expert_up_proj.weight'))
        put('latent_norm', read(m+'routed_expert_norm.weight'))
        put('shared_gate', read(m+'shared_experts.gate_proj.weight'), 0)
        put('shared_up', read(m+'shared_experts.up_proj.weight'), 0)
        put('shared_down', read(m+'shared_experts.down_proj.weight'), 1)
        for key, w, dim in (('gate', 'w1', 0), ('up', 'w3', 0), ('down', 'w2', 1)):
            packed = torch.stack([read(f'{m}experts.{e}.{w}.weight_packed') for e in range(p['experts'])])
            scales = torch.stack([read(f'{m}experts.{e}.{w}.weight_scale') for e in range(p['experts'])])
            E, rows, half = packed.shape
            blocks = packed.view(E, rows, half//16, 16)
            put(key+'_blocks', blocks, dim+1)
            put(key+'_scales', scales, dim+1)
    return out


def _global_tensors(read, tp: int) -> list[dict[str, torch.Tensor]]:
    out = [{} for _ in range(tp)]
    embed, head = read(P+'embed_tokens.weight'), read('language_model.lm_head.weight')
    for r in range(tp):
        out[r] = {'embed': _split(embed, 0, r, tp), 'lm_head': _split(head, 0, r, tp),
                  'fnorm': read(P+'norm.weight'), 'output_res_norm': read(P+'output_attn_res_norm.weight'),
                  'output_res_proj': read(P+'output_attn_res_proj.weight')[0].contiguous()}
    return out


def _case(layers: int, tp: int):
    from ..cases import select_cases
    base = select_cases('all', ['kimi-k3-tp8-b1-s8192'])[0]
    return replace(base, gpus=tp, tp=tp, params=base.params | {'layers': layers})


def _convert_one(args):
    src, out, tp, layer, total = args
    from safetensors.torch import save_file
    case = _case(93, tp)  # layer types follow the real 93-layer model
    read = _Reader(src)
    pieces = _global_tensors(read, tp) if layer < 0 else _layer_tensors(read, case, layer, tp)
    name = 'global.safetensors' if layer < 0 else f'layer{layer:03d}.safetensors'
    for r, tensors in enumerate(pieces):
        directory = Path(out)/f'tp{tp}'/f'rank{r}'
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(directory/(name+'.tmp')))
        os.replace(directory/(name+'.tmp'), directory/name)
    return name


def convert(src: str, out: str, tp: int, layers: int = 93, workers: int = 4) -> None:
    todo = [-1]+list(range(layers))
    done = {p.name for p in (Path(out)/f'tp{tp}'/f'rank{tp-1}').glob('*.safetensors')}
    jobs = [(src, out, tp, layer, layers) for layer in todo
            if ('global.safetensors' if layer < 0 else f'layer{layer:03d}.safetensors') not in done]
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_convert_one, jobs):
            print('wrote', name, flush=True)
    (Path(out)/f'tp{tp}'/'DONE.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION,
                                                             'tp': tp, 'layers': layers}))


def load_weights(case, root: str, rank: int, device: str) -> dict[str, torch.Tensor]:
    """Fixture weight dict for one rank, from convert() output."""
    from safetensors.torch import load_file
    from .kimi import weight_shapes
    directory = Path(root)/f'tp{case.tp}'/f'rank{rank}'
    if not (directory/'global.safetensors').exists():
        raise FileNotFoundError(f'{directory}: no converted weights; run setup/convert_checkpoint.py --tp {case.tp}')
    values = load_file(str(directory/'global.safetensors'), device=str(device))
    stacked = {key: [] for key in ('ln1', 'ln2', 'attention_res_norm', 'mlp_res_norm',
                                    'attention_res_proj', 'mlp_res_proj')}
    for layer in range(case.params['layers']):
        tensors = load_file(str(directory/f'layer{layer:03d}.safetensors'), device=str(device))
        for key, tensor in tensors.items():
            if key in stacked:
                stacked[key].append(tensor)
            else:
                values[f'l{layer}_{key}'] = tensor
    values |= {key: torch.stack(items) for key, items in stacked.items()}
    expected = weight_shapes(case)
    if set(values) != set(expected):
        raise ValueError(f'converted weights do not match the fixture: missing {sorted(set(expected)-set(values))[:5]}, '
                         f'extra {sorted(set(values)-set(expected))[:5]}')
    for key, (shape, dtype, _, _) in expected.items():
        if tuple(values[key].shape) != shape or values[key].dtype != dtype:
            raise ValueError(f'{key}: converted {tuple(values[key].shape)} {values[key].dtype}, fixture {shape} {dtype}')
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--src', required=True, help='HF snapshot directory of moonshotai/Kimi-K3')
    parser.add_argument('--out', required=True, help='output directory (about 1.6 TB for 93 layers)')
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--layers', type=int, default=93, help='convert only the first N layers (development)')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    convert(args.src, args.out, args.tp, args.layers, args.workers)


if __name__ == '__main__':
    main()
