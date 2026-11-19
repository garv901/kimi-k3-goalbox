"""CPU-only, read-only check that convert() preserved the checkpoint: for every fixture tensor of the converted
layers, the concatenation of the per-rank shards along the documented split dimension is bit-identical to the
Hugging Face source tensor (after the documented re-layout: [1,h] -> [h], A_log[:heads], the stacked q/k/v
convolution taps, the MXFP4 [rows, cols/2] -> [rows, cols/32, 16] view). Unsplit tensors must be identical on
every rank. The source tensors are read directly from the HF shards with safetensors; the fixture-key table below
is written from the converter's documented name map, not imported from it, so that a bug in the converter's
split code would show up as a mismatch.

Usage: PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES="" python -s planning/bringup/verify_conversion.py
           --hf /mnt/shared/weights/kimi-k3 --out <converted root> --tp 4 --layers 4 [--experts all|N]
Exit 0 only if every tensor matched. Prints one line per tensor class and a JSON summary."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

P = 'language_model.model.'
HEADS = 96
MLA_LAYERS = {3 + 4*i for i in range(23)} | {92}   # HF full_attn_layers (1-indexed 4,8,...,92,93) minus one


def hf_reader(directory: Path):
    index = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    handles = {}

    def read(name: str) -> torch.Tensor:
        file = index[name]
        if file not in handles:
            handles[file] = safe_open(str(directory/file), 'pt')
        return handles[file].get_tensor(name)
    return read


def layer_table(layer: int, experts: int):
    """fixture key -> (split dim or None, callable(read) -> source tensor)"""
    L = f'{P}layers.{layer}.'
    a = L + 'self_attn.'
    t = {
        'ln1': (None, lambda r: r(L+'input_layernorm.weight')),
        'ln2': (None, lambda r: r(L+'post_attention_layernorm.weight')),
        'attention_res_norm': (None, lambda r: r(L+'self_attention_res_norm.weight')),
        'mlp_res_norm': (None, lambda r: r(L+'mlp_res_norm.weight')),
        'attention_res_proj': (None, lambda r: r(L+'self_attention_res_proj.weight')[0]),
        'mlp_res_proj': (None, lambda r: r(L+'mlp_res_proj.weight')[0]),
        'output_gate': (0, lambda r: r(a+'g_proj.weight')),
        'o': (1, lambda r: r(a+'o_proj.weight')),
    }
    if layer in MLA_LAYERS:
        t |= {'qa': (None, lambda r: r(a+'q_a_proj.weight')), 'qn': (None, lambda r: r(a+'q_a_layernorm.weight')),
              'qb': (0, lambda r: r(a+'q_b_proj.weight')), 'ka': (None, lambda r: r(a+'kv_a_proj_with_mqa.weight')),
              'kn': (None, lambda r: r(a+'kv_a_layernorm.weight')), 'kb': (0, lambda r: r(a+'kv_b_proj.weight'))}
    else:
        t |= {'q': (0, lambda r: r(a+'q_proj.weight')), 'k': (0, lambda r: r(a+'k_proj.weight')),
              'v': (0, lambda r: r(a+'v_proj.weight')), 'fa': (None, lambda r: r(a+'f_a_proj.weight')),
              'fb': (0, lambda r: r(a+'f_b_proj.weight')), 'beta': (0, lambda r: r(a+'b_proj.weight')),
              'onorm': (None, lambda r: r(a+'o_norm.weight').float()),
              'A_log': (0, lambda r: r(a+'A_log')[:HEADS].float()),
              'dt_bias': (0, lambda r: r(a+'dt_bias').float()),
              'conv_weight': (1, lambda r: torch.stack([r(a+f'{k}_conv1d.weight')[:, 0, :] for k in 'qkv']).float())}
    if layer < 1:  # first_dense
        t |= {'gate': (0, lambda r: r(L+'mlp.gate_proj.weight')), 'up': (0, lambda r: r(L+'mlp.up_proj.weight')),
              'down': (1, lambda r: r(L+'mlp.down_proj.weight'))}
    else:
        m = L + 'block_sparse_moe.'
        t |= {'router': (None, lambda r: r(m+'gate.weight')),
              'router_bias': (None, lambda r: r(m+'gate.e_score_correction_bias').float()),
              'latent_down': (None, lambda r: r(m+'routed_expert_down_proj.weight')),
              'latent_up': (None, lambda r: r(m+'routed_expert_up_proj.weight')),
              'latent_norm': (None, lambda r: r(m+'routed_expert_norm.weight')),
              'shared_gate': (0, lambda r: r(m+'shared_experts.gate_proj.weight')),
              'shared_up': (0, lambda r: r(m+'shared_experts.up_proj.weight')),
              'shared_down': (1, lambda r: r(m+'shared_experts.down_proj.weight'))}

        def packed(w, r):
            stacked = torch.stack([r(f'{m}experts.{e}.{w}.weight_packed') for e in range(experts)])
            E, rows, half = stacked.shape
            return stacked.view(E, rows, half//16, 16)

        def scales(w, r):
            return torch.stack([r(f'{m}experts.{e}.{w}.weight_scale') for e in range(experts)])
        # gate (w1) and up (w3) are row-split: dim 0 of the matrix = dim 1 of [E, rows, ...]; down (w2) column-split
        t |= {'gate_blocks': (1, lambda r: packed('w1', r)), 'up_blocks': (1, lambda r: packed('w3', r)),
              'down_blocks': (2, lambda r: packed('w2', r)),
              'gate_scales': (1, lambda r: scales('w1', r)), 'up_scales': (1, lambda r: scales('w3', r)),
              'down_scales': (2, lambda r: scales('w2', r))}
    return t


def global_table():
    return {'embed': (0, lambda r: r(P+'embed_tokens.weight')), 'lm_head': (0, lambda r: r('language_model.lm_head.weight')),
            'fnorm': (None, lambda r: r(P+'norm.weight')), 'output_res_norm': (None, lambda r: r(P+'output_attn_res_norm.weight')),
            'output_res_proj': (None, lambda r: r(P+'output_attn_res_proj.weight')[0])}


def check_file(name: str, table, ranks: list[dict], read, experts_limit: int | None, summary: dict) -> bool:
    ok = True
    keys = set(table)
    for r, tensors in enumerate(ranks):
        if set(tensors) != keys:
            print(f'{name} rank{r}: key set differs: missing {sorted(keys-set(tensors))[:5]} extra {sorted(set(tensors)-keys)[:5]}')
            ok = False
    for key, (dim, source) in sorted(table.items()):
        src = source(read)
        if dim is None:
            same = all(torch.equal(tensors[key], src) and tensors[key].dtype == src.dtype for tensors in ranks)
        else:
            shards = [tensors[key] for tensors in ranks]
            if experts_limit is not None and key.endswith(('_blocks', '_scales')):
                shards = [s[:experts_limit] for s in shards]
                src = src[:experts_limit]
            joined = torch.cat(shards, dim=dim)
            same = tuple(joined.shape) == tuple(src.shape) and joined.dtype == src.dtype and torch.equal(joined, src)
        summary['tensors'] += 1
        summary['bytes'] += src.numel()*src.element_size()
        if not same:
            ok = False
            summary['mismatches'].append(f'{name}:{key}')
            print(f'MISMATCH {name} {key} fixture {[tuple(t[key].shape) for t in ranks]} {ranks[0][key].dtype} '
                  f'source {tuple(src.shape)} {src.dtype}')
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tp', type=int, required=True); ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--experts', default='all', help="'all' or the number of experts to compare per MoE layer")
    args = ap.parse_args()
    read = hf_reader(Path(args.hf))
    limit = None if args.experts == 'all' else int(args.experts)
    root = Path(args.out)/f'tp{args.tp}'
    summary = {'hf': args.hf, 'out': args.out, 'tp': args.tp, 'layers': args.layers, 'experts': args.experts,
               'tensors': 0, 'bytes': 0, 'mismatches': [], 'files': []}
    ok = True
    t0 = time.time()
    for layer in [-1] + list(range(args.layers)):
        name = 'global.safetensors' if layer < 0 else f'layer{layer:03d}.safetensors'
        ranks = [load_file(str(root/f'rank{r}'/name)) for r in range(args.tp)]
        table = global_table() if layer < 0 else layer_table(layer, 896)
        good = check_file(name, table, ranks, read, limit, summary)
        ok &= good
        print(f'{name}: {"OK" if good else "FAILED"} ({len(table)} tensors, {time.time()-t0:.0f} s)', flush=True)
        summary['files'].append({'file': name, 'ok': good, 'tensors': len(table)})
        del ranks
    summary['ok'] = ok
    summary['seconds'] = round(time.time()-t0, 1)
    print(json.dumps(summary))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
