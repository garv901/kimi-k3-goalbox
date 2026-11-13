"""CPU-only, read-only: per-rank tensor inventory of every case from the evaluator's own geometry functions.

For each case: weight tensors (kimi.weight_shapes) and per-sequence state (the shapes kimi.make_state allocates),
their byte totals, how many take the host-copy snapshot path (<= 64 MiB, or size not a multiple of 4) versus the
device-checksum path, the largest tensors, the largest FP32 cast the FP32 reference makes of one weight, the FP32
logits size, and megabench.sol's HBM floor. Nothing is allocated at model size. Run with PYTHONPATH=<repo>,
CUDA_VISIBLE_DEVICES="". Output: JSON on stdout (planning/evidence/case_sizes.json)."""
import json

import torch

from megabench import sol
from megabench.cases import CASES
from megabench.tasks import kimi

LIMIT = 1 << 26


def nbytes(shape, dtype):
    n = 1
    for s in shape:
        n *= s
    return n * torch.tensor([], dtype=dtype).element_size()


out = {}
for case in CASES:
    shapes = kimi.weight_shapes(case)
    weights = [(k, nbytes(s, d)) for k, (s, d, _, _) in shapes.items()]
    p, tp, B = case.params, case.tp, case.params['batch']
    qdim, heads, d = p['q_heads'] * p['head_dim'] // tp, p['q_heads'] // tp, p['head_dim']
    state = [('token', B * 8), ('reset', B * 1)]
    for layer in range(p['layers']):
        if kimi.full_layer(case, layer):
            state += [(f'l{layer}_kv_cache', B * p['context'] * p['kv_lora_rank'] * 2),
                      (f'l{layer}_pe_cache', B * p['context'] * p['qk_rope_dim'] * 2)]
        else:
            state += [(f'l{layer}_conv_state', B * 3 * qdim * p['conv_kernel'] * 2),
                      (f'l{layer}_recurrent_state', B * heads * d * d * 4)]
    every = weights + state
    host = [(k, b) for k, b in every if b <= LIMIT or b % 4]
    device = [(k, b) for k, b in every if not (b <= LIMIT or b % 4)]
    estimate = sol.estimate_case(case)
    fp32_largest = max(nbytes(s, torch.float32) for k, (s, dt, _, _) in shapes.items() if dt == torch.bfloat16)
    out[case.id] = dict(
        gpus=case.gpus, tp=tp, layers=p['layers'], batch=B, checkpoint=bool(p.get('checkpoint')),
        weight_tensors=len(weights), weight_bytes=sum(b for _, b in weights),
        state_tensors=len(state), state_bytes=sum(b for _, b in state),
        host_snapshot_tensors=len(host), host_snapshot_bytes=sum(b for _, b in host),
        checksum_tensors=len(device), checksum_bytes=sum(b for _, b in device),
        checksum_int32_words=sum(b for _, b in device) // 4,
        largest_tensors=[(k, b) for k, b in sorted(every, key=lambda x: -x[1])[:4]],
        largest_fp32_weight_cast_bytes=fp32_largest, logits_fp32_bytes=B * p['vocab'] // tp * 4,
        hbm_floor_ms=estimate['hbm_floor_ms'], sol_minimum_bytes=estimate['minimum_bytes'],
        sol_components_bytes={c['name']: c['bytes'] for c in estimate['components']})
print(json.dumps(out, indent=1))
