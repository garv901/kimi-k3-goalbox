"""Milestone 1 (GPU, needs its own GO): load the converted real weights of one TP rank per GPU through the
evaluator's own loader (megabench.tasks.kimi_checkpoint.load_weights, which also validates names, shapes and
dtypes against kimi.weight_shapes), record load time and device memory, then free everything. No process group,
no oracle step, no candidate. One process per GPU via torch.multiprocessing.

Usage: KIMI_K3_WEIGHTS=<converted root> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=<repo> \
         python -s planning/bringup/tp4_real_load_only.py --layers 56 --tp 4 --out <dir> [--device cpu]
--device cpu (CPU smoke test of the same code path, host memory instead of GPU memory) needs no GPU and no GO.
Writes <dir>/load-rank<N>.json per rank and <dir>/load-summary.json; exit 0 only if every rank loaded and validated."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import torch.multiprocessing as mp


def bringup_case(layers: int, tp: int, batch: int = 1):
    from megabench.cases import select_cases
    base = select_cases('all', ['kimi-k3-tp8-b1-s8192'])[0]
    return replace(base, id=f'kimi-k3-real-l{layers}-tp{tp}-b{batch}-s8192-bringup', suite='p3', gpus=tp, tp=tp,  # p3 = planned, the only non-graded suite the Case validator accepts
                   params=base.params | {'layers': layers, 'batch': batch},
                   note=f'Bring-up only (suite p3 = planned): first {layers} real layers at TP{tp}; not a graded case.')


def _rank(rank: int, layers: int, tp: int, out: str, device_kind: str) -> None:
    import resource
    from megabench.tasks.kimi_checkpoint import load_weights
    cuda = device_kind == 'cuda'
    device = f'cuda:{rank}' if cuda else 'cpu'
    record = {'rank': rank, 'device': device, 'status': 'error'}
    try:
        if cuda:
            torch.cuda.set_device(rank)
            free0, total = torch.cuda.mem_get_info(rank)
            record |= {'gpu_total_bytes': total, 'gpu_free_before_bytes': free0,
                       'gpu_name': torch.cuda.get_device_name(rank)}
        case = bringup_case(layers, tp)
        root = os.environ['KIMI_K3_WEIGHTS']
        t0 = time.time()
        values = load_weights(case, root, rank, device)
        if cuda:
            torch.cuda.synchronize(rank)
        record['load_seconds'] = round(time.time() - t0, 1)
        record['tensors'] = len(values)
        record['weight_bytes'] = sum(t.numel()*t.element_size() for t in values.values())
        if cuda:
            record['memory_allocated_bytes'] = torch.cuda.memory_allocated(rank)
            record['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(rank)
            record['max_memory_reserved_bytes'] = torch.cuda.max_memory_reserved(rank)
            record['gpu_free_after_load_bytes'] = torch.cuda.mem_get_info(rank)[0]
        record['max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record['sample'] = {k: [list(values[k].shape), str(values[k].dtype)] for k in ('embed', 'lm_head', 'l1_gate_blocks', 'l3_kb')
                            if k in values}
        del values
        if cuda:
            torch.cuda.empty_cache()
            record['gpu_free_after_release_bytes'] = torch.cuda.mem_get_info(rank)[0]
        record['status'] = 'pass'
    except Exception as exc:  # recorded, not re-raised: the parent reads the per-rank files
        record |= {'reason': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(limit=8)}
    Path(out, f'load-rank{rank}.json').write_text(json.dumps(record, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, required=True); ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--out', required=True); ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda' and torch.cuda.device_count() < args.tp:
        print(f'need {args.tp} visible GPUs, have {torch.cuda.device_count()}'); return 2
    case = bringup_case(args.layers, args.tp)
    Path(args.out, 'case.json').write_text(json.dumps(case.to_dict(), indent=1))
    t0 = time.time()
    mp.spawn(_rank, nprocs=args.tp, args=(args.layers, args.tp, args.out, args.device), join=True)
    ranks = [json.loads(Path(args.out, f'load-rank{r}.json').read_text()) for r in range(args.tp)]
    summary = {'case': case.id, 'device': args.device, 'wall_seconds': round(time.time() - t0, 1), 'weights_root': os.environ.get('KIMI_K3_WEIGHTS'),
               'all_pass': all(r['status'] == 'pass' for r in ranks),
               'per_rank': [{k: r.get(k) for k in ('rank', 'status', 'load_seconds', 'weight_bytes', 'memory_allocated_bytes',
                                                  'max_memory_reserved_bytes', 'gpu_free_after_load_bytes', 'max_rss_bytes', 'reason')} for r in ranks]}
    Path(args.out, 'load-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if summary['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
