"""Milestone 3 (GPU, needs its own GO): NCCL smoke test across two GB300 trays (8 ranks, 4 per tray), independent of
the evaluator. Checks that the process group forms over the tray network, that all-reduce results are exact, reports
the transport NCCL chose (from NCCL_DEBUG=INFO logs) and the all-reduce bus bandwidth for 256 MiB, and exchanges the
hostnames so the record proves both trays took part. Launch on each tray (torchrun from the venv):

  torchrun --nnodes 2 --node_rank <0|1> --nproc_per_node 4 --master_addr <tray A address> --master_port 29517 \
      planning/bringup/nccl_smoke.py --out <shared dir>
with NCCL_DEBUG=INFO NCCL_DEBUG_FILE=<dir>/nccl-%h-%p.log.txt and, as in the earlier two-tray deployment on these racks,
NCCL_MNNVL_ENABLE=1. Writes <dir>/nccl-smoke-rank<R>.json; rank 0 writes <dir>/nccl-smoke-summary.json."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--mib', type=int, default=256); ap.add_argument('--iters', type=int, default=20)
    args = ap.parse_args()
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    record = {'rank': rank, 'world_size': world, 'local_rank': local, 'host': platform.node(),
              'gpu': torch.cuda.get_device_name(local), 'torch': torch.__version__,
              'nccl': '.'.join(map(str, torch.cuda.nccl.version())),
              'env': {k: v for k, v in os.environ.items() if k.startswith('NCCL_') and 'DEBUG_FILE' not in k}}
    hosts = [None]*world
    dist.all_gather_object(hosts, platform.node())
    record['hosts'] = hosts
    # exactness: every rank contributes rank+1 -> sum = world*(world+1)/2 in every element
    n = args.mib * (1 << 20) // 4
    x = torch.full((n,), float(rank + 1), device=device)
    dist.all_reduce(x)
    torch.cuda.synchronize(device)
    expected = world * (world + 1) / 2
    record['allreduce_exact'] = bool(torch.all(x == expected).item())
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize(device)
    dist.barrier()
    t0 = time.time()
    for _ in range(args.iters):
        dist.all_reduce(x)
    torch.cuda.synchronize(device)
    seconds = (time.time() - t0) / args.iters
    size = n * 4
    record['allreduce_bytes'] = size
    record['allreduce_ms'] = round(seconds * 1e3, 3)
    record['algbw_GBps'] = round(size / seconds / 1e9, 2)
    record['busbw_GBps'] = round(size / seconds / 1e9 * 2 * (world - 1) / world, 2)
    small = torch.tensor([rank], device=device, dtype=torch.int64)
    dist.all_reduce(small, op=dist.ReduceOp.MIN)
    record['min_rank_ok'] = int(small.item()) == 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f'nccl-smoke-rank{rank}.json').write_text(json.dumps(record, indent=1))
    dist.barrier()
    if rank == 0:
        ranks = [json.loads(Path(args.out, f'nccl-smoke-rank{r}.json').read_text()) for r in range(world)]
        summary = {'world_size': world, 'distinct_hosts': sorted(set(hosts)), 'all_exact': all(r['allreduce_exact'] for r in ranks),
                   'busbw_GBps_min': min(r['busbw_GBps'] for r in ranks), 'allreduce_ms_max': max(r['allreduce_ms'] for r in ranks)}
        Path(args.out, 'nccl-smoke-summary.json').write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
    dist.destroy_process_group()
    return 0 if record['allreduce_exact'] else 1


if __name__ == '__main__':
    sys.exit(main())
