"""One rank of profile_candidate_all_ranks, run under its own ncu (application replay).

Every replay pass re-executes this process; a per-rank pass counter gives each pass a fresh file
rendezvous, so all ranks meet again in pass k. Only the warmed launch between cudaProfilerStart/Stop
is profiled (ncu --profile-from-start off).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--world', type=int, required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--submission', required=True)
    ap.add_argument('--seed', type=int, default=7301)
    ap.add_argument('--rdv', required=True, help='directory for the per-pass rendezvous files')
    a = ap.parse_args()
    import torch
    import torch.distributed as dist
    from megabench.cases import select_cases
    from megabench.harness.runner import _load_submission
    from megabench.workloads import make_inputs

    rank = a.rank
    rdv = Path(a.rdv)
    counter = rdv / f'pass_rank{rank}'
    p = int(counter.read_text()) + 1 if counter.exists() else 0
    counter.write_text(str(p))
    print(f'rank {rank}: replay pass {p}', flush=True)
    case = select_cases('all', [a.case])[0]
    device = f'cuda:{rank}'
    torch.cuda.set_device(rank)
    execution = None
    if a.world > 1:
        from megabench.tasks.parallel import initialize
        dist.init_process_group('nccl', init_method=(rdv / f'rdv_{p}').as_uri(), rank=rank, world_size=a.world,
                                timeout=timedelta(minutes=10), device_id=torch.device(device))
        ctx = initialize(case)
        execution = {'rank': rank, 'world_size': a.world, 'tp_rank': ctx.tp_rank, 'ep_rank': ctx.ep_rank,
                     'tp_group': ctx.tp_group, 'ep_group': ctx.ep_group, 'backend': 'nccl', 'device': device}
    try:
        with torch.inference_mode():
            inputs = make_inputs(case, a.seed, device)
            candidate = _load_submission(Path(a.submission), case, execution=execution)
            candidate(inputs)
            torch.cuda.synchronize(device)
            if a.world > 1:
                dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStart()
            candidate(inputs)
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
            if a.world > 1:
                dist.barrier()
    finally:
        if a.world > 1:
            dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
