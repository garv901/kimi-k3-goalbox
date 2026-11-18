"""Per-rank capture worker (started by ./capture under torchrun, one process per rank).

Builds the candidate with -DSS_CAPTURE through its ss_task.build_ext, runs it like the
evaluator does (fresh inputs from the case's make_inputs + trial scenario), arms the capture on
launch --launch-index, and saves every SS_CAP record of the selected layers plus everything
task_ab needs to restore the state on one GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cand', default=a3lib.WS)
    ap.add_argument('--layers', default='1,8')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--launch-index', type=int, default=1, help='0-based launch to capture (earlier ones warm up)')
    ap.add_argument('--cap-mb', type=int, default=4096, help='device capture buffer per rank')
    ap.add_argument('--flags', default='', help='extra build flags')
    ap.add_argument('--timeout-s', type=int, default=480, help='hard limit; the process exits 124 after it')
    ap.add_argument('--build-only', action='store_true', help='build (no GPU) and exit')
    a = ap.parse_args()
    if a.build_only:   # the wrapper builds outside the GPU lease
        print('[capture] built', a3lib.import_hooks(os.path.abspath(a.cand)).build_ext(('-DSS_CAPTURE ' + a.flags).strip()))
        return

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', rank))
    a3lib.watchdog(a.timeout_s, f'capture rank {rank}')
    torch.cuda.set_device(local)
    dev = torch.device(f'cuda:{local}')
    import datetime   # NCCL collectives time out before the watchdog fires
    dist.init_process_group('nccl', device_id=dev, timeout=datetime.timedelta(seconds=max(60, a.timeout_s - 30)))
    group = dist.new_group(list(range(world)))
    cand = os.path.abspath(a.cand)
    hooks = a3lib.import_hooks(cand)
    case = hooks.case() if hasattr(hooks, 'case') else a3lib.default_case()
    layers = [int(x) for x in a.layers.split(',') if x.strip()]
    flags = ('-DSS_CAPTURE ' + a.flags).strip()

    t0 = time.time()
    so = None
    if rank == 0:   # build once, then the other ranks load the cached build
        so = hooks.build_ext(flags)
    dist.barrier(group=group)
    if rank != 0:
        so = hooks.build_ext(flags)
    lib = ctypes.CDLL(so)
    if not hasattr(lib, 'ss_capture_config'):
        raise SystemExit(f'{so} has no ss_capture_config: include ss_task.cuh in the kernel source and make '
                         'build_ext pass its flags to nvcc')
    lib.ss_capture_config.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                      ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ss_capture_status.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    if rank == 0:
        print(f'[capture] built {so} ({time.time() - t0:.0f}s)', flush=True)

    with torch.inference_mode():
        V = a3lib.make_inputs(hooks, case, a.seed, a.trial, str(dev), rank)
        if hasattr(hooks, 'build_runner'):
            runner = hooks.build_runner(case, rank, world, group, str(dev))
        else:
            cd = case.to_dict()
            cd['execution'] = dict(rank=rank, world_size=world, tp_rank=rank, ep_rank=0, device=str(dev),
                                   backend='nccl', tp_group=group, ep_group=None)
            import submission
            runner = submission.build(cd)
        for _ in range(a.launch_index):
            runner(V)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        cap = a.cap_mb << 20
        buf = torch.zeros(cap, dtype=torch.uint8, device=dev)
        pbuf = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        arr = (ctypes.c_int * len(layers))(*layers)
        r = lib.ss_capture_config(local, buf.data_ptr(), cap, pbuf.data_ptr(), pbuf.numel(), 1, arr, len(layers))
        assert r == 0, f'ss_capture_config: cuda error {r}'
        outs = runner(V)
        torch.cuda.synchronize()
        used, dropped = ctypes.c_uint64(0), ctypes.c_uint64(0)
        r = lib.ss_capture_status(local, ctypes.byref(used), ctypes.byref(dropped))
        assert r == 0, f'ss_capture_status: cuda error {r}'
        lib.ss_capture_config(local, None, 0, None, 0, 0, arr, 0)
        records = a3lib.parse_records(buf[:used.value].cpu().numpy().tobytes())
        pbytes = int(pbuf[:8].cpu().view(torch.int64)[0])
        params = pbuf[8:8 + pbytes].cpu().numpy().tobytes() if pbytes else b''
        del buf

        # --- where do the params and regions point?
        inputs = {}
        for k, t in V.items():
            if isinstance(t, torch.Tensor) and t.is_cuda:
                nb = t.numel() * t.element_size()
                fp = a3lib.fingerprint(t) if nb else 0
                inputs[k] = dict(ptr=t.data_ptr(), nbytes=nb, shape=list(t.shape), dtype=str(t.dtype), fp=fp)
        outputs, final = {}, {}
        if isinstance(outs, dict):
            for k, t in outs.items():
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    outputs[k] = dict(ptr=t.data_ptr(), nbytes=t.numel() * t.element_size(), shape=list(t.shape),
                                      dtype=str(t.dtype))
                    final[k] = t.detach().cpu().clone()

        def known(v):
            for d in (inputs, outputs):
                for m in d.values():
                    if m['ptr'] <= v < m['ptr'] + max(m['nbytes'], 1):
                        return True
            return False

        cands = set(a3lib.word_values(params))
        for rec in records:
            for g in rec['regions']:
                cands.add(g['addr'])
        blocks = a3lib.torch_blocks(local)
        allocs = []

        def covered(v):
            return any(x['base'] <= v < x['base'] + x['nbytes'] for x in allocs)

        for v in sorted(cands):
            if v < (1 << 16) or known(v) or covered(v):
                continue
            ext = next(((b, s) for b, s in blocks if b <= v < b + s), None)
            kind = 'torch'
            if ext is None:
                ext = a3lib.driver_extent(v)
                kind = 'driver'
            if ext is None:
                continue
            base, size = ext
            if size > (1 << 30):
                print(f'[capture] rank {rank}: skip snapshot of {size >> 20} MB allocation at {base:#x} '
                      '(too large; register what the phase needs with SS_REG)', flush=True)
                allocs.append(dict(base=base, nbytes=size, kind=kind, data=None))
                continue
            data = torch.empty(size, dtype=torch.uint8)
            _copy_dev_to_host(base, size, data)
            allocs.append(dict(base=base, nbytes=size, kind=kind, data=data))

        meta = dict(rank=rank, world=world, seed=a.seed, trial=a.trial, launch_index=a.launch_index, layers=layers,
                    flags=flags, so=so, used=used.value, dropped=dropped.value, params=params, records=records,
                    inputs=inputs, outputs=outputs, allocs=allocs, final_outputs=final,
                    sm_count=torch.cuda.get_device_properties(dev).multi_processor_count)
        os.makedirs(a.out, exist_ok=True)
        torch.save(meta, os.path.join(a.out, f'rank{rank}.pt'))
        summary = dict(rank=rank, records=[dict(name=r['name'], layer=r['layer'], seq=r['seq'],
                                                regions=[dict(name=g['name'], bytes=g['bytes'],
                                                              dtype=a3lib.DTYPE_NAMES.get(g['dtype'], g['dtype']))
                                                         for g in r['regions']]) for r in records],
                       param_bytes=len(params), allocs=[dict(base=hex(x['base']), nbytes=x['nbytes'], kind=x['kind'])
                                                        for x in allocs],
                       used_bytes=used.value, dropped=dropped.value)
        sums = [None] * world
        dist.all_gather_object(sums, summary, group=group)
        if rank == 0:
            man = dict(cand=cand, seed=a.seed, trial=a.trial, launch_index=a.launch_index, layers=layers,
                       flags=flags, world=world, tolerances=a3lib.tolerances(case),
                       case_id=getattr(case, 'id', None), ranks=sums, seconds=round(time.time() - t0, 1))
            a3lib.save_json(os.path.join(a.out, 'manifest.json'), man)
            names = sorted({(r['name'], r['layer']) for r in summary['records']}, key=lambda x: (x[1], x[0]))
            print(f'[capture] {len(summary["records"])} records/rank: ' + ', '.join(f'{n}@{l}' for n, l in names))
            print(f'[capture] params {len(params)} B, {len(allocs)} extra allocations snapshotted, '
                  f'{used.value >> 20} MB captured/rank, dropped {dropped.value}')
            if not params:
                print('[capture] WARNING: no parameter block (add SS_CAP_PARAMS(&p, sizeof(p)) at the kernel start);'
                      ' task_ab cannot run without it')
            if dropped.value:
                print('[capture] WARNING: records dropped: raise --cap-mb')
            print(f'[capture] wrote {a.out}')
    dist.barrier(group=group)
    sys.stdout.flush()
    os._exit(0)   # skip NCCL/IPC teardown (can hang); everything is written


def _copy_dev_to_host(base: int, size: int, out: torch.Tensor):
    cuda = a3lib._libcuda()
    torch.cuda.synchronize()
    r = cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), ctypes.c_uint64(base), ctypes.c_size_t(size))
    if r != 0:
        raise RuntimeError(f'cuMemcpyDtoH failed ({r}) for {size} bytes at {base:#x}')


if __name__ == '__main__':
    main()
