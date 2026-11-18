"""Single-GPU phase harness (started by ./task_ab inside a 1-GPU lease).

Restores the captured state of one rank at record --in (layer --layer) on this GPU, launches the
registered task kernel --task of variant A (and B), checks the state at record --out against the
capture with the case's tolerances, and times A and B alternately with CUDA events.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_variant(cand: str, flags: str) -> str:
    code = ('import sys; sys.path.insert(0, %r); import a3lib; h = a3lib.import_hooks(%r); '
            'print("SO=" + h.build_ext(%r), flush=True)') % (HERE, cand, flags)
    env = dict(os.environ)
    p = subprocess.run([sys.executable, '-c', code], cwd=cand, env=env, capture_output=True, text=True)
    so = [ln[3:] for ln in p.stdout.splitlines() if ln.startswith('SO=')]
    if p.returncode != 0 or not so:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-8000:])
        raise SystemExit(f'build of {cand} with {flags!r} failed')
    return so[-1]


class Variant:
    def __init__(self, label, cand, flags, task):
        self.label, self.cand = label, cand
        t = time.time()
        self.so = build_variant(cand, ('-DSS_TASK ' + flags).strip())
        self.build_s = time.time() - t
        self.lib = ctypes.CDLL(self.so)
        if not hasattr(self.lib, 'ss_task_launch'):
            raise SystemExit(f'{self.so}: no ss_task_launch; include ss_task.cuh and register tasks under SS_TASK')
        self.lib.ss_task_launch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.ss_task_name.restype = ctypes.c_char_p
        names = [self.lib.ss_task_name(i).decode() for i in range(self.lib.ss_task_count())]
        if task not in names:
            raise SystemExit(f'{label} ({cand}): task {task!r} not registered; registered: {names}')
        self.task = task.encode()


def restore_state(meta, rec_in, V):
    """Map every captured address range to a buffer on this GPU. Returns (ranges, mutable)."""
    ranges = []   # (old_base, nbytes, tensor uint8 view, kind, name)
    for k, m in meta['inputs'].items():
        t = V[k]
        ranges.append((m['ptr'], m['nbytes'], t.view(-1).view(torch.uint8) if m['nbytes'] else None, 'input', k))
    mutable = []
    for k, m in meta['outputs'].items():
        t = torch.zeros(m['nbytes'], dtype=torch.uint8, device='cuda')
        ranges.append((m['ptr'], m['nbytes'], t, 'output', k))
        mutable.append(t)
    for x in meta['allocs']:
        t = (x['data'].cuda() if x['data'] is not None
             else torch.zeros(x['nbytes'], dtype=torch.uint8, device='cuda'))
        ranges.append((x['base'], x['nbytes'], t, 'alloc', hex(x['base'])))
        mutable.append(t)
    for g in rec_in['regions']:
        hit = locate(ranges, g['addr'], g['bytes'])
        if hit is None:
            t = torch.empty(g['bytes'], dtype=torch.uint8, device='cuda')
            ranges.append((g['addr'], g['bytes'], t, 'region', g['name']))
            mutable.append(t)
            hit = (t, 0)
        t, off = hit
        t[off:off + g['bytes']].copy_(g['data'].cuda())
    return ranges, mutable


def locate(ranges, addr, nbytes=1):
    for base, size, t, _, _ in ranges:
        if t is not None and base <= addr and addr + nbytes <= base + size:
            return t, addr - base
    return None


def translate(params: bytes, ranges) -> tuple[bytes, list]:
    words = a3lib.word_values(params)
    out, moved = [], []
    for i, v in enumerate(words):
        hit = None
        for base, size, t, kind, name in ranges:
            if t is not None and base <= v < base + max(size, 1):
                hit = (t.data_ptr() + (v - base), kind, name)
                break
        if hit:
            out.append(hit[0])
            moved.append((i, hit[1], hit[2]))
        else:
            out.append(v)
    import struct
    return struct.pack(f'<{len(out)}Q', *out) + params[8 * len(out):], moved


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latest_capture(ws):
    caps = sorted(glob.glob(os.path.join(ws, 'reports', 'capture-*', 'manifest.json')), key=os.path.getmtime)
    if not caps:
        raise SystemExit('no reports/capture-*/manifest.json: run ./capture first')
    return os.path.dirname(caps[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('variants', nargs='+', help='A [B]: candidate directories (default A = workspace)')
    ap.add_argument('--task', required=True, help='name given to SS_TASK_REGISTER')
    ap.add_argument('--in', dest='rin', required=True, help='SS_CAP record holding the phase inputs')
    ap.add_argument('--out', dest='rout', required=True, help='SS_CAP record holding the phase outputs')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--rank', default='0', help='captured rank(s) to replay, e.g. 0 or 0,3 or all')
    ap.add_argument('--capture', default=None, help='capture dir (default: newest reports/capture-*)')
    ap.add_argument('--regions', default='', help='comma list: compare only these --out regions')
    ap.add_argument('--aux', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=0, help='grid size (default: SM count x occupancy)')
    ap.add_argument('--reps', type=int, default=9)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--flags-a', default='')
    ap.add_argument('--flags-b', default='')
    ap.add_argument('--no-flush', action='store_true', help='keep L2 warm between reps (default: flush)')
    ap.add_argument('--noise-pct', type=float, default=2.0, help='minimum relative difference that counts')
    ap.add_argument('--timeout-s', type=int, default=540, help='hard limit; exits 124 after it')
    ap.add_argument('--build-only', action='store_true', help='build the variants (no GPU) and exit')
    a = ap.parse_args()
    if a.build_only:   # the wrapper builds outside the GPU lease; the leased run then hits the build cache
        for i, d in enumerate(a.variants):
            print('[task_ab] built', build_variant(os.path.abspath(d), ('-DSS_TASK ' + (a.flags_a, a.flags_b)[min(i, 1)]).strip()))
        return

    a3lib.watchdog(a.timeout_s, 'task_ab')
    ws = a3lib.WS
    cap = a.capture or latest_capture(ws)
    man = json.load(open(os.path.join(cap, 'manifest.json')))
    tol = man.get('tolerances', {})
    ranks = list(range(man['world'])) if a.rank == 'all' else [int(x) for x in a.rank.split(',')]
    if len(a.variants) > 2:
        raise SystemExit('at most two variants (A B)')
    dirs = [os.path.abspath(d) for d in a.variants]
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join(ws, 'reports', f'task_ab-{a.task}-L{a.layer}-{stamp}')
    os.makedirs(outdir, exist_ok=True)

    vs = [Variant('A', dirs[0], a.flags_a, a.task)]
    if len(dirs) > 1:
        vs.append(Variant('B', dirs[1], a.flags_b, a.task))
    for v in vs:
        print(f'[task_ab] {v.label}: {v.cand} built in {v.build_s:.0f}s -> {os.path.basename(v.so)}', flush=True)

    hooks = a3lib.import_hooks(dirs[0])
    case = hooks.case() if hasattr(hooks, 'case') else a3lib.default_case()
    flush = None if a.no_flush else torch.empty(2 * props.L2_cache_size + (64 << 20), dtype=torch.uint8,
                                                device='cuda')
    stream = torch.cuda.current_stream()
    result = dict(capture=cap, task=a.task, rin=a.rin, rout=a.rout, layer=a.layer, gpu=props.name,
                  tolerances=tol, variants=[dict(label=v.label, dir=v.cand, so=v.so) for v in vs], ranks={})
    overall_ok = True
    for rank in ranks:
        meta = torch.load(os.path.join(cap, f'rank{rank}.pt'), weights_only=False)
        rec_in = a3lib.find_record(meta['records'], a.rin, a.layer)
        rec_out = a3lib.find_record(meta['records'], a.rout, a.layer)
        if not meta['params']:
            raise SystemExit('capture has no parameter block (SS_CAP_PARAMS missing)')
        t = time.time()
        with torch.inference_mode():
            V = a3lib.make_inputs(hooks, case, meta['seed'], meta['trial'], 'cuda:0', rank)
        bad = []
        for k, m in meta['inputs'].items():
            tt = V.get(k)
            if tt is None or tt.numel() * tt.element_size() != m['nbytes']:
                bad.append(k)
                continue
            if m['nbytes'] and a3lib.fingerprint(tt) != m['fp']:
                bad.append(k)
        if bad:
            raise SystemExit(f'regenerated inputs differ from the capture for {bad[:8]}')
        print(f'[task_ab] rank {rank}: inputs regenerated and verified ({time.time() - t:.0f}s)', flush=True)
        ranges, mutable = restore_state(meta, rec_in, V)
        pristine = [m.clone() for m in mutable]
        kp, moved = translate(meta['params'], ranges)
        kp_buf = ctypes.create_string_buffer(kp, len(kp))

        def reset():
            for m, p in zip(mutable, pristine):
                m.copy_(p, non_blocking=True)
            if flush is not None:
                flush.fill_(1)   # evict L2: the phase starts with cold caches, as in the full kernel

        def launch(v):
            r = v.lib.ss_task_launch(0, v.task, ctypes.cast(kp_buf, ctypes.c_void_p), len(kp), a.layer, a.aux,
                                     a.blocks, ctypes.c_void_p(stream.cuda_stream))
            if r != 0:
                raise SystemExit(f'{v.label}: ss_task_launch returned {r}')

        want = rec_out['regions']
        if a.regions:
            sel = set(a.regions.split(','))
            want = [g for g in want if g['name'] in sel]
        rank_res = dict(params_repointed=len(moved), variants={})
        times = {v.label: [] for v in vs}
        for v in vs:   # correctness: one run from the restored state
            reset()
            launch(v)
            torch.cuda.synchronize()
            checks = {}
            for g in want:
                hit = locate(ranges, g['addr'], g['bytes'])
                if hit is None:
                    checks[g['name']] = dict(ok=False, error='region not inside any restored buffer')
                    continue
                tt, off = hit
                checks[g['name']] = a3lib.compare(tt[off:off + g['bytes']].cpu(), g['data'], g['dtype'], tol)
            ok = all(c['ok'] for c in checks.values())
            overall_ok &= ok
            rank_res['variants'][v.label] = dict(correct=ok, checks=checks)
        for i in range(a.warmup + a.reps):   # timing: alternate A/B, B/A
            order = vs if i % 2 == 0 else vs[::-1]
            for v in order:
                reset()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                launch(v)
                en.record()
                en.synchronize()
                if i >= a.warmup:
                    times[v.label].append(st.elapsed_time(en) * 1000.0)
        for v in vs:
            xs = times[v.label]
            med = statistics.median(xs)
            rank_res['variants'][v.label].update(us=xs, median_us=med, min_us=min(xs),
                                                 iqr_pct=100.0 * (pct(xs, .75) - pct(xs, .25)) / med)
        if len(vs) == 2:
            A, B = rank_res['variants']['A'], rank_res['variants']['B']
            delta = 100.0 * (B['median_us'] - A['median_us']) / A['median_us']
            thr = max(a.noise_pct, A['iqr_pct'], B['iqr_pct'])
            if not B['correct']:
                verdict = 'B INCORRECT'
            elif delta < -thr:
                verdict = 'B FASTER'
            elif delta > thr:
                verdict = 'B SLOWER'
            else:
                verdict = 'SAME (within noise)'
            rank_res.update(delta_pct=delta, threshold_pct=thr, verdict=verdict)
        result['ranks'][rank] = rank_res
        # print
        print(f'[task_ab] rank {rank} layer {a.layer} task {a.task}  ({len(moved)} param pointers re-pointed)')
        for v in vs:
            r = rank_res['variants'][v.label]
            bad = [k for k, c in r['checks'].items() if not c['ok']]
            worst = max((c.get('rel_l2', 0.0) for c in r['checks'].values()), default=0.0)
            exact = all(c.get('bitwise_equal') for c in r['checks'].values())
            print(f'  {v.label}: {"PASS" if r["correct"] else "FAIL"} vs capture '
                  f'({len(r["checks"])} regions, {"bit-exact" if exact else f"max rel_l2 {worst:.2e}"}'
                  f'{", failing: " + ",".join(bad) if bad else ""})  '
                  f'median {r["median_us"]:.1f} us  min {r["min_us"]:.1f}  IQR {r["iqr_pct"]:.1f}%  n={len(r["us"])}')
        if len(vs) == 2:
            print(f'  verdict: {rank_res["verdict"]}  (B-A {rank_res["delta_pct"]:+.1f}%, threshold '
                  f'{rank_res["threshold_pct"]:.1f}%)')
        del V, ranges, mutable, pristine
        torch.cuda.empty_cache()
    result['correct'] = overall_ok
    a3lib.save_json(os.path.join(outdir, 'result.json'), result)
    print(f'[task_ab] wrote {outdir}/result.json')
    sys.stdout.flush()
    os._exit(0 if overall_ok else 3)


if __name__ == '__main__':
    main()
