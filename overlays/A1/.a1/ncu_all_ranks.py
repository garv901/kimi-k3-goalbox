"""profile_candidate_all_ranks: Nsight Compute on EVERY rank of a multi-GPU candidate (application replay).

  ./profile_candidate_all_ranks                     default metrics (DRAM bytes, L2 hit, SM/tensor pipes)
  ./profile_candidate_all_ranks --set stalls        warp-stall PC sampling (where warps wait)
  ./profile_candidate_all_ranks --metrics a,b,c     your own ncu metric list
  ./profile_candidate_all_ranks --import DIR        re-print a finished report (no GPU)

Each rank runs under its own ncu (--replay-mode application --target-processes all): ncu serialises
profiled kernels across the processes it controls, so ONE ncu over all ranks deadlocks a kernel that waits
for its peers. Every replay pass re-runs each rank, which joins a fresh per-pass rendezvous, and only the
warmed launch inside the profiler window (cudaProfilerStart/Stop) is measured. Each pass costs one full run (~1-2 min for 8 ranks),
so the default sets are kept to as few passes as possible. Kernel replay (ncu's default) hangs or fails on
kernels that wait for peer ranks; do not use it for multi-rank kernels.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NCU_CANDIDATES = ['/opt/nvidia/nsight-compute/2025.3.1/ncu', '/usr/local/cuda/bin/ncu']
# replay passes measured on B300 (sm_103): each pass = one full run of every rank (~1-1.5 min for 8 ranks)
SETS = {
    'default': ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum', 'smsp__inst_executed.sum',
                'sm__inst_executed_pipe_tensor.sum', 'sm__inst_executed_pipe_tc.sum'],                       # 1 pass
    'l2': ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum', 'lts__t_sector_hit_rate.pct',
           'smsp__inst_executed.sum', 'sm__inst_executed_pipe_tensor.sum', 'sm__inst_executed_pipe_tc.sum'],  # 3 passes
    'pipes': ['smsp__inst_executed.sum', 'sm__inst_executed_pipe_tensor.sum', 'sm__inst_executed_pipe_tc.sum',
              'sm__inst_executed_pipe_alu.sum', 'sm__inst_executed_pipe_fma.sum', 'sm__inst_executed_pipe_lsu.sum',
              'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'],                              # ~3 passes
    'stalls': ['group:smsp__pcsamp_warp_stall_reasons', 'gpu__time_duration.sum'],                         # 2 passes
}


def find_ncu() -> str:
    for c in [os.environ.get('NCU')] + NCU_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise SystemExit('ncu not found (set NCU=/path/to/ncu)')


def to_num(s: str):
    s = s.replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_raw_csv(text: str) -> list[dict]:
    lines = [l for l in text.splitlines() if l.startswith('"')]
    if len(lines) < 2:
        return []
    rows = list(csv.reader(io.StringIO('\n'.join(lines))))
    hdr, units = rows[0], rows[1]
    out = []
    for r in rows[2:]:
        d = {}
        for h, u, v in zip(hdr, units, r):
            d[h] = v
            d['__unit__' + h] = u
        out.append(d)
    return out


def summarize(rows: list[dict], metrics: list[str]) -> dict:
    """Per device (= rank for spawned rank i on cuda:i) metric values, converted to base units."""
    scale = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'Tbyte': 1e12, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9,
             'ns': 1e-9, 'us': 1e-6, 'usecond': 1e-6, 'nsecond': 1e-9, 'msecond': 1e-3, 'ms': 1e-3, 'second': 1, 's': 1}
    ranks = {}
    for r in rows:
        dev = r.get('Device') or r.get('device__attribute_display_name') or '?'
        key = f"rank{dev}"
        R = ranks.setdefault(key, dict(device=dev, pid=r.get('Process ID'), kernels=[], metrics={}))
        R['kernels'].append(r.get('Kernel Name', '?')[:80])
        for h, v in r.items():
            if h.startswith('__unit__') or h in ('ID', 'Process ID', 'Process Name', 'Host Name', 'Kernel Name',
                                                 'Context', 'Stream', 'Block Size', 'Grid Size', 'Device', 'CC'):
                continue
            x = to_num(v)
            if x is None:
                continue
            u = r.get('__unit__' + h, '')
            if u in scale:
                x *= scale[u]
                u = 'byte' if 'byte' in u or u.endswith('B') else 'second'
            M = R['metrics'].setdefault(h, dict(value=0.0, unit=u, n=0))
            if h.endswith('.pct') or 'pct' in h or 'per_thread' in h or 'launch__' in h:
                M['value'] = (M['value'] * M['n'] + x) / (M['n'] + 1)   # mean over kernels
            else:
                M['value'] += x                                       # sum over kernels
            M['n'] += 1
    return ranks


def fmt(v: float, unit: str) -> str:
    if unit == 'byte':
        return f'{v / 1e9:.3f} GB'
    if unit == 'second':
        return f'{v * 1e3:.3f} ms'
    if abs(v) >= 1e6:
        return f'{v / 1e6:.2f} M'
    return f'{v:.2f}'


def report(out: Path, ncu: str, metrics: list[str]) -> str:
    reps = sorted(out.glob('prof_r*.ncu-rep'), key=lambda p: int(re.search(r'prof_r(\d+)', p.name).group(1)))
    rows, texts = [], []
    for rep in reps:
        p = subprocess.run([ncu, '--import', str(rep), '--csv', '--page', 'raw'], capture_output=True, text=True)
        if p.returncode != 0:
            print(f'ncu --import {rep} failed: {p.stderr[-500:]}', file=sys.stderr)
            continue
        texts.append(p.stdout)
        r = int(re.search(r'prof_r(\d+)', rep.name).group(1))
        for row in parse_raw_csv(p.stdout):
            row['Device'] = str(r)
            rows.append(row)
    (out / 'raw.csv').write_text('\n'.join(texts))
    rep = out / 'prof_r*.ncu-rep'
    ranks = summarize(rows, metrics)
    lines = [f'profile_candidate_all_ranks: {len(rows)} kernel record(s) on {len(ranks)} process(es); report {rep}']
    keys = sorted(ranks, key=lambda k: (int(ranks[k]['device']) if str(ranks[k]['device']).isdigit() else 99, k))
    want = [m for m in metrics if not m.startswith('group:')]
    shown = [m for m in want if any(m in ranks[k]['metrics'] for k in keys)]
    extra = sorted({h for k in keys for h in ranks[k]['metrics']} - set(shown))
    if extra and len(extra) <= 60:
        shown += extra
    lines.append(f'{"metric":62s} ' + ' '.join(f'{"rank" + str(ranks[k]["device"]):>11s}' for k in keys))
    for m in shown:
        vals = [ranks[k]['metrics'].get(m) for k in keys]
        unit = next((v['unit'] for v in vals if v), '')
        lines.append(f'{m[:62]:62s} ' + ' '.join(f'{fmt(v["value"], unit) if v else "-":>11s}' for v in vals))
    lines.append('kernels per process: ' + '; '.join(f'rank{ranks[k]["device"]}: {", ".join(sorted(set(ranks[k]["kernels"])))[:60]}'
                                                     for k in keys[:2]) + (' ...' if len(keys) > 2 else ''))
    lines.append('NOTE: under ncu the ranks start the kernel at different times and early ranks spin on peer flags. '
                 'DRAM/L2 byte counts and tensor-pipe counts are valid; duration, % metrics and total instruction counts '
                 'include the spin (compare the rank with the shortest duration, or compare bytes across builds). '
                 'Use ./phase_table for time.')
    summary = dict(report=str(rep), metrics=metrics,
                   ranks=[dict(rank=ranks[k]['device'], pid=ranks[k]['pid'], kernels=sorted(set(ranks[k]['kernels'])),
                               metrics={m: v['value'] for m, v in ranks[k]['metrics'].items()},
                               units={m: v['unit'] for m, v in ranks[k]['metrics'].items()}) for k in keys])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    text = '\n'.join(lines)
    (out / 'table.txt').write_text(text + '\n')
    return text


def launch(a, ncu: str) -> int:
    """Inside the GPU lease: one ncu per rank, all started together; returns the worst exit code."""
    import tempfile
    rdv = Path(tempfile.mkdtemp(prefix='a1-ncu-rdv-', dir='/tmp'))
    procs = []
    for r in range(a.world):
        cmd = [ncu, '--target-processes', 'all', '--replay-mode', 'application', '--profile-from-start', 'off',
               '--cache-control', 'none', '--clock-control', 'none', '--metrics', a.metrics, '-f',
               '-o', str(a.out / f'prof_r{r}')] + (['-k', a.kernel] if a.kernel else []) + a.extra.split() + [
               common.python(), str(Path(__file__).resolve().parent / 'ncu_rank.py'), '--rank', str(r),
               '--world', str(a.world), '--case', a.case, '--submission', str(common.submission_path()),
               '--seed', str(a.seed), '--rdv', str(rdv)]
        log = open(a.out / f'rank{r}.log', 'w')
        if r == 0:
            print('+ ' + ' '.join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=False))
    rcs = []
    try:
        for p in procs:
            rcs.append(p.wait())
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    for r, rc in enumerate(rcs):
        if rc:
            print(f'rank {r}: ncu exit {rc}', flush=True)
    passes = (rdv / 'pass_rank0').read_text() if (rdv / 'pass_rank0').exists() else '?'
    print(f'application replay passes: {int(passes) + 1 if passes != "?" else "?"}', flush=True)
    return max(rcs) if rcs else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--set', default='default', choices=sorted(SETS))
    ap.add_argument('--metrics', help='comma-separated ncu metrics (overrides --set)')
    ap.add_argument('--kernel', help='ncu -k filter (regex:NAME); default: every kernel in the profiler window')
    ap.add_argument('--timeout', type=int, default=900, help='hard timeout in s for the profiled run, inside the GPU lease (default 900)')
    ap.add_argument('--seed', type=int, default=7301)
    ap.add_argument('--extra', default='', help='extra ncu arguments (quoted)')
    ap.add_argument('--import', dest='imp', type=Path, help='re-print a finished reports/ncu_all_ranks-* dir')
    ap.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--out', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--case', help=argparse.SUPPRESS)
    ap.add_argument('--world', type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    ncu = find_ncu()
    if a.launch:
        return launch(a, ncu)
    metrics = a.metrics.split(',') if a.metrics else SETS[a.set]
    if a.imp:
        print(report(a.imp, ncu, json.loads((a.imp / 'metrics.json').read_text()) if (a.imp / 'metrics.json').exists() else metrics))
        return 0
    info = common.case_info()
    out = common.WS / 'reports' / f'ncu_all_ranks-{common.run_id()}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics))
    env = dict(os.environ)
    env['PYTHONPATH'] = common.benchmark_path() + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp')
    cmd = common.gpu_run_prefix(int(info.get('gpus', 1))) + [
        'timeout', '-k', '30', str(a.timeout), common.python(), str(Path(__file__).resolve()), '--launch',
        '--out', str(out), '--case', info['id'], '--world', str(int(info.get('gpus', 1))), '--seed', str(a.seed),
        '--metrics', ','.join(metrics), '--extra', a.extra] + (['--kernel', a.kernel] if a.kernel else [])
    print('+', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=str(common.WS))
    print(f'exit {rc} after {time.time() - t0:.0f} s (logs {out}/rank*.log)')
    if rc in (124, 137):
        print(f'TIMEOUT after {a.timeout} s: use fewer metrics (--set bytes) or a larger --timeout', file=sys.stderr)
    if not list(out.glob('prof_r*.ncu-rep')):
        tail = ''.join(open(out / 'rank0.log').readlines()[-30:]) if (out / 'rank0.log').exists() else ''
        print('no report written; rank 0 log tail:\n' + tail, file=sys.stderr)
        return rc or 1
    print(report(out, ncu, metrics))
    print(f'\nwrote {out}/summary.json, table.txt, prof_r<rank>.ncu-rep (open in Nsight Compute)')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
