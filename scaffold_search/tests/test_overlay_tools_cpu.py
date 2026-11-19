"""CPU tests of the A1/A2/A3 overlay tools for the TP4 protocol (no GPU, no torch needed except where
noted): rank handling (4 ranks from case.json), profiler command generation, artifact layout and
the tools' own self-tests.

  python3.12 -m pytest -q scaffold_search/tests/test_overlay_tools_cpu.py
"""
import importlib.util, json, os, shutil, stat, subprocess, sys, tempfile, types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OV = ROOT / 'overlays'
PY = sys.executable


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fake_ws(gpus=4):
    d = Path(tempfile.mkdtemp(prefix='ss-ovws-'))
    (d / 'case.json').write_text(json.dumps({'id': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4', 'gpus': gpus, 'tp': gpus}))
    (d / 'submission.py').write_text('import mk_host\n')
    (d / 'gpu_run').write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/gpu_run.args"\nexit 0\n')
    os.chmod(d / 'gpu_run', 0o755)
    (d / 'reports').mkdir()
    return d


# ------------------------------------------------------------------------------------------- A1
def test_phase_table_self_test():
    pytest.importorskip('numpy')
    p = subprocess.run([PY, str(OV / 'A1/.a1/phase_table.py'), '--self-test'], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and 'phase_table self-test: PASS' in p.stdout, p.stdout[-800:] + p.stderr[-800:]


def test_flag_audit_self_test():
    p = subprocess.run([PY, str(OV / 'A1/.a1/flag_audit.py'), '--self-test'], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and 'flag_audit self-test: PASS' in p.stdout, p.stdout[-800:] + p.stderr[-800:]


def test_a1_common_reads_4_ranks_from_case_json(monkeypatch):
    common = load(OV / 'A1/.a1/common.py', 'a1_common_t')
    ws = fake_ws()
    monkeypatch.setattr(common, 'WS', ws)
    info = common.case_info()
    assert info['gpus'] == 4 and info['id'].endswith('-b8-tp4')
    assert common.gpu_run_prefix(int(info['gpus'])) == [str(ws / 'gpu_run'), '-g', '4', '--']
    # no case.json: the fallback is the protocol's node (4), never 8
    (ws / 'case.json').unlink()
    (ws / 'check_candidate').write_text('exec python3.12 -m megabench.integrations.gpu_noise check --case dsv41-flash-dspark-step-k5-ctx2k-b8-tp4 -- x\n')
    assert common.case_info() == {'id': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4', 'gpus': 4}
    monkeypatch.setenv('A1_NO_GPU_RUN', '1')
    assert common.gpu_run_prefix(4) == []


def test_ncu_all_ranks_command_generation(monkeypatch):
    """The outer command leases all 4 GPUs and passes --world 4; the inner launch starts one ncu per
    rank (4) in application-replay mode with --target-processes all."""
    sys.path.insert(0, str(OV / 'A1/.a1'))
    for m in ('common', 'ncu_all_ranks'):
        sys.modules.pop(m, None)
    import common, ncu_all_ranks as nar  # noqa: E402
    ws = fake_ws()
    monkeypatch.setattr(common, 'WS', ws)
    (ws / 'ncu').write_text('#!/bin/sh\nexit 0\n'); os.chmod(ws / 'ncu', 0o755)
    monkeypatch.setenv('NCU', str(ws / 'ncu'))
    monkeypatch.setenv('MEGABENCH_ROOT', str(ws))   # benchmark_path(): any dir with megabench/
    (ws / 'megabench').mkdir()
    calls = []
    monkeypatch.setattr(nar.subprocess, 'call', lambda cmd, **kw: calls.append((cmd, kw)) or 1)
    rc = nar.main([])
    assert rc == 1 and len(calls) == 1
    cmd = calls[0][0]
    assert cmd[:4] == [str(ws / 'gpu_run'), '-g', '4', '--'], cmd
    assert cmd[cmd.index('--world') + 1] == '4' and '--launch' in cmd and cmd[cmd.index('--case') + 1].endswith('-b8-tp4')
    assert calls[0][1]['env']['PYTHONPATH'].startswith(str(ws))
    out = sorted((ws / 'reports').glob('ncu_all_ranks-*'))
    assert len(out) == 1 and json.loads((out[0] / 'metrics.json').read_text()) == nar.SETS['default']
    # inner launch: one ncu per rank
    popens = []

    class P:
        def __init__(self, cmd, **kw):
            popens.append(cmd); self.returncode = 0
        def wait(self): return 0
        def poll(self): return 0
        def kill(self): pass
    a = types.SimpleNamespace(world=4, out=out[0], metrics='gpu__time_duration.sum', kernel=None, extra='', case='c', seed=1)
    from unittest import mock
    with mock.patch.object(nar.subprocess, 'Popen', P):     # launch() only; report() below uses subprocess.run
        rc = nar.launch(a, str(ws / 'ncu'))
    assert rc == 0 and len(popens) == 4
    for r, c in enumerate(popens):
        assert c[0] == str(ws / 'ncu') and c[c.index('--rank') + 1] == str(r) and c[c.index('--world') + 1] == '4'
        assert c[c.index('--replay-mode') + 1] == 'application' and c[c.index('--target-processes') + 1] == 'all'
        assert c[c.index('-o') + 1] == str(out[0] / f'prof_r{r}') and c[c.index('--profile-from-start') + 1] == 'off'
        assert (out[0] / f'rank{r}.log').exists()
    # report(): per-rank table from imported CSV (ncu stub prints nothing => 0 records, no crash)
    for r in range(4):
        (out[0] / f'prof_r{r}.ncu-rep').write_bytes(b'')
    text = nar.report(out[0], str(ws / 'ncu'), nar.SETS['default'])
    assert 'profile_candidate_all_ranks: 0 kernel record(s)' in text and (out[0] / 'summary.json').exists()


def test_ncu_summarize_groups_per_rank():
    sys.path.insert(0, str(OV / 'A1/.a1'))
    import ncu_all_ranks as nar  # noqa: E402
    csv = ('"ID","Process ID","Kernel Name","Device","dram__bytes_read.sum","gpu__time_duration.sum"\n'
           '"","","","","Gbyte","usecond"\n' + ''.join(f'"{r}","{100+r}","mk_kernel","{r}","1.5","28000"\n' for r in range(4)))
    rows = nar.parse_raw_csv(csv)
    assert len(rows) == 4
    ranks = nar.summarize(rows, ['dram__bytes_read.sum', 'gpu__time_duration.sum'])
    assert sorted(ranks) == ['rank0', 'rank1', 'rank2', 'rank3']
    assert abs(ranks['rank3']['metrics']['dram__bytes_read.sum']['value'] - 1.5e9) < 1
    assert abs(ranks['rank0']['metrics']['gpu__time_duration.sum']['value'] - 0.028) < 1e-9


def test_phase_table_analyze_4_ranks():
    """Analysis of 4 captured ranks (synthetic buffers): per-rank spans, cross-rank skew, slowest rank."""
    np = pytest.importorskip('numpy')  # noqa: F841
    pt = load(OV / 'A1/.a1/phase_table.py', 'phase_table_t')
    import argparse
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ph = [(10, 10.0, 2.0), (20, 50.0, 0.0), (30, 5.0, 0.0)]
        for r in range(4):
            (d / f'rank{r}').mkdir()
            slow = [(10, 10.0, 2.0), (20, 50.0 + (8.0 if r == 2 else 0.0), 0.0), (30, 5.0, 0.0)]
            (d / f'rank{r}' / 'launch0.u64').write_bytes(pt.synth_buffer(4, slow if r == 2 else ph, 0.0))
        (d / 'names.json').write_text(json.dumps({'10': 'A', '20': 'MOE', '30': 'C'}))
        pt.analyze_dir(d, argparse.Namespace(src=[], group=['moe=MOE'], top=10))
        S = json.loads((d / 'summary.json').read_text())
        assert len(S['span_per_rank']) == 4 and S['ranks_aligned']
        moe = {p['label']: p for p in S['phases']}['MOE']
        assert moe['slowest_rank'] == 2 and len(moe['wall_per_rank']) == 4 and moe['xrank_spread'] > 1.0


# ------------------------------------------------------------------------------------------- A3
def test_capture_wrapper_leases_4_ranks_from_case_json():
    """./capture (bash) reads the rank count from case.json (4), builds outside the lease, then runs
    torchrun with --nproc_per_node 4 under ./gpu_run -g 4."""
    ws = fake_ws()
    shutil.copy2(OV / 'A3/capture', ws / 'capture'); os.chmod(ws / 'capture', 0o755)
    (ws / 'a3tools').mkdir()
    (ws / 'a3tools/capture.py').write_text('import sys; print("stub", sys.argv[1:])\n')
    p = subprocess.run(['./capture', '--layers', '1,8'], cwd=ws, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    args = (ws / 'gpu_run.args').read_text().split()
    assert args[:5] == ['-g', '4', '-m', '10', '--'], args
    assert args[args.index('--nproc_per_node') + 1] == '4' and '--standalone' in args
    assert '--build-only' not in args and '--build-only' in p.stdout      # built first, outside the lease
    # explicit override
    p = subprocess.run(['./capture', '--gpus', '2', '--layers', '1'], cwd=ws, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and (ws / 'gpu_run.args').read_text().split()[:2] == ['-g', '2']
    # no case.json: default 4
    (ws / 'case.json').unlink()
    p = subprocess.run(['./capture'], cwd=ws, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and (ws / 'gpu_run.args').read_text().split()[:2] == ['-g', '4']


def test_task_ab_wrapper_uses_one_gpu():
    ws = fake_ws()
    shutil.copy2(OV / 'A3/task_ab', ws / 'task_ab'); os.chmod(ws / 'task_ab', 0o755)
    (ws / 'a3tools').mkdir()
    (ws / 'a3tools/task_ab.py').write_text('import sys; print("stub", sys.argv[1:])\n')
    p = subprocess.run(['./task_ab', 'A', '--task', 't', '--in', 'i', '--out', 'o', '--layer', '1', '--rank', '3'],
                       cwd=ws, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    args = (ws / 'gpu_run.args').read_text().split()
    assert args[:5] == ['-g', '1', '-m', '10', '--'] and args[args.index('--rank') + 1] == '3'


def test_a3lib_compare_rejects_corruption():
    torch = pytest.importorskip('torch')
    a3lib = load(OV / 'A3/a3tools/a3lib.py', 'a3lib_t')
    tol = {'atol': 0.003, 'rtol': 0.003, 'bf16_rtol': 0.008, 'noise_factor': 2.0}
    want = torch.randn(1024, dtype=torch.bfloat16)
    same = a3lib.compare(want.view(torch.uint8), want.view(torch.uint8), 9, tol)
    assert same['ok'] and same['bitwise_equal']
    noisy = (want.float() * 1.05).to(torch.bfloat16)
    r = a3lib.compare(noisy.view(torch.uint8), want.view(torch.uint8), 9, tol)
    assert not r['ok'] and r['rel_l2'] > 0.008
    nan = want.clone(); nan[0] = float('nan')
    assert not a3lib.compare(nan.view(torch.uint8), want.view(torch.uint8), 9, tol)['ok']
    ints = torch.arange(100, dtype=torch.int32)
    bad = ints.clone(); bad[42] += 1
    r = a3lib.compare(bad.view(torch.uint8), ints.view(torch.uint8), 5, tol)
    assert not r['ok'] and r['mismatches'] == 1
    # record parser: 4 ranks' worth of synthetic records, one region each
    import struct
    def rec(name, layer, seq, data):
        total = 128 + 128 + len(data)
        hdr = struct.pack('<IIiIQ', a3lib.MAGIC, 1, layer, seq, total) + name.encode().ljust(48, b'\0')
        hdr = hdr.ljust(128, b'\0')
        reg = b'x_out'.ljust(48, b'\0') + struct.pack('<QQ', 0x7000, len(data)) + struct.pack('<i', 9) + b'\0' * 4 \
            + struct.pack('<Q', 256)
        reg = reg.ljust(128, b'\0')
        return hdr + reg + data
    buf = b''.join(rec('gemv', l, i, bytes(64)) for i, l in enumerate((0, 1, 2, 3)))
    recs = a3lib.parse_records(buf)
    assert [r['layer'] for r in recs] == [0, 1, 2, 3] and recs[0]['regions'][0]['name'] == 'x_out'
    assert a3lib.find_record(recs, 'gemv', 3)['seq'] == 3
    with pytest.raises(SystemExit):
        a3lib.find_record(recs, 'gemv', 7)


def test_a3_capture_worker_is_rank_generic():
    """capture.py takes RANK/WORLD_SIZE/LOCAL_RANK from torchrun (no fixed world); task_ab replays
    ranks named by --rank (0..world-1 from the capture manifest)."""
    src = (OV / 'A3/a3tools/capture.py').read_text()
    assert "os.environ.get('WORLD_SIZE'" in src and "os.environ.get('RANK'" in src and 'nprocs' not in src
    assert 'range(8)' not in src and 'world_size=8' not in src
    tab = (OV / 'A3/a3tools/task_ab.py').read_text()
    assert "range(man['world'])" in tab
    for f in OV.rglob('*'):
        if f.is_file() and f.suffix in ('.py', '.sh', '') and 'skills' not in f.parts and f.stat().st_size < 200000:
            t = f.read_text(errors='replace')
            assert 'GPUS=8' not in t and "'gpus': 8" not in t and 'nproc_per_node 8' not in t, f


# ------------------------------------------------------------------------------------------- A2
def test_a2_primdb_rows_and_run_seed_groups():
    rows = [json.loads(l) for l in (OV / 'A2/progress/microbench/primdb_local.jsonl').read_text().splitlines() if l.strip()]
    assert len(rows) > 10 and all('gpu' in json.dumps(r) for r in rows[:5])
    rs = (OV / 'A2/primbench_b300/run_seed.sh').read_text()
    assert 'gpu_run' in rs or 'GPU' in rs
    p = subprocess.run(['bash', '-n', str(OV / 'A2/primbench_b300/run_seed.sh')], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    for f in ('capture', 'task_ab'):
        p = subprocess.run(['bash', '-n', str(OV / 'A3' / f)], capture_output=True, text=True)
        assert p.returncode == 0, (f, p.stderr)
