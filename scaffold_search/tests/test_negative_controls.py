"""Negative controls of the harness side of scoring (CPU, canned data). Protocol dsv41-tp4-rack10-v1.

Every control must be REJECTED by the harness (review of the rescore reports, snapshotting of
pending checks, parsing of the stock check output, source review):
  - wrong output values / a failing trial on ONE rank only (rank 3)
  - launch counts other than exactly one GPU kernel launch on each of the 4 ranks
  - evaluator statuses other than ok_provisional (incorrect, error, non_megakernel, timeout, ...)
  - invalid candidate artifacts: non-zero check rc, missing/unparseable report, missing or
    truncated snapshot tarball, snapshot without submission.py, oracle/reference imports

  python3.12 -m pytest -q scaffold_search/tests/test_negative_controls.py
"""
import contextlib, json, os, subprocess, sys, tarfile, tempfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold_search import rescore, review, snapshot, workspace  # noqa: E402

NGPU = workspace.NGPU
SCENARIOS = ((11, 'raw'), (12, 'accept_full'), (13, 'accept_mixed'))
OUTPUTS = ('logits', 'draft_logits', 'verify_tokens', 'draft_tokens', 'accepted_count', 'committed_tokens')


def rank_rec(fail_outputs=(), fail_trials=(0, 1, 2)):
    trials = []
    for i, (seed, sc) in enumerate(SCENARIOS):
        outs = {o: {'pass': not (o in fail_outputs and i in fail_trials)} for o in OUTPUTS}
        trials.append({'seed': seed, 'scenario': sc, 'outputs': outs})
    return {'status': 'pass' if not fail_outputs else 'fail', 'trials': trials}


def report(status='ok_provisional', counts=(1,) * NGPU, bad_rank=None, bad_outputs=('logits',), bad_trials=(0,),
           correctness_status=None):
    per_rank = [rank_rec(bad_outputs if r == bad_rank else (), bad_trials) for r in range(len(counts))]
    return {'status': status,
            'launch_audit': {'status': 'within_budget' if all(c == 1 for c in counts) else 'launch_budget_failed',
                             'per_rank': [{'gpu_kernel_count': c, 'status': 'within_budget' if c == 1 else
                                           'launch_budget_failed'} for c in counts]},
            'correctness': {'status': correctness_status or ('pass' if bad_rank is None else 'fail'),
                            'per_rank': per_rank}}


def good():
    return [report(), report(), report()]


def test_tp4_baseline_is_accepted():
    r = review.review_reports(good())
    assert r['ok'] and r['fresh_seeds_per_scenario'] == {'raw': 1, 'accept_full': 1, 'accept_mixed': 1}


def test_one_rank_wrong_values_rejected():
    """rank 3 fails logits in ONE trial of ONE rescore run; ranks 0-2 pass everything."""
    for run in range(3):
        reps = good(); reps[run] = report(bad_rank=3)
        r = review.review_reports(reps)
        assert not r['ok'] and any('failed' in i for i in r['issues']), r
    # the evaluator also marks the whole report; both signals are required to be absent for a pass
    reps = good(); reps[1] = report(bad_rank=3, correctness_status='pass')   # inconsistent report
    assert not review.review_reports(reps)['ok']
    # a rank-3 failure of an EXACT output (accepted_count) in the last trial only
    reps = good(); reps[2] = report(bad_rank=3, bad_outputs=('accepted_count',), bad_trials=(2,))
    assert not review.review_reports(reps)['ok']
    # every rank fails
    reps = [report(bad_rank=r) for r in range(3)]
    assert not review.review_reports(reps)['ok']


def test_launch_counts_rejected():
    for counts in ((1, 1, 1, 2), (2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 0), (1, 1, 1), (1,), (1,) * 8, (2,) * 4,
                   (1, 1, 1, None)):
        reps = good(); reps[1] = report(counts=counts)
        r = review.review_reports(reps)
        assert not r['ok'] and any('launches per rank' in i for i in r['issues']), (counts, r)
    reps = good(); reps[0]['launch_audit'] = {}          # audit missing entirely
    assert not review.review_reports(reps)['ok']
    reps = good(); del reps[0]['launch_audit']
    assert not review.review_reports(reps)['ok']


def test_non_ok_statuses_rejected():
    for st in ('incorrect', 'error', 'non_megakernel', 'authenticity_unverified', 'correctness_only', 'timeout',
               'worker_failed', 'unavailable', 'pass', None, ''):
        reps = good(); reps[2] = report(status=st)
        r = review.review_reports(reps)
        assert not r['ok'], (st, r)
    assert not review.review_reports([None, None, None])['ok']


def test_no_reports_is_not_a_pass():
    # review_reports([]) has no issues by construction; rescore_run never gets there because
    # check_n always returns one record (or None) per rep. Document the contract:
    r = review.review_reports([None])
    assert not r['ok'] and r['issues'] == ['rescore run 0: no report']


def _pending(ws, cid, rc, status, with_tgz=True, with_report=True, truncate=False, log_status=None):
    os.makedirs(f'{ws}/.harness/pending', exist_ok=True); os.makedirs(f'{ws}/reports', exist_ok=True)
    base = f'{ws}/.harness/pending/{cid}'
    rep = f'{ws}/reports/check-{cid}.jsonl'
    if with_report:
        json.dump(report(status=status), open(rep, 'w')); open(rep, 'a').write('\n')
        json.dump({'fast_p50_ms': 28.02, 'verdict': 'stable', 'processes': [{}], 'probe_errors': []},
                  open(f'{ws}/reports/check-{cid}.stability.json', 'w'))
        log = (f'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4: {log_status or status} (158.5x GPU-event vs best available baseline)\n'
               f'{{"results": "{rep}", "passed": 1, "total": 1}}\nstability: reports/check-{cid}.stability.json\n')
    else:
        log = f'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4: {log_status or status} (x)\nTraceback: boom\n'
    open(base + '.log', 'w').write(log)
    if with_tgz:
        open(f'{ws}/submission.py', 'w').write('import mk_host\n')
        subprocess.run(['tar', 'czf', base + '.tgz', '--exclude=./.harness', '.'], cwd=ws, check=True)
        if truncate:
            data = open(base + '.tgz', 'rb').read()
            open(base + '.tgz', 'wb').write(data[:len(data) // 2])
    json.dump({'id': cid, 'rc': rc, 'start': '2026-11-19T01:00:00Z', 'end': '2026-11-19T01:05:00Z'},
              open(base + '.json', 'w'))
    return base


def _run_dir():
    d = tempfile.mkdtemp(prefix='ss-neg-')
    os.makedirs(f'{d}/ws'); os.makedirs(f'{d}/snapshots'); os.makedirs(f'{d}/logs')
    json.dump({'run_id': 'neg', 'arm': 'A0S', 'case': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4',
               'task': 'improve-b8-tp4', 'seed': 'dsv41_b8_tp4_nopf'}, open(f'{d}/logs/workspace.json', 'w'))
    return d


def _process(d):
    """snapshot.process; it chowns the snapshot to root, which only root may do."""
    cm = contextlib.nullcontext() if os.geteuid() == 0 else mock.patch.object(os, 'chown', lambda *a, **k: None)
    with cm:
        return snapshot.process(d)


def _checks(d):
    p = f'{d}/snapshots/checks.jsonl'
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _index(d):
    p = f'{d}/snapshots/index.jsonl'
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def test_snapshot_passing_check_is_recorded():
    d = _run_dir()
    _pending(f'{d}/ws', 'c1', 0, 'ok_provisional')
    new = _process(d)
    assert len(new) == 1 and new[0]['passing'] and new[0]['p50_ms'] == 28.02 and new[0]['status'] == 'ok_provisional'
    assert len(_index(d)) == 1 and os.path.exists(f'{d}/snapshots/{new[0]["tar"]}')
    assert not os.listdir(f'{d}/ws/.harness/pending')


def test_snapshot_rejects_invalid_candidate_artifacts():
    cases = {
        'rc1_ok_status': dict(rc=1, status='ok_provisional'),               # wrapper/lease failed
        'rc0_incorrect': dict(rc=0, status='incorrect'),                    # report says incorrect
        'rc1_incorrect': dict(rc=1, status='incorrect'),
        'rc0_non_megakernel': dict(rc=0, status='non_megakernel'),          # launch audit failed
        'rc0_error': dict(rc=0, status='error'),
        'rc0_timeout': dict(rc=0, status='timeout'),
        'no_report': dict(rc=0, status='ok_provisional', with_report=False, log_status='ok_provisional'),
        'no_tgz': dict(rc=0, status='ok_provisional', with_tgz=False),
        'rc_none': dict(rc=None, status='ok_provisional'),
    }
    for name, kw in cases.items():
        d = _run_dir()
        _pending(f'{d}/ws', name, **kw)
        new = _process(d)
        rows = _checks(d)
        assert new == [] and len(rows) == 1 and rows[0]['passing'] is False, (name, new, rows)
        assert _index(d) == [], name
        assert not glob_tgz(d), name
        assert not os.listdir(f'{d}/ws/.harness/pending'), name   # consumed, never left for a retry
    # the one exception that IS a pass must still be a pass (control of the control)
    d = _run_dir(); _pending(f'{d}/ws', 'ok', 0, 'ok_provisional')
    assert len(_process(d)) == 1


def glob_tgz(d):
    return [f for f in os.listdir(f'{d}/snapshots') if f.endswith('.tgz')]


def test_log_only_ok_without_report_is_not_a_pass():
    """A log line claiming ok_provisional but no evaluator report file (e.g. a tampered stock tool
    that prints the line and exits 0): not a snapshot. The rescore would reject it anyway (fresh
    stock tools, 3 fresh checks), but it must not even enter the candidate list."""
    d = _run_dir()
    _pending(f'{d}/ws', 'logonly', 0, 'ok_provisional', with_report=False, log_status='ok_provisional')
    new = _process(d)
    rows = _checks(d)
    assert new == [] and rows[0]['passing'] is False and rows[0]['status'] == 'ok_provisional' and rows[0]['report'] is None


def test_truncated_snapshot_tarball_fails_loudly():
    d = _run_dir()
    base = _pending(f'{d}/ws', 'trunc', 0, 'ok_provisional', truncate=True)
    new = _process(d)       # the pending record is complete; the tarball is moved as-is
    assert len(new) == 1
    tar = f'{d}/snapshots/{new[0]["tar"]}'
    try:
        rescore.tar_digest(tar)
    except (tarfile.TarError, EOFError, OSError):
        pass
    else:
        raise AssertionError('truncated tarball digested without error')
    try:
        rescore.extract(tar)(tempfile.mkdtemp())
    except (tarfile.TarError, EOFError, OSError):
        pass
    else:
        raise AssertionError('truncated tarball extracted without error')


def test_snapshot_without_submission_is_not_scorable():
    d = tempfile.mkdtemp(); ws = f'{d}/ws'; os.makedirs(ws)
    open(f'{ws}/NOTES.md', 'w').write('nothing here\n')
    workspace.tar_source(ws, f'{d}/s.tgz')
    out = tempfile.mkdtemp(); rescore.extract(f'{d}/s.tgz')(out)
    assert not os.path.exists(f'{out}/submission.py')
    # review_sources on a tree without submission.py: nothing reachable, but the stock check itself fails
    # (no submission) => status error => review_reports rejects; the tree digest still works
    assert rescore.tree_digest(out)


def test_parse_check_rejects_statuses_and_missing_fields():
    d = tempfile.mkdtemp()
    r = snapshot.parse_check(d, 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4: incorrect (rank 3: trial 0: AssertionError)\n')
    assert r['status'] == 'incorrect' and r['p50_ms'] is None and r['report'] is None
    r = snapshot.parse_check(d, 'Traceback (most recent call last):\n  boom\n')
    assert r['status'] is None and r['p50_ms'] is None
    # report present but unparseable
    os.makedirs(f'{d}/reports'); rep = f'{d}/reports/check-z.jsonl'; open(rep, 'w').write('{not json')
    r = snapshot.parse_check(d, f'{{"results": "{rep}", "passed": 0, "total": 1}}\n')
    assert r['status'] is None and 'parse_error' in r
    # report with status incorrect and a stability file: p50 is parsed but status is not ok
    json.dump(report(status='incorrect'), open(rep, 'w'))
    json.dump({'fast_p50_ms': 1.0, 'verdict': 'stable', 'processes': [{}], 'probe_errors': []},
              open(f'{d}/reports/check-z.stability.json', 'w'))
    r = snapshot.parse_check(d, f'{{"results": "{rep}", "passed": 0, "total": 1}}\nstability: reports/check-z.stability.json\n')
    assert r['status'] == 'incorrect' and r['p50_ms'] == 1.0 and r['launches'] is None


def test_review_sources_catches_oracle_and_tensor_files():
    d = tempfile.mkdtemp()
    open(f'{d}/submission.py', 'w').write('import mk_host\ndef build(case):\n    return mk_host.Runner(case)\n')
    bad = {
        'oracle import': 'from megabench.tasks.oracle import fp32_reference\n',
        'reference import': 'from reference.megabench.tasks import dsv41\n',
        'dynamic import': 'import importlib\nm = importlib.import_module("megabench.workloads")\n',
        'sys.path': 'import sys\nsys.path.insert(0, "reference")\n',
        'torch.load': 'import torch\nW = torch.load("cache.pt")\n',
        'numpy.load': 'import numpy as np\nW = np.load("w.npy")\n',
        'pickle': 'import pickle\nW = pickle.load(open("w.pkl", "rb"))\n',
        'dsv41 import': 'import megabench.tasks.dsv41 as d\n',
    }
    for what, text in bad.items():
        open(f'{d}/mk_host.py', 'w').write('import torch\n' + text + 'class Runner:\n    pass\n')
        r = review.review_sources(d)
        assert not r['ok'] and r['disqualifying'], (what, r)
    open(f'{d}/mk_host.py', 'w').write('import torch\nclass Runner:\n    pass\n')
    assert review.review_sources(d)['ok']


def test_rescore_fallback_when_no_snapshot_passes():
    """No snapshot passes review => the run's L is the seed calibration (seed-fallback), never a
    snapshot's agent-side p50; without a calibration the AUC uses SEED_L_FALLBACK_MS = 28.02."""
    assert rescore.SEED_L_FALLBACK_MS == 28.02 and rescore.B3_MS == 28.95
    import math
    assert abs(rescore.auc([], 28.02, 0.0, 120) - math.log(28.95 / 28.02)) < 1e-12


if __name__ == '__main__':
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            try:
                v(); print('PASS', k)
            except Exception as e:
                fails += 1; print('FAIL', k, repr(e))
    sys.exit(1 if fails else 0)
