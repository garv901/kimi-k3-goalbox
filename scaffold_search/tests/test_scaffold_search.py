"""CPU tests on canned data: arm parsing, schedule, check parsing, AUC, review, transcript stats.

  python3.12 scaffold_search/tests/test_scaffold_search.py     (or pytest)
"""
import json, math, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold_search import schedule, snapshot, review, telemetry, workspace  # noqa: E402
from scaffold_search import rescore, report  # noqa: E402


def test_arms_parse():
    for a in ('A0S', 'A1', 'A2', 'A3'):
        arm = workspace.load_arm(f'{ROOT}/arms/{a}.toml')
        assert arm['name'] == a and arm['task'] == 'improve-b8' and arm['budget_minutes'] == 120
    assert workspace.load_arm(f'{ROOT}/arms/A0S.toml')['overlays'] == ['S']
    assert os.path.isdir(f'{ROOT}/overlays/S/.claude/skills/cuda-toolkit')


def test_prompt_len_and_placeholder():
    p = open(f'{ROOT}/prompts/goal_improve.md').read()
    assert '{DEADLINE}' in p and len(p.replace('{DEADLINE}', '2026-11-17T12:00:00Z')) < 4000


def test_seed_manifest():
    assert len(workspace.verify_seed('dsv41_b8_nopf')) == 64


def test_schedule_mixes_and_is_reproducible():
    s1 = schedule.make({'A0S': 3, 'A1': 3}, 3, 42, 'p')
    s2 = schedule.make({'A0S': 3, 'A1': 3}, 3, 42, 'p')
    assert s1['waves'] == s2['waves']
    assert all(len({r['arm'] for r in w}) == 2 for w in s1['waves'])
    assert sorted(r['arm'] for r in s1['order']) == ['A0S'] * 3 + ['A1'] * 3
    s3 = schedule.make({'A0S': 4, 'A1': 4, 'A2': 4, 'A3': 4}, 3, 7, 'm')
    assert len(s3['order']) == 16 and all(len({r['arm'] for r in w}) >= min(2, len(w)) for w in s3['waves'])


def test_parse_check_canned():
    d = tempfile.mkdtemp()
    rep = f'{d}/reports/check-x.jsonl'
    os.makedirs(f'{d}/reports')
    json.dump({'status': 'ok_provisional', 'launch_audit': {'gpu_kernel_count': None},
               'candidate_timing': {'cuda_event_p50_ms': 21.2}}, open(rep, 'w'))
    json.dump({'fast_p50_ms': 21.18, 'verdict': 'stable', 'processes': [{}, {}, {}], 'probe_errors': []},
              open(f'{d}/reports/check-x.stability.json', 'w'))
    log = ('dsv41-flash-dspark-step-k5-ctx2k-b8: ok_provisional (102.98x GPU-event vs best available baseline)\n'
           f'{{"results": "{rep}", "passed": 1, "total": 1}}\nstability: reports/check-x.stability.json\n')
    r = snapshot.parse_check(d, log)
    assert r['status'] == 'ok_provisional' and r['p50_ms'] == 21.18 and r['n_processes'] == 3


def test_auc():
    t0 = 1_000_000.0
    iso = lambda t: __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime(t))
    # no improvement: AUC = log(28.95/seed)
    assert abs(rescore.auc([], 21.0, t0, 120) - math.log(28.95 / 21.0)) < 1e-9
    # 20 ms from minute 60 on: half the time at the seed, half at 20
    snaps = [{'end': iso(t0 + 3600), 'p50_ms': 20.0}, {'end': iso(t0 + 4000), 'p50_ms': 25.0}]
    want = 0.5 * math.log(28.95 / 21.0) + 0.5 * math.log(28.95 / 20.0)
    assert abs(rescore.auc(snaps, 21.0, t0, 120) - want) < 1e-6


def _rec(status='ok_provisional', counts=(1,) * 8, ok=True):
    return {'status': status, 'launch_audit': {'per_rank': [{'gpu_kernel_count': c} for c in counts]},
            'correctness': {'status': 'pass' if ok else 'fail', 'per_rank': [
                {'trials': [{'seed': s, 'scenario': sc, 'outputs': {'logits': {'pass': ok}}}
                            for s, sc in ((1, 'raw'), (2, 'accept_full'), (3, 'accept_mixed'))]}]}}


def test_review_reports():
    assert review.review_reports([_rec(), _rec(), _rec()])['ok']
    assert not review.review_reports([_rec(), _rec(counts=(1,) * 7 + (2,)), _rec()])['ok']
    assert not review.review_reports([_rec(status='incorrect'), _rec(), _rec()])['ok']
    assert not review.review_reports([_rec(ok=False), _rec(), None])['ok']


def test_review_sources():
    d = tempfile.mkdtemp()
    open(f'{d}/submission.py', 'w').write('import mk_host\ndef build(case):\n    return mk_host.Runner(case)\n')
    open(f'{d}/mk_host.py', 'w').write('import torch\nSRC = "mk.cu"\n')
    open(f'{d}/mk.cu', 'w').write('#include "mk_params.h"\n')
    open(f'{d}/mk_params.h', 'w').write('// ok\n')
    open(f'{d}/devtest.py', 'w').write('from megabench.workloads import reference\n')  # not reachable
    r = review.review_sources(d)
    assert r['ok'], r
    open(f'{d}/mk_host.py', 'w').write('import torch\nW = torch.load("weights.pt")\n')
    assert not review.review_sources(d)['ok']
    open(f'{d}/mk_host.py', 'w').write('from megabench.tasks import dsv41\n')
    assert not review.review_sources(d)['ok']


def test_transcript_stats_and_scan():
    d = tempfile.mkdtemp()
    p = f'{d}/agent.jsonl'
    lines = [
        {'type': 'system', 'subtype': 'init', 'session_id': 'S1'},
        {'type': 'assistant', 'message': {'model': 'claude-opus-5-5', 'content': [
            {'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'cat /root/megabench-runs/x/NOTES.md'}},
            {'type': 'tool_use', 'name': 'Skill', 'input': {'skill': 'cuda-toolkit'}}]}},
        {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected'}},
        {'type': 'assistant', 'message': {'model': 'claude-opus-5-5', 'content': [
            {'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'curl https://github.com'}},
            {'type': 'tool_use', 'name': 'Task', 'input': {'subagent_type': 'engine'}}]}},
        {'type': 'result', 'total_cost_usd': 1.5, 'usage': {'output_tokens': 10, 'input_tokens': 5}},
    ]
    open(p, 'w').write('\n'.join(json.dumps(x) for x in lines) + '\n')
    open(f'{d}/agent.times', 'w').write('0\n10\n100\n400\n500\n')
    t = telemetry.transcript_summary([p])
    assert t['session_id'] == 'S1' and t['tool_calls']['Bash'] == 2 and t['skill_calls'] == {'cuda-toolkit': 1}
    assert t['subagent_calls'] == {'engine': 1} and t['cost_usd'] == 1.5 and t['rate_limit_stall_min'] == 5.0
    s = review.scan_transcript([p], '/srv/ss/runs/r1/ws', '/srv/ss/runs/r1')
    kinds = sorted(f['kind'] for f in s['flags'])
    assert kinds == ['network', 'outside-path'], kinds


def test_tar_digest_ignores_notes():
    import subprocess
    d = tempfile.mkdtemp()
    for name, notes in (('a', 'x'), ('b', 'y')):
        os.makedirs(f'{d}/{name}/reports')
        open(f'{d}/{name}/mk.cu', 'w').write('kernel')
        open(f'{d}/{name}/NOTES.md', 'w').write(notes)
        open(f'{d}/{name}/reports/r.jsonl', 'w').write(notes)
        subprocess.run(['tar', 'czf', f'{d}/{name}.tgz', '.'], cwd=f'{d}/{name}', check=True)
    assert rescore.tar_digest(f'{d}/a.tgz') == rescore.tar_digest(f'{d}/b.tgz')
    open(f'{d}/b/mk.cu', 'w').write('kernel2')
    subprocess.run(['tar', 'czf', f'{d}/b.tgz', '.'], cwd=f'{d}/b', check=True)
    assert rescore.tar_digest(f'{d}/a.tgz') != rescore.tar_digest(f'{d}/b.tgz')


def test_snapshots_skip_captures_and_big_files():
    import subprocess, tarfile
    d = tempfile.mkdtemp(); src = f'{d}/ws'
    os.makedirs(f'{src}/reports/capture-1'); os.makedirs(f'{src}/capture-2'); os.makedirs(f'{src}/sub')
    open(f'{src}/mk.cu', 'w').write('k')
    open(f'{src}/reports/capture-1/x.bin', 'w').write('x')
    open(f'{src}/capture-2/y.bin', 'w').write('y')
    with open(f'{src}/sub/big.bin', 'wb') as f:
        f.truncate(51 * 1024 * 1024)
    workspace.tar_source(src, f'{d}/s.tgz')
    names = tarfile.open(f'{d}/s.tgz').getnames()
    assert './mk.cu' in names and not any('capture' in n or 'big.bin' in n for n in names), names
    snapshot._copy_tree(src, f'{d}/copy')
    got = [os.path.relpath(os.path.join(r, f), f'{d}/copy') for r, _, fs in os.walk(f'{d}/copy') for f in fs]
    assert got == ['mk.cu'], got


def test_report_verdicts():
    def r(i, arm, score, auc):
        return {'run_id': f'r{i}', 'arm': arm, 'primary': {'score': score, 'L_ms': 28.95 / score, 'source': 'snapshot'},
                'auc': auc, 'final_tree': None, 'times_utc': {'agent_start_utc': f'2026-11-17T0{i}:00:00Z'},
                'secondary': {'checks': {'n': 5, 'passing': 3}, 'abort_flags': [], 'skill_calls': {}},
                'transcript_scan': {'flagged': False, 'n': 0}}
    rows = [r(1, 'A0S', 1.44, .31), r(2, 'A0S', 1.46, .30), r(3, 'A0S', 1.45, .30), r(4, 'A0S', 1.43, .29),
            r(5, 'A1', 1.70, .40), r(6, 'A1', 1.75, .41), r(7, 'A1', 1.80, .42), r(8, 'A1', 1.72, .40),
            r(9, 'A2', 1.74, .41), r(10, 'A2', 1.76, .43), r(11, 'A2', 1.71, .40), r(12, 'A2', 1.73, .42)]
    text, stats, v = report.report(rows)
    assert v['A1-A0S'].startswith('A1 vs A0S: WIN'), v
    assert v['A2-A1'].startswith('A2 vs A1: no win'), v
    assert 'A3-A1' not in v


if __name__ == '__main__':
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            try:
                v(); print('PASS', k)
            except Exception as e:
                fails += 1; print('FAIL', k, repr(e))
    sys.exit(1 if fails else 0)
