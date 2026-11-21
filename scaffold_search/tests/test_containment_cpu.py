"""CPU, no GPU, no root: detection-only containment demonstration.

A process tree of the (simulated) run user appears as a GPU compute process WITHOUT SS_LEASE_ID in its
environment and outside every active lease (nvidia-smi and the lease daemon are mocked). The sampler must flag it as `bypass`; run_arm's Runner.contain
must SIGKILL it with every descendant, record the event and mark the run failed; rescore must then
score the run as failed (primary score None, source 'failed: ...').

  python3.12 -m pytest -q scaffold_search/tests/test_containment_cpu.py
"""
import json, os, subprocess, sys, tempfile, time, types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scaffold_search import run_arm, telemetry  # noqa: E402


def _tree():
    """bash -> (sleep, bash -> sleep): three descendants of the leader."""
    p = subprocess.Popen(['bash', '-c', 'sleep 600 & bash -c "sleep 600 & wait" & wait'], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if len(telemetry.descendants([p.pid])) >= 4:
            break
        time.sleep(0.1)
    return p


def _alive(pid):
    return os.path.exists(f'/proc/{pid}') and 'Z' not in telemetry._state(pid)


def _smi_mock(bypass_pid):
    def _smi(args):
        if args[0].startswith('--query-gpu='):
            return [f'{i}, 00000000:0{i}:00.0, 97, 150309' for i in range(4)]
        return [f'{bypass_pid}, 00000000:02:00.0']
    return _smi


def _sample_twice(s, smi, active, env=None):
    """two consecutive samples with the daemon status `active` (list / None = daemon down) and environ `env`"""
    with mock.patch.object(telemetry, '_smi', smi), mock.patch.object(telemetry, 'lease_active', lambda: active), \
         mock.patch.object(telemetry, 'lease_env_id', lambda pid: env):
        return s.sample(), s.sample()


def test_sampler_flags_unleased_gpu_process_of_the_run_user():
    """environ unreadable (None, as for root without CAP_SYS_PTRACE in the container): classification comes
    from the lease daemon's active holders; a bypass needs two consecutive samples."""
    p = _tree(); kid = telemetry.descendants([p.pid])[0]      # deepest descendant: the 'GPU process'
    uid = os.getuid()
    try:
        d = tempfile.mkdtemp(); os.makedirs(f'{d}/logs')
        s = telemetry.GpuSampler(d, uid)
        r1, r2 = _sample_twice(s, _smi_mock(kid), [], env=None)         # no lease at all
        assert r1['unleased'] == [kid] and r1['bypass'] == [] and r1['mine'] == [2] and r1['lease_daemon']
        assert r2['bypass'] == [kid] and s.last_bypass == [kid] and s.first_seen[kid] <= r1['t'] + 0.01
        assert r1['gpus'][2]['uids'] == [uid]
        # covered by an active lease of this uid whose holder is the tree leader -> not a bypass (environ unreadable)
        s = telemetry.GpuSampler(d, uid)
        r1, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L1', 'uid': uid, 'pid': p.pid}], env=None)
        assert r1['unleased'] == [] and r2['bypass'] == [] and s.last_bypass == [] and r2['leases'] == ['L1']
        # an active lease of this uid held by an unrelated tree does not cover it
        q = _tree()
        try:
            s = telemetry.GpuSampler(d, uid)
            _, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L2', 'uid': uid, 'pid': q.pid}], env=None)
            assert r2['bypass'] == [kid]
            # ... unless its (readable) environment names that active lease id; an expired id does not count
            s = telemetry.GpuSampler(d, uid)
            _, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L2', 'uid': uid, 'pid': q.pid}], env='L2')
            assert r2['bypass'] == []
            s = telemetry.GpuSampler(d, uid)
            _, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L2', 'uid': uid, 'pid': q.pid}], env='L0-expired')
            assert r2['bypass'] == [kid]
        finally:
            telemetry.kill_tree([q.pid])
        # a lease of this uid granted but not yet spawned (pid None): unknown, nothing is flagged
        s = telemetry.GpuSampler(d, uid)
        _, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L3', 'uid': uid, 'pid': None}], env=None)
        assert r2['bypass'] == [] and r2['unleased'] == []
        # another uid's lease is irrelevant; daemon unreachable -> no lease can be active -> flagged
        s = telemetry.GpuSampler(d, uid)
        _, r2 = _sample_twice(s, _smi_mock(kid), [{'id': 'L4', 'uid': uid + 1, 'pid': p.pid}], env=None)
        assert r2['bypass'] == [kid]
        s = telemetry.GpuSampler(d, uid)
        _, r2 = _sample_twice(s, _smi_mock(kid), None, env=None)
        assert r2['bypass'] == [kid] and r2['lease_daemon'] is False
    finally:
        telemetry.kill_tree([p.pid])


def test_bypass_needs_two_consecutive_samples():
    p = _tree(); kid = telemetry.descendants([p.pid])[0]; uid = os.getuid()
    try:
        d = tempfile.mkdtemp(); os.makedirs(f'{d}/logs')
        s = telemetry.GpuSampler(d, uid)
        none = _smi_mock(2 ** 22)                                     # a pid that does not exist
        with mock.patch.object(telemetry, 'lease_active', lambda: []), mock.patch.object(telemetry, 'lease_env_id', lambda pid: None):
            with mock.patch.object(telemetry, '_smi', _smi_mock(kid)):
                assert s.sample()['bypass'] == []
            with mock.patch.object(telemetry, '_smi', none):          # gone in between: not confirmed
                assert s.sample()['bypass'] == [] and s.pending == set()
            with mock.patch.object(telemetry, '_smi', _smi_mock(kid)):
                assert s.sample()['bypass'] == []
                assert s.sample()['bypass'] == [kid]
    finally:
        telemetry.kill_tree([p.pid])


def test_leased_membership_rules():
    """holder pid, its session/process group (new session per lease), its descendants; nothing else."""
    uid = os.getuid()
    p = _tree(); q = _tree()
    try:
        kid = telemetry.descendants([p.pid])[0]
        act = [{'id': 'L', 'uid': uid, 'pid': p.pid}]
        with mock.patch.object(telemetry, 'lease_env_id', lambda pid: None):
            assert telemetry.leased(p.pid, uid, act) is True
            assert telemetry.leased(kid, uid, act) is True
            assert telemetry.leased(q.pid, uid, act) is False
            assert telemetry.leased(kid, uid, []) is False
            assert telemetry.leased(kid, uid, [{'id': 'L', 'uid': uid, 'pid': None}]) is None
        # /proc/<pid>/stat is readable for any pid; the sid of a tree started with start_new_session is its leader
        assert telemetry._proc_ids(kid)[2] == p.pid
    finally:
        telemetry.kill_tree([p.pid, q.pid])


def test_containment_kills_tree_records_and_fails_the_run():
    p = _tree()
    tree = telemetry.descendants([p.pid])
    assert len(tree) >= 4 and all(_alive(x) for x in tree)
    d = tempfile.mkdtemp(); os.makedirs(f'{d}/logs')
    r = run_arm.Runner.__new__(run_arm.Runner)
    r.run = {'abort_flags': []}; r.run_dir = d; r.logf = None; r.contained = set()
    r.sampler = types.SimpleNamespace(last_bypass=[p.pid], first_seen={p.pid: time.time() - 2.5})
    r.a = types.SimpleNamespace(no_push=True, fake_agent=True)
    with mock.patch.object(run_arm.snapshot, 'process', lambda *a, **k: []):
        try:
            r.tick(time.time(), {'creds': time.time(), 'push': time.time()})
            assert False, 'tick must raise Contained'
        except run_arm.Contained:
            pass
    time.sleep(0.3)
    assert not any(_alive(x) for x in tree), [x for x in tree if _alive(x)]
    run = json.load(open(f'{d}/logs/run.json'))
    ev = run['containment'][0]
    assert ev['bypass_pids'] == [p.pid] and set(tree) <= set(ev['tree']) and set(tree) <= set(ev['killed'])
    assert ev['still_alive'] == []
    assert 'sleep 600' in json.dumps(ev['cmdlines']) and str(p.pid) in ev['cmdlines']
    assert run['failed'].startswith('unauthorized GPU use') and run['abort_flags'][0].startswith('containment:')
    assert 'detect_to_kill_s' in ev and ev['detect_to_kill_s'] >= 0 and ev['first_seen_utc'] <= ev['contained_utc']
    # idempotent: the same pids are not contained twice
    assert r.contain([p.pid]) is False and len(run['containment']) == 1
    p.wait(timeout=5)


def test_rescore_marks_a_contained_run_failed():
    from scaffold_search import rescore
    d = tempfile.mkdtemp()
    for sub in ('logs', 'snapshots', 'ws'):
        os.makedirs(f'{d}/{sub}')
    json.dump({'run_id': 'c1', 'arm': 'A0S', 'case': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4',
               'task': 'improve-b8-tp4', 'seed': 'dsv41_b8_tp4_nopf'}, open(f'{d}/logs/workspace.json', 'w'))
    json.dump({'run_id': 'c1', 'abort_flags': ['containment: ...'], 'failed': 'unauthorized GPU use: test',
               'containment': [{'bypass_pids': [1234]}], 'calibration': {'L_ms': 28.02, 'status': 'ok_provisional'},
               't_agent_start': 1.0, 'budget_minutes': 10, 'fake_agent': True}, open(f'{d}/logs/run.json', 'w'))
    with mock.patch.object(rescore.telemetry, 'summarize', lambda *a: {'transcript': {}, 'gpu': {}, 'checks': [],
                                                                       'lease': {}, 'snapshots': []}), \
         mock.patch.object(rescore.review, 'scan_transcript', lambda *a, **k: {}):
        res = rescore.rescore_run(d, reps=1, final=False, push=False)
    assert res['failed'].startswith('unauthorized GPU use') and res['containment'] == [{'bypass_pids': [1234]}]
    assert res['primary']['score'] is None and res['primary']['L_ms'] is None
    assert res['primary']['source'].startswith('failed: ')
    assert res['candidates_tried'] == [] and res['final_tree'] is None   # nothing of a failed run is rescored
    assert json.load(open(f'{d}/result.json'))['failed'] == res['failed']


def test_rescore_invalidates_snapshots_of_a_contained_run():
    from scaffold_search import rescore
    d = tempfile.mkdtemp()
    for sub in ('logs', 'snapshots', 'ws'):
        os.makedirs(f'{d}/{sub}')
    json.dump({'run_id': 'c2', 'arm': 'A0S', 'case': 'c', 'task': 't', 'seed': 's'}, open(f'{d}/logs/workspace.json', 'w'))
    open(f'{d}/snapshots/index.jsonl', 'w').write(json.dumps({'n': 0, 'tar': '000-x.tgz', 'p50_ms': 20.0, 'sha256': 'ab',
                                                               'end': '2026-11-19T14:48:46Z'}) + '\n')
    json.dump({'run_id': 'c2', 'abort_flags': [], 'failed': 'unauthorized GPU use: test', 'containment': [],
               'calibration': {'L_ms': 28.0, 'status': 'ok_provisional'}, 't_agent_start': 1.0, 'budget_minutes': 10},
              open(f'{d}/logs/run.json', 'w'))
    calls = []
    with mock.patch.object(rescore, 'check_n', lambda *a, **k: calls.append(a) or None), \
         mock.patch.object(rescore.telemetry, 'summarize', lambda *a: {'transcript': {}, 'gpu': {}, 'checks': [], 'lease': {},
                                                                       'snapshots': [{'tar': '000-x.tgz', 'p50_ms': 20.0}]}), \
         mock.patch.object(rescore.review, 'scan_transcript', lambda *a, **k: {}):
        res = rescore.rescore_run(d, reps=3, final=True, push=False)
    assert calls == []                       # no GPU rescore of any candidate or of the final tree
    assert res['primary']['score'] is None and res['final_tree'] is None
    assert res['snapshots'][0]['invalidated'] == 'unauthorized GPU use: test'
