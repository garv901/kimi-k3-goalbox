"""gpu_lease policy tests against a private daemon (temp socket/log) using dummy commands.

Node size 4 (GPU_LEASE_NGPU default; the whole-node class is the 4-GPU lease).

Run on the node as root:  python3.12 -m pytest -q scaffold_search/tests/test_gpu_lease.py
(or `python3.12 scaffold_search/tests/test_gpu_lease.py`). Uses the real GPUs only for the
visibility test (a tiny torch call); everything else is `sleep`.
"""
import json, os, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
GL = os.path.join(os.path.dirname(HERE), 'gpu_lease.py')
PY = sys.executable


class Env:
    def __init__(self, **extra_env):
        self.dir = tempfile.mkdtemp(prefix='gl_test_')
        self.sock = os.path.join(self.dir, 's.sock')
        self.log = os.path.join(self.dir, 'l.jsonl')
        self.env = dict(os.environ, GPU_LEASE_SOCK=self.sock, GPU_LEASE_LOG=self.log, GPU_LEASE_ADOPT='0')
        self.env.update(extra_env)
        self.env.pop('SS_LEASE_ID', None)
        self.d = subprocess.Popen([PY, GL, 'daemon'], env=self.env, stdout=subprocess.DEVNULL)
        for _ in range(100):
            if os.path.exists(self.sock):
                break
            time.sleep(0.05)

    def run(self, run_id, n, cmd, max_min=None, **kw):
        args = [PY, GL, 'run', '--gpus', str(n)]
        if max_min:
            args += ['--max-minutes', str(max_min)]
        return subprocess.Popen(args + ['--', 'sh', '-c', cmd], env=dict(self.env, SS_RUN_ID=run_id),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)

    def events(self):
        return [json.loads(l) for l in open(self.log)]

    def close(self):
        self.d.kill(); self.d.wait()


def intervals(ev):
    g = {e['id']: e for e in ev if e['ev'] == 'grant'}
    r = {e['id']: e for e in ev if e['ev'] == 'release'}
    return [(g[i]['run'], g[i]['n'], g[i]['t'], r[i]['t'], g[i]['gpus']) for i in g if i in r]


def overlap(a, b):
    return a[2] < b[3] - 0.05 and b[2] < a[3] - 0.05


def test_exclusive_node_lease_and_gpu_disjoint():
    E = Env()
    try:
        ps = [E.run('A', 4, 'sleep 1.5'), E.run('B', 4, 'sleep 1.5'), E.run('C', 1, 'sleep 1'),
              E.run('D', 2, 'sleep 1')]
        for p in ps:
            assert p.wait(timeout=60) == 0
        iv = intervals(E.events())
        assert len(iv) == 4
        for i, a in enumerate(iv):
            for b in iv[i + 1:]:
                if overlap(a, b):
                    assert a[1] < 4 and b[1] < 4, (a, b)
                    assert not set(a[4]) & set(b[4]), (a, b)
    finally:
        E.close()


def test_per_run_cap_and_round_robin():
    E = Env()
    try:
        blocker = E.run('X', 4, 'sleep 2')
        time.sleep(0.5)
        ps = [E.run('A', 1, 'sleep 1') for _ in range(6)]
        time.sleep(0.3)
        ps += [E.run('B', 1, 'sleep 1') for _ in range(3)]
        for p in ps + [blocker]:
            assert p.wait(timeout=60) == 0
        ev = E.events()
        iv = intervals(ev)
        # cap: A never holds more than 3 GPUs at once
        a = [x for x in iv if x[0] == 'A']
        for x in a:
            conc = sum(1 for y in a if y[2] <= x[2] < y[3])
            assert conc <= 3, conc
        # round robin: B's 3 requests are all granted in the first wave (with A's 3), not after
        # all six of A's
        grants = [e for e in ev if e['ev'] == 'grant' and e['run'] in 'AB']
        first6 = [g['run'] for g in grants[:6]]
        assert first6.count('B') == 3 and first6.count('A') == 3, first6
    finally:
        E.close()


def test_starvation_guard():
    E = Env()
    try:
        ps = [E.run(r, 1, 'sleep 2') for r in 'ABC' for _ in range(3)]  # 9 small 1-GPU requests on 4 GPUs: 4 run, 5 wait
        time.sleep(0.7)
        big = E.run('Z', 4, 'sleep 1')
        time.sleep(0.3)
        late = [E.run('D', 1, 'sleep 0.5') for _ in range(2)]
        for p in ps + [big] + late:
            assert p.wait(timeout=60) == 0
        ev = E.events()
        zreq = next(e for e in ev if e['ev'] == 'request' and e['run'] == 'Z')
        zg = next(e for e in ev if e['ev'] == 'grant' and e['run'] == 'Z')
        # no small request NEWER than Z's starts before Z (older ones may)
        reqs = {e['id']: e for e in ev if e['ev'] == 'request'}
        between = [e for e in ev if e['ev'] == 'grant' and zreq['t'] < e['t'] < zg['t']
                   and reqs[e['id']]['t'] > zreq['t']]
        assert not between, between
        dg = [e for e in ev if e['ev'] == 'grant' and e['run'] == 'D']
        assert all(e['t'] >= zg['t'] for e in dg)
    finally:
        E.close()


def test_mixed_stream_bounded_waits():
    """Two runs keep whole-node (4-GPU) requests queued back to back while two others submit 1-GPU jobs:
    neither class may starve (the old policy let 1-GPU requests wait for the whole stream)."""
    E = Env()
    try:
        stop = time.time() + 14
        loops = []
        for r in ('X', 'Y'):  # each keeps one 4-GPU request outstanding at all times
            loops.append(E.run(r, 4, f'true', ))
        import threading
        def big_stream(run):
            while time.time() < stop:
                E.run(run, 4, 'sleep 1').wait()
        def small_stream(run):
            while time.time() < stop:
                E.run(run, 1, 'sleep 1').wait()
                time.sleep(0.2)
        th = [threading.Thread(target=big_stream, args=(r,)) for r in ('X', 'Y')]
        th += [threading.Thread(target=small_stream, args=(r,)) for r in ('A', 'B')]
        for t in th: t.start()
        for t in th: t.join(timeout=120)
        for p in loops: p.wait(timeout=30)
        ev = E.events()
        g = [e for e in ev if e['ev'] == 'grant']
        small = [e for e in g if e['cls'] == 'small']
        big = [e for e in g if e['cls'] == 'node']
        assert len(small) >= 4 and len(big) >= 4, (len(small), len(big))
        # bounded: a small request waits at most for the 4-GPU leases that were already queued
        # before it (one per big stream) plus the one running; under the old policy it waited for
        # the whole 14 s stream
        hold = max(e['hold_s'] for e in ev if e['ev'] == 'release' and e['cls'] == 'node')
        assert max(e['wait_s'] for e in small) < 3 * hold + 1.0, (max(e['wait_s'] for e in small), hold)
        assert max(e['wait_s'] for e in big) < 3 * hold + 2.0, (max(e['wait_s'] for e in big), hold)
        print('   small waits max %.2fs, 4-GPU waits max %.2fs, grants small=%d big=%d' % (
            max(e['wait_s'] for e in small), max(e['wait_s'] for e in big), len(small), len(big)))
    finally:
        E.close()


def test_daemon_restart_while_queued():
    """A request queued when the daemon restarts is resubmitted to the new daemon."""
    E = Env()
    try:
        holder = E.run('A', 4, 'sleep 3')
        time.sleep(0.5)
        waiter = E.run('B', 4, 'echo RAN')
        time.sleep(0.5)
        holder.kill(); holder.wait()          # drop the holder, then restart with B queued
        E.d.kill(); E.d.wait()
        os.unlink(E.sock)
        E.d = subprocess.Popen([PY, GL, 'daemon'], env=E.env, stdout=subprocess.DEVNULL)
        out, err = waiter.communicate(timeout=60)
        assert waiter.returncode == 0 and 'RAN' in out, (out, err)
        assert 'resubmitting' in err
    finally:
        E.close()


def test_adopts_running_gpu_processes():
    """GPUs busy with processes at daemon start (old leases) are not handed out until free."""
    code = "import torch,time; x=torch.ones(1,device='cuda:3'); time.sleep(12)"
    gp = subprocess.Popen([PY, '-c', code])
    time.sleep(6)  # CUDA init
    E = Env(GPU_LEASE_ADOPT='1', GPU_LEASE_ADOPT_PIDS=str(gp.pid))
    try:
        t0 = time.time()
        p = E.run('A', 4, 'true')
        assert p.wait(timeout=60) == 0
        waited = time.time() - t0
        ev = E.events()
        assert any(e['ev'] == 'adopt' and 3 in e['gpus'] for e in ev), ev[:3]
        assert waited > 3, waited
        q = E.run('B', 1, 'true'); assert q.wait(timeout=30) == 0
    finally:
        gp.kill(); E.close()


def test_experiment_runs_before_builders():
    E = Env()
    try:
        blocker = E.run('r01', 4, 'sleep 5')
        time.sleep(0.5)
        b = E.run('builder-x', 1, 'true')       # older, low priority
        time.sleep(0.3)
        e = E.run('r02', 4, 'true')             # newer, experiment run
        for p in (blocker, b, e):
            assert p.wait(timeout=60) == 0
        ev = E.events()
        g = {x['run']: x['t'] for x in ev if x['ev'] == 'grant'}
        assert g['r02'] < g['builder-x'], g
        rel = [x for x in ev if x['ev'] == 'release' and x['run'] == 'r01']
        assert rel[0].get('util_mean') is not None, rel
    finally:
        E.close()


def _wait_pid(marker):
    for _ in range(100):
        if os.path.exists(marker) and open(marker).read().strip():
            return int(open(marker).read())
        time.sleep(0.1)
    raise AssertionError('no pid')


def test_handler_crash_releases_and_kills():
    E = Env(GPU_LEASE_TEST_CRASH_AFTER_SPAWN='1')
    try:
        marker = os.path.join(E.dir, 'c.pid')
        p = E.run('A', 1, f'echo $$ > {marker}; sleep 300')
        assert p.wait(timeout=60) != 0
        time.sleep(1)
        cpid = _wait_pid(marker)
        assert not os.path.exists(f'/proc/{cpid}'), 'child survived'
        rel = [e for e in E.events() if e['ev'] == 'release']
        assert rel and rel[-1]['reason'] == 'handler-exit', rel
    finally:
        E.close()


def test_kill_run_kills_active_lease():
    E = Env()
    try:
        marker = os.path.join(E.dir, 'c.pid')
        p = E.run('X', 1, f'echo $$ > {marker}; sleep 300')
        cpid = _wait_pid(marker)
        t0 = time.time()
        k = subprocess.run([PY, GL, 'kill-run', 'X'], env=E.env, capture_output=True, text=True)
        assert '"killed": 1' in k.stdout, k.stdout
        assert p.wait(timeout=60) != 0 and time.time() - t0 < 20
        time.sleep(0.5)
        assert not os.path.exists(f'/proc/{cpid}')
        rel = [e for e in E.events() if e['ev'] == 'release']
        assert rel[-1]['reason'] == 'kill-run', rel
    finally:
        E.close()


def test_idle_lease_is_reaped():
    E = Env(GPU_LEASE_IDLE_S='3', GPU_LEASE_TEST_NO_SMI='1')
    try:
        t0 = time.time()
        p = E.run('A', 1, 'sleep 300')
        assert p.wait(timeout=60) != 0
        assert time.time() - t0 < 25
        assert 'idle-reaped' in p.stderr.read()
        rel = [e for e in E.events() if e['ev'] == 'release']
        assert rel[-1]['reason'] == 'idle-reaped', rel
    finally:
        E.close()


def test_small_lease_preempted_for_waiting_node_lease():
    E = Env(GPU_LEASE_PREEMPT_WAIT_S='1', GPU_LEASE_SMALL_CAP_S='2', GPU_LEASE_PREEMPT_GRACE_S='2')
    try:
        small = E.run('A', 1, 'sleep 300')
        time.sleep(0.5)
        t0 = time.time()
        big = E.run('B', 4, 'true')
        assert big.wait(timeout=60) == 0
        waited = time.time() - t0
        assert small.wait(timeout=30) != 0
        assert 'ended after' in small.stderr.read()
        ev = E.events()
        rel = [e for e in ev if e['ev'] == 'release' and e['run'] == 'A']
        assert rel[0]['reason'] == 'preempted', rel
        assert any(e['ev'] == 'preempt' and e['for_run'] == 'B' for e in ev)
        assert waited < 15, waited
    finally:
        E.close()


def test_no_preemption_without_waiting_node_lease():
    E = Env(GPU_LEASE_PREEMPT_WAIT_S='1', GPU_LEASE_SMALL_CAP_S='1', GPU_LEASE_PREEMPT_GRACE_S='1')
    try:
        p = E.run('A', 1, 'sleep 8')
        q = E.run('builder-x', 4, 'true')   # low priority: does not trigger preemption
        assert p.wait(timeout=60) == 0
        assert q.wait(timeout=60) == 0
        rel = [e for e in E.events() if e['ev'] == 'release' and e['run'] == 'A']
        assert rel[0]['reason'] == 'exit', rel
    finally:
        E.close()


def test_timeout_kills_process_group():
    E = Env()
    try:
        marker = os.path.join(E.dir, 'child.pid')
        t0 = time.time()
        p = E.run('A', 1, f'sleep 300 & echo $! > {marker}; wait', max_min=0.05)  # 3 s
        rc = p.wait(timeout=60)
        dt = time.time() - t0
        assert rc != 0 and dt < 20, (rc, dt)
        assert 'expired' in p.stderr.read()
        cpid = int(open(marker).read())
        time.sleep(0.5)
        assert not os.path.exists(f'/proc/{cpid}') or 'Z' in open(f'/proc/{cpid}/stat').read().split()[2]
        rel = [e for e in E.events() if e['ev'] == 'release']
        assert rel[-1]['reason'] == 'expired'
    finally:
        E.close()


def test_client_death_releases():
    E = Env()
    try:
        marker = os.path.join(E.dir, 'child.pid')
        p = E.run('A', 4, f'echo $$ > {marker}; sleep 300')
        for _ in range(100):
            if os.path.exists(marker) and open(marker).read().strip():
                break
            time.sleep(0.1)
        cpid = int(open(marker).read())
        p.send_signal(signal.SIGKILL); p.wait()
        q = E.run('B', 4, 'true')
        assert q.wait(timeout=30) == 0
        time.sleep(0.3)
        assert not os.path.exists(f'/proc/{cpid}')
        rel = [e for e in E.events() if e['ev'] == 'release' and e['run'] == 'A']
        assert rel and rel[0]['reason'] == 'disconnect'
    finally:
        E.close()


def test_cancel_by_sigint_and_exit_code():
    E = Env()
    try:
        p = E.run('A', 1, 'exit 7')
        assert p.wait(timeout=30) == 7
        p = E.run('A', 1, 'sleep 300')
        time.sleep(1.0)
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=30) != 0
        rel = [e for e in E.events() if e['ev'] == 'release']
        assert rel[-1]['reason'] == 'cancel'
    finally:
        E.close()


def test_visibility_and_nested():
    E = Env()
    try:
        code = "import torch; print('NDEV', torch.cuda.device_count())"
        p = E.run('A', 1, f'{PY} -c "{code}"')
        out, err = p.communicate(timeout=120)
        assert 'NDEV 1' in out, (out, err)
        # nested: a 4-GPU lease whose command asks for 1 GPU runs directly (no deadlock)
        p = E.run('A', 4, f'{PY} {GL} run --gpus 1 -- sh -c "echo inner \\$CUDA_VISIBLE_DEVICES"')
        out, err = p.communicate(timeout=60)
        assert 'inner 0,1,2,3' in out, (out, err)
        grants = [e for e in E.events() if e['ev'] == 'grant']
        assert len(grants) == 2
    finally:
        E.close()


if __name__ == '__main__':
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            t = time.time()
            try:
                v(); print(f'PASS {k} ({time.time() - t:.1f}s)')
            except Exception as e:
                fails += 1; print(f'FAIL {k}: {e!r}')
    sys.exit(1 if fails else 0)
