"""Run one arm once, end to end (root).

  python3.12 -m scaffold_search.run_arm --arm A0S --run-id r01 [--budget-minutes N] [--fake-agent]
         [--fake-agent-script PATH] [--no-calibrate] [--no-rescore] [--no-push] [--rescore-reps 3]

Steps: node lockdown (idempotent) -> run user -> workspace (arm + seed + overlays) -> seed
calibration check (1x, before the clock starts) -> `claude -p "<goal prompt>"` as the run user
(stream-json transcript to logs/agent.jsonl, line timestamps to logs/agent.times) -> monitor
(pending checks -> snapshots + run-branch push; credential sync every minute; periodic push every
10 min; 1 Hz GPU sampler; containment: a compute process of the run user without a lease is
SIGKILLed with all its descendants and the run is marked failed, see Runner.contain) -> at T-15 min (T-max(2, budget/4) for budgets < 60) SIGINT, then
`claude -p --resume <session> "<summary message>"` for at most 10 min -> kill the process group and
every process of the run user, `gpu_lease kill-run`, check no GPU process of the uid remains ->
final snapshot processing, telemetry, push -> rescore.py (result.json, results branch).
"""
import argparse, hashlib, json, os, signal, subprocess, sys, threading, time

from scaffold_search import isolation, snapshot, telemetry, workspace, rescore

REPO = workspace.REPO
SS = '/srv/ss'
MODEL = 'claude-opus-5-5'
RESUME_MSG = 'Time is up: write the NOTES.md final summary now; do not start new work.'
FAKE_AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_agent.sh')


def utc(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t if t is not None else time.time()))


class Contained(Exception):
    """raised from tick() once unauthorized GPU use was contained: the agent phase ends as a failed run"""


class Runner:
    def __init__(self, a):
        self.a = a
        self.contained = set()
        self.sampler = None
        arm_path = a.arm if a.arm.endswith('.toml') else f'{REPO}/arms/{a.arm}.toml'
        self.arm = workspace.load_arm(arm_path)
        self.arm_sha = hashlib.sha256(open(arm_path, 'rb').read()).hexdigest()
        self.run_id = a.run_id
        self.run_dir = f'{SS}/runs/{a.run_id}'
        self.budget = a.budget_minutes or self.arm['budget_minutes']
        self.lead = 15 if self.budget >= 60 else max(2.0, self.budget / 4)
        self.run = {'run_id': self.run_id, 'arm': self.arm['name'], 'budget_minutes': self.budget,
                    'summary_lead_minutes': self.lead, 'fake_agent': a.fake_agent, 'abort_flags': [],
                    'arm_sha256': self.arm_sha, 'model': MODEL, 'claude_version': isolation.CLAUDE_VERSION}
        self.logf = None

    def log(self, msg):
        line = f'{utc()} {msg}'
        print(line, flush=True)
        if self.logf:
            self.logf.write(line + '\n'); self.logf.flush()

    def save(self):
        json.dump(self.run, open(f'{self.run_dir}/logs/run.json', 'w'), indent=1)

    # ---------------------------------------------------------------- setup
    def setup(self):
        if os.path.exists(self.run_dir):
            sys.exit(f'{self.run_dir} exists; pick a new run id')
        lock = isolation.setup_node()
        self.run['device_lockdown'] = {'locked': len(lock['devices_locked']), 'unlockable': lock['devices_unlockable']}
        if lock['devices_unlockable']:
            self.run['abort_flags'].append('GPU device nodes not lockable (read-only): lease bypass by the run user is '
                                           'detectable (bypass_pids), not prevented')
        uid, gid = isolation.create_user(self.run_id, self.run_dir, require_creds=not self.a.fake_agent)
        self.uid = uid
        os.makedirs(f'{self.run_dir}/logs', exist_ok=True); os.chmod(f'{self.run_dir}/logs', 0o700)
        self.logf = open(f'{self.run_dir}/logs/run_arm.log', 'a')
        self.log(f'run {self.run_id} arm {self.arm["name"]} budget {self.budget} min uid {uid}')
        ws, info = workspace.build(self.run_id, self.arm, self.run_dir, uid, gid)
        self.ws = ws
        self.run['seed_sha256'] = info.get('seed_sha256')
        self.run['harness_commit'] = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True,
                                                    text=True).stdout.strip()
        self.run['harness_dirty'] = bool(subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--untracked-files=no'],
                                                        capture_output=True, text=True).stdout.strip())
        self.run['trusted_commit'] = open(f'{workspace.TRUSTED}/.ss_commit').read().strip()
        tp = f'{workspace.TRUSTED}/.ss_patches'
        self.run['trusted_patches'] = open(tp).read().split() if os.path.exists(tp) else []
        self.run['claude_version_actual'] = subprocess.run([f'{SS}/bin/claude', '--version'], capture_output=True,
                                                           text=True).stdout.strip()
        self.save()

    # ---------------------------------------------------------------- agent
    def spawn(self, argv, out_name):
        out = open(f'{self.run_dir}/logs/{out_name}.jsonl', 'ab')
        times = open(f'{self.run_dir}/logs/{out_name}.times', 'a')
        err = open(f'{self.run_dir}/logs/{out_name}.stderr', 'ab')
        p = subprocess.Popen(argv, cwd=self.ws, env=isolation.agent_env(self.run_id, self.run_dir),
                             preexec_fn=isolation.demote(self.run_id), start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)

        def pump():
            for line in p.stdout:
                out.write(line); out.flush()
                times.write(f'{time.time():.3f}\n'); times.flush()
        th = threading.Thread(target=pump, daemon=True); th.start()
        return p, th

    def claude_argv(self, prompt, resume=None):
        argv = [f'{SS}/bin/claude', '-p']
        if resume:
            argv += ['--resume', resume]
        argv += [prompt, '--model', MODEL, '--permission-mode', 'auto', '--permission-prompts', 'none',
                 '--output-format', 'stream-json', '--verbose', '--disallowedTools', 'WebFetch,WebSearch']
        return argv + list(self.arm.get('claude_extra_args', []))

    def session_id(self):
        try:
            for line in open(f'{self.run_dir}/logs/agent.jsonl'):
                d = json.loads(line)
                if d.get('session_id'):
                    return d['session_id']
        except (OSError, ValueError):
            pass
        return None

    def contain(self, bypass):
        """Unauthorized GPU use by the run user (compute process without SS_LEASE_ID): SIGKILL the
        processes and all their descendants, record it, and mark the run failed. Idempotent per pid."""
        new_pids = [p for p in bypass if p not in self.contained]
        if not new_pids:
            return False
        self.contained.update(new_pids)
        tree = telemetry.descendants(new_pids)
        cmdlines = {str(p): telemetry.cmdline(p) for p in tree}   # before the kill: /proc/<pid>/cmdline empties
        killed, alive = telemetry.kill_tree(new_pids)
        now = time.time()
        first = min((getattr(self.sampler, 'first_seen', {}) or {}).get(p, now) for p in new_pids)
        ev = {'t': utc(now), 'bypass_pids': new_pids, 'tree': tree, 'killed': killed, 'still_alive': alive,
              'cmdlines': cmdlines, 'first_seen_utc': utc(first), 'contained_utc': utc(now),
              'detect_to_kill_s': round(now - first, 2)}
        self.run.setdefault('containment', []).append(ev)
        if not self.run.get('failed'):
            self.run['failed'] = 'unauthorized GPU use: compute process of the run user outside a lease'
            self.run['abort_flags'].append(f'containment: GPU bypass pids {new_pids} killed with descendants {tree}')
        self.save()
        self.log(f'CONTAINMENT bypass pids {new_pids} tree {tree} killed {killed} alive {alive}')
        return True

    def tick(self, now, state):
        if getattr(self, 'sampler', None) is not None and self.sampler.last_bypass:
            if self.contain(list(self.sampler.last_bypass)):
                raise Contained()
        new = snapshot.process(self.run_dir)
        if new:
            for s in new:
                self.log(f'snapshot {s["tar"]} status {s["status"]} p50 {s["p50_ms"]}')
        if now - state['creds'] >= 60 and not self.a.fake_agent:
            state['creds'] = now
            try:
                isolation.sync_creds()
            except Exception as e:
                self.log(f'creds sync error {e!r}')
        if (new or now - state['push'] >= 600) and not self.a.no_push:
            state['push'] = now
            r = snapshot.push(self.run_dir, 'pass' if new else 'periodic')
            if r != 'ok':
                self.log(f'push: {r}')

    def wait_proc(self, p, until, state):
        while p.poll() is None and time.time() < until:
            self.tick(time.time(), state)
            time.sleep(5)
        return p.poll()

    def stop_group(self, p, first=signal.SIGINT, grace=60):
        if p.poll() is not None:
            return p.returncode
        try:
            os.kill(p.pid, first)
        except ProcessLookupError:
            pass
        end = time.time() + grace
        while p.poll() is None and time.time() < end:
            time.sleep(1)
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            time.sleep(10)
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
        return p.returncode

    def agent_phase(self):
        state = {'creds': 0, 'push': time.time()}
        t0 = time.time()
        t_int = t0 + (self.budget - self.lead) * 60
        t_end = t0 + self.budget * 60
        deadline = utc(t_int)
        prompt = open(f'{REPO}/prompts/goal_improve.md').read().strip().replace('{DEADLINE}', deadline)
        assert len(prompt) < 4000, len(prompt)
        open(f'{self.run_dir}/logs/prompt.txt', 'w').write(prompt)
        self.run.update({'t_agent_start': t0, 'agent_start_utc': utc(t0), 'deadline_in_prompt': deadline,
                         'interrupt_utc': utc(t_int), 'prompt_sha256': hashlib.sha256(
                             open(f'{REPO}/prompts/goal_improve.md', 'rb').read()).hexdigest()})
        self.save()
        sampler = telemetry.GpuSampler(self.run_dir, self.uid); sampler.start()
        self.sampler = sampler
        if self.a.fake_agent:
            fa = f'{self.run_dir}/fake_agent.sh'
            open(fa, 'w').write(open(self.a.fake_agent_script or FAKE_AGENT).read()); os.chmod(fa, 0o755)
            self.fake = fa
            self.run['fake_agent_script'] = self.a.fake_agent_script or FAKE_AGENT
        argv = ['/bin/bash', self.fake, 'work'] if self.a.fake_agent else self.claude_argv(prompt)
        self.log(f'agent start; prompt deadline {deadline}; argv {argv[:2]}...')
        p, th = self.spawn(argv, 'agent')
        try:
            rc = self.wait_proc(p, t_int, state)
        except Contained:
            self.log('run failed (containment): killing the agent, no summary resume')
            self.run['agent_exit'] = 'killed: containment'
            self.stop_group(p, signal.SIGKILL, grace=5)
            th.join(timeout=10)
            self.run['agent_end_utc'] = utc()
            self.save()
            self.teardown(p)
            sampler.stop(); sampler.join(timeout=5)
            self.run['t_end'] = time.time(); self.run['end_utc'] = utc()
            snapshot.process(self.run_dir, final=True)
            self.save()
            return
        if rc is None:
            self.log('T-lead reached: SIGINT to the agent')
            self.run['agent_exit'] = 'interrupted'
            rc = self.stop_group(p, signal.SIGINT, grace=60)
        else:
            self.run['agent_exit'] = f'exited rc={rc} at {utc()}'
            self.run['abort_flags'].append(f'agent exited {round((t_int - time.time()) / 60, 1)} min before the summary point')
        th.join(timeout=10)
        self.run['agent_end_utc'] = utc()
        self.save()
        # summary resume
        sid = self.session_id()
        r_until = min(time.time() + 600, t_end)
        if self.a.fake_agent:
            rp, rth = self.spawn(['/bin/bash', self.fake, 'summary'], 'resume')
        elif sid:
            self.log(f'resume {sid} for the summary (until {utc(r_until)})')
            rp, rth = self.spawn(self.claude_argv(RESUME_MSG, resume=sid), 'resume')
        else:
            rp = None
            self.run['abort_flags'].append('no session id: summary resume skipped')
        if rp is not None:
            try:
                rrc = self.wait_proc(rp, r_until, state)
            except Contained:
                self.run['agent_exit'] = str(self.run.get('agent_exit')) + ' / summary killed: containment'
                self.stop_group(rp, signal.SIGKILL, grace=5)
                rrc = rp.returncode
            if rrc is None:
                self.run['abort_flags'].append('summary resume timed out')
                self.stop_group(rp, signal.SIGINT, grace=20)
            rth.join(timeout=10)
            self.run['resume_rc'] = rp.returncode
        self.teardown(p)
        sampler.stop(); sampler.join(timeout=5)
        self.run['t_end'] = time.time(); self.run['end_utc'] = utc()
        self.tick(time.time(), state)
        snapshot.process(self.run_dir, final=True)
        self.save()

    def teardown(self, p):
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        isolation.kill_user_procs(self.run_id)
        subprocess.run([f'{SS}/bin/gpu_lease', 'kill-run', self.run_id], capture_output=True)
        time.sleep(15)  # lease daemon kill grace
        isolation.kill_user_procs(self.run_id)
        left = []
        for _ in range(6):
            left = [pid for pid in telemetry_gpu_pids() if telemetry.proc_uid(pid) == self.uid]
            if not left:
                break
            for pid in left:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            time.sleep(5)
        self.run['gpu_clean'] = not left
        self.log(f'teardown done; GPU processes of uid {self.uid} left: {left}')

    def go(self):
        self.setup()
        if not self.a.no_calibrate:
            self.log('calibration: seed check (1x)')
            self.run['calibration'] = rescore.calibrate(self.run_id, self.run_dir)
            self.log(f'calibration {self.run["calibration"]}')
            self.save()
        self.agent_phase()
        telemetry.summarize(self.run_dir, self.run_id)
        if not self.a.no_push:
            self.log(f'final push: {snapshot.push(self.run_dir, "final")}')
        if not self.a.no_rescore:
            self.log('rescore')
            r = rescore.rescore_run(self.run_dir, reps=self.a.rescore_reps, push=not self.a.no_push)
            self.log(f'result: primary {r["primary"]} auc {r["auc"]} final {r["final_tree"]}')
            if not self.a.no_push:
                snapshot.push(self.run_dir, 'result')
        isolation.retire_user(self.run_id, self.run_dir)
        self.log('done')


def telemetry_gpu_pids():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True,
                         text=True).stdout.split()
    return [int(x) for x in out if x.strip().isdigit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True); ap.add_argument('--run-id', required=True)
    ap.add_argument('--budget-minutes', type=float); ap.add_argument('--fake-agent', action='store_true')
    ap.add_argument('--fake-agent-script', help='stand-in script instead of fake_agent.sh (tests, e.g. fake_agent_bypass.sh)')
    ap.add_argument('--no-calibrate', action='store_true'); ap.add_argument('--no-rescore', action='store_true')
    ap.add_argument('--no-push', action='store_true'); ap.add_argument('--rescore-reps', type=int, default=3)
    a = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit('run as root')
    Runner(a).go()


if __name__ == '__main__':
    main()
