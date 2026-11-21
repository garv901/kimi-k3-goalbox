"""Run telemetry (root).

- `GpuSampler(run_dir, uid)`: 1 Hz nvidia-smi sampler -> logs/gpu_samples.jsonl: per GPU util/mem
  and the uids owning its compute processes; flags processes of the run uid that no active lease
  covers (not the holder pid / its session / its descendants per the gpu_lease daemon's status) as
  `unleased`, and as `bypass` once seen in two consecutive samples. run_arm reads `sampler.last_bypass`
  every tick and contains: `kill_tree(descendants(bypass))`, the run is marked failed (containment).
  /proc/<pid>/environ (SS_LEASE_ID) is NOT the primary test: root in the docker container has no
  CAP_SYS_PTRACE and cannot read a run user's environ (found in the GPU containment test ctest1).
- `summarize(run_dir, run_id)` -> logs/telemetry.json: GPU-busy fraction, lease slice of
  /srv/ss/lease.jsonl (waits/holds), transcript stats from logs/agent*.jsonl (+ .times line
  timestamps written by run_arm): turns, tool counts incl. Skill/Task, tokens, cost, rate-limit
  events and stall minutes, and the snapshot/check list.

  python3.12 -m scaffold_search.telemetry summarize RUN_DIR RUN_ID
"""
import collections, json, os, subprocess, sys, threading, time

from scaffold_search.workspace import NGPU

LEASE_LOG = '/srv/ss/lease.jsonl'
STALL_GAP_S = 300


def _smi(args):
    return subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'], capture_output=True,
                          text=True, timeout=20).stdout.strip().splitlines()


def proc_uid(pid):
    try:
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('Uid:'):
                return int(line.split()[1])
    except OSError:
        return None


def cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()[:200]
    except OSError:
        return None


def lease_env_id(pid):
    """SS_LEASE_ID of pid's environment: the id, '' if absent, None if /proc/<pid>/environ is unreadable.
    Root WITHOUT CAP_SYS_PTRACE (the default docker capability set) cannot read another uid's environ, so
    in the container this is None for every run-user process: never the primary leased/unleased test."""
    try:
        env = open(f'/proc/{pid}/environ', 'rb').read()
    except OSError:
        return None
    for kv in env.split(b'\0'):
        if kv.startswith(b'SS_LEASE_ID='):
            return kv[12:].decode('utf-8', 'replace')
    return ''


def _proc_ids(pid):
    """(ppid, pgid, sid) of pid from /proc/<pid>/stat (readable without ptrace rights), or None."""
    try:
        stat = open(f'/proc/{pid}/stat').read()
        f = stat[stat.rindex(')') + 2:].split()
        return int(f[1]), int(f[2]), int(f[3])
    except (OSError, ValueError, IndexError):
        return None


def lease_active():
    """Active leases of the gpu_lease daemon ({id, run, uid, pid, gpus, ...}); None if it is unreachable."""
    try:
        from scaffold_search import gpu_lease
        s = gpu_lease.connect()
        try:
            gpu_lease.send_msg(s, {'op': 'status'})
            return gpu_lease.recv_msg(s).get('active', [])
        finally:
            s.close()
    except Exception:
        return None


def leased(pid, uid, active):
    """Is compute process `pid` of `uid` covered by an active lease (list from lease_active())?
    The daemon spawns a leased command as a new session, so a leased process is the holder pid, a member of
    the holder's session or process group, or a descendant of it: the same membership rule gpu_lease uses to
    kill a lease's processes. An SS_LEASE_ID naming an active lease of this uid is an extra accept path when
    the environment happens to be readable; an expired id is not. Returns None (unknown: do not flag) while
    a lease of this uid has no holder pid yet, i.e. in the instant between grant and spawn."""
    own = [l for l in active if l.get('uid') == uid]
    if any(not l.get('pid') for l in own):
        return None
    holders = [l['pid'] for l in own]
    if pid in holders:
        return True
    ids = _proc_ids(pid)
    if ids and (ids[1] in holders or ids[2] in holders):
        return True
    if holders and pid in descendants(holders):
        return True
    env_id = lease_env_id(pid)
    return bool(env_id) and env_id in {l['id'] for l in own}


def descendants(pids):
    """pids plus every live descendant (parent chain from /proc/*/stat), deepest first."""
    kids = collections.defaultdict(list)
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            stat = open(f'/proc/{d}/stat').read()
            ppid = int(stat[stat.rindex(')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids[ppid].append(int(d))
    out, seen = [], set()
    def walk(p):
        if p in seen:
            return
        seen.add(p)
        for k in kids.get(p, []):
            walk(k)
        out.append(p)
    for p in pids:
        walk(p)
    return out


def kill_tree(pids, sig=9):
    """SIGKILL (default) the processes and all their descendants; returns (killed, still_alive)."""
    import signal
    tree = descendants(pids)
    killed = []
    for p in tree:
        try:
            os.kill(p, sig); killed.append(p)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(0.5)
    alive = [p for p in tree if os.path.exists(f'/proc/{p}') and 'Z' not in _state(p)]
    return killed, alive


def _state(pid):
    try:
        stat = open(f'/proc/{pid}/stat').read()
        return stat[stat.rindex(')') + 2:].split()[0]
    except (OSError, ValueError):
        return ''


class GpuSampler(threading.Thread):
    """Also keeps the newest `bypass` list in self.last_bypass for run_arm's containment check."""
    def __init__(self, run_dir, uid, interval=1.0):
        super().__init__(daemon=True)
        self.path = f'{run_dir}/logs/gpu_samples.jsonl'
        self.uid, self.interval = uid, interval
        self.stop_ev = threading.Event()
        self.last_bypass = []
        self.first_seen = {}     # unleased pid -> time of the sample that first showed it
        self.pending = set()     # unleased pids of the previous sample (a bypass is confirmed by two samples)

    def sample(self):
        gpus = []
        bus2idx = {}
        for line in _smi(['--query-gpu=index,pci.bus_id,utilization.gpu,memory.used']):
            i, bus, u, m = [x.strip() for x in line.split(',')]
            bus2idx[bus] = int(i)
            gpus.append({'i': int(i), 'util': int(u), 'mem_mib': int(m), 'uids': []})
        active = lease_active()
        daemon_up = active is not None
        if not daemon_up:
            active = []          # no daemon -> no lease can be active: every GPU process of the run uid is unleased
        mine, unleased = [], []
        for line in _smi(['--query-compute-apps=pid,gpu_bus_id']):
            try:
                pid, bus = [x.strip() for x in line.split(',')]
                pid = int(pid)
            except ValueError:
                continue
            uid = proc_uid(pid)
            gi = bus2idx.get(bus)
            if gi is not None and uid is not None and uid not in gpus[gi]['uids']:
                gpus[gi]['uids'].append(uid)
            if uid == self.uid:
                mine.append(gi)
                if leased(pid, self.uid, active) is False:
                    unleased.append(pid)
        now = time.time()
        for pid in unleased:
            self.first_seen.setdefault(pid, now)
        # a bypass is an unleased pid seen in two consecutive samples (never kill on one stale lease status)
        bypass = [p for p in unleased if p in self.pending]
        self.pending = set(unleased)
        self.last_bypass = bypass
        return {'t': round(now, 2), 'gpus': gpus, 'mine': sorted(set(x for x in mine if x is not None)),
                'bypass': bypass, 'unleased': unleased, 'leases': [l.get('id') for l in active],
                'lease_daemon': daemon_up}

    def run(self):
        with open(self.path, 'a') as f:
            while not self.stop_ev.is_set():
                t0 = time.time()
                try:
                    f.write(json.dumps(self.sample()) + '\n'); f.flush()
                except Exception as e:
                    f.write(json.dumps({'t': time.time(), 'error': repr(e)}) + '\n')
                self.stop_ev.wait(max(0.0, self.interval - (time.time() - t0)))

    def stop(self):
        self.stop_ev.set()


def gpu_summary(run_dir):
    p = f'{run_dir}/logs/gpu_samples.jsonl'
    if not os.path.exists(p):
        return {}
    n = busy = 0; bypass = set(); util_mine = []
    for line in open(p):
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if 'gpus' not in s:
            continue
        n += 1
        if s['mine']:
            busy += 1
            util_mine.append(sum(s['gpus'][i]['util'] for i in s['mine']) / len(s['mine']))
        bypass.update(s.get('bypass', []))
    return {'samples': n, 'gpu_busy_fraction': round(busy / n, 4) if n else None,
            'mean_util_when_busy': round(sum(util_mine) / len(util_mine), 1) if util_mine else None,
            'bypass_pids': sorted(bypass)}


def lease_summary(run_id, t0=None, t1=None):
    """lease covariates of one run inside [t0, t1]; wait_share_node = whole-node (4-GPU) wait / phase length."""
    rows = []
    if os.path.exists(LEASE_LOG):
        for line in open(LEASE_LOG):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('run') != run_id:
                continue
            if t0 and e['t'] < t0 or t1 and e['t'] > t1:
                continue
            rows.append(e)
    rel = [e for e in rows if e['ev'] == 'release']
    by_n = collections.Counter(e['n'] for e in rel)
    return {'leases': len(rel), 'by_gpus': dict(by_n),
            'wait_min_total': round(sum(e['wait_s'] for e in rel) / 60, 2),
            'wait_min_node': round(sum(e['wait_s'] for e in rel if e['n'] == NGPU) / 60, 2),
            'max_wait_min': round(max([e['wait_s'] for e in rel] or [0]) / 60, 2),
            'hold_min_total': round(sum(e['hold_s'] for e in rel) / 60, 2),
            'gpu_min_held': round(sum(e['hold_s'] * e['n'] for e in rel) / 60, 2),
            'wait_share_node': round(sum(e['wait_s'] for e in rel if e['n'] == NGPU) / (t1 - t0), 4) if t0 and t1 else None,
            'util_mean_node': (round(sum(e['util_mean'] * e['hold_s'] for e in rel if e['n'] == NGPU and e.get('util_mean') is not None)
                                     / max(1e-9, sum(e['hold_s'] for e in rel if e['n'] == NGPU and e.get('util_mean') is not None)), 1)
                               if any(e['n'] == NGPU and e.get('util_mean') is not None for e in rel) else None),
            'expired': sum(1 for e in rel if e.get('reason') == 'expired'),
            'preempted': sum(1 for e in rel if e.get('reason') == 'preempted'),
            'preempted_min_lost': round(sum(e['hold_s'] * e['n'] for e in rel if e.get('reason') == 'preempted') / 60, 2),
            'idle_reaped': sum(1 for e in rel if e.get('reason') == 'idle-reaped'),
            'abandoned': sum(1 for e in rows if e['ev'] == 'abandon')}


def transcript_summary(paths):
    tools = collections.Counter(); skills = collections.Counter(); subagents = collections.Counter()
    tok = collections.Counter(); cost = 0.0; turns = 0; results = []; rl = collections.Counter()
    models = collections.Counter(); session = None
    stall_s = 0.0; rl_blocked_at = None
    for path in paths:
        if not os.path.exists(path):
            continue
        times = []
        tp = path[:-6] + '.times' if path.endswith('.jsonl') else None
        if tp and os.path.exists(tp):
            times = [float(x) for x in open(tp).read().split()]
        last_t = None
        for i, line in enumerate(open(path, errors='replace')):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = times[i] if i < len(times) else None
            ty = d.get('type')
            if ty == 'system' and d.get('subtype') == 'init':
                session = session or d.get('session_id')
            if ty == 'assistant':
                msg = d.get('message', {})
                models[msg.get('model')] += 1
                if d.get('parent_tool_use_id') is None:
                    turns += 1
                for c in msg.get('content') or []:
                    if c.get('type') == 'tool_use':
                        tools[c['name']] += 1
                        inp = c.get('input') or {}
                        if c['name'] == 'Skill':
                            skills[inp.get('skill') or inp.get('name') or json.dumps(inp)[:60]] += 1
                        if c['name'] in ('Task', 'Agent'):
                            subagents[inp.get('subagent_type') or 'general'] += 1
                if rl_blocked_at is not None and t is not None:
                    stall_s += t - rl_blocked_at; rl_blocked_at = None
            if ty == 'rate_limit_event':
                info = d.get('rate_limit_info', {})
                rl[info.get('status')] += 1
                if info.get('status') not in ('allowed', 'allowed_warning') and t is not None and rl_blocked_at is None:
                    rl_blocked_at = t
            if ty == 'result':
                results.append({k: d.get(k) for k in ('subtype', 'is_error', 'num_turns', 'duration_ms',
                                                       'total_cost_usd', 'terminal_reason', 'stop_reason')})
                cost += d.get('total_cost_usd') or 0
                u = d.get('usage') or {}
                for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
                    tok[k] += u.get(k) or 0
            last_t = t if t is not None else last_t
    return {'session_id': session, 'assistant_messages': turns, 'tool_calls': dict(tools.most_common()),
            'skill_calls': dict(skills), 'subagent_calls': dict(subagents), 'tokens': dict(tok),
            'cost_usd': round(cost, 4), 'results': results, 'rate_limit_events': dict(rl),
            'rate_limit_stall_min': round(stall_s / 60, 2), 'models': dict(models)}


def summarize(run_dir, run_id):
    run = json.load(open(f'{run_dir}/logs/run.json')) if os.path.exists(f'{run_dir}/logs/run.json') else {}
    snaps = [json.loads(l) for l in open(f'{run_dir}/snapshots/index.jsonl')] \
        if os.path.exists(f'{run_dir}/snapshots/index.jsonl') else []
    checks = [json.loads(l) for l in open(f'{run_dir}/snapshots/checks.jsonl')] \
        if os.path.exists(f'{run_dir}/snapshots/checks.jsonl') else []
    agent_logs = [f'{run_dir}/logs/agent.jsonl', f'{run_dir}/logs/resume.jsonl']
    tel = {'run_id': run_id, 'gpu': gpu_summary(run_dir),
           'lease': lease_summary(run_id, run.get('t_agent_start'), run.get('t_end')),
           'transcript': transcript_summary(agent_logs),
           'checks': {'n': len(checks), 'passing': sum(1 for c in checks if c.get('passing'))},
           'snapshots': [{k: s.get(k) for k in ('n', 'id', 'start', 'end', 'status', 'p50_ms', 'tar', 'sha256')}
                         for s in snaps]}
    json.dump(tel, open(f'{run_dir}/logs/telemetry.json', 'w'), indent=1)
    return tel


if __name__ == '__main__':
    if sys.argv[1] == 'summarize':
        print(json.dumps(summarize(sys.argv[2], sys.argv[3]), indent=1))
