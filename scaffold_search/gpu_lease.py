#!/usr/bin/env python3
"""gpu_lease: one daemon per node that hands out 1..3-GPU and whole-node (all NGPU GPUs) leases.

dsv41-tp4-rack10-v1: NGPU = 4 (one 4x GB300 tray); the whole-node class is the 4-GPU lease.

Standalone (stdlib only) so the same file serves as the CLI for run users.

  gpu_lease daemon [--sock PATH] [--log PATH]          # root, in tmux `ss-lease`
  gpu_lease run --gpus N [--max-minutes M] -- CMD...   # blocks until granted, runs CMD
  gpu_lease status                                     # holders + queue (json)
  gpu_lease kill-run RUN_ID                            # root: kill every lease/request of a run

Design
- The daemon (root) spawns CMD itself, as the caller's uid, in a new session, with the caller's
  cwd, env and stdio fds (passed over the unix socket with SCM_RIGHTS), CUDA_VISIBLE_DEVICES set
  and the supplementary group `ss-gpu` added. When /dev/nvidia* are root:ss-gpu 0660 (isolation
  step), only lease-spawned processes of run users can open the GPUs.
- Policy (PLAN 4.5, age-ordered across classes): a whole-node lease is exclusive; the next waiting
  whole-node request blocks small requests that arrived after it, while older small requests are still
  served first (so neither class starves); inside a class grants go round-robin across runs
  (least-recently granted run first, FIFO inside a run); a run holds at most 3 GPUs in small leases; default max minutes 20 (whole node) / 60 (small),
  capped for non-root at 30 / 120. On expiry, client disconnect or cancel the session/process
  group is SIGTERMed, then SIGKILLed after 10 s, and stray GPU processes of the same (non-root)
  uid on the lease's GPUs are killed.
- Run identity: for uid != 0 the run id is the user name minus the `ss-` prefix (SS_RUN_ID is
  ignored, so an agent cannot impersonate another run); root callers use SS_RUN_ID (default root).
- Nested calls (CMD itself calls gpu_lease run): detected by SS_LEASE_ID in env; the inner command
  runs directly inside the outer lease when it asks for <= the outer lease's GPU count.
- Priority: experiment runs first; requests of runs named builder-*, smoke* or root are served only
  when no experiment-run request is waiting.
- Conditional small-lease cap: while an experiment run's whole-node request has waited >= 2 min, small
  leases older than 10 min are ended (SIGTERM, 30 s grace, SIGKILL; release reason 'preempted',
  plus a 'preempt' event). Without a waiting whole-node request small leases keep their 60/120 min limits.
- Robustness: a reaper thread releases leases/requests whose handler died (and kills their
  process group), forces kill-run after 30 s, and reaps leases whose GPUs had no compute process
  and 0% utilization for GPU_LEASE_IDLE_S seconds (default 300; reason 'idle-reaped').
- Log: one json line per event (request, grant, release) in /srv/ss/lease.jsonl; releases carry
  `util_mean`, the mean nvidia-smi utilization of the lease's GPUs sampled every 2 s.
"""
import argparse, array, grp, json, os, pwd, signal, socket, struct, subprocess, sys, threading, time

SOCK = os.environ.get('GPU_LEASE_SOCK', '/srv/ss/lease/lease.sock')
LOG = os.environ.get('GPU_LEASE_LOG', '/srv/ss/lease.jsonl')
NGPU = int(os.environ.get('GPU_LEASE_NGPU', '4'))   # GPUs of the node; the whole-node lease class
SIZES = tuple(sorted({1, 2, 3, NGPU}))              # lease sizes a run may ask for
GPU_GROUP = 'ss-gpu'
SMALL_CAP = 3                    # GPUs per run in concurrent small leases
DEFAULT_MIN = {'node': 20, 'small': 60}
NONROOT_MAX_MIN = {'node': 30, 'small': 120}
KILL_GRACE = 10
IDLE_S = float(os.environ.get('GPU_LEASE_IDLE_S', '300'))  # 0 disables the idle reaper
# conditional small-lease cap: while an experiment run's whole-node request has waited PREEMPT_WAIT_S,
# small leases older than SMALL_CAP_S are ended (SIGTERM, PREEMPT_GRACE_S, SIGKILL; 'preempted')
PREEMPT_WAIT_S = float(os.environ.get('GPU_LEASE_PREEMPT_WAIT_S', '120'))
SMALL_CAP_S = float(os.environ.get('GPU_LEASE_SMALL_CAP_S', '600'))
PREEMPT_GRACE_S = float(os.environ.get('GPU_LEASE_PREEMPT_GRACE_S', '30'))


# ---------------------------------------------------------------- framing
def send_msg(sock, obj, fds=None):
    data = json.dumps(obj).encode()
    hdr = struct.pack('!I', len(data))
    if fds:
        sock.sendmsg([hdr + data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', fds))])
    else:
        sock.sendall(hdr + data)


def recv_exact(sock, n, fds_out=None):
    buf = b''
    while len(buf) < n:
        if fds_out is not None:
            chunk, anc, _, _ = sock.recvmsg(n - len(buf), socket.CMSG_LEN(16 * 4))
            for lvl, typ, d in anc:
                if lvl == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                    a = array.array('i'); a.frombytes(d[:len(d) - len(d) % 4]); fds_out.extend(a)
        else:
            chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def recv_msg(sock, fds_out=None):
    n = struct.unpack('!I', recv_exact(sock, 4, fds_out))[0]
    return json.loads(recv_exact(sock, n, fds_out))


# ---------------------------------------------------------------- daemon
class Daemon:
    def __init__(self, sock_path, log_path, ngpu):
        self.sock_path, self.log_path, self.ngpu = sock_path, log_path, ngpu
        self.cv = threading.Condition()
        self.free = set(range(ngpu))
        self.queue = []          # pending request dicts (in arrival order)
        self.active = {}         # lease_id -> dict
        self.last_grant = {}     # run -> grant seq
        self.seq = 0
        self.logf = open(log_path, 'a', buffering=1)
        try:
            self.gpu_gid = grp.getgrnam(GPU_GROUP).gr_gid
        except KeyError:
            self.gpu_gid = None

    def log(self, **kw):
        kw['t'] = round(time.time(), 3)
        self.logf.write(json.dumps(kw) + '\n')

    # policy ---------------------------------------------------------
    def small_held(self, run):
        return sum(len(l['gpus']) for l in self.active.values() if l['run'] == run and l['n'] < self.ngpu)

    def cls(self, n):
        return 'node' if n == self.ngpu else 'small'

    def rr_order(self, reqs):
        """round-robin: least recently granted run first; FIFO (seq) inside a run."""
        heads = {}
        for r in sorted(reqs, key=lambda r: r['seq']):
            heads.setdefault(r['run'], r)
        return sorted(heads.values(), key=lambda r: (self.last_grant.get(r['run'], -1), r['seq']))

    def schedule(self):
        """Grant whatever the policy allows. Caller holds self.cv.

        Age order across classes: the next whole-node request (round-robin among runs) blocks only
        small requests that arrived after it; small requests that are older are still served
        first. A whole-node lease starts when all GPUs are free and is exclusive. Inside each class,
        grants are round-robin across runs; a run holds at most SMALL_CAP GPUs in small leases."""
        changed = True
        while changed:
            changed = False
            pend = [r for r in self.queue if not r['granted'] and not r['dead']]
            if not pend:
                return
            # experiment runs first: builder-* / smoke* requests are served only when no
            # experiment-run request is waiting (small ones held back only by their run's cap
            # do not count as waiting)
            hi = [r for r in pend if not low_priority(r['run']) and not
                  (r['n'] < self.ngpu and self.small_held(r['run']) + r['n'] > SMALL_CAP)]
            if hi:
                pend = [r for r in pend if not low_priority(r['run'])]
            big = self.rr_order([r for r in pend if r['n'] == self.ngpu])
            small = [r for r in pend if r['n'] < self.ngpu]
            nb = big[0] if big else None
            if nb is not None:
                small = [r for r in small if r['seq'] < nb['seq']]   # only older small requests
            busy_node = any(l['n'] == self.ngpu for l in self.active.values())
            if not busy_node:
                for r in self.rr_order(small):
                    if len(self.free) >= r['n'] and self.small_held(r['run']) + r['n'] <= SMALL_CAP:
                        self.grant(r, sorted(self.free)[:r['n']]); changed = True
                        break
                if changed:
                    continue
            if nb is not None and len(self.free) == self.ngpu:
                self.grant(nb, sorted(self.free)); changed = True

    def grant(self, r, gpus):
        r['granted'] = True
        r['gpus'] = gpus
        self.free -= set(gpus)
        self.seq += 1
        self.last_grant[r['run']] = self.seq
        r['t_grant'] = time.time()
        self.active[r['id']] = r
        self.queue.remove(r)
        self.log(ev='grant', id=r['id'], run=r['run'], uid=r['uid'], n=r['n'], cls=self.cls(r['n']), gpus=gpus,
                 wait_s=round(r['t_grant'] - r['t_req'], 3), max_min=r['max_min'])
        self.cv.notify_all()

    def release(self, r, reason, rc=None):
        with self.cv:
            if r['id'] in self.active:
                del self.active[r['id']]
                self.free |= set(r['gpus'])
                self.log(ev='release', id=r['id'], run=r['run'], uid=r['uid'], n=r['n'],
                         cls=self.cls(r['n']), gpus=r['gpus'],
                         wait_s=round(r['t_grant'] - r['t_req'], 3),
                         hold_s=round(time.time() - r['t_grant'], 3), reason=reason, rc=rc,
                         util_mean=round(r['util_sum'] / r['util_n'], 1) if r.get('util_n') else None,
                         cmd=r['cmd'][:300])
            elif r in self.queue:
                self.queue.remove(r)
                self.log(ev='abandon', id=r['id'], run=r['run'], uid=r['uid'], n=r['n'],
                         wait_s=round(time.time() - r['t_req'], 3), reason=reason)
            self.schedule()
            self.cv.notify_all()

    # process control -------------------------------------------------
    def kill_lease_procs(self, r, proc):
        pid = proc.pid if proc else r.get('pid')
        def members():
            out = []
            for d in os.listdir('/proc'):
                if not d.isdigit():
                    continue
                try:
                    st = open(f'/proc/{d}/stat').read()
                    f = st[st.rfind(')') + 2:].split()
                    if f[0] in 'ZX':
                        continue
                    pgid, sid = int(f[2]), int(f[3])
                except Exception:
                    continue
                if pid and (pgid == pid or sid == pid):
                    out.append(int(d))
            return out
        if pid:
            for sig in (signal.SIGTERM, None, signal.SIGKILL):
                ms = members()
                if not ms:
                    break
                if sig is None:
                    end = time.time() + r.get('grace', KILL_GRACE)
                    while time.time() < end and members():
                        time.sleep(0.2)
                    continue
                for m in ms:
                    try: os.kill(m, sig)
                    except ProcessLookupError: pass
            if proc is not None:
                try: proc.wait(timeout=5)
                except Exception: pass
        # strays of a non-root uid still on the lease's GPUs
        if r['uid'] != 0:
            for gpid in gpu_pids(r['gpus']):
                if proc_uid(gpid) == r['uid']:
                    try: os.kill(gpid, signal.SIGKILL)
                    except ProcessLookupError: pass

    def kill_run(self, run):
        with self.cv:
            victims = [l for l in self.active.values() if l['run'] == run]
            for q in self.queue:
                if q['run'] == run:
                    q['dead'] = True
            self.cv.notify_all()
        for v in victims:
            v['kill_reason'] = 'kill-run'
            v['kill_t'] = time.time()
        return len(victims)

    # connection ------------------------------------------------------
    def handle(self, conn):
        try:
            cred = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
            ppid, uid, gid = struct.unpack('3i', cred)
            fds = []
            req = recv_msg(conn, fds)
            op = req.get('op')
            if op == 'status':
                with self.cv:
                    st = {'free': sorted(self.free),
                          'active': [{k: l[k] for k in ('id', 'run', 'uid', 'n', 'gpus', 'cmd', 'pid')} |
                                     {'held_s': round(time.time() - l['t_grant'], 1)} for l in self.active.values()],
                          'queue': [{k: q[k] for k in ('id', 'run', 'uid', 'n', 'cmd')} |
                                    {'waited_s': round(time.time() - q['t_req'], 1)} for q in self.queue]}
                send_msg(conn, st); return
            if op == 'kill-run':
                if uid != 0:
                    send_msg(conn, {'error': 'root only'}); return
                send_msg(conn, {'killed': self.kill_run(req['run'])}); return
            if op == 'check-lease':
                with self.cv:
                    l = self.active.get(req.get('lease_id'))
                ok = bool(l) and l['uid'] == uid
                send_msg(conn, {'ok': ok, 'gpus': l['gpus'] if ok else []}); return
            if op != 'run':
                send_msg(conn, {'error': f'bad op {op}'}); return
            self.handle_run(conn, req, uid, gid, fds)
        except (EOFError, ConnectionError):
            pass
        except Exception as e:
            try: send_msg(conn, {'error': repr(e)})
            except Exception: pass
        finally:
            conn.close()

    def handle_run(self, conn, req, uid, gid, fds):
        n = int(req['gpus'])
        if n not in SIZES or n > self.ngpu:
            send_msg(conn, {'error': f'gpus must be one of {", ".join(map(str, SIZES))}'}); return
        kind = self.cls(n)
        mm = float(req.get('max_minutes') or DEFAULT_MIN[kind])
        if uid != 0:
            mm = min(mm, NONROOT_MAX_MIN[kind])
            name = pwd.getpwuid(uid).pw_name
            run = name[3:] if name.startswith('ss-') else name
        else:
            run = req.get('run') or 'root'
        with self.cv:
            self.seq += 1
            r = {'id': f'{int(time.time())}-{self.seq}', 'seq': self.seq, 'run': run, 'uid': uid, 'gid': gid,
                 'n': n, 'max_min': mm, 'cmd': ' '.join(req['argv'])[:1000], 't_req': time.time(),
                 'granted': False, 'dead': False, 'gpus': [], 'handler': threading.current_thread(),
                 'pid': None}
            self.queue.append(r)
            self.log(ev='request', id=r['id'], run=run, uid=uid, n=n, max_min=mm, cmd=r['cmd'][:300])
            self.schedule()
        proc = None
        try:
            send_msg(conn, {'queued': r['id'], 'run': run})
            conn.setblocking(False)
            # wait for grant; watch for disconnect
            while True:
                with self.cv:
                    if r['granted'] or r['dead']:
                        break
                    self.cv.wait(timeout=1.0)
                if peer_gone(conn):
                    self.release(r, 'disconnect-waiting'); return
            if r['dead']:
                self.release(r, 'killed-waiting')
                conn.setblocking(True); send_msg(conn, {'error': 'request killed'}); return
            conn.setblocking(True)
            env = dict(req['env'])
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, r['gpus']))
            env['SS_LEASE_ID'] = r['id']
            env['SS_LEASE_GPUS'] = str(n)
            try:
                pw = pwd.getpwuid(uid)
                groups = os.getgrouplist(pw.pw_name, pw.pw_gid)
                if self.gpu_gid is not None and self.gpu_gid not in groups:
                    groups.append(self.gpu_gid)
                proc = subprocess.Popen(req['argv'], cwd=req['cwd'], env=env,
                                        stdin=fds[0], stdout=fds[1], stderr=fds[2],
                                        user=uid, group=gid, extra_groups=groups,
                                        start_new_session=True, close_fds=True)
            except Exception as e:
                self.release(r, 'spawn-failed')
                send_msg(conn, {'error': f'spawn failed: {e!r}'}); return
            finally:
                for f in fds:
                    os.close(f)
            r['pid'] = proc.pid
            if os.environ.get('GPU_LEASE_TEST_CRASH_AFTER_SPAWN'):
                raise RuntimeError('test crash')
            send_msg(conn, {'granted': r['id'], 'gpus': r['gpus'], 'pid': proc.pid,
                            'wait_s': round(r['t_grant'] - r['t_req'], 3)})
            deadline = r['t_grant'] + mm * 60
            conn.setblocking(False)
            reason, rc = None, None
            buf = b''
            while True:
                rc = proc.poll()
                if rc is not None:
                    reason = 'exit'; break
                if r.get('kill_reason'):
                    reason = r['kill_reason']; break
                if time.time() > deadline:
                    reason = 'expired'; break
                try:
                    d = conn.recv(4096)
                    if not d:
                        reason = 'disconnect'; break
                    buf += d
                    if b'cancel' in buf:
                        reason = 'cancel'; break
                except BlockingIOError:
                    pass
                except OSError:
                    reason = 'disconnect'; break
                time.sleep(0.25)
            self.kill_lease_procs(r, proc)
            if rc is None:
                rc = proc.returncode if proc.returncode is not None else -9
            self.release(r, reason, rc)
            try:
                conn.setblocking(True)
                send_msg(conn, {'exit': rc, 'reason': reason})
            except Exception:
                pass
        finally:
            # never leave a queued request or an active lease without its handler
            if r in self.queue or r['id'] in self.active:
                self.kill_lease_procs(r, proc)
                self.release(r, 'handler-exit')

    def util_sampler(self, interval=2.0):
        """mean GPU utilization per lease (logged on release as util_mean)."""
        while True:
            time.sleep(interval)
            if os.environ.get('GPU_LEASE_TEST_NO_SMI'):   # tests: pretend every GPU is idle
                util, busy_gpus = {}, set()
            else:
                try:
                    out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                                         capture_output=True, text=True, timeout=20).stdout
                    util = {int(a): float(b) for a, b in (l.split(',') for l in out.strip().splitlines())}
                except Exception:
                    continue
                busy_gpus = {g for g, _ in gpu_pid_map()}
            now = time.time()
            with self.cv:
                for l in self.active.values():
                    if l['gpus']:
                        u = sum(util.get(g, 0.0) for g in l['gpus']) / len(l['gpus'])
                        l['util_sum'] = l.get('util_sum', 0.0) + u
                        l['util_n'] = l.get('util_n', 0) + 1
                        if u > 0 or any(g in busy_gpus for g in l['gpus']):
                            l['last_active'] = now

    def reaper(self, interval=5.0):
        """Every few seconds: release leases/requests whose handler thread died (killing their
        process group); force leases that ignore a kill-run; reap leases whose GPUs had no
        compute process and 0% utilization for IDLE_S seconds (logged reason 'idle-reaped')."""
        while True:
            time.sleep(interval)
            now = time.time()
            with self.cv:
                items = [r for r in list(self.active.values()) + list(self.queue) if r.get('run') != 'orphan']
            with self.cv:
                big_wait = [q for q in self.queue if q['n'] == self.ngpu and not q['dead'] and not low_priority(q['run'])
                            and now - q['t_req'] >= PREEMPT_WAIT_S]
                if big_wait:
                    for l in self.active.values():
                        if l['n'] < self.ngpu and l.get('run') != 'orphan' and not l.get('kill_reason') \
                                and now - l['t_grant'] > SMALL_CAP_S:
                            l['kill_reason'] = 'preempted'
                            l['kill_t'] = now
                            l['grace'] = PREEMPT_GRACE_S
                            self.log(ev='preempt', id=l['id'], run=l['run'], n=l['n'],
                                     held_s=round(now - l['t_grant'], 1), for_id=big_wait[0]['id'],
                                     for_run=big_wait[0]['run'])
            for r in items:
                h = r.get('handler')
                if h is not None and not h.is_alive():
                    self.kill_lease_procs(r, None)
                    self.release(r, 'reaped-no-handler')
                    continue
                if not r.get('granted'):
                    continue
                if r.get('kill_reason') and now - r.setdefault('kill_t', now) > 30 + r.get('grace', KILL_GRACE):
                    self.kill_lease_procs(r, None)
                    self.release(r, r['kill_reason'] + '-forced')
                    continue
                if IDLE_S > 0 and not r.get('kill_reason') and \
                        now - max(r['t_grant'], r.get('last_active', 0)) > IDLE_S:
                    r['kill_reason'] = 'idle-reaped'
                    r['kill_t'] = now

    def adopt_orphans(self):
        """At start-up, GPUs that already run compute processes (e.g. leases of a previous daemon
        instance) are held by a synthetic 'orphan' lease until those processes exit."""
        if os.environ.get('GPU_LEASE_ADOPT', '1') == '0':
            return
        only = os.environ.get('GPU_LEASE_ADOPT_PIDS')  # tests: restrict adoption to these pids
        only = {int(x) for x in only.split(',')} if only else None
        pmap = lambda: [(g, p) for g, p in gpu_pid_map() if only is None or p in only]
        busy = {}
        for gi, pid in pmap():
            busy.setdefault(gi, set()).add(pid)
        if not busy:
            return
        gpus = sorted(busy)
        with self.cv:
            self.seq += 1
            r = {'id': f'orphan-{self.seq}', 'seq': self.seq, 'run': 'orphan', 'uid': -1, 'gid': -1,
                 'n': len(gpus), 'max_min': 0, 'cmd': 'adopted at daemon start',
                 't_req': time.time(), 't_grant': time.time(), 'granted': True, 'dead': False, 'gpus': gpus}
            self.free -= set(gpus)
            self.active[r['id']] = r
            self.log(ev='adopt', id=r['id'], gpus=gpus, pids=sorted(p for v in busy.values() for p in v))

        def watch():
            while True:
                time.sleep(2)
                left = {gi for gi, _ in pmap()}
                with self.cv:
                    done = [g for g in r['gpus'] if g not in left]
                    if done:
                        r['gpus'] = [g for g in r['gpus'] if g in left]
                        self.free |= set(done)
                        self.log(ev='orphan-free', id=r['id'], gpus=done)
                        if not r['gpus']:
                            del self.active[r['id']]
                        self.schedule(); self.cv.notify_all()
                    if not r['gpus']:
                        return
        threading.Thread(target=watch, daemon=True).start()

    def serve(self):
        self.adopt_orphans()
        threading.Thread(target=self.util_sampler, daemon=True).start()
        threading.Thread(target=self.reaper, daemon=True).start()
        os.makedirs(os.path.dirname(self.sock_path), exist_ok=True)
        try: os.unlink(self.sock_path)
        except FileNotFoundError: pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.sock_path)
        os.chmod(self.sock_path, 0o666)
        s.listen(256)
        self.log(ev='daemon-start', pid=os.getpid(), ngpu=self.ngpu, gpu_gid=self.gpu_gid)
        print(f'gpu_lease daemon on {self.sock_path}, log {self.log_path}, gpu group gid {self.gpu_gid}', flush=True)
        while True:
            c, _ = s.accept()
            threading.Thread(target=self.handle, args=(c,), daemon=True).start()


LOW_PRIORITY_PREFIXES = ('builder-', 'smoke', 'root')


def low_priority(run):
    return str(run).startswith(LOW_PRIORITY_PREFIXES)


def peer_gone(conn):
    try:
        d = conn.recv(1, socket.MSG_PEEK)
        return d == b''
    except BlockingIOError:
        return False
    except OSError:
        return True


def gpu_pid_map():
    """[(gpu index, pid)] of all compute processes."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
        bus = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    idx = {}
    for line in bus.strip().splitlines():
        i, b = [x.strip() for x in line.split(',')]
        idx[b] = int(i)
    res = []
    for line in out.strip().splitlines():
        try:
            p, b = [x.strip() for x in line.split(',')]
            if b in idx:
                res.append((idx[b], int(p)))
        except ValueError:
            pass
    return res


def gpu_pids(gpus):
    """pids of compute apps on the given physical GPU indices."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
        bus = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    idx = {}
    for line in bus.strip().splitlines():
        i, b = [x.strip() for x in line.split(',')]
        idx[b] = int(i)
    pids = []
    for line in out.strip().splitlines():
        try:
            p, b = [x.strip() for x in line.split(',')]
            if idx.get(b) in gpus:
                pids.append(int(p))
        except ValueError:
            pass
    return pids


def proc_uid(pid):
    try:
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('Uid:'):
                return int(line.split()[1])
    except OSError:
        return None


# ---------------------------------------------------------------- client
def connect():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    return s


def cmd_run(a):
    argv = a.cmd
    if argv and argv[0] == '--':
        argv = argv[1:]
    if not argv:
        sys.exit('gpu_lease run: no command')
    # nested inside an existing lease: run directly
    if os.environ.get('SS_LEASE_ID'):
        outer = int(os.environ.get('SS_LEASE_GPUS', '0'))
        s = connect(); send_msg(s, {'op': 'check-lease', 'lease_id': os.environ['SS_LEASE_ID']})
        ok = recv_msg(s); s.close()
        if ok.get('ok'):
            if a.gpus > outer:
                sys.exit(f'gpu_lease: nested request for {a.gpus} GPUs inside a {outer}-GPU lease')
            os.execvp(argv[0], argv)
    state = {'cancelled': False, 'sock': None, 'granted': False}

    def on_sig(signum, frame):
        if not state['cancelled']:
            state['cancelled'] = True
            if state['sock'] is None:
                os._exit(128 + signum)
            try: state['sock'].sendall(b'cancel')
            except OSError: os._exit(128 + signum)
            if not state['granted']:
                os._exit(128 + signum)
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, on_sig)
    rc = 1
    retry_until = None
    while True:
        # (re)submit; a daemon restart while we are still queued is survived by resubmitting
        try:
            s = connect()
        except OSError:
            if retry_until is None or time.time() > retry_until:
                print('gpu_lease: cannot reach the lease daemon', file=sys.stderr); sys.exit(125)
            time.sleep(1); continue
        state['sock'] = s
        send_msg(s, {'op': 'run', 'gpus': a.gpus, 'max_minutes': a.max_minutes, 'argv': argv,
                     'cwd': os.getcwd(), 'env': dict(os.environ), 'run': os.environ.get('SS_RUN_ID')},
                 fds=[0, 1, 2])
        lost = False
        while True:
            try:
                m = recv_msg(s)
            except (EOFError, ConnectionError):
                lost = True; break
            except InterruptedError:
                continue
            if 'error' in m:
                print(f'gpu_lease: {m["error"]}', file=sys.stderr); sys.exit(125)
            if 'queued' in m and a.verbose:
                print(f'gpu_lease: queued {m["queued"]} run={m["run"]} gpus={a.gpus}', file=sys.stderr)
            if 'granted' in m:
                state['granted'] = True
                if a.verbose:
                    print(f'gpu_lease: granted GPUs {m["gpus"]} after {m["wait_s"]:.1f}s', file=sys.stderr)
            if 'exit' in m:
                rc = m['exit']
                if m['reason'] == 'preempted':
                    print(f'gpu_lease: this {a.gpus}-GPU lease was ended after {SMALL_CAP_S / 60:.0f} min because '
                          f'another job has been waiting for all {NGPU} GPUs; rerun it shorter or checkpoint long jobs',
                          file=sys.stderr)
                elif m['reason'] not in ('exit',):
                    print(f'gpu_lease: lease ended ({m["reason"]})', file=sys.stderr)
                break
        if not lost:
            break
        s.close()
        if state['granted']:
            print('gpu_lease: daemon connection lost during the lease', file=sys.stderr); sys.exit(125)
        print('gpu_lease: daemon connection lost while queued; resubmitting', file=sys.stderr)
        retry_until = time.time() + 120
        time.sleep(1)
    if rc < 0:
        rc = 128 - rc
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(prog='gpu_lease')
    sub = ap.add_subparsers(dest='op', required=True)
    d = sub.add_parser('daemon'); d.add_argument('--sock', default=SOCK); d.add_argument('--log', default=LOG)
    d.add_argument('--ngpu', type=int, default=NGPU)
    r = sub.add_parser('run'); r.add_argument('--gpus', '-g', type=int, required=True)
    r.add_argument('--max-minutes', type=float); r.add_argument('-v', '--verbose', action='store_true')
    r.add_argument('cmd', nargs=argparse.REMAINDER)
    sub.add_parser('status')
    k = sub.add_parser('kill-run'); k.add_argument('run')
    a = ap.parse_args()
    if a.op == 'daemon':
        Daemon(a.sock, a.log, a.ngpu).serve()
    elif a.op == 'run':
        cmd_run(a)
    else:
        s = connect()
        send_msg(s, {'op': a.op, 'run': getattr(a, 'run', None)})
        print(json.dumps(recv_msg(s), indent=1))


if __name__ == '__main__':
    main()
