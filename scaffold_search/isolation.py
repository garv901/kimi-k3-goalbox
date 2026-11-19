"""Per-run isolation on the node (run as root).

  python3.12 -m scaffold_search.isolation setup-node          # idempotent node lockdown
  python3.12 -m scaffold_search.isolation create-user RUN_ID RUN_DIR
  python3.12 -m scaffold_search.isolation delete-user RUN_ID
  python3.12 -m scaffold_search.isolation proxy               # allowlist CONNECT proxy (tmux ss-proxy)
  python3.12 -m scaffold_search.isolation sync-creds          # newest-wins credential sync

What it does
- Unix user `ss-<run_id>` (primary group ss-agents), home `<run_dir>/home` (0700) holding only
  `.claude/.credentials.json` (copy of root's, kept in sync, see below), a minimal
  `.claude/settings.json` and a minimal `~/.claude.json` (onboarding done, account id, the
  workspace trusted). No history, projects, skills or plugins from root's home.
- Node lockdown: /root, /srv/ss/{build,logs,scratch,harness,results} 0700; /srv/ss/runs 0711 (no
  listing); each run dir 0711 (root), ws/home owned by the run user, snapshots/ and agent logs
  root-only. /tmp 1733 (others cannot list). The benchmark is read from the read-only snapshot
  /srv/ss/trusted/vibe-megakernel (root-owned, a-w). GPU device nodes root:ss-gpu 0660: run users
  reach the GPUs only through gpu_lease, which adds ss-gpu to the spawned command's groups.
  Where the device nodes cannot be changed (docker `--gpus` bind-mounts the host's 0666 nodes
  read-only; seen on rack 10 GB300 trays), setup_node warns and reports them as unlockable; run_arm
  records this in run.json ('device_lockdown') and the lease is then only enforced for cooperative
  users (telemetry.bypass_pids detects direct GPU use).
- Network: the node has no CAP_NET_ADMIN (no iptables), so the web block is (1) Claude Code's
  --disallowedTools WebFetch,WebSearch, (2) HTTPS_PROXY/HTTP_PROXY pointing at a local CONNECT
  proxy that allows only Anthropic/Claude hosts and logs every denied host with the client uid,
  (3) the post-run transcript scan. An agent that deliberately unsets the proxy variables can
  still reach the internet; the scan flags such commands.
- Credentials: Claude Code OAuth tokens expire after a few hours and refresh tokens may rotate, so
  per-run copies could log each other (and root) out. `sync-creds` copies the newest-expiresAt
  credential file to root and every run home (run_arm's monitor calls it every minute).
"""
import json, os, pwd, grp, shutil, socket, subprocess, sys, threading, time

SS = '/srv/ss'
TRUSTED = f'{SS}/trusted/vibe-megakernel'
CREDS_DIR = f'{SS}/creds'
SHARED_CREDS = f'{CREDS_DIR}/.credentials.json'
ROOT_CREDS = '/root/.claude/.credentials.json'
# Optional: KEY=VALUE lines with CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) or ANTHROPIC_API_KEY,
# root-only 0600. When present, these variables are added to every run user's environment (agent_env).
# ANTHROPIC_BASE_URL points Claude Code at a credential relay (docs/credential-relay.md): the real key
# stays on the relay host, ANTHROPIC_API_KEY is then a per-run relay token. The relay host is added to
# the proxy allowlist; NODE_EXTRA_CA_CERTS lets Claude Code trust the relay's private CA.
CLAUDE_ENV_FILE = f'{CREDS_DIR}/claude.env'
CLAUDE_ENV_KEYS = ('CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'NODE_EXTRA_CA_CERTS')
PROXY_PORT = 3128
PROXY_LOG = f'{SS}/proxy.jsonl'
ALLOW = ('anthropic.com', 'claude.ai', 'claude.com')
AGENT_GROUP = 'ss-agents'
GPU_GROUP = 'ss-gpu'


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def user_name(run_id):
    n = f'ss-{run_id}'
    assert len(n) <= 32 and all(c.isalnum() or c in '-_' for c in run_id), run_id
    return n


def ensure_group(name):
    try:
        return grp.getgrnam(name).gr_gid
    except KeyError:
        sh('groupadd', name)
        return grp.getgrnam(name).gr_gid


def gpu_devices():
    return sorted(f'/dev/{d}' for d in os.listdir('/dev') if d.startswith('nvidia') and
                  not os.path.isdir(f'/dev/{d}'))


def setup_node(lock_devices=True):
    ensure_group(AGENT_GROUP)
    gid = ensure_group(GPU_GROUP)
    agid = grp.getgrnam(AGENT_GROUP).gr_gid
    os.chmod('/root', 0o700)
    for d in ('build', 'logs', 'scratch', 'harness', 'results', 'runs', 'trusted', 'bin', 'lease'):
        os.makedirs(f'{SS}/{d}', exist_ok=True)
    for d in ('build', 'logs', 'scratch', 'harness', 'results', 'creds', 'schedules'):
        if os.path.isdir(f'{SS}/{d}'):
            os.chmod(f'{SS}/{d}', 0o700)
    os.chmod(SS, 0o755)
    os.chmod(f'{SS}/runs', 0o711)
    for d in ('trusted', 'bin', 'lease'):
        os.chmod(f'{SS}/{d}', 0o755)
    for f in ('lease.jsonl', 'proxy.jsonl'):
        p = f'{SS}/{f}'
        open(p, 'a').close(); os.chmod(p, 0o600)
    # trusted benchmark: root-owned, read-only for everyone
    sh('chown', '-R', 'root:root', TRUSTED)
    sh('chmod', '-R', 'a-w,a+rX', TRUSTED)
    os.chmod('/tmp', 0o1733)
    hide_shared_tmp()
    install_claude()
    locked, unlockable = [], []
    if lock_devices:
        for dev in gpu_devices():
            try:
                os.chown(dev, 0, gid); os.chmod(dev, 0o660)
                locked.append(dev)
            except OSError as e:   # e.g. docker --gpus: read-only bind mounts of the host's 0666 nodes
                unlockable.append(f'{dev}: {e.strerror}')
    if unlockable:
        print(f'WARNING: GPU device nodes could not be locked to root:{GPU_GROUP} ({len(unlockable)}): '
              f'{unlockable[0]} ...; run users can reach the GPUs without a lease (telemetry bypass_pids '
              f'detects this, nothing prevents it). See docs: credential/device isolation on rack 10.', file=sys.stderr)
    sync_creds()
    return {'devices_locked': locked, 'devices_unlockable': unlockable}


def hide_shared_tmp():
    """/tmp and /dev/shm entries not owned by run users lose their 'other' permissions, so
    leftovers of earlier experiments and builders (e.g. old check logs) are unreadable to runs."""
    run_uids = {p.pw_uid for p in pwd.getpwall() if p.pw_name.startswith('ss-')}
    for d in ('/tmp', '/dev/shm'):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            try:
                st = os.lstat(p)
                if st.st_uid not in run_uids and not os.path.islink(p) and st.st_mode & 0o007:
                    os.chmod(p, st.st_mode & ~0o007 & 0o7777)
            except OSError:
                pass


CLAUDE_VERSION = '2.1.326'


def install_claude():
    """Pinned Claude Code binary readable by run users (root's copy lives under /root)."""
    dst = f'{SS}/bin/claude-{CLAUDE_VERSION}'
    if not os.path.exists(dst):
        src = f'/root/.local/share/claude/versions/{CLAUDE_VERSION}'   # native installer layout
        if os.path.exists(src):
            shutil.copyfile(src, dst + '.tmp')
            os.chmod(dst + '.tmp', 0o755); os.replace(dst + '.tmp', dst)
        else:   # other install methods (e.g. npm -g): link the `claude` on PATH if it is the pinned version
            w = shutil.which('claude') or ''
            v = subprocess.run([w, '--version'], capture_output=True, text=True).stdout if w else ''
            real = os.path.realpath(w) if w else ''
            if not v.startswith(CLAUDE_VERSION) or real.startswith('/root/'):
                raise SystemExit(f'Claude Code {CLAUDE_VERSION} not usable by run users ({src} missing; '
                                 f'claude on PATH: {w!r} -> {real!r}, version {v.strip()!r})')
            os.symlink(real, dst)
    link = f'{SS}/bin/claude'
    if not (os.path.islink(link) and os.readlink(link) == dst):
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(dst, link)


def read_exp(path):
    try:
        return int(json.load(open(path))['claudeAiOauth'].get('expiresAt') or 0)
    except Exception:
        return -1


def atomic_copy(src, dst, uid, gid, mode):
    tmp = f'{dst}.tmp{os.getpid()}'
    shutil.copyfile(src, tmp)
    os.chown(tmp, uid, gid); os.chmod(tmp, mode)
    os.replace(tmp, dst)


def active_homes():
    out = []
    for p in pwd.getpwall():
        if p.pw_name.startswith('ss-') and p.pw_shell != '/usr/sbin/nologin' and os.path.isdir(f'{p.pw_dir}/.claude'):
            out.append((p.pw_dir, p.pw_uid, p.pw_gid))
    return out


def sync_creds():
    """Newest expiresAt wins across root's credentials and every run user's copy.

    Claude Code refreshes its OAuth token itself and may rotate the refresh token; propagating the
    newest token everywhere (every minute, from run_arm) keeps root and all runs logged in."""
    files = [(ROOT_CREDS, 0, 0)] + [(f'{h}/.claude/.credentials.json', u, g) for h, u, g in active_homes()]
    exp = [(read_exp(f), f) for f, _, _ in files]
    best_exp, best = max(exp)
    if best_exp <= 0:
        return 'no valid credentials'
    n = 0
    for (e, f), (_, u, g) in zip(exp, files):
        if e < best_exp:
            atomic_copy(best, f, u, g, 0o600); n += 1
    return f'newest={best} exp={best_exp} updated={n}'


def claude_env():
    out = {}
    try:
        for line in open(CLAUDE_ENV_FILE):
            k, sep, v = line.strip().partition('=')
            if sep and k.strip() in CLAUDE_ENV_KEYS and v.strip():
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def minimal_claude_json(ws):
    src = json.load(open('/root/.claude.json')) if os.path.exists('/root/.claude.json') else {}
    keep = {k: src[k] for k in ('oauthAccount', 'userID', 'hasCompletedOnboarding', 'lastOnboardingVersion',
                                'firstStartTime', 'migrationVersion', 'opusProMigrationComplete',
                                'hasSeenAutoDefaultNotice', 'hasResetAutoModeOptInForDefaultOffer')
            if k in src}
    keep.update({'numStartups': 1, 'autoUpdates': False, 'installMethod': src.get('installMethod', 'native'),
                 'hasCompletedOnboarding': True,
                 'projects': {ws: {'hasTrustDialogAccepted': True, 'allowedTools': [],
                                   'hasCompletedProjectOnboarding': True}}})
    key = claude_env().get('ANTHROPIC_API_KEY')
    if key:   # pre-approve the env key (relay token) so claude -p does not stop at the "use this API key?" prompt
        keep['customApiKeyResponses'] = {'approved': [key[-20:]], 'rejected': []}
    return keep


SETTINGS = {
    'theme': 'dark',
    'autoUpdates': False,
    'includeCoAuthoredBy': False,
    'env': {'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1'},
    'permissions': {'deny': ['WebFetch', 'WebSearch']},
}


def create_user(run_id, run_dir, require_creds=True):
    """require_creds=False: a fake-agent run (nothing in it talks to Anthropic) needs no credential at all."""
    name = user_name(run_id)
    agid = ensure_group(AGENT_GROUP)
    home = f'{run_dir}/home'
    os.makedirs(run_dir, exist_ok=True)
    os.chmod(run_dir, 0o711)
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        sh('useradd', '-M', '-d', home, '-g', AGENT_GROUP, '-s', '/bin/bash', name)
        pw = pwd.getpwnam(name)
    uid = pw.pw_uid
    os.makedirs(f'{home}/.claude', exist_ok=True)
    ws = f'{run_dir}/ws'
    json.dump(SETTINGS, open(f'{home}/.claude/settings.json', 'w'), indent=1)
    json.dump(minimal_claude_json(ws), open(f'{home}/.claude.json', 'w'), indent=1)
    if os.path.exists(ROOT_CREDS):
        shutil.copyfile(ROOT_CREDS, f'{home}/.claude/.credentials.json')
        os.chmod(f'{home}/.claude/.credentials.json', 0o600)
    elif require_creds and not claude_env():
        raise SystemExit(f'no Claude credentials: log in as root ({ROOT_CREDS}) or write {CLAUDE_ENV_FILE}')
    sh('chown', '-hR', f'{uid}:{agid}', home)
    os.chmod(home, 0o700)
    for f in (f'{home}/.claude.json', f'{home}/.claude/settings.json'):
        os.chmod(f, 0o600)
    os.makedirs(f'{run_dir}/tmp', exist_ok=True)
    os.chown(f'{run_dir}/tmp', uid, agid); os.chmod(f'{run_dir}/tmp', 0o700)
    return uid, agid


def agent_env(run_id, run_dir):
    """Environment for every process of the run user (claude and its tools)."""
    home = f'{run_dir}/home'
    proxy = f'http://127.0.0.1:{PROXY_PORT}'
    return claude_env() | {
        'HOME': home, 'USER': user_name(run_id), 'LOGNAME': user_name(run_id), 'SHELL': '/bin/bash',
        'PATH': '/srv/ss/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'LANG': 'C.UTF-8', 'TERM': 'dumb',
        'TMPDIR': f'{run_dir}/tmp', 'TMP': f'{run_dir}/tmp', 'TEMP': f'{run_dir}/tmp',
        'PYTHONPATH': TRUSTED, 'SS_RUN_ID': run_id,
        'HTTPS_PROXY': proxy, 'HTTP_PROXY': proxy, 'https_proxy': proxy, 'http_proxy': proxy,
        'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1',
        'DISABLE_AUTOUPDATER': '1', 'CUDA_HOME': '/usr/local/cuda',
    }


def demote(run_id):
    """preexec_fn: become the run user (primary group ss-agents, no supplementary groups)."""
    pw = pwd.getpwnam(user_name(run_id))
    def f():
        os.setgroups([])
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
        os.umask(0o022)
    return f


def exec_as(run_id, run_dir, argv, cwd=None):
    env = agent_env(run_id, run_dir)
    demote(run_id)()
    os.chdir(cwd or (f'{run_dir}/ws' if os.path.isdir(f'{run_dir}/ws') else run_dir))
    os.execvpe(argv[0], argv, env)


def kill_user_procs(run_id):
    name = user_name(run_id)
    sh('pkill', '-KILL', '-u', name, check=False)


def retire_user(run_id, run_dir):
    """End of a run: kill the user's processes, hand its files to root, lock the account.
    Users are never deleted, so a uid is never reused by a later run."""
    kill_user_procs(run_id)
    time.sleep(0.5)
    sh('chown', '-R', 'root:root', run_dir, check=False)
    sh('usermod', '-L', '-s', '/usr/sbin/nologin', user_name(run_id), check=False)


def delete_user(run_id):
    """Only for test users whose files are already root-owned (uids get reused after userdel)."""
    kill_user_procs(run_id)
    time.sleep(0.5)
    sh('userdel', user_name(run_id), check=False)


# ------------------------------------------------------------------ allowlist proxy
def _peer_uid(port):
    """uid owning the local tcp socket with source port `port` (from /proc/net/tcp)."""
    for f in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            for line in open(f).readlines()[1:]:
                p = line.split()
                if int(p[1].split(':')[1], 16) == port:
                    return int(p[7])
        except OSError:
            pass
    return None


def relay_endpoint():
    """(host, port) of ANTHROPIC_BASE_URL in claude.env, or None."""
    url = claude_env().get('ANTHROPIC_BASE_URL')
    if not url:
        return None
    from urllib.parse import urlsplit
    u = urlsplit(url)
    return (u.hostname or '').lower(), u.port or (443 if u.scheme == 'https' else 80)


def _allowed(host, port=443):
    host = host.lower().rstrip('.')
    if any(host == a or host.endswith('.' + a) for a in ALLOW):
        return port in (443, 80)
    return (host, port) == relay_endpoint()


def _pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass


def _proxy_conn(c, addr, logf, lock):
    try:
        buf = b''
        while b'\r\n\r\n' not in buf and len(buf) < 65536:
            d = c.recv(4096)
            if not d:
                return
            buf += d
        head, rest = buf.split(b'\r\n\r\n', 1)
        line = head.split(b'\r\n')[0].decode('latin1')
        method, target, _ = line.split(' ', 2)
        if method.upper() == 'CONNECT':
            host, _, port = target.rpartition(':')
            port = int(port or 443)
        else:  # plain http: absolute URI
            from urllib.parse import urlsplit
            u = urlsplit(target); host, port = u.hostname or '', u.port or 80
        ok = _allowed(host, port)
        with lock:
            if not ok or os.environ.get('SS_PROXY_LOG_ALL'):
                logf.write(json.dumps({'t': round(time.time(), 3), 'uid': _peer_uid(addr[1]), 'host': host,
                                       'port': port, 'method': method, 'allowed': ok}) + '\n'); logf.flush()
        if not ok:
            c.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n'); return
        u = socket.create_connection((host, port), timeout=30)
        u.settimeout(None)
        if method.upper() == 'CONNECT':
            c.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
            if rest:
                u.sendall(rest)
        else:
            u.sendall(buf)
        t = threading.Thread(target=_pipe, args=(u, c), daemon=True); t.start()
        _pipe(c, u); t.join()
    except Exception:
        pass
    finally:
        c.close()


def proxy(port=PROXY_PORT):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', port)); s.listen(512)
    logf = open(PROXY_LOG, 'a'); lock = threading.Lock()
    print(f'allowlist proxy on 127.0.0.1:{port}, allow {ALLOW} relay {relay_endpoint()}, log {PROXY_LOG}', flush=True)
    while True:
        c, addr = s.accept()
        threading.Thread(target=_proxy_conn, args=(c, addr, logf, lock), daemon=True).start()


def main():
    a = sys.argv[1:]
    if a[0] == 'setup-node':
        r = setup_node(lock_devices='--no-devices' not in a)
        print('node set up; GPU device nodes locked:', len(r['devices_locked']), 'unlockable:', len(r['devices_unlockable']))
    elif a[0] == 'create-user':
        print(create_user(a[1], a[2]))
    elif a[0] == 'delete-user':
        delete_user(a[1])
    elif a[0] == 'proxy':
        proxy()
    elif a[0] == 'exec':  # exec RUN_ID RUN_DIR -- CMD...
        i = a.index('--')
        exec_as(a[1], a[2], a[i + 1:])
    elif a[0] == 'sync-creds':
        print(sync_creds())
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
