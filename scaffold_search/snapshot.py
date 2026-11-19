"""Snapshots of passing checks + pushes of the run branch (root).

  python3.12 -m scaffold_search.snapshot process RUN_DIR      # handle pending check records
  python3.12 -m scaffold_search.snapshot push RUN_DIR REASON  # push branch run/<run_id>-<arm>

Passing checks: the workspace's check_candidate wrapper tars the source tree before the check and
leaves `.harness/pending/<id>.{tgz,log,json}`. `process` (called every few seconds by run_arm's
monitor) parses the check output and report, and for a passing (`ok_provisional`, rc 0) check
moves the tarball to `<run_dir>/snapshots/<n>-<id>.tgz` (root-only) and appends a row to
`snapshots/index.jsonl`; every check (passing or not) is appended to `snapshots/checks.jsonl`, and a
copy of its report/stability files is kept under `snapshots/reports/`.

Run branch: `<run_dir>/push/` is a root-owned git repo whose tree is
  ws/        copy of the workspace (no .git, build dirs, .so/.o, caches, files > 50 MB)
  harness/   snapshots index, checks, transcript (agent*.jsonl), telemetry, workspace.json,
             agent_git_log.txt
committed and pushed to `run/<run_id>-<arm>` of the snapshot repo ($SS_SNAPSHOT_REPO, see REMOTE).
"""
import glob, hashlib, json, os, re, shutil, subprocess, sys, time

PLACEHOLDER_REMOTE = 'git@github.com:<you>/<repo>.git'


def _remote():
    """Snapshot/results repo: $SS_SNAPSHOT_REPO, else the first line of /srv/ss/snapshot_repo
    (written by scripts/bootstrap_node.sh, so tmux windows that do not inherit the variable still
    see it), else a placeholder that makes every push fail harmlessly. Our runs pushed to a
    private repo of ours; the harness pushes branches `run/<run_id>-<arm>` and `results` there."""
    r = os.environ.get('SS_SNAPSHOT_REPO', '').strip()
    if not r:
        try:
            r = open('/srv/ss/snapshot_repo').readline().strip()
        except OSError:
            r = ''
    return r or PLACEHOLDER_REMOTE


REMOTE = _remote()
MAX_BYTES = 50 * 1024 * 1024
SKIP_DIRS = {'.git', 'build', '__pycache__', '.cache', '.torch_extensions', '.build-cache', '.gpu_run', 'pending'}
SKIP_EXT = ('.so', '.o', '.a', '.pyc', '.tgz')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_info(run_dir):
    return json.load(open(f'{run_dir}/logs/workspace.json'))


def parse_check(ws, log_text):
    """status / p50 from the stock check output and its report files."""
    out = {'status': None, 'p50_ms': None, 'report': None, 'stability': None, 'launches': None}
    m = re.search(r'\{"results": "([^"]+)"', log_text)
    if m:
        out['report'] = m.group(1)
    m = re.search(r'^stability: (\S+)', log_text, re.M)
    if m:
        out['stability'] = os.path.join(ws, m.group(1))
    if out['report'] and os.path.exists(out['report']):
        try:
            rec = json.loads(open(out['report']).readline())
            out['status'] = rec.get('status')
            la = rec.get('launch_audit') or {}
            out['launches'] = la.get('gpu_kernel_count')
            ct = rec.get('candidate_timing') or {}
            out['cuda_event_p50_ms'] = ct.get('cuda_event_p50_ms')
        except Exception as e:
            out['parse_error'] = repr(e)
    if out['stability'] and os.path.exists(out['stability']):
        try:
            st = json.load(open(out['stability']))
            out['p50_ms'] = st.get('fast_p50_ms') or st.get('median_ms')
            out['verdict'] = st.get('verdict')
            out['n_processes'] = len(st.get('processes') or [])
            out['probe_errors'] = len(st.get('probe_errors') or [])
        except Exception as e:
            out['parse_error'] = repr(e)
    if out['p50_ms'] is None:
        out['p50_ms'] = out.get('cuda_event_p50_ms')
    if out['status'] is None:
        m = re.search(r': (ok_provisional|[a-z_]+) \(', log_text)
        if m:
            out['status'] = m.group(1)
    return out


def process(run_dir, final=False):
    """Handle completed pending checks. Returns the list of new passing snapshots.

    final=True (after teardown): checks that never finished (killed while queued or running)
    are logged to checks.jsonl as `incomplete`."""
    ws = f'{run_dir}/ws'
    snap = f'{run_dir}/snapshots'
    os.makedirs(f'{snap}/reports', exist_ok=True)
    new = []
    for jf in sorted(glob.glob(f'{ws}/.harness/pending/*.json')):
        cid = os.path.basename(jf)[:-5]
        try:
            rec = json.load(open(jf))
        except Exception:
            continue  # being written
        base = f'{ws}/.harness/pending/{cid}'
        log_text = open(base + '.log', errors='replace').read() if os.path.exists(base + '.log') else ''
        info = parse_check(ws, log_text)
        row = {'id': cid, 'rc': rec.get('rc'), 'start': rec.get('start'), 'end': rec.get('end'),
               'processed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | info
        for k in ('report', 'stability'):
            if info.get(k) and os.path.exists(info[k]):
                shutil.copy2(info[k], f'{snap}/reports/{cid}-{os.path.basename(info[k])}')
        shutil.copy2(base + '.log', f'{snap}/reports/{cid}.log') if os.path.exists(base + '.log') else None
        # a pass needs the evaluator's report file (status read from it), not just a log line saying so
        passing = (rec.get('rc') == 0 and info['status'] == 'ok_provisional' and bool(info.get('report'))
                   and os.path.exists(info['report']) and os.path.exists(base + '.tgz'))
        if passing:
            n = len(open(f'{snap}/index.jsonl').readlines()) if os.path.exists(f'{snap}/index.jsonl') else 0
            dst = f'{snap}/{n:03d}-{cid}.tgz'
            shutil.move(base + '.tgz', dst)
            os.chown(dst, 0, 0); os.chmod(dst, 0o600)
            row.update({'n': n, 'tar': os.path.basename(dst), 'sha256': sha256(dst), 'bytes': os.path.getsize(dst)})
            with open(f'{snap}/index.jsonl', 'a') as f:
                f.write(json.dumps(row) + '\n')
            new.append(row)
        row['passing'] = passing
        with open(f'{snap}/checks.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        for ext in ('.tgz', '.log', '.json'):
            if os.path.exists(base + ext):
                os.unlink(base + ext)
    if final:
        for tf in sorted(glob.glob(f'{ws}/.harness/pending/*.tgz')):
            cid = os.path.basename(tf)[:-4]
            base = f'{ws}/.harness/pending/{cid}'
            log_text = open(base + '.log', errors='replace').read() if os.path.exists(base + '.log') else ''
            row = {'id': cid, 'rc': None, 'status': 'incomplete', 'passing': False,
                   'note': 'check did not finish before teardown', 'log_tail': log_text[-500:],
                   'processed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            with open(f'{snap}/checks.jsonl', 'a') as f:
                f.write(json.dumps(row) + '\n')
            for ext in ('.tgz', '.log'):
                if os.path.exists(base + ext):
                    os.unlink(base + ext)
    return new


def _copy_tree(src, dst):
    skipped = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('capture-')]
        rel = os.path.relpath(root, src)
        for fn in files:
            p = os.path.join(root, fn)
            if fn.endswith(SKIP_EXT) or os.path.islink(p) or not os.path.isfile(p):
                continue
            if os.path.getsize(p) > MAX_BYTES:
                skipped.append(os.path.join(rel, fn)); continue
            q = os.path.join(dst, rel, fn)
            os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copy2(p, q)
    return skipped


def git(repo, *args, check=True):
    return subprocess.run(['git', '-c', 'safe.directory=*', '-C', repo, *args], capture_output=True,
                          text=True, check=check)


def push(run_dir, reason, remote=REMOTE, dry=False):
    info = run_info(run_dir)
    branch = f'run/{info["run_id"]}-{info["arm"]}'
    repo = f'{run_dir}/push'
    if not os.path.isdir(f'{repo}/.git'):
        os.makedirs(repo, exist_ok=True); os.chmod(repo, 0o700)
        git(repo, 'init', '-q', '-b', branch)
        git(repo, 'config', 'user.name', 'scaffold-search harness')
        git(repo, 'config', 'user.email', 'harness@localhost')
        git(repo, 'remote', 'add', 'origin', remote)
    for d in ('ws', 'harness'):
        shutil.rmtree(f'{repo}/{d}', ignore_errors=True)
    skipped = _copy_tree(f'{run_dir}/ws', f'{repo}/ws')
    h = f'{repo}/harness'
    os.makedirs(h, exist_ok=True)
    for f in ('snapshots/index.jsonl', 'snapshots/checks.jsonl', 'logs/workspace.json', 'logs/telemetry.json',
              'logs/run.json', 'logs/run_arm.log', 'logs/agent.stderr', 'logs/resume.stderr'):
        if os.path.exists(f'{run_dir}/{f}'):
            shutil.copy2(f'{run_dir}/{f}', f'{h}/{os.path.basename(f)}')
    for f in glob.glob(f'{run_dir}/logs/agent*.jsonl') + glob.glob(f'{run_dir}/logs/resume*.jsonl'):
        if os.path.getsize(f) > MAX_BYTES:
            subprocess.run(['sh', '-c', f'gzip -c {f} > {h}/{os.path.basename(f)}.gz'], check=True)
        else:
            shutil.copy2(f, h)
    lg = subprocess.run(['git', '-c', 'safe.directory=*', '-C', f'{run_dir}/ws', 'log', '--stat', '--date=iso-strict'],
                        capture_output=True, text=True)
    open(f'{h}/agent_git_log.txt', 'w').write(lg.stdout)
    if skipped:
        open(f'{h}/skipped_large_files.txt', 'w').write('\n'.join(skipped) + '\n')
    git(repo, 'add', '-A', '-f', '.')
    if git(repo, 'diff', '--cached', '--quiet', check=False).returncode != 0:
        git(repo, 'commit', '-qm', f'{reason} {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    if dry:
        return 'dry'
    p = git(repo, 'push', '-q', 'origin', f'HEAD:refs/heads/{branch}', check=False)
    return 'ok' if p.returncode == 0 else f'push failed: {p.stderr.strip()[-300:]}'


def main():
    a = sys.argv[1:]
    if a[0] == 'process':
        print(json.dumps(process(a[1]), indent=1))
    elif a[0] == 'push':
        print(push(a[1], a[2] if len(a) > 2 else 'manual'))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
