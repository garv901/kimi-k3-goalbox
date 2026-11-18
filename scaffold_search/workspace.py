"""Build an agent workspace for one run (root).

  python3.12 -m scaffold_search.workspace --run-id r01 --arm arms/A0S.toml [--run-dir DIR]

Layout of /srv/ss/runs/<run_id>/ (0711 root):
  ws/          the agent workspace (owner: run user, 0700)
  home/        the run user's home (isolation.py)
  tmp/         the run user's TMPDIR
  snapshots/   root-only: <n>-<utc>.tgz + index.jsonl  (written by snapshot.py)
  logs/        root-only: agent.jsonl, run_arm.log, telemetry ...

The workspace is `make_agent_workspace --agent <base_agent> --case <case>` (trusted snapshot as
benchmark root), plus the seed sources (task improve-b8), plus the harness files:
  ./gpu_run                  -> /srv/ss/bin/gpu_lease run ...
  ./check_candidate          wrapper: tar the source tree, run the stock check under a full-node
                             (task gpus) lease, leave a pending record in .harness/pending/ for the root
                             monitor (which snapshots passing checks and pushes the run branch)
  ./profile_candidate, ./timeline_candidate   stock tools under a full-node lease
  .harness/*.stock           the stock tools
  TASK.d/harness.md          harness notes (identical in every arm)
then the arm's overlays, copied verbatim in order.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRUSTED = '/srv/ss/trusted/vibe-megakernel'
PY = '/usr/bin/python3.12'
# task -> case, seed (seeds/<seed>/, None = from scratch), gpus (= the case's ranks = the full-node lease;
# the node or slot must have exactly this many GPUs), b3_ms (score = b3_ms / L; None = not measured yet:
# rescore then reports 1/L and flags it, see PROTOCOL_TP4.md)
TASKS = {
    'improve-b8': {'case': 'dsv41-flash-dspark-step-k5-ctx2k-b8', 'seed': 'dsv41_b8_nopf', 'gpus': 8, 'b3_ms': 28.95},
    'scratch-b1': {'case': 'dsv41-flash-dspark-step-k5-ctx2k', 'seed': None, 'gpus': 8, 'b3_ms': 28.95},
    # TP4 (PROTOCOL_TP4.md): the experiment task, and the one-time builder task that ports the TP8 seed
    'improve-b8-tp4': {'case': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4', 'seed': 'dsv41_b8_tp4_nopf', 'gpus': 4,
                       'b3_ms': None},
    'port-b8-tp4': {'case': 'dsv41-flash-dspark-step-k5-ctx2k-b8-tp4', 'seed': 'dsv41_b8_nopf', 'gpus': 4,
                    'b3_ms': None},
}
CASES = {k: (v['case'], v['seed']) for k, v in TASKS.items()}


def task_of_case(case):
    for v in TASKS.values():
        if v['case'] == case:
            return v
    raise KeyError(case)


def render(text, ngpu):
    """fill the GPU-count tokens of harness texts (TP8 renders byte-identical to the original texts)."""
    return text.replace('@A_NGPU@', ('an ' if ngpu == 8 else 'a ') + str(ngpu)).replace('@NGPU@', str(ngpu))

# what the snapshot tarball holds: the source tree, not builds, reports or caches
TAR_EXCLUDES = ['./.git', './build', './reports', './.harness', 'capture-*', './__pycache__', '*/__pycache__',
                './.cache', './.torch_extensions', '*.so', '*.o', '*.ncu-rep', '*.nsys-rep', '*.sqlite',
                './progress/microbench/.build-cache', './.gpu_run']


def tar_source(src, out):
    """snapshot-style tarball of a source tree: TAR_EXCLUDES and every file > 50 MB left out."""
    big = subprocess.run(['find', '.', '-path', './.git', '-prune', '-o', '-type', 'f', '-size', '+50M', '-print'],
                         cwd=src, capture_output=True, text=True).stdout
    lst = out + '.big'
    open(lst, 'w').write(big)
    subprocess.run(['tar', 'czf', out] + [f'--exclude={e}' for e in TAR_EXCLUDES] + ['-X', lst, '.'],
                   cwd=src, check=True)
    os.unlink(lst)


def load_arm(path):
    with open(path, 'rb') as f:
        arm = tomllib.load(f)
    for k in ('name', 'base_agent', 'overlays', 'task', 'budget_minutes'):
        assert k in arm, f'{path}: missing {k}'
    assert arm['task'] in CASES, arm['task']
    arm.setdefault('max_concurrent_1gpu', 3)
    arm.setdefault('claude_extra_args', [])
    arm.setdefault('prompt', 'goal_improve.md')
    return arm


def verify_seed(seed):
    d = f'{REPO}/seeds/{seed}'
    bad = []
    for line in open(f'{d}/MANIFEST.sha256'):
        h, p = line.split(None, 1)
        p = p.strip()
        if hashlib.sha256(open(f'{d}/src/{p}', 'rb').read()).hexdigest() != h:
            bad.append(p)
    if bad:
        raise SystemExit(f'seed {seed}: sha256 mismatch: {bad}')
    return open(f'{d}/SEED_SHA256').read().strip()


def sh(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def wexec(path, text):
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)


SEED_PREBUILD = """#!/bin/sh
# Compile the submission's CUDA extension without a GPU. ./check_candidate runs this (if it exists
# and is executable) before it takes the @NGPU@-GPU lease, so nvcc does not hold the GPUs. Keep it
# building exactly what submission.py loads; if it fails, the check still builds inside the lease.
set -e
cd "$(dirname "$0")"
exec /usr/bin/python3.12 -c "import mk_host; mk_host.load_ext()"
"""


def harness_files(ws, task_prebuild=None, ngpu=8):
    os.makedirs(f'{ws}/.harness/pending', exist_ok=True)
    for tool in ('check_candidate', 'profile_candidate', 'timeline_candidate'):
        if os.path.exists(f'{ws}/{tool}'):
            os.replace(f'{ws}/{tool}', f'{ws}/.harness/{tool}.stock')
    # in-run checks time 1 process (no 2 extra gpu_noise probe processes); rescore.py uses the
    # unmodified stock tool (3 processes). Correctness, launch audit and timing are unchanged.
    stock = open(f'{ws}/.harness/check_candidate.stock').read()
    assert stock.count('gpu_noise check ') == 1, 'unexpected stock check_candidate'
    open(f'{ws}/.harness/check_candidate.stock', 'w').write(
        stock.replace('gpu_noise check ', 'gpu_noise check --processes 1 '))
    if task_prebuild and not os.path.exists(f'{ws}/prebuild'):
        wexec(f'{ws}/prebuild', render(task_prebuild.replace('{ws}', ws), ngpu))
    wexec(f'{ws}/gpu_run', render('''#!/bin/sh
# ./gpu_run -g N [-m MINUTES] [--] CMD...   run CMD on N leased GPUs (N = 1, 2, 3 or @NGPU@)
n=1; m=""
while [ $# -gt 0 ]; do
  case "$1" in
    -g|--gpus) n="$2"; shift 2;;
    -m|--max-minutes) m="--max-minutes $2"; shift 2;;
    --) shift; break;;
    -h|--help) sed -n 2p "$0"; exit 0;;
    *) break;;
  esac
done
[ $# -gt 0 ] || { echo "usage: ./gpu_run -g N [-m MINUTES] -- CMD..." >&2; exit 2; }
exec /srv/ss/bin/gpu_lease run -v --gpus "$n" $m -- "$@"
''', ngpu))
    excl = ' '.join(f"--exclude='{e}'" for e in TAR_EXCLUDES)
    wexec(f'{ws}/check_candidate', f'''#!/bin/bash
# Harness wrapper around the stock ./check_candidate (in .harness/): records the source tree that is
# checked, runs the check on {render('@A_NGPU@', ngpu)}-GPU lease, and leaves a record for the harness snapshotter.
set -u
cd "$(dirname "$0")"
id=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p .harness/pending
find . -path ./.git -prune -o -type f -size +50M -print > .harness/pending/$id.big 2>/dev/null
tar czf .harness/pending/$id.tgz --warning=no-file-changed {excl} -X .harness/pending/$id.big . 2>/dev/null
rm -f .harness/pending/$id.big
start=$(date -u +%FT%TZ)
set -o pipefail
if [ -x ./prebuild ]; then
  echo "check_candidate: ./prebuild (compile without GPUs)"
  ./prebuild 2>&1 | tail -20 || echo "check_candidate: ./prebuild failed; the check builds inside the lease"
fi
./gpu_run -g {ngpu} -m 30 -- .harness/check_candidate.stock "$@" 2>&1 | tee .harness/pending/$id.log
rc=$?
printf '{{"id":"%s","rc":%d,"start":"%s","end":"%s"}}\\n' "$id" "$rc" "$start" "$(date -u +%FT%TZ)" > .harness/pending/$id.json
exit $rc
''')
    for tool in ('profile_candidate', 'timeline_candidate'):
        wexec(f'{ws}/{tool}', f'''#!/bin/sh
# Harness wrapper: the stock tool (.harness/{tool}.stock) on {render('@A_NGPU@', ngpu)}-GPU lease.
cd "$(dirname "$0")"
exec ./gpu_run -g {ngpu} -m 30 -- .harness/{tool}.stock "$@"
''')
    shutil.copytree(f'{HERE}/harness_files', ws, dirs_exist_ok=True)
    for root, _, files in os.walk(f'{HERE}/harness_files'):
        for fn in files:
            dst = os.path.join(ws, os.path.relpath(os.path.join(root, fn), f'{HERE}/harness_files'))
            text = render(open(dst).read(), ngpu)
            open(dst, 'w').write(text)


def build(run_id, arm, run_dir, uid=None, gid=None):
    t = TASKS[arm['task']]
    case, seed = t['case'], t['seed']
    ws = f'{run_dir}/ws'
    if os.path.exists(ws):
        raise SystemExit(f'{ws} exists')
    os.makedirs(run_dir, exist_ok=True)
    os.chmod(run_dir, 0o711)
    for d in ('snapshots', 'logs'):
        os.makedirs(f'{run_dir}/{d}', exist_ok=True)
        os.chmod(f'{run_dir}/{d}', 0o700)
    sh(PY, '-m', 'megabench.integrations.make_agent_workspace', '--agent', arm['base_agent'], '--case', case,
       '--output', ws, '--benchmark-root', TRUSTED, cwd=TRUSTED, env=dict(os.environ, PYTHONPATH=TRUSTED))
    sh('chmod', '-R', 'u+w', ws)  # tools copied from the read-only trusted tree keep a-w
    info = {'run_id': run_id, 'arm': arm['name'], 'case': case, 'task': arm['task'], 'seed': seed, 'gpus': t['gpus']}
    if seed:
        info['seed_sha256'] = verify_seed(seed)
        shutil.copytree(f'{REPO}/seeds/{seed}/src', ws, dirs_exist_ok=True)
    harness_files(ws, SEED_PREBUILD if seed else None, t['gpus'])
    owner = {}
    for ov in arm['overlays']:
        src = f'{REPO}/overlays/{ov}'
        if not os.path.isdir(src):
            raise SystemExit(f'overlay {ov} missing at {src}')
        conflicts = []
        for root, dirs, files in os.walk(src):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), src)
                if os.path.lexists(f'{ws}/{rel}'):
                    conflicts.append(f'{rel} (from {owner.get(rel, "workspace/seed/harness")})')
                owner[rel] = ov
        if conflicts:
            raise SystemExit(f'overlay {ov} would overwrite: {conflicts}')
        shutil.copytree(src, ws, dirs_exist_ok=True, symlinks=True)
    info_overlays = owner
    # initial commit as the starting point of the agent's git history
    g = ['git', '-c', 'safe.directory=*', '-C', ws]
    sh(*g, 'config', 'user.name', 'agent'); sh(*g, 'config', 'user.email', 'agent@localhost')
    with open(f'{ws}/.gitignore', 'a') as f:
        f.write('/.harness/\n')
    sh(*g, 'add', '-A'); sh(*g, 'commit', '-qm', f'workspace for run {run_id} (arm {arm["name"]})')
    info['ws_tree'] = sh(*g, 'rev-parse', 'HEAD^{tree}').stdout.strip()
    if uid is not None:
        sh('chown', '-R', f'{uid}:{gid}', ws)
    os.chmod(ws, 0o700)
    info['overlay_files'] = len(info_overlays)
    json.dump(info, open(f'{run_dir}/logs/workspace.json', 'w'), indent=1)
    return ws, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--run-dir'); ap.add_argument('--user', action='store_true', help='also create the run user')
    a = ap.parse_args()
    arm = load_arm(a.arm)
    run_dir = a.run_dir or f'/srv/ss/runs/{a.run_id}'
    uid = gid = None
    if a.user:
        from scaffold_search import isolation
        uid, gid = isolation.create_user(a.run_id, run_dir)
    print(json.dumps(build(a.run_id, arm, run_dir, uid, gid)[1], indent=1))


if __name__ == '__main__':
    sys.exit(main())
