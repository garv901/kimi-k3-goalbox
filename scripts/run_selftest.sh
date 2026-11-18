#!/bin/bash
# run_selftest.sh [--cpu-only] [--no-smoke] [--claude]: tests to run after bootstrap_node.sh,
# before the main schedule. Stops at the first failing stage.
#
#   1. PROTOCOL manifest (scripts/verify_protocol.py)                                    CPU, seconds
#   2. CPU unit tests: tests/test_scaffold_search.py + tests/test_package.py           CPU, seconds
#   3. gpu_lease policy tests (private daemon, mostly `sleep`; 2 tests touch a GPU)   root, ~5 min
#   4. isolation tests as a run user (needs ss-lease + ss-proxy; --claude adds 2 tiny
#      `claude -p` calls as the run user, ~$0.05)                                     root, ~2 min
#   5. arm workspaces S/A1/A2/A3 build without conflicts; overlay tools start          root, ~2 min
#   6. fake-agent smoke: run_arm --arm A0S --fake-agent --budget-minutes 10 (submits the seed;
#      calibration + 10 min + rescore 3x + final check), expect score ~1.37 = 28.95 / ~21.1 ms
#                                                                                     full node, ~35 min
# --cpu-only: stages 1-2 (any machine with python3.12 + pytest). --no-smoke: skip stage 6.
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/usr/bin/python3.12}; [ -x "$PY" ] || PY=$(command -v python3.12 || command -v python3)
CPU_ONLY=0; SMOKE=1; CLAUDE=""
for a in "$@"; do case "$a" in --cpu-only) CPU_ONLY=1;; --no-smoke) SMOKE=0;; --claude) CLAUDE=--claude;; *) echo "unknown $a"; exit 2;; esac; done
cd "$H"
sh scripts/fix_modes.sh
stage() { printf '\n######## %s\n' "$*"; }

stage "1. PROTOCOL manifest"
bash scripts/unpack_seed.sh
if [ -d /srv/ss/trusted/vibe-megakernel ]; then "$PY" scripts/verify_protocol.py --trusted /srv/ss/trusted/vibe-megakernel
else "$PY" scripts/verify_protocol.py; fi

stage "2. CPU unit tests"
"$PY" -m pytest -q -p no:cacheprovider scaffold_search/tests/test_scaffold_search.py scaffold_search/tests/test_package.py
[ $CPU_ONLY = 1 ] && { echo; echo "CPU-only selftest OK"; exit 0; }

[ "$(id -u)" = 0 ] || { echo "stages 3-6 need root on the node" >&2; exit 1; }
[ "$H" = /srv/ss/harness ] || { echo "run from /srv/ss/harness (bootstrap_node.sh sets it up)" >&2; exit 1; }
/srv/ss/bin/gpu_lease status >/dev/null || { echo "gpu_lease daemon not running (bootstrap_node.sh)" >&2; exit 1; }
idle=$(/srv/ss/bin/gpu_lease status | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(int(not d['active'] and not d['queue']))")
[ "$idle" = 1 ] || echo "WARNING: the lease daemon has active/queued leases; tests 3 and 6 will wait for them"

stage "3. gpu_lease policy tests"
"$PY" -m pytest -q -p no:cacheprovider scaffold_search/tests/test_gpu_lease.py

stage "4. isolation tests"
"$PY" scaffold_search/tests/test_isolation.py $CLAUDE

stage "5. arm workspaces"
"$PY" - <<'EOF'
import os, shutil, subprocess, sys
sys.path.insert(0, '/srv/ss/harness')
from scaffold_search import workspace
base = '/srv/ss/scratch/selftest-ws'
for arm in ('A0S', 'A1', 'A2', 'A3'):
    d = f'{base}/{arm}'
    shutil.rmtree(d, ignore_errors=True)
    ws, info = workspace.build(f'st{arm}', workspace.load_arm(f'/srv/ss/harness/arms/{arm}.toml'), d)
    print(arm, 'overlays', workspace.load_arm(f'/srv/ss/harness/arms/{arm}.toml')['overlays'],
          'overlay files', info['overlay_files'], 'seed', info['seed_sha256'][:12])
    def run(c):
        p = subprocess.run(['bash', '-c', c], cwd=ws, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PYTHONPATH='/srv/ss/trusted/vibe-megakernel'))
        return p.returncode, (p.stdout + p.stderr).strip()
    if arm in ('A1', 'A2', 'A3'):
        rc, out = run('./flag_audit'); print(f'  ./flag_audit rc={rc}'); assert rc == 0, out[-800:]
        rc, out = run('./phase_table --help'); assert rc == 0, out[-800:]
    if arm == 'A2':
        rc, out = run('./primdb q dequant gpu=B300 | wc -l'); print(f'  ./primdb q dequant gpu=B300: {out} lines')
        assert rc == 0 and int(out.split()[-1]) > 2, out
    if arm == 'A3':
        for t in ('./capture --help', './task_ab --help'):
            rc, out = run(t); assert rc == 0, out[-800:]
        assert os.path.exists(f'{ws}/.claude/agents/engine.md')
    if True:
        assert os.path.isdir(f'{ws}/.claude/skills/cuda-toolkit'), 'skills S missing'
shutil.rmtree(base, ignore_errors=True)
print('arm workspaces OK')
EOF

if [ $SMOKE = 1 ]; then
  stage "6. fake-agent smoke (10-min budget, ~35 min total)"
  RID=testF$(date -u +%m%d%H%M)
  PUSH=""; [ -s /srv/ss/snapshot_repo ] || PUSH=--no-push
  "$PY" -m scaffold_search.run_arm --arm A0S --run-id "$RID" --budget-minutes 10 --fake-agent $PUSH 2>&1 | tail -25
  "$PY" - "$RID" <<'EOF'
import json, sys
r = json.load(open(f'/srv/ss/runs/{sys.argv[1]}/result.json'))
p, cal = r['primary'], r['calibration']
print(json.dumps({'primary': p, 'calibration': cal, 'final_tree': r['final_tree'],
                  'review': (r['candidates_tried'] or [{}])[0].get('review_reports')}, indent=1))
assert cal.get('status') == 'ok_provisional', 'seed calibration did not pass'
assert p['L_ms'] and abs(p['L_ms'] / cal['L_ms'] - 1) < 0.03, 'rescore L not within 3% of the calibration'
print(f"smoke OK: seed calibration {cal['L_ms']:.3f} ms, rescore L {p['L_ms']:.3f} ms, score {p['score']} "
      f"source {p['source']}")
EOF
fi
echo; echo "selftest OK"
