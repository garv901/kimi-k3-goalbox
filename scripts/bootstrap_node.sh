#!/bin/bash
# bootstrap_node.sh: prepare a 4x GB300 node (protocol dsv41-tp4-rack10-v1; inside the container, as
# root) for the scaffold-search runs. Idempotent: safe to rerun after a container restart.
#
#   bash scripts/bootstrap_node.sh [--check-login]
#
# Env (all optional):
#   SS_SNAPSHOT_REPO   git URL the harness pushes run branches + `results` to (see scripts/login_notes.md);
#                      unset = no pushes (results stay on the node under /srv/ss/runs/<id>/).
#   SS_BENCHMARK_URL   clone URL of kamahori/vibe-megakernel (default https://github.com/kamahori/vibe-megakernel.git)
#   SS_SEED_REPO_URL   clone URL of the seed repo (only used if the vendored seed fails to verify)
#   SS_ALLOW_MISMATCH=1  continue although the tree differs from the PROTOCOL.md manifest (runs would be invalid)
#   SS_SKIP_GPU_CHECK=1  skip the 4x GB300 check (for a dry run on another machine)
# --check-login: also run one tiny `claude -p` as root to prove the login works (costs ~$0.01).
#
# Steps: tools -> /srv/ss/harness checkout -> seed (vendored, verified) ->
# benchmark + patch -> trusted snapshot -> PROTOCOL manifest check -> snapshot repo config ->
# gpu_lease install, groups, node lockdown (ss-gpu owns /dev/nvidia*) -> tmux ss-lease + ss-proxy ->
# sanity checks (4x GB300 sm_103, claude 2.1.326, credentials, proxy allowlist, lease daemon).
set -euo pipefail
SS=/srv/ss
H=$SS/harness
PY=/usr/bin/python3.12
CHECK_LOGIN=0; [ "${1:-}" = "--check-login" ] && CHECK_LOGIN=1
say() { printf '\n== %s\n' "$*"; }
die() { echo "bootstrap: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root (inside the container)"
HERE=$(cd "$(dirname "$0")/.." && pwd)
sh "$HERE/scripts/fix_modes.sh"   # exec bits (the branch was published through the GitHub API)

say "tools"
for t in git tmux curl patch tar xz base64 useradd groupadd usermod pkill nvidia-smi "$PY"; do
  command -v "$t" >/dev/null || die "missing $t (use container/Dockerfile)"
done
"$PY" -c "import torch, tomllib; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

say "harness checkout at $H"
mkdir -p $SS; chmod 755 $SS
if [ "$HERE" != "$H" ]; then
  if [ ! -d "$H/.git" ]; then
    git clone -q "$HERE" "$H"
    git -C "$H" remote set-url origin "$(git -C "$HERE" remote get-url origin 2>/dev/null || echo "$HERE")"
  fi
  [ "$(git -C "$H" rev-parse HEAD)" = "$(git -C "$HERE" rev-parse HEAD)" ] || \
    die "$H is at $(git -C "$H" rev-parse --short HEAD), this checkout at $(git -C "$HERE" rev-parse --short HEAD); update $H (git -C $H pull) or run from $H"
  exec bash "$H/scripts/bootstrap_node.sh" "$@"
fi
git -C "$H" log -1 --format='harness commit %H (%cd)'
for d in bin lease logs build scratch results runs trusted schedules creds; do mkdir -p $SS/$d; done
chmod 700 $SS/creds $SS/build $SS/schedules

say "GitHub auth for git (gh)"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then gh auth setup-git; echo "gh: logged in, git uses gh credentials"
else echo "gh: not logged in (fine if the benchmark repo and SS_SNAPSHOT_REPO are reachable otherwise)"; fi

say "seed"
bash "$H/scripts/unpack_seed.sh" || die "vendored seed failed to verify against its manifest"

say "benchmark + trusted snapshot"
bash "$H/scripts/fetch_benchmark.sh"

say "PROTOCOL manifest"
if ! "$PY" "$H/scripts/verify_protocol.py" --trusted $SS/trusted/vibe-megakernel; then
  [ "${SS_ALLOW_MISMATCH:-0}" = 1 ] || die "tree differs from PROTOCOL.md (set SS_ALLOW_MISMATCH=1 to continue anyway; such runs are not valid)"
fi

say "snapshot repo"
if [ -n "${SS_SNAPSHOT_REPO:-}" ]; then
  echo "$SS_SNAPSHOT_REPO" > $SS/snapshot_repo; chmod 600 $SS/snapshot_repo
fi
if [ -s $SS/snapshot_repo ]; then
  R=$(head -1 $SS/snapshot_repo)
  git ls-remote "$R" >/dev/null || die "cannot reach snapshot repo $R (see scripts/login_notes.md)"
  if ! git ls-remote --exit-code --heads "$R" results >/dev/null; then
    T=$(mktemp -d); git -C "$T" init -q -b results
    printf '# scaffold-search results\n\nWritten by the harness: results/<run_id>/result.json, results/_ledger, results/_schedules.\n' > "$T/README.md"
    git -C "$T" add README.md && git -C "$T" -c user.name=scaffold-search -c user.email=harness@localhost commit -qm "results branch"
    git -C "$T" push -q "$R" results && echo "created branch results in $R"; rm -rf "$T"
  fi
  echo "snapshot repo: $R (branches run/<id>-<arm>, results)"
else
  echo "no snapshot repo configured: run with pushes disabled (scripts/run_main.sh does this automatically)"
fi

say "gpu_lease + node lockdown"
getent group ss-gpu >/dev/null || groupadd ss-gpu
getent group ss-agents >/dev/null || groupadd ss-agents
install -m 755 "$H/scaffold_search/gpu_lease.py" $SS/bin/gpu_lease.py
printf '#!/bin/sh\nexec /usr/bin/python3.12 /srv/ss/bin/gpu_lease.py "$@"\n' > $SS/bin/gpu_lease
chmod 755 $SS/bin/gpu_lease $SS/bin $SS/lease
touch $SS/lease.jsonl
# setup-node: /root 0700, /srv/ss dirs, /tmp 1733, trusted tree read-only, pinned claude copy in
# /srv/ss/bin, /dev/nvidia* root:ss-gpu 0660, credential sync (run_arm repeats it every run)
(cd "$H" && "$PY" -m scaffold_search.isolation setup-node)
ls -l /dev/nvidia* | head -3

say "daemons (tmux ss-lease, ss-proxy)"
if ! tmux has-session -t ss-lease 2>/dev/null; then
  tmux new -d -s ss-lease "/srv/ss/bin/gpu_lease daemon 2>&1 | tee -a /srv/ss/logs/ss-lease.log"
  sleep 2
fi
if ! tmux has-session -t ss-proxy 2>/dev/null; then
  tmux new -d -s ss-proxy "cd /srv/ss/harness && python3.12 -m scaffold_search.isolation proxy 2>&1 | tee -a /srv/ss/logs/ss-proxy.log"
  sleep 2
fi
$SS/bin/gpu_lease status >/dev/null && echo "gpu_lease daemon: up" || die "gpu_lease daemon not answering (tmux attach -t ss-lease)"

say "sanity checks"
fail=0
if [ "${SS_SKIP_GPU_CHECK:-0}" != 1 ]; then
  n=$(nvidia-smi -L | grep -c -E 'GB300|B300' || true)
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
  echo "GPUs: $(nvidia-smi -L | wc -l) total, $n (G)B300, compute capability $cc"
  [ "$n" = 4 ] && [ "$(nvidia-smi -L | wc -l)" = 4 ] || { echo "FAIL: need exactly 4x GB300 (TP4 protocol)"; fail=1; }
  [ "$cc" = "10.3 " ] || { echo "FAIL: compute capability $cc, need 10.3 (sm_103)"; fail=1; }
  [ "$(uname -m)" = aarch64 ] || echo "note: $(uname -m) host (the pinned image is linux/arm64)"
  nvidia-smi topo -m 2>/dev/null | head -10 || true
fi
v=$($SS/bin/claude --version 2>/dev/null || true)
echo "claude (run users): $v"
case "$v" in 2.1.326*) ;; *) echo "FAIL: need Claude Code 2.1.326 at /srv/ss/bin/claude"; fail=1;; esac
if [ -f /root/.claude/.credentials.json ]; then echo "credentials: /root/.claude/.credentials.json (OAuth)"
elif [ -s $SS/creds/claude.env ]; then echo "credentials: $SS/creds/claude.env (token/API key; see scripts/login_notes.md)"
else echo "FAIL: Claude Code not logged in (scripts/login_notes.md)"; fail=1; fi
code=$(curl -s -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 --max-time 10 https://github.com || true)
echo "proxy: github.com via proxy -> ${code:-error} (want 000/403: denied)"
code2=$(curl -s -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 --max-time 10 https://api.anthropic.com || true)
echo "proxy: api.anthropic.com via proxy -> $code2 (want an HTTP status, e.g. 404)"
case "$code2" in 000|"") echo "FAIL: proxy does not reach api.anthropic.com"; fail=1;; esac
if [ $CHECK_LOGIN = 1 ]; then
  out=$(cd /tmp && { [ -f $SS/creds/claude.env ] && set -a && . $SS/creds/claude.env && set +a; true; } && timeout 120 claude -p "Reply with exactly: OK" --model claude-opus-5-5 2>&1 | tail -1 || true)
  echo "claude -p as root: $out"; [ "$out" = OK ] || { echo "FAIL: claude -p did not answer OK"; fail=1; }
fi
[ $fail = 0 ] || die "sanity checks failed"
echo; echo "bootstrap OK. Next: bash $H/scripts/run_selftest.sh"
