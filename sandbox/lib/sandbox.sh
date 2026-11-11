#!/bin/bash
# Build the goalbox filesystem view with bubblewrap and run a command in it.
#
#   sandbox.sh agent RUN_DIR -- CMD...   no direct network; HTTP(S) only through the
#                                        allowlist proxy socket in RUN_DIR/sock
#   sandbox.sh gpu   RUN_DIR -- CMD...   no network at all; NVIDIA devices of the Slurm job
#
# Visible inside: /usr (minus /usr/local/{bin,lib}), /etc (minus /etc/claude-code), /sys,
# /work (RUN_DIR/work, rw), /home/agent (RUN_DIR/home, rw), /tmp (RUN_DIR/tmp, rw),
# /goalbox/{venv,bench,bin,claude} (read-only). Everything else on the host is absent.
# gpu mode also gets a private /dev/shm (NCCL, CUDA IPC) and, when KIMI_K3_WEIGHTS is set,
# the converted real weights read-only at /weights.
#
# Settings (environment): GOALBOX_VENV (default REPO/.venv), GOALBOX_CLAUDE_BIN (default: claude on PATH).
set -euo pipefail
GB=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$GB/.." && pwd)
MODE=$1 RUN=$(realpath "$2"); shift 2
[ "${1:-}" = "--" ] && shift
VENV=$(realpath "${GOALBOX_VENV:-$REPO/.venv}")
CLAUDE_BIN=$(readlink -f "${GOALBOX_CLAUDE_BIN:-$(command -v claude || echo /nonexistent)}")

args=(
  --die-with-parent --unshare-all --hostname goalbox
  --ro-bind /usr /usr
  --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin
  --tmpfs /usr/local/bin --tmpfs /usr/local/lib
  --ro-bind /etc /etc --tmpfs /etc/claude-code
  --ro-bind /sys /sys
  --proc /proc --dev /dev
  --dir /var/tmp --dir /run
  --bind "$RUN/tmp" /tmp
  --bind "$RUN/home" /home/agent
  --bind "$RUN/work" /work
  --ro-bind "$VENV" /goalbox/venv
  --ro-bind "$REPO/megabench" /goalbox/bench/megabench
  --ro-bind "$GB/sandbox-bin" /goalbox/bin
  --clearenv
  --setenv HOME /home/agent --setenv USER agent --setenv LOGNAME agent
  --setenv PATH /home/agent/venv/bin:/goalbox/bin:/usr/local/cuda/bin:/usr/bin
  --setenv LANG C.UTF-8 --setenv TERM "${TERM:-xterm-256color}"
  --setenv TMPDIR /tmp --setenv CUDA_HOME /usr/local/cuda
  --setenv PYTHONPATH /goalbox/bench
  --setenv PIP_INDEX_URL http://pypi.goalbox/simple/ --setenv PIP_TRUSTED_HOST pypi.goalbox
  --chdir "${GOALBOX_CWD:-/work}"
)

case $MODE in
  agent)
    [ -x "$CLAUDE_BIN" ] || { echo "Claude Code binary not found; set GOALBOX_CLAUDE_BIN" >&2; exit 1; }
    args+=(--ro-bind "$CLAUDE_BIN" /goalbox/claude --ro-bind "$RUN/sock" /goalbox/sock
           --setenv HTTPS_PROXY http://127.0.0.1:3128 --setenv HTTP_PROXY http://127.0.0.1:3128
           --setenv https_proxy http://127.0.0.1:3128 --setenv http_proxy http://127.0.0.1:3128
           --setenv NO_PROXY localhost,127.0.0.1 --setenv no_proxy localhost,127.0.0.1)
    # The OAuth token must not appear on a command line (any user can read argv with ps), so it
    # goes in as a read-only file through a file descriptor and is exported inside the sandbox.
    if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
      exec 3<<<"$CLAUDE_CODE_OAUTH_TOKEN"
      args+=(--perms 0400 --ro-bind-data 3 /goalbox/token)
    fi
    for v in CLAUDE_CODE_SUBAGENT_MODEL DISABLE_AUTOUPDATER \
             CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC ENABLE_CLAUDEAI_MCP_SERVERS COLORTERM; do
      [ -n "${!v:-}" ] && args+=(--setenv "$v" "${!v}")
    done
    # The proxy socket is only reachable as a unix socket; bridge it to 127.0.0.1:3128.
    exec bwrap "${args[@]}" bash -c \
      '[ -r /goalbox/token ] && export CLAUDE_CODE_OAUTH_TOKEN=$(cat /goalbox/token)
       socat TCP-LISTEN:3128,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:/goalbox/sock/net.sock 2>/dev/null & exec "$@"' \
      goalbox "$@"
    ;;
  gpu)
    args+=(--tmpfs /dev/shm)
    for d in /dev/nvidia* /dev/nvidia-caps /dev/nvidia-caps-imex-channels; do [ -e "$d" ] && args+=(--dev-bind "$d" "$d"); done
    if [ -n "${KIMI_K3_WEIGHTS:-}" ]; then
      args+=(--ro-bind "$(realpath "$KIMI_K3_WEIGHTS")" /weights --setenv KIMI_K3_WEIGHTS /weights)
    fi
    for v in CUDA_VISIBLE_DEVICES SLURM_JOB_ID NCCL_DEBUG; do
      [ -n "${!v:-}" ] && args+=(--setenv "$v" "${!v}")
    done
    exec bwrap "${args[@]}" "$@"
    ;;
  *) echo "mode must be agent or gpu" >&2; exit 2 ;;
esac
