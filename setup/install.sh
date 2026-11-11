#!/bin/bash
# Create the clean, read-only Python environment REPO/.venv (harness, oracle, agent base).
#
#   setup/install.sh            PYTHON=python3 (a system interpreter under /usr: the sandbox
#                               shows only /usr, so a venv built on ~/.local or conda Python breaks)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130 (default; CUDA 13 wheels support sm_100 and sm_103)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$(readlink -f "$(command -v "${PYTHON:-python3}")")
case $PY in /usr/*) ;; *) echo "PYTHON must resolve under /usr (got $PY)"; exit 1 ;; esac
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
for tool in bwrap socat tmux rsync jq git curl nvidia-smi; do
  command -v $tool >/dev/null || echo "WARNING: $tool not found (needed by the sandbox or tools)"
done
[ -x /usr/local/cuda/bin/nvcc ] || echo "WARNING: /usr/local/cuda/bin/nvcc not found; the agent compiles with it"
command -v claude >/dev/null || echo "WARNING: Claude Code (claude) not on PATH; set GOALBOX_CLAUDE_BIN for goalbox-start"
"$PY" -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
"$REPO/.venv/bin/pip" install -q numpy safetensors "huggingface_hub[hf_xet]" ninja
"$REPO/.venv/bin/python" -c 'import torch, safetensors; print("torch", torch.__version__, "cuda", torch.version.cuda)'
echo "environment ready: $REPO/.venv"
