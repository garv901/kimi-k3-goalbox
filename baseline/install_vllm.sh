#!/bin/bash
# Install the baseline: vLLM with the Kimi K3 latency-specialized decode path
# (vllm-project/vllm PR 61867, merged as commit 642c18d). Separate venv: REPO/.venv-vllm.
# This environment is never visible inside the agent sandbox.
#
#   baseline/install_vllm.sh                 nightly wheel (must include the PR)
#   VLLM_COMMIT=<full sha> baseline/install_vllm.sh   wheel of one commit at or after 642c18d
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
"$PY" -m venv "$REPO/.venv-vllm"
P=$REPO/.venv-vllm/bin
"$P/pip" install -q --upgrade pip uv
INDEX=https://wheels.vllm.ai/${VLLM_COMMIT:-nightly}
"$P/uv" pip install --python "$P/python" -U vllm --pre --torch-backend=auto --extra-index-url "$INDEX"
"$P/python" - <<'EOF'
import importlib.util, pathlib, vllm
root = pathlib.Path(vllm.__file__).parent
path = root / "model_executor/models/kimi_k3/nvidia/specialized_decode"
print("vllm", vllm.__version__)
if not path.exists():
    raise SystemExit(f"this vLLM build has no {path.relative_to(root)}: it predates PR 61867 (commit 642c18d); "
                     "set VLLM_COMMIT to a later commit")
print("latency-specialized Kimi K3 decode path present")
EOF
