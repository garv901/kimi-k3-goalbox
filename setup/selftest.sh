#!/bin/bash
# Check the installation: unit tests (CPU), the agent sandbox, and the GPU sandbox.
#
#   setup/selftest.sh            GPU part: GOALBOX_GPU_MODE=local runs here; =slurm uses
#                                srun (GOALBOX_PARTITION, GOALBOX_SBATCH_ARGS)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$REPO/.venv/bin/python
echo "== unit tests"
(cd "$REPO" && PYTHONPATH=$REPO "$PY" -s -m unittest megabench.tests.test_kimi)
T=$(mktemp -d); mkdir -p "$T"/{work,home,tmp,sock}
trap 'rm -rf "$T"' EXIT
echo "== agent sandbox: no direct network, no host files, benchmark read-only"
"$REPO/sandbox/lib/sandbox.sh" gpu "$T" -- bash -c '
  ls /home /raid /scratch 2>&1 | head -3
  (exec 3<>/dev/tcp/1.1.1.1/443) 2>/dev/null && echo "FAIL: network reachable" || echo "ok: network unreachable"
  touch /goalbox/bench/megabench/x 2>/dev/null && echo "FAIL: bench writable" || echo "ok: bench read-only"
  /goalbox/venv/bin/python -c "import torch; print(\"ok: torch\", torch.__version__)"'
echo "== GPU sandbox"
CMD=("$REPO/sandbox/lib/sandbox.sh" gpu "$T" -- /goalbox/venv/bin/python -c \
  'import torch; n=torch.cuda.device_count(); print("ok: GPUs", n, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(4096,4096,device="cuda"); print("ok: matmul", float((x@x).norm())>0); import os; print("ok: /dev/shm", os.access("/dev/shm", os.W_OK))')
MODE=${GOALBOX_GPU_MODE:-$(command -v sbatch >/dev/null && echo slurm || echo local)}
if [ "$MODE" = slurm ]; then
  srun -p "${GOALBOX_PARTITION:-main}" --gres=gpu:1 -t 5 ${GOALBOX_SBATCH_ARGS:-} "${CMD[@]}"
else
  "${CMD[@]}"
fi
