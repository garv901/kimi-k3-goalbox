#!/bin/bash
# Convert the HF checkpoint to the per-rank fixture layout that the harness loads.
#
#   setup/convert_checkpoint.sh HF_DIR OUT_DIR [TP=8] [WORKERS=4]
#
# Writes OUT_DIR/tp<TP>/rank<R>/{global,layer000..layer092}.safetensors (about 1.6 TB for TP8).
# Restartable: finished layers are kept. Then export KIMI_K3_WEIGHTS=OUT_DIR for the harness
# and for sandbox/bin/goalbox-* (the GPU sandbox mounts it read-only at /weights).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:?usage: convert_checkpoint.sh HF_DIR OUT_DIR [TP] [WORKERS]} OUT=${2:?out dir}
PYTHONPATH=$REPO exec "$REPO/.venv/bin/python" -s -m megabench.tasks.kimi_checkpoint \
  --src "$SRC" --out "$OUT" --tp "${3:-8}" --workers "${4:-4}"
