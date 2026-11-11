#!/bin/bash
# Step 1 smoke test of the evaluator: the PyTorch oracle as the candidate on kimi-k3-dev-l4-tp1-b8-s8192.
# Host-level on rack 10 tray14 as dc, one GPU, no sandbox, no agent. Everything is recorded under reports/gb300/.
set -uo pipefail
export HOME=/mnt/shared/home/dc
R=$HOME/workspace/kimi-k3-goalbox
PY=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv/bin/python
CASE=kimi-k3-dev-l4-tp1-b8-s8192
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$R/reports/gb300/step1-dev-l4-tp1-$TS
mkdir -p "$OUT/cache"
export CUDA_CACHE_PATH=$OUT/cache/cuda TRITON_CACHE_DIR=$OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor XDG_CACHE_HOME=$OUT/cache/xdg
export CUDA_VISIBLE_DEVICES=0
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; echo "user: $(id -un)"; echo "kernel: $(uname -r) $(uname -m)"
  echo "repo: $R"; echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "python: $PY -> $(readlink -f "$PY")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpus_on_host:"; nvidia-smi -L
  echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
  echo "libnccl_system: $(ls /usr/lib/aarch64-linux-gnu/libnccl.so.2.* 2>/dev/null)"
  "$PY" -s -c 'import torch
print("torch:", torch.__version__); print("torch_cuda:", torch.version.cuda)
print("torch_nccl:", ".".join(map(str, torch.cuda.nccl.version()))); print("cudnn:", torch.backends.cudnn.version())
print("visible_gpus:", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import triton; print("triton:", triton.__version__)
except Exception as e:
    print("triton: not importable:", e)'
  echo "caches: CUDA_CACHE_PATH=$CUDA_CACHE_PATH TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
} > "$OUT/env.txt" 2>&1
occupancy() {
  echo "date_utc: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  echo "compute_apps:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  echo "k8s_pods_on_node:"
  kubectl get pods -A --field-selector "spec.nodeName=$(hostname)" -o json 2>/dev/null | "$PY" -s -c 'import sys, json
for p in json.load(sys.stdin)["items"]:
    g = sum(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)) for c in p["spec"]["containers"])
    print(" ", p["metadata"]["namespace"], p["metadata"]["name"], p["status"]["phase"], "gpu=%d" % g)' || echo "  (kubectl failed)"
}
occupancy > "$OUT/occupancy-before.txt" 2>&1
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || grep -q "gpu=[1-9]" "$OUT/occupancy-before.txt"; then
  echo "ABORTED: GPU or node already in use (see occupancy-before.txt)" | tee "$OUT/ABORTED"; exit 99
fi
( while :; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 10; done ) > "$OUT/gpu-monitor.log" 2>&1 &
MON=$!
CMD=(timeout 1500 "$PY" -s -m megabench evaluate --case "$CASE" --submission "$R/runs/oracle-check/submission.py"
     --method oracle-smoke --session-id "gb300-step1-$TS" --device cuda:0 --timeout 1200 --output "$OUT/result.jsonl")
{ printf 'cd %q && PYTHONPATH=%q ' "$R" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$OUT/command.txt"
echo "start_utc: $(date -u +%FT%TZ)" > "$OUT/timing.txt"
PYTHONPATH=$R "${CMD[@]}" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
RC=$?
echo "$RC" > "$OUT/exit_code"
echo "end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
occupancy > "$OUT/occupancy-after.txt" 2>&1
cp "$R/runs/oracle-check/submission.py" "$OUT/submission.py"
echo "exit_code=$RC out=$OUT"
