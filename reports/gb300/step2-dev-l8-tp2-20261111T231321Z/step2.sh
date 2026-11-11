#!/bin/bash
# Step 2 smoke test of the distributed evaluator: the PyTorch oracle as the candidate on
# kimi-k3-dev-l8-tp2-b8-s8192 (2 ranks, TP2). Host-level on rack 10 tray14 as dc, GPUs 0 and 1,
# no sandbox, no agent. Everything is recorded under reports/gb300/step2-dev-l8-tp2-<ts>/.
set -uo pipefail
export HOME=/mnt/shared/home/dc
export PATH=$HOME/.local/bin:$PATH          # kubectl (Step 1's in-script check lacked this)
R=$HOME/workspace/kimi-k3-goalbox
PY=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv/bin/python
CASE=kimi-k3-dev-l8-tp2-b8-s8192
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$R/reports/gb300/step2-dev-l8-tp2-$TS
mkdir -p "$OUT/cache"
export CUDA_CACHE_PATH=$OUT/cache/cuda TRITON_CACHE_DIR=$OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor XDG_CACHE_HOME=$OUT/cache/xdg
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$OUT/nccl-%h-%p.log.txt   # NCCL init and errors, one file per rank process
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; echo "user: $(id -un)"; echo "kernel: $(uname -r) $(uname -m)"
  echo "repo: $R"; echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "python: $PY -> $(readlink -f "$PY")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpus_on_host (index, uuid, pci bus id):"; nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
  echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
  echo "libnccl_system: $(ls /usr/lib/aarch64-linux-gnu/libnccl.so.2.* 2>/dev/null)"
  "$PY" -s -c 'import torch
print("torch:", torch.__version__); print("torch_cuda:", torch.version.cuda)
print("torch_nccl:", ".".join(map(str, torch.cuda.nccl.version()))); print("cudnn:", torch.backends.cudnn.version())
print("visible_gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.get_device_capability(0))
try:
    import triton; print("triton:", triton.__version__)
except Exception as e:
    print("triton: not importable:", e)'
  echo "caches: CUDA_CACHE_PATH=$CUDA_CACHE_PATH TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NCCL_DEBUG=$NCCL_DEBUG NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE"
  echo "kubectl: $(command -v kubectl || echo MISSING)"
} > "$OUT/env.txt" 2>&1
k8s_gpu_pods() {  # lines: namespace phase gpu=N started=...  (names redacted); exit 1 if kubectl fails
  kubectl get pods -A --field-selector "spec.nodeName=$(hostname)" -o json 2>/dev/null | "$PY" -s -c 'import sys, json
items = json.load(sys.stdin)["items"]
gpu = 0
for p in items:
    g = sum(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)) for c in p["spec"]["containers"])
    gpu += g
    print("  %-20s (pod name redacted) %-10s gpu=%d started=%s" % (p["metadata"]["namespace"], p["status"]["phase"], g, p["status"].get("startTime")))
print("  total_pods=%d total_gpu_requests=%d" % (len(items), gpu))'
}
occupancy() {
  echo "date_utc: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  echo "compute_apps:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  echo "k8s_pods_on_node:"; k8s_gpu_pods || echo "  (kubectl failed)"
}
occupancy > "$OUT/occupancy-before.txt" 2>&1
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || ! grep -q "total_gpu_requests=0" "$OUT/occupancy-before.txt"; then
  echo "ABORTED: GPU busy, GPU pod on node, or kubectl check unavailable (see occupancy-before.txt)" | tee "$OUT/ABORTED"; exit 99
fi
( while :; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 10; done ) > "$OUT/gpu-monitor.log.txt" 2>&1 &
MON=$!
( while :; do echo "$(date -u +%FT%TZ) $(k8s_gpu_pods 2>/dev/null | tail -1 || echo kubectl-failed)"; sleep 30; done ) > "$OUT/k8s-monitor.log.txt" 2>&1 &
KMON=$!
CMD=(timeout -k 30 1800 "$PY" -s -m megabench evaluate --case "$CASE" --submission "$R/runs/oracle-check/submission.py"
     --method oracle-smoke --session-id "gb300-step2-$TS" --device cuda:0 --timeout 1500 --output "$OUT/result.jsonl")
{ printf 'cd %q && CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO NCCL_DEBUG_FILE=%q PYTHONPATH=%q ' "$R" "$NCCL_DEBUG_FILE" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$OUT/command.txt"
echo "start_utc: $(date -u +%FT%TZ)" > "$OUT/timing.txt"
PYTHONPATH=$R "${CMD[@]}" > "$OUT/stdout.log.txt" 2> "$OUT/stderr.log.txt"
RC=$?
echo "$RC" > "$OUT/exit_code"
echo "end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
# Cleanup check: the evaluator's own --timeout kills the worker's process group (start_new_session + killpg);
# the outer `timeout` signals its own process group. Catch anything that survived both.
{
  echo "date_utc: $(date -u +%FT%TZ) evaluator_exit=$RC"
  left=$(pgrep -u "$(id -un)" -f "[m]egabench (_worker|evaluate)" || true)
  if [ -n "$left" ]; then
    echo "leftover megabench processes: $left"; ps -o pid,pgid,etime,cmd -p $left
    for p in $left; do kill -TERM -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null; done; sleep 10
    left2=$(pgrep -u "$(id -un)" -f "[m]egabench (_worker|evaluate)" || true)
    [ -n "$left2" ] && { echo "still alive after TERM, sending KILL: $left2"; for p in $left2; do kill -KILL -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null; done; }
  else
    echo "no leftover megabench processes"
  fi
} > "$OUT/cleanup.txt" 2>&1
kill "$MON" "$KMON" 2>/dev/null; wait "$MON" "$KMON" 2>/dev/null
occupancy > "$OUT/occupancy-after.txt" 2>&1
cp "$R/runs/oracle-check/submission.py" "$OUT/submission.py"; cp "$0" "$OUT/step2.sh"
echo "exit_code=$RC out=$OUT"
