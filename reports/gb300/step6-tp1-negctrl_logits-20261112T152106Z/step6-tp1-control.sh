#!/bin/bash
# Step 6 (PROPOSED, not yet approved to run): one clean evaluator invocation of one control on the single-GPU
# dev case kimi-k3-dev-l4-tp1-b8-s8192, host-level on rack 10 tray14 as dc, GPU 0, no sandbox, no agent.
# Usage: step6-tp1-control.sh <positive_control_oracle|negctrl_logits|negctrl_routing|negctrl_immutability>
# Derived from step5.sh: single GPU (no NCCL variables), the submission comes from the committed
# reports/gb300/negative-controls/ directory, GOALBOX_NEGCTRL_RECORD makes the control write its own record
# (the evaluator does not keep the candidate's stdout/stderr), and a result summary is written at the end.
# Recorded under reports/gb300/step6-tp1-<control>-<ts>/.
set -uo pipefail
CTRL=${1:?control name}
case "$CTRL" in positive_control_oracle|negctrl_logits|negctrl_routing|negctrl_immutability) ;; *) echo "unknown control $CTRL"; exit 97;; esac
export HOME=/mnt/shared/home/dc
export PATH=$HOME/.local/bin:$PATH          # kubectl
R=$HOME/workspace/kimi-k3-goalbox
PY=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv/bin/python
CASE=kimi-k3-dev-l4-tp1-b8-s8192
SUB=$R/reports/gb300/negative-controls/$CTRL.py
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$R/reports/gb300/step6-tp1-$CTRL-$TS
mkdir -p "$OUT/cache"
export CUDA_CACHE_PATH=$OUT/cache/cuda TRITON_CACHE_DIR=$OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor XDG_CACHE_HOME=$OUT/cache/xdg
export CUDA_VISIBLE_DEVICES=0
export GOALBOX_NEGCTRL_RECORD=$OUT/negctrl-record
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; echo "user: $(id -un)"; echo "kernel: $(uname -r) $(uname -m)"
  echo "repo: $R"; echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "control: $CTRL"; echo "submission: $SUB"; echo "submission_sha256: $(sha256sum "$SUB" | cut -d' ' -f1)"
  echo "python: $PY -> $(readlink -f "$PY")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpus_on_host (index, uuid, pci bus id):"; nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
  echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
  "$PY" -s -c 'import torch
print("torch:", torch.__version__); print("torch_cuda:", torch.version.cuda); print("cudnn:", torch.backends.cudnn.version())
print("visible_gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.get_device_capability(0))
try:
    import triton; print("triton:", triton.__version__)
except Exception as e:
    print("triton: not importable:", e)'
  echo "caches: CUDA_CACHE_PATH=$CUDA_CACHE_PATH TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GOALBOX_NEGCTRL_RECORD=$GOALBOX_NEGCTRL_RECORD"
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
( echo "date_utc total_mb used_mb free_mb available_mb swap_used_mb"; while :; do echo "$(date -u +%FT%TZ) $(free -m | awk 'NR==2{t=$2;u=$3;f=$4;a=$7} NR==3{s=$3} END{print t, u, f, a, s}')"; sleep 10; done ) > "$OUT/host-mem-monitor.log.txt" 2>&1 &
HMON=$!
CMD=(timeout -k 30 1800 "$PY" -s -m megabench evaluate --case "$CASE" --submission "$SUB"
     --method "negctrl-$CTRL" --session-id "gb300-step6-$CTRL-$TS" --device cuda:0 --timeout 1500 --output "$OUT/result.jsonl")
{ printf 'cd %q && CUDA_VISIBLE_DEVICES=0 GOALBOX_NEGCTRL_RECORD=%q PYTHONPATH=%q ' "$R" "$GOALBOX_NEGCTRL_RECORD" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$OUT/command.txt"
echo "start_utc: $(date -u +%FT%TZ)" > "$OUT/timing.txt"
PYTHONPATH=$R "${CMD[@]}" > "$OUT/stdout.log.txt" 2> "$OUT/stderr.log.txt"
RC=$?
echo "$RC" > "$OUT/exit_code"
echo "end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
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
kill "$MON" "$KMON" "$HMON" 2>/dev/null; wait "$MON" "$KMON" "$HMON" 2>/dev/null
occupancy > "$OUT/occupancy-after.txt" 2>&1
"$PY" -s -c 'import json, sys
r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
print("status:", r.get("status")); print("reason:", r.get("reason")); print("submission_sha256:", r.get("submission_sha256"))
c = r.get("correctness"); print("correctness:", c["status"] if isinstance(c, dict) else c)
print("launch_audit:", (r.get("launch_audit") or {}).get("status")); print("peak_allocated_bytes:", (r.get("hardware") or {}).get("peak_allocated_bytes"))
print("traceback_tail:", (r.get("traceback") or "")[-600:])' "$OUT/result.jsonl" > "$OUT/result-summary.txt" 2>&1
cp "$SUB" "$OUT/$CTRL.py"; cp "$0" "$OUT/step6-tp1-control.sh"
echo "exit_code=$RC out=$OUT"
