#!/bin/bash
# Milestone driver for the TP4 real-weight bring-up on one GB300 tray (4 GPUs), host level, no sandbox, no agent.
# Only runs that have their own GO may be started. Derived from reports/gb300/negative-controls/step7-tp4-control.sh
# (same monitors, NCCL logging, cache isolation, cleanup); adds a scheduler monitor for Slurm (rack 3) or
# Kubernetes (rack 10) depending on what the host has, and the KIMI_K3_WEIGHTS provenance in env.txt.
#
# Usage: m1-tp4-real.sh <load|evaluate> <layers> [batch=1]
#   load      milestone 1: planning/bringup/tp4_real_load_only.py (loader + validation only, no process group)
#   evaluate  milestone 2: planning/bringup/evaluate_custom_case.py with the trusted oracle as candidate,
#             trials 1, warmup 1, reps 1 (one correctness trial, minimal timing, one launch audit)
# Environment: KIMI_K3_WEIGHTS (converted root, default below), M1_OUT_ROOT (default <repo>/reports/gb300),
#   M1_REPO (default kevin's clone on the shared filesystem), M1_TIMEOUT (seconds; default 3600 load, 5400 evaluate).
set -uo pipefail
MODE=${1:?load|evaluate}; LAYERS=${2:?layers}; B=${3:-1}
case "$MODE" in load|evaluate) ;; *) echo "unknown mode $MODE"; exit 97;; esac
R=${M1_REPO:-/mnt/shared/home/kevin/workspace/kimi-k3-goalbox}
PY=$R/.venv/bin/python
export KIMI_K3_WEIGHTS=${KIMI_K3_WEIGHTS:-/mnt/shared/home/kevin/weights/kimi-k3-fixture-trial}
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${M1_OUT_ROOT:-$R/reports/gb300}/m1-tp4-real-$MODE-l$LAYERS-b$B-$TS
mkdir -p "$OUT/cache" || exit 98
export CUDA_CACHE_PATH=$OUT/cache/cuda TRITON_CACHE_DIR=$OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor XDG_CACHE_HOME=$OUT/cache/xdg
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,COLL NCCL_DEBUG_FILE=$OUT/nccl-%h-%p.log.txt
TIMEOUT=${M1_TIMEOUT:-$([ "$MODE" = load ] && echo 3600 || echo 5400)}
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; echo "user: $(id -un)"; echo "kernel: $(uname -r) $(uname -m)"
  echo "repo: $R"; echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "mode: $MODE layers: $LAYERS batch: $B timeout_s: $TIMEOUT"
  echo "weights_root: $KIMI_K3_WEIGHTS"; echo "weights_done_json: $(cat "$KIMI_K3_WEIGHTS/tp4/DONE.json" 2>/dev/null)"
  echo "weights_manifest_sha256: $(sha256sum "$KIMI_K3_WEIGHTS/MANIFEST-tp4.json" 2>/dev/null | cut -d' ' -f1)"
  echo "weights_rank_files: $(ls "$KIMI_K3_WEIGHTS/tp4/rank0" 2>/dev/null | wc -l) per rank, $(du -sh "$KIMI_K3_WEIGHTS/tp4" 2>/dev/null | cut -f1)"
  echo "scripts_sha256:"; sha256sum planning/bringup/tp4_real_load_only.py planning/bringup/evaluate_custom_case.py reports/gb300/negative-controls/positive_control_oracle.py
  echo "python: $PY -> $(readlink -f "$PY")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpus_on_host (index, uuid, pci bus id, total memory):"; nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total --format=csv
  "$PY" -s -c 'import torch
print("torch:", torch.__version__); print("torch_cuda:", torch.version.cuda)
print("torch_nccl:", ".".join(map(str, torch.cuda.nccl.version())))
print("visible_gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
  echo "host_mem: $(free -g | sed -n 2p)"; echo "shared_fs: $(df -h "$KIMI_K3_WEIGHTS" | tail -1)"
  echo "scheduler: $(command -v squeue >/dev/null && echo slurm || (command -v kubectl >/dev/null && echo kubernetes || echo none))"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NCCL_DEBUG=$NCCL_DEBUG NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE"
} > "$OUT/env.txt" 2>&1
sched() {  # one line: scheduler view of this node
  if command -v squeue >/dev/null; then
    echo "slurm_jobs_on_node=$(squeue -w "$(hostname)" -h -o '%i %u %T %b' | wc -l) $(squeue -w "$(hostname)" -h -o '%i:%u:%T:%b' | tr '\n' ' ')"
  elif command -v kubectl >/dev/null; then
    kubectl get pods -A --field-selector "spec.nodeName=$(hostname)" -o json 2>/dev/null | "$PY" -s -c 'import sys, json
items = json.load(sys.stdin)["items"]
gpu = sum(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)) for p in items for c in p["spec"]["containers"])
print("total_pods=%d total_gpu_requests=%d" % (len(items), gpu))' || echo "kubectl-failed"
  else echo "no-scheduler"; fi
}
occupancy() {
  echo "date_utc: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  echo "compute_apps:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  echo "scheduler: $(sched)"
}
occupancy > "$OUT/occupancy-before.txt" 2>&1
busy=0
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && busy=1
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024 {exit 1}' || busy=1
grep -qE "slurm_jobs_on_node=0|total_gpu_requests=0" "$OUT/occupancy-before.txt" || busy=1
if [ "$busy" = 1 ]; then echo "ABORTED: GPU in use, more than 1 GiB used on a GPU, a scheduler job/pod on this node, or the scheduler check failed (see occupancy-before.txt)" | tee "$OUT/ABORTED"; exit 99; fi
avail=$(free -g | awk 'NR==2{print $7}'); if [ "$avail" -lt 300 ]; then echo "ABORTED: only $avail GB host memory available (need 300)" | tee "$OUT/ABORTED"; exit 99; fi
( while :; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';') apps: $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 10; done ) > "$OUT/gpu-monitor.log.txt" 2>&1 &
MON=$!
( while :; do echo "$(date -u +%FT%TZ) $(sched)"; sleep 30; done ) > "$OUT/scheduler-monitor.log.txt" 2>&1 &
SMON=$!
( echo "date_utc total_mb used_mb free_mb available_mb swap_used_mb"; while :; do echo "$(date -u +%FT%TZ) $(free -m | awk 'NR==2{t=$2;u=$3;f=$4;a=$7} NR==3{s=$3} END{print t, u, f, a, s}')"; sleep 10; done ) > "$OUT/host-mem-monitor.log.txt" 2>&1 &
HMON=$!
if [ "$MODE" = load ]; then
  CMD=(timeout -k 30 "$TIMEOUT" "$PY" -s planning/bringup/tp4_real_load_only.py --layers "$LAYERS" --tp 4 --out "$OUT/load")
else
  CMD=(timeout -k 30 "$TIMEOUT" "$PY" -s planning/bringup/evaluate_custom_case.py --layers "$LAYERS" --tp 4 --batch "$B"
       --submission reports/gb300/negative-controls/positive_control_oracle.py --method bringup-oracle
       --session-id "gb300-m1-$MODE-l$LAYERS-b$B-$TS" --device cuda:0 --trials 1 --warmup 1 --reps 1 --output "$OUT/result.jsonl")
fi
{ printf 'cd %q && KIMI_K3_WEIGHTS=%q CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,COLL NCCL_DEBUG_FILE=%q PYTHONPATH=%q ' "$R" "$KIMI_K3_WEIGHTS" "$NCCL_DEBUG_FILE" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$OUT/command.txt"
echo "start_utc: $(date -u +%FT%TZ)" > "$OUT/timing.txt"
PYTHONPATH=$R "${CMD[@]}" > "$OUT/stdout.log.txt" 2> "$OUT/stderr.log.txt"
RC=$?
echo "$RC" > "$OUT/exit_code"
echo "end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
{
  echo "date_utc: $(date -u +%FT%TZ) exit=$RC"
  left=$(pgrep -u "$(id -un)" -f "tp4_real_load_[o]nly|evaluate_custom_[c]ase|megabench (_worker|[e]valuate)" || true)
  if [ -n "$left" ]; then
    echo "leftover processes: $left"; ps -o pid,pgid,etime,cmd -p $left
    for p in $left; do kill -TERM -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null; done; sleep 10
    left2=$(pgrep -u "$(id -un)" -f "tp4_real_load_[o]nly|evaluate_custom_[c]ase|megabench (_worker|[e]valuate)" || true)
    [ -n "$left2" ] && { echo "still alive after TERM, sending KILL: $left2"; for p in $left2; do kill -KILL -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null; done; }
  else echo "no leftover processes"; fi
} > "$OUT/cleanup.txt" 2>&1
kill "$MON" "$SMON" "$HMON" 2>/dev/null; wait "$MON" "$SMON" "$HMON" 2>/dev/null
occupancy > "$OUT/occupancy-after.txt" 2>&1
cp "$0" "$OUT/m1-tp4-real.sh"
echo "exit_code=$RC out=$OUT"
