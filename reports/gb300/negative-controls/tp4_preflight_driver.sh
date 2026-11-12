#!/bin/bash
# CPU-only preflight driver for the TP4 controls (rack 3, no GPU visible). Writes
# reports/gb300/negative-controls/tp4-preflight-<ts>/: env.txt, syntax.txt, preflight.log.txt + preflight.json
# (tp4_preflight.py: snapshot-path proof and tiny-geometry 4-process gloo runs of every control), then one
# real-geometry run of the actual CLI: kimi-k3-syn-l8-tp4-b64-s8192 with negctrl_immutability.py on --device cpu
# (4 gloo processes, 1 trial), which exercises the 512 MB l3_kv_cache checksum path and the agree() flow on the
# real case without a GPU. Usage: tp4_preflight_driver.sh [skip-real]
set -uo pipefail
R=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox
PY=$R/.venv/bin/python
NC=$R/reports/gb300/negative-controls
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$NC/tp4-preflight-$TS
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=32
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; echo "user: $(id -un)"; echo "kernel: $(uname -r) $(uname -m)"
  echo "nproc: $(nproc)"; echo "mem: $(free -g | awk 'NR==2{print $2" GB total, "$7" GB available"}')"
  echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "evaluator_diff_vs_52f8d8c: $(git diff --stat 52f8d8c -- megabench setup sandbox | tail -1)"
  echo "python: $PY -> $(readlink -f "$PY")"; "$PY" -s -c 'import torch; print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())'
  echo "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "nvidia_compute_apps_on_host: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  for f in "$NC"/*.py "$NC"/step7-tp4-control.sh; do echo "sha256 $(sha256sum "$f" | cut -d' ' -f1)  $(basename "$f")"; done
} > "$OUT/env.txt" 2>&1
"$PY" -s -m py_compile "$NC"/*.py > "$OUT/syntax.txt" 2>&1 && echo "py_compile ok" >> "$OUT/syntax.txt"; bash -n "$NC/step7-tp4-control.sh" && echo "bash -n step7-tp4-control.sh ok" >> "$OUT/syntax.txt"
echo "start_utc: $(date -u +%FT%TZ)" > "$OUT/timing.txt"
PYTHONPATH=$R "$PY" -s "$NC/tp4_preflight.py" "$OUT" > "$OUT/preflight.log.txt" 2>&1
echo "preflight_rc=$? preflight_end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
if [ "${1:-}" != "skip-real" ]; then
  D=$OUT/real-geometry-b64-negctrl_immutability; mkdir -p "$D"
  CMD=("$PY" -s -m megabench evaluate --case kimi-k3-syn-l8-tp4-b64-s8192 --submission "$NC/negctrl_immutability.py"
       --method negctrl-tp4-cpu-preflight --session-id "tp4-cpu-preflight-$TS" --device cpu --trials 1 --warmup 0 --reps 1 --timeout 3000 --output "$D/result.jsonl")
  { printf 'cd %q && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=%s GOALBOX_NEGCTRL_RECORD=%q PYTHONPATH=%q ' "$R" "$OMP_NUM_THREADS" "$D/negctrl-record" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$D/command.txt"
  ( echo "date_utc total_mb used_mb free_mb available_mb swap_used_mb"; while :; do echo "$(date -u +%FT%TZ) $(free -m | awk 'NR==2{t=$2;u=$3;f=$4;a=$7} NR==3{s=$3} END{print t, u, f, a, s}')"; sleep 10; done ) > "$D/host-mem-monitor.log.txt" 2>&1 &
  HMON=$!
  echo "real_start_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
  GOALBOX_NEGCTRL_RECORD=$D/negctrl-record PYTHONPATH=$R /usr/bin/time -v -o "$D/time.txt" timeout -k 30 3300 "${CMD[@]}" > "$D/stdout.log.txt" 2> "$D/stderr.log.txt"
  echo $? > "$D/exit_code"
  echo "real_end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
  kill "$HMON" 2>/dev/null; wait "$HMON" 2>/dev/null
  "$PY" -s -c 'import json, sys
r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
print("status:", r.get("status")); print("reason:", r.get("reason"))
for k in r.get("ranks") or []:
    print("rank %s (%s): status=%s trials=%d reason=%s" % (k.get("rank"), k.get("device"), k.get("status"), len((k.get("correctness") or {}).get("trials", [])), k.get("reason")))' "$D/result.jsonl" > "$D/result-summary.txt" 2>&1
  left=$(pgrep -u "$(id -un)" -f "[m]egabench (_worker|evaluate)" || true); echo "leftover_megabench_processes: ${left:-none}" > "$D/cleanup.txt"
fi
echo "end_utc: $(date -u +%FT%TZ)" >> "$OUT/timing.txt"
touch "$OUT/DONE"
echo "out=$OUT"
