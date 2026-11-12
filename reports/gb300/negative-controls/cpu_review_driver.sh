#!/bin/bash
# CPU-only review of the negative controls (no GPU is touched: CUDA_VISIBLE_DEVICES is empty and every command
# uses --device cpu). Run from the repository root on a host with the venv; writes
# reports/gb300/negative-controls/cpu-review-<ts>/ with:
#   env.txt, syntax.txt                      versions, host, py_compile of every control
#   cpu-review.log.txt, cpu-review.json      cpu_review.py: the harness's own check_trials/snapshot/compare on a
#   negctrl-record.rank0.jsonl               tiny geometry, 5 fixed seeds, with the controls' record file enabled
#   real-geometry/<control>/                 the real evaluator CLI (`megabench evaluate --device cpu`) on the real
#                                            dev case kimi-k3-dev-l4-tp1-b8-s8192 (full width, 4 layers), 1 trial:
#                                            command.txt, exit_code, stdout/stderr, result.jsonl, time.txt (max RSS),
#                                            negctrl-record.rank0.jsonl
# The real-geometry runs take about 3 to 7 minutes each on CPU (synthetic weight generation dominates) and about
# 60 GB of host RAM; the three negative controls stop at their first trial, the positive control also runs the
# 1-warmup/1-rep timing on CPU and the audit returns not_applicable_cpu.
set -uo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$R/.venv/bin/python
NC=$R/reports/gb300/negative-controls
CASE=kimi-k3-dev-l4-tp1-b8-s8192
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$NC/cpu-review-$TS
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export CUDA_CACHE_PATH=$OUT/cache/cuda TRITON_CACHE_DIR=$OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor XDG_CACHE_HOME=$OUT/cache/xdg
cd "$R" || exit 98
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "host: $(hostname) $(uname -r) $(uname -m)"; echo "user: $(id -un)"
  echo "repo: $R"; echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"
  echo "worktree_status: $(git status --short | tr '\n' ' ')"
  echo "evaluator_files_changed_vs_HEAD: $(git diff --stat HEAD -- megabench setup sandbox | tail -1)"
  echo "python: $PY -> $(readlink -f "$PY") $("$PY" -V 2>&1)"
  "$PY" -s -c 'import torch, triton; print("torch:", torch.__version__, "triton:", triton.__version__, "cuda_available:", torch.cuda.is_available(), "threads:", torch.get_num_threads())'
  echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES] OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "host_mem_available_gb: $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "control_sha256:"; sha256sum "$NC"/*.py
} > "$OUT/env.txt" 2>&1
if "$PY" -m py_compile "$NC"/*.py; then echo "py_compile: ok" > "$OUT/syntax.txt"; else echo "py_compile: FAILED" > "$OUT/syntax.txt"; fi
GOALBOX_NEGCTRL_RECORD=$OUT/negctrl-record "$PY" -s "$NC/cpu_review.py" --seeds 5 --out "$OUT" > "$OUT/cpu-review.log.txt" 2>&1
echo "exit $?" >> "$OUT/cpu-review.log.txt"
for ctrl in negctrl_logits negctrl_routing negctrl_immutability positive_control_oracle; do
  G=$OUT/real-geometry/$ctrl; mkdir -p "$G"
  CMD=(timeout -k 30 1500 "$PY" -s -m megabench evaluate --case "$CASE" --submission "$NC/$ctrl.py" --method negctrl-cpu-review
       --session-id "cpu-review-$TS-$ctrl" --device cpu --trials 1 --warmup 0 --reps 1 --timeout 1200 --output "$G/result.jsonl")
  { printf 'cd %q && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=%s GOALBOX_NEGCTRL_RECORD=%q PYTHONPATH=%q ' "$R" "$OMP_NUM_THREADS" "$G/negctrl-record" "$R"; printf '%q ' "${CMD[@]}"; echo; } > "$G/command.txt"
  echo "start_utc: $(date -u +%FT%TZ)" > "$G/timing.txt"
  GOALBOX_NEGCTRL_RECORD=$G/negctrl-record PYTHONPATH=$R /usr/bin/time -v -o "$G/time.txt" "${CMD[@]}" > "$G/stdout.log.txt" 2> "$G/stderr.log.txt"
  echo $? > "$G/exit_code"
  echo "end_utc: $(date -u +%FT%TZ)" >> "$G/timing.txt"
  "$PY" -s -c 'import json, sys
r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
print("status:", r.get("status")); print("reason:", r.get("reason")); print("submission_sha256:", r.get("submission_sha256"))
c = r.get("correctness"); print("correctness:", c["status"] if isinstance(c, dict) else c)
print("launch_audit:", (r.get("launch_audit") or {}).get("status"))' "$G/result.jsonl" > "$G/result-summary.txt" 2>&1
done
echo "done_utc: $(date -u +%FT%TZ)" > "$OUT/DONE"
