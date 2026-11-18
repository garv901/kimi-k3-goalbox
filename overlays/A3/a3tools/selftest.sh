#!/bin/bash
# ./task_ab --selftest : capture + task_ab on the toy kernel (a3tools/toy) with 1 GPU, including an A/B
# against a deliberately slowed variant (-DTOY_SLOW). Exit 0 when both checks pass.
set -e
WS=$(cd "$(dirname "$0")/.." && pwd)
D="$WS/reports/a3-selftest-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$D"
cp -r "$WS/a3tools/toy" "$D/toy"
cp "$WS/ss_task.cuh" "$D/toy/"
cd "$WS"
./capture --gpus 1 --cand "$D/toy" --layers 0,2 2>&1 | tee "$D/capture.log"
CAP=$(ls -td "$WS"/reports/capture-* | head -1)
./task_ab "$D/toy" "$D/toy" --flags-b=-DTOY_SLOW --task gemv --in gemv_in --out gemv_out --layer 2 \
  --capture "$CAP" --reps 9 2>&1 | tee "$D/task_ab.log"
grep -q "A: PASS" "$D/task_ab.log" && grep -q "B: PASS" "$D/task_ab.log" && grep -q "verdict: B SLOWER" "$D/task_ab.log" \
  && echo "SELFTEST PASS ($D)" || { echo "SELFTEST FAIL ($D)"; exit 1; }
