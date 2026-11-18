#!/bin/bash
# CPU-only checks of the 4-layer TP4 conversion trial (no GPU): verifier, manifest, load-only smoke, evaluator smoke.
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox KIMI_K3_WEIGHTS=/mnt/shared/home/kevin/weights/kimi-k3-fixture-trial OMP_NUM_THREADS=32
cd /mnt/shared/home/kevin/workspace/kimi-k3-goalbox
PY=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv/bin/python
step() { echo "== $1 start $(date -u +%FT%TZ)" >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/progress.txt; }
: verify already done
: manifest already done
step loadonly_cpu; /usr/bin/time -v $PY -s planning/bringup/tp4_real_load_only.py --layers 4 --tp 4 --device cpu --out /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/loadonly > /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/loadonly.log 2> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/loadonly.time; echo "loadonly exit $?" >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/progress.txt
step evaluate_cpu; /usr/bin/time -v timeout -k 30 3000 $PY -s planning/bringup/evaluate_custom_case.py --layers 4 --tp 4 --batch 1 --device cpu --submission reports/gb300/negative-controls/positive_control_oracle.py --session-id bringup-cpu-smoke-l4-tp4 --trials 1 --warmup 1 --reps 1 --output /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/evaluate-cpu.jsonl > /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/evaluate.log 2> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/evaluate.time; echo "evaluate exit $?" >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/progress.txt
echo "== done $(date -u +%FT%TZ)" >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/cpu-checks-20261118T203748Z/progress.txt
