#!/bin/bash
# CPU-only conversion trial: first 4 layers at TP4 from the CephFS HF copy (started 2026-11-18T20:32:43Z).
export CUDA_VISIBLE_DEVICES=""
cd /mnt/shared/home/kevin/workspace/kimi-k3-goalbox
echo "start $(date -u +%FT%TZ) host $(hostname) commit $(git rev-parse HEAD)" > /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/trial.log
/usr/bin/time -v env PYTHONPATH=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox /mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv/bin/python -s -m megabench.tasks.kimi_checkpoint --src /mnt/shared/weights/kimi-k3 --out /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial --tp 4 --layers 4 --workers 4 >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/trial.log 2> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/trial.time.txt
echo "exit $? end $(date -u +%FT%TZ)" >> /mnt/shared/home/kevin/weights/kimi-k3-fixture-trial/trial.log
