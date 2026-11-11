# PROGRESS

Development log of the `gb300-tp4-synthetic` branch: validating the kimi-k3-goalbox harness on GB300 NVL72 trays before any agent run. Newest entry last. Evidence lives under `reports/gb300/`.

## 2026-11-11: hardware audit and branch

- Finding: a GB300 NVL72 tray is one node with 4 GPUs (aarch64 Grace host). The harness launcher (`megabench/harness/distributed.py`) starts all ranks on one host, so no TP8 case can run on a tray. The real checkpoint does not fit at TP4 (two 200 GB shards per GPU).
- Decision (repo owner, 2026-11-11): the synthetic graded cases become TP4. Real-weight cases stay TP8 and need an 8-GPU node or a two-node launcher that does not exist yet.
- Change: commit `7f3820e`. New graded cases `kimi-k3-syn-l8-tp4-b{1,8,64}-s8192` (suite p1); the TP8 synthetic cases move to suite p2; `goalbox-grade` requests as many GPUs as its largest case; the Slurm partition is no longer hardcoded to `main` in the broker, `goalbox-grade` and `selftest.sh`; `ENVIRONMENT.md`, `gpu-run` and the README describe 4 GPUs per node; new goal prompt `goals/kimi-k3-tp4-synthetic.txt`; new unit test that every case shards cleanly at its TP degree.
- Verification: CPU unit tests 7/7 on rack 3 and on rack 10 with the branch venv (PyTorch 2.14.1+cu130, aarch64). HBM floors from `megabench sol` for the new cases: 0.546 ms (B1), 0.981 ms (B8), 3.144 ms (B64) at 8 TB/s.
- Not done: `bubblewrap` and `socat` are not installed on either rack, so the sandbox and `setup/selftest.sh` cannot run yet (needs sudo). vLLM baseline not attempted.

## 2026-11-11: Step 1, evaluator smoke test (PASS as expected)

- Run: `kimi-k3-dev-l4-tp1-b8-s8192`, candidate = the oracle itself, rack 10 tray14 GPU 0, host level, 75 s. Report: `reports/gb300/step1-dev-l4-tp1-20261111T182425Z/REPORT.md`.
- Result: correctness band, `next_token`, `expert_ids`, `cache_length` and input immutability all pass over 3 seeded trials; launch audit fails with 25128 kernels against a budget of 1, which is the expected outcome for the oracle; status `non_megakernel`, exit code 1. Timing recorded (346 ms candidate, 345 ms eager reference) and is not a performance baseline.
- Infrastructure: no other GPU process and no GPU pod on the node during the run; run classified as completed, not interrupted.
- Observations to review: CUDA-graph capture of the oracle fails on this case (`cudaErrorStreamCaptureInvalidated`), so the graph baseline is unavailable; the advisory oracle-import hint does not match `from megabench.tasks import kimi`.
- Next (needs approval): Step 2, `kimi-k3-dev-l8-tp2-b8-s8192` on 2 GPUs with the same candidate, then the three TP4 synthetic cases.
