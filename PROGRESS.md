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

## 2026-11-11: Step 2, distributed evaluator smoke test at TP2 (PASS as expected)

- Run: `kimi-k3-dev-l8-tp2-b8-s8192`, candidate = the oracle, rack 10 tray14 GPUs 0 and 1, host level, 99 s. Code at commit `7f3820e` (no code change since). Report: `reports/gb300/step2-dev-l8-tp2-20261111T231321Z/REPORT.md`.
- Result: both ranks `pass` on all three shared-seed trials (band ratios 0.44 to 0.49, next_token inside the tie margin, expert sets identical to the oracle, cache_length exact, inputs unchanged). Launch audit fails on both ranks with 58533 GPU events against 1, including 26 NCCL all-reduce device kernels per rank; status `non_megakernel`, exit code 1. Timing recorded (889 ms candidate, 884 ms eager reference, slowest rank), not a baseline. Peak 66.2 GB allocated per GPU.
- Infrastructure: ranks on distinct GPUs (UUIDs and PCI ids recorded), NCCL 2.30.7 over NVLink P2P with 0 warnings, no other GPU process and 0 GPU-pod requests on the node before, during (30 s polling) and after. Not interrupted. No leftover processes.
- Observation: each NCCL collective appears twice in `gpu_kernel_count` (device kernel plus `nccl:all_reduce` record). The distributed evaluator does not attempt a CUDA-graph baseline by design.
- Next (needs approval): `kimi-k3-syn-l8-tp4-b1-s8192` on 4 GPUs with the same candidate, then B8 and B64 after separate approvals.

## 2026-11-11: Step 3, TP4 batch 1, first graded synthetic case (PASS as expected)

- Run: `kimi-k3-syn-l8-tp4-b1-s8192`, candidate = the oracle, rack 10 tray14 GPUs 0 to 3, host level, 86 s, runner at `31fed7a`. Report: `reports/gb300/step3-syn-l8-tp4-b1-20261111T234011Z/REPORT.md`.
- Result: all four ranks `pass` on three shared-seed trials; worst band ratio 0.481 on every rank; next_token exact against the FP32 argmax; expert sets identical to the oracle; cache_length exact; inputs unchanged. Launch audit fails on all ranks: 9089 profiler device events per rank = 8189 compute kernels + 26 NCCL device kernels + 848 memcpy events + 26 collective records; status `non_megakernel`. Timing recorded (162 ms candidate, 174 ms eager reference, same code: noise), not a baseline. Peak 33.96 GB allocated per GPU.
- NCCL: 603 collectives per process (573 AllReduce, 22 AllGather, 8 Broadcast), identical sequence on all four processes, 0 warnings; the audited step has the same 26 NCCL device kernels in the same order on every rank.
- Infrastructure: four distinct GPUs by UUID and PCI id, no other GPU process, 0 GPU-pod requests before, during (30 s polling) and after, no leftover processes. Not interrupted.
- New: `reports/gb300/ISSUES.md` with non-blocking notes for the repo owner, first of all the launch-audit accounting semantics (collectives counted twice, copies counted as launches).
- Next (needs approval): `kimi-k3-syn-l8-tp4-b8-s8192`, then B64; then a deliberately corrupted-output negative control before any TP8 or real-weight work.

## 2026-11-12: Step 4, TP4 batch 8 (PASS as expected)

- Run: `kimi-k3-syn-l8-tp4-b8-s8192`, candidate = the oracle, rack 10 tray14 GPUs 0 to 3, host level, 117 s, runner at `211ecda`. Report: `reports/gb300/step4-syn-l8-tp4-b8-20261112T002655Z/REPORT.md`.
- Result: 12 of 12 rank-trial combinations pass; worst band ratios logits 0.489, cache writes 0.481, KDA states 0.459; next_token inside the tie margin; expert sets identical; cache_length exact; inputs unchanged. Launch audit fails on all ranks: 58948 profiler events per rank = 52933 compute kernels + 26 NCCL device kernels + 5963 memcpy + 26 collective records (actual launches 52959). Status `non_megakernel`.
- NCCL: 603 collectives per process, identical sequence on all four processes, 0 warnings; the audited step has the same 26 NCCL kernels as B1. The trace decomposition shows the 603 span 21 decode steps across trials, warmup, timing and audit plus 57 evaluator collectives (ISSUES.md item 2).
- Memory: 34.30 GB allocated per GPU (B1: 33.96), 37726 MiB sampled (B1: 37210); reserved not exposed by the evaluator. Timing recorded, not a baseline (1050 ms candidate, 1033 ms eager reference, one 19 percent outlier rep).
- Infrastructure: four distinct GPUs, no other GPU process, 0 GPU-pod requests before, during and after, no leftover processes. Not interrupted.
- ISSUES.md updated (B1 vs B8 accounting table, collective coverage, reserved memory, timing noise).
- Next (needs approval): `kimi-k3-syn-l8-tp4-b64-s8192` with the memory feasibility assessment in the Step 4 report message; then the corrupted-output negative control before any TP8 or real-weight work.

## 2026-11-12: Step 5, TP4 batch 64, last graded synthetic case (PASS as expected)

- Run: `kimi-k3-syn-l8-tp4-b64-s8192`, candidate = the oracle, rack 10 tray14 GPUs 0 to 3, host level, 239 s, runner at `76c3e01`. Report: `reports/gb300/step5-syn-l8-tp4-b64-20261112T071257Z/REPORT.md`.
- Result: 12 of 12 rank-trial combinations pass; worst band ratios logits 0.489, cache writes 0.479, KDA recurrent 0.453, conv 0.455; next_token 0 rows outside the tie margin (4 to 9 of 64 rows per trial differ from the FP32 argmax inside the margin); expert sets identical; cache_length exact; inputs unchanged. Launch audit fails on all ranks: 286253 profiler events per rank = 256893 compute + 26 NCCL device kernels + 29308 memcpy + 26 collective records (256919 actual launches). Status `non_megakernel`.
- NCCL: 603 collectives per process, identical sequence on all four processes, 0 warnings; the audited step has the same 26 NCCL kernels as B1 and B8; only element counts scale with batch.
- Memory: 37.25 GB allocated per GPU (projection about 37), 41506 MiB sampled (projection about 41); host RAM used 135.5 to 213.5 GB with 767 GB still available. Timing recorded, not a baseline (4264 ms candidate, 4707 ms eager reference).
- Infrastructure: four distinct GPUs, no foreign GPU process, 0 GPU-pod requests before, during and after, no leftover processes. No stop condition triggered. Not interrupted.
- ISSUES.md updated (B64 column, near-tie next_token note). The TP4 synthetic series (B1, B8, B64) is complete.
- Next (needs approval): the corrupted-output negative control (a candidate that returns the oracle output with a deliberate perturbation) to show the grader rejects an incorrect candidate independently, before any TP8 or real-weight work.
