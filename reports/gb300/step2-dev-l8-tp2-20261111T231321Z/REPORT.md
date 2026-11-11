# Step 2: distributed evaluator smoke test (TP2) with the oracle as the candidate

| Item | Value |
|---|---|
| Case | `kimi-k3-dev-l8-tp2-b8-s8192` (2 GPUs, TP2, 8 layers, batch 8, context 8192, synthetic weights) |
| Candidate | `submission.py` in this directory, identical to Step 1 (sha256 `6f9e8a84…c906`): `build()` returns `kimi.reference` |
| Repository commit | code at `7f3820ec32eee5cf98913962addf0a71c5dca780`; the runner clone had not yet pulled `4a83f5f`, which only adds Step 1 evidence and docs (no code change, verified with `git diff --stat 7f3820e 4a83f5f -- megabench sandbox setup goals`) |
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPUs 0 and 1, host level as `dc`, no sandbox, no agent |
| When (UTC) | 2026-11-11 23:13:25 to 23:15:04 (99 s wall clock) |
| Command | `command.txt`: evaluator defaults (3 trials, 2 warmup, 5 timed reps), `--timeout 1500`, outer `timeout -k 30 1800`, `CUDA_VISIBLE_DEVICES=0,1`, `NCCL_DEBUG=INFO` to per-process files |
| Exit code | 1 (`exit_code`): 0 of 1 passed because the status is `non_megakernel`. Expected for the oracle. |
| Evaluator status | `non_megakernel`; both ranks `pass`; aggregate correctness `pass` |

## Distributed correctness

| Check | Verdict | Evidence |
|---|---|---|
| Ranks on distinct physical GPUs | PASS | rank 0 `cuda:0` = `GPU-2305f466…` (PCI 0008:06:00.0), rank 1 `cuda:1` = `GPU-4db89045…` (PCI 0009:06:00.0); `gpu-monitor.log.txt` shows pid 3791240 on the first UUID and pid 3791241 on the second; NCCL logs name `[0]` and `[1]` |
| NCCL initialization | PASS | `nccl-*-3791240.log.txt`, `nccl-*-3791241.log.txt`: NCCL 2.30.7+cuda13.3, 32 channels `0[0] -> 1[1] via P2P/CUMEM` (NVLink peer memory), NVLS multicast available, "Connected all rings"; 0 WARN, 0 ERROR lines; both ranks reached timing and audit, so no collective failed |
| Same three seeded trials on both ranks | PASS | seeds 3164688805, 540641042, 569168861 on rank 0 and rank 1 (the evaluator shares the seed across ranks) |
| FP32 band: logits, MLA cache writes, KDA states | PASS on both ranks | candidate distance = oracle distance; `worst_ratio` 0.44 to 0.49 of the band in every trial |
| `next_token` | PASS | 0 rows outside the tie margin on both ranks; trial 2 has 1 row whose BF16 argmax differs from the FP32 argmax, inside the margin |
| `expert_ids` near-top-k | PASS | `sets_differing_from_oracle` 0; shortfall 0.0022 to 0.0075 against allowed 0.0054 to 0.0160 |
| `cache_length` | PASS | `mismatched_elements` 0 |
| Input immutability | PASS | `check_unchanged` raised nothing on either rank (it would have ended the rank with `error`) |
| Per-rank results | available | `result.jsonl` has `ranks[0]` and `ranks[1]`, each with its own correctness trials, launch audit, timings and peak memory; the aggregate `correctness.per_rank` lists both. No reporting limitation. |

## Launch audit

| Item | Rank 0 | Rank 1 |
|---|---|---|
| Verdict | `launch_budget_failed` (expected) | `launch_budget_failed` (expected) |
| GPU events counted | 58533 against a budget of 1 | 58533 |
| Communication entries | 52: 26 `ncclDevKernel_AllReduce_Sum_*_RING_LL` device kernels (25 f32, 1 u64 for the greedy token) plus 26 `nccl:all_reduce` collective records | same |
| Dominant kernels | `at::native` elementwise and index kernels (about 41000) | same |
| Aggregate | `launch_audit.status` `launch_budget_failed`, `review_required` true | |

The audit counts every CUDA-device event of the profiler, so NCCL device kernels are included. The 26 all-reduces match the oracle's structure for 8 layers (2 per layer plus embedding, LM head and token selection). Each collective also appears once as an `nccl:all_reduce` record, so communication is counted twice in `gpu_kernel_count`; that does not change any verdict but is worth knowing when a candidate is near the budget.

## Timing (recorded, not a performance baseline)

| Measurement | CUDA-event p50 | p95 | Note |
|---|---|---|---|
| Candidate, slowest rank per rep | 889.2 ms | 893.8 ms | 5 reps: 894.9, 889.2, 889.6, 885.8, 873.3 |
| Eager reference, slowest rank per rep | 883.8 ms | 884.3 ms | the distributed evaluator uses 1 warmup and 3 reps for the reference by design |
| CUDA-graph reference | not attempted | | by evaluator design in distributed mode: "host-driven routing and process-group collectives" |

`build_ms` 1980 (two ranks, process-group setup), `first_call_ms_including_jit` 882.

## Memory

| Device | Peak allocated by PyTorch (`hardware.peak_allocated_bytes`) | nvidia-smi sampled peak (`gpu-monitor.log.txt`, 10 s) |
|---|---|---|
| GPU 0, rank 0 | 66.24 GB | 69320 MiB |
| GPU 1, rank 1 | 66.24 GB | 69320 MiB |

## Infrastructure

| Check | Result |
|---|---|
| Shared venv under `dc` | functional and unmodified: `/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv` |
| Versions (`env.txt`) | driver 595.71.05; nvcc 13.2; PyTorch 2.14.1+cu130 (cuDNN 9.24, bundled NCCL 2.30.7); Triton 3.8.0; 2 visible GB300, capability 10.3 |
| Caches | dedicated under `cache/` in this directory (876 MB after the run), not committed |
| kubectl | on PATH this time; in-script Kubernetes checks worked |
| Before | both GPUs 0 MiB, no compute process; 17 pods on the node, 0 GPU requests (`occupancy-before.txt`) |
| During | GPU monitor every 10 s: only pids 3791240 and 3791241 (ours) on GPUs 0 and 1, nothing on GPUs 2 and 3; Kubernetes monitor every 30 s: 0 GPU requests on the node at every sample (`k8s-monitor.log.txt`). Kubernetes monitoring is polling at 30 s, not continuous; cluster events are not readable with this service account. |
| After | both GPUs 0 MiB, no compute process; 0 GPU requests (`occupancy-after.txt`) |
| Cleanup | `cleanup.txt`: no leftover `megabench` processes. The evaluator's own `--timeout` kills the worker's process group (`start_new_session` + `killpg`); the outer `timeout -k 30` covers the parent; the script would have killed any surviving process group. Not exercised: the run finished in 99 s. |
| Classification | completed, not interrupted |

## Deviations from the Step 1 methodology

- 2 GPUs and the TP2 case; `NCCL_DEBUG=INFO` written to per-process files (logging only).
- Kubernetes GPU-pod monitoring every 30 s during the run; kubectl on PATH; `timeout -k 30`; post-run leftover-process check.
- Evaluator design differences in distributed mode, not changes by us: reference timing uses 1 warmup and 3 reps; no CUDA-graph baseline; timings are the slowest rank per repetition.
- IP addresses in the NCCL logs and environment files are replaced with `<ip-redacted>` in the committed copy. `result.jsonl` is committed gzip-compressed (`result.jsonl.gz`) because the two-rank record with all kernel names is large; the raw file stays on the runner.

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt` (analyzer output), `env.txt`, `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `nccl-*-3791240.log.txt`, `nccl-*-3791241.log.txt`, `cleanup.txt`, `step2.sh` (the exact script that ran), `submission.py` (the candidate).
