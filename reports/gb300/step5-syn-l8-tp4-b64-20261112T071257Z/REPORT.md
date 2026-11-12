# Step 5: TP4 batch 64 (graded synthetic case) with the oracle as the candidate

| Item | Value |
|---|---|
| Case | `kimi-k3-syn-l8-tp4-b64-s8192` (4 GPUs, TP4, 8 layers, batch 64, context 8192, fresh synthetic weights; graded, suite p1) |
| Candidate | `submission.py` here, identical to Steps 1 to 4 (sha256 `6f9e8a848ef76fcf234473950f0beedeccdeb163641ce6ee405234c74161c906`): `build()` returns `kimi.reference` |
| Runner commit | `76c3e01127ba0c16155d5756276f65e44d802d85` (code unchanged since `7f3820e`) |
| Analyzer | `analyze_step5.py` (same content as Step 4's corrected analyzer), sha256 `af2efdef0134794e0a8df0f444b1cddde7c08218b1e9535e3c8381a4ce632a92`; `result-summary.txt` is its output on `result.jsonl` with the four NCCL logs beside it |
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPUs 0 to 3, host level as `dc`, no sandbox, no agent |
| When (UTC) | 2026-11-12 07:13:18 to 07:17:17 (239 s wall clock) |
| Command | `command.txt`: evaluator defaults (3 trials, 2 warmup, 5 timed reps), `--timeout 1500`, outer `timeout -k 30 1800`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,COLL` to per-process files; `step5.sh` = Step 4 script plus a 10 s host-memory monitor |
| Exit code | 1: 0 of 1 passed because the status is `non_megakernel`. Expected for the oracle. |
| Evaluator status | `non_megakernel`; all four ranks `pass`; aggregate correctness `pass` |

## Correctness: four ranks, three shared-seed trials, 12 of 12 pass

| Check | Verdict | Evidence |
|---|---|---|
| Four ranks on four distinct physical GPUs | PASS | rank 0..3 = `cuda:0..3` = `GPU-2305f466…`, `GPU-4db89045…`, `GPU-651ce205…`, `GPU-a7a707cd…`; pids 3887249..3887252, one per UUID in `gpu-monitor.log.txt`; NCCL logs cudaDev 0..3 |
| Seeds shared across ranks | PASS | 849511295, 1532634598, 3332118212 on every rank |
| FP32 band, worst normalized ratio (all ranks identical) | PASS | logits 0.489; MLA cache writes kv 0.479, pe 0.479; KDA recurrent state 0.453; conv state 0.455 (trials 0 and 1; trial 2 slightly lower) |
| `next_token` | PASS | 0 rows outside the tie margin in every trial; 5, 9 and 4 of 64 rows have a BF16 argmax different from the FP32 argmax, all inside the margin (see note) |
| `expert_ids` | PASS | `sets_differing_from_oracle` 0; shortfall 0.0057 to 0.0090 against allowed 0.0123 to 0.0190 |
| `cache_length` | PASS | `mismatched_elements` 0 |
| Input immutability | PASS | `check_unchanged` raised nothing on any rank |

Note on next_token: the fraction of rows whose BF16 argmax differs from the FP32 argmax rises with batch (0 of 8 at B1 and about 1 of 8 at B8, 4 to 9 of 64 here), while every such row stays inside the tie margin. With random synthetic weights the top logits are nearly tied, so this is a property of the synthetic cases, not a grader or candidate defect. It shows the tie-margin rule doing its job; a candidate that was not the oracle would be judged by the same rule.

## NCCL: initialization and collective sequence

| Check | Result |
|---|---|
| Initialization | NCCL 2.30.7+cuda13.3, `Init COMPLETE` 3 per process, 0 `NCCL WARN`, 0 error lines, NVLink `P2P/CUMEM` |
| Collective trace | 603 collectives per process (573 AllReduce, 22 AllGather, 8 Broadcast); the ordered sequence of (kind, element count, datatype, reduction op) is identical across all four processes; same histogram as B1 and B8 |
| Audited step | 26 NCCL device kernels per rank (25 `AllReduce_Sum_f32`, 1 `AllReduce_Sum_u64`), identical order on all ranks, same as B1 and B8 |
| Differences from B1 and B8 | only the element counts scale with batch: hidden-state all-reduces of 458752 elements (7168 × 64) and latent all-reduces of 229376 (3584 × 64); greedy reductions over 64 rows |

## Launch accounting (per rank, all four ranks identical)

| Quantity | B64 (this run) | B8 | B1 |
|---|---|---|---|
| Total profiler device events (`gpu_kernel_count`) | 286253 | 58948 | 9089 |
| Compute kernel launches | 256893 | 52933 | 8189 |
| NCCL device kernel launches | 26 | 26 | 26 |
| Actual device kernel launches (compute + NCCL device) | 256919 | 52959 | 8215 |
| `Memcpy` device events (not launches) | 29308 | 5963 | 848 |
| `nccl:all_reduce` collective records (not launches) | 26 | 26 | 26 |
| Verdict | `launch_budget_failed` on all ranks (expected) | same | same |

Compute and memcpy counts scale about linearly with batch (host-driven per-row routing in the eager oracle); the NCCL kernel count does not. The audit was not modified.

## Timing (evaluator diagnostics, not a performance baseline)

| Measurement (slowest rank per rep) | CUDA-event p50 | p95 | Raw reps |
|---|---|---|---|
| Candidate | 4264.3 ms | 4575.7 ms | 4264.3, 4262.2, 4282.2, 4261.5, 4648.9 |
| Eager reference | 4706.7 ms | 5052.6 ms | 5091.0, 4706.7, 4616.6 (1 warmup, 3 reps by evaluator design) |

Same code on both sides; the reported 1.10x is noise. The candidate series has one 9 percent outlier (rep 5); the reference's first timed rep is 8 percent above its median, consistent with a single warmup. First call 4431 ms, `build_ms` 2296.

## Memory

| Quantity | Projection (Step 4 report) | Observed B64 | B8 | B1 |
|---|---|---|---|---|
| Peak allocated by PyTorch, per GPU (all four equal) | about 37 GB | 37.25 GB | 34.30 GB | 33.96 GB |
| nvidia-smi sampled process memory, per GPU | about 41 GB | 41506 MiB | 37726 MiB | 37210 MiB |
| Host RAM used (`free -m`, 24 samples at 10 s) | not projected | 135.5 GB before to 213.5 GB peak, 845.6 GB to 767.6 GB available, swap 0 | | |

Both GPU projections held. Host memory grew by 78 GB during the run (the evaluator snapshots every rank's inputs, including the synthetic weights, to host memory for the immutability check), with 767 GB still available at the peak. PyTorch reserved memory is not in the evaluator record (ISSUES.md item 6).

## Infrastructure

| Check | Result |
|---|---|
| Idle before start | all four GPUs 0 MiB, no compute process; 17 pods on the node, 0 GPU requests; checked by hand and by the script |
| During | GPU monitor every 10 s (24 samples): only pids 3887249..3887252 (ours), one per GPU, no foreign process; Kubernetes poll every 30 s (8 samples): 0 GPU requests throughout (polling, not continuous); host-memory monitor every 10 s |
| After | all four GPUs 0 MiB, no compute process, 0 GPU requests |
| Cleanup | `cleanup.txt`: no leftover `megabench` processes |
| Shared venv | unmodified; driver 595.71.05; PyTorch 2.14.1+cu130, Triton 3.8.0, bundled NCCL 2.30.7 |
| Caches | dedicated under `cache/` in this directory (876 MB), not committed |
| Stop conditions (OOM, memory pressure, hang, contention) | none triggered |
| Classification | completed, not interrupted |

## Deviations from the Step 4 methodology

- Case (B64) and output names; a host-memory monitor added to the script (`host-mem-monitor.log.txt`).
- The runner clone was fast-forwarded to `76c3e01` before the run; its raw Step 4 copy was moved outside the repository first.
- IP addresses redacted in the committed copy; `result.jsonl` (1.1 GB raw: four ranks times 286253 event names) committed gzip-compressed.

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt`, `analyze_step5.py`, `env.txt`, `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `host-mem-monitor.log.txt`, `nccl-*-38872{49,50,51,52}.log.txt`, `cleanup.txt`, `step5.sh`, `submission.py`.
