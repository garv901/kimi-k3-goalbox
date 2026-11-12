# Step 4: TP4 batch 8 (graded synthetic case) with the oracle as the candidate

| Item | Value |
|---|---|
| Case | `kimi-k3-syn-l8-tp4-b8-s8192` (4 GPUs, TP4, 8 layers, batch 8, context 8192, fresh synthetic weights; graded, suite p1) |
| Candidate | `submission.py` here, identical to Steps 1 to 3 (sha256 `6f9e8a84…c906`): `build()` returns `kimi.reference` |
| Runner commit | `211ecda45839455a6674a3c4f2e09a057da288c3` (code unchanged since `7f3820e`) |
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPUs 0 to 3, host level as `dc`, no sandbox, no agent |
| When (UTC) | 2026-11-12 00:27:01 to 00:28:58 (117 s wall clock) |
| Command | `command.txt`: evaluator defaults (3 trials, 2 warmup, 5 timed reps), `--timeout 1500`, outer `timeout -k 30 1800`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,COLL` to per-process files; same script as Step 3 except the case and output names (`step4.sh`) |
| Exit code | 1: 0 of 1 passed because the status is `non_megakernel`. Expected for the oracle. |
| Evaluator status | `non_megakernel`; all four ranks `pass`; aggregate correctness `pass` |

## Correctness: four ranks, three shared-seed trials, 12 of 12 pass

| Check | Verdict | Evidence (`result.jsonl.gz`, `result-summary.txt`) |
|---|---|---|
| Four ranks on four distinct physical GPUs | PASS | rank 0..3 = `cuda:0..3` = `GPU-2305f466…`, `GPU-4db89045…`, `GPU-651ce205…`, `GPU-a7a707cd…`; pids 3836445..3836448, one per UUID in `gpu-monitor.log.txt`; NCCL logs cudaDev 0..3 |
| Seeds shared across ranks | PASS | 348423407, 2575015891, 580083562 on every rank |
| FP32 band, worst normalized ratio (all ranks identical) | PASS | logits 0.489 (trial 2); MLA cache writes kv 0.481, pe 0.481; KDA recurrent state 0.457, conv state 0.459; all trials 0.44 to 0.49 |
| `next_token` | PASS | 0 rows outside the tie margin in every trial; trials 1 and 2 each have 1 of 8 rows whose BF16 argmax differs from the FP32 argmax, inside the margin |
| `expert_ids` | PASS | `sets_differing_from_oracle` 0; shortfall 0.0030 to 0.0073 against allowed 0.0070 to 0.0157 |
| `cache_length` | PASS | `mismatched_elements` 0 |
| Input immutability | PASS | `check_unchanged` raised nothing on any rank |

## NCCL: initialization and collective sequence

| Check | Result |
|---|---|
| Initialization | NCCL 2.30.7+cuda13.3, `Init COMPLETE` 3 per process, 0 `NCCL WARN`, 0 error lines, NVLink `P2P/CUMEM` between all pairs |
| Collective trace (`COLL`) | 603 collectives per process (573 AllReduce, 22 AllGather, 8 Broadcast), the ordered sequence of (kind, element count, datatype, reduction op) identical across all four processes |
| Audited step | 26 NCCL device kernels per rank (25 `AllReduce_Sum_f32`, 1 `AllReduce_Sum_u64`), identical order on all ranks, same as B1 |

What the 603 collectives cover (run-length view in `result-summary.txt`): one decode step of the oracle is 24 float32 sum all-reduces (17 of hidden size 7168×B, 7 of latent size 3584×B), one float32 max (greedy max logit) and one int64 min (greedy token), 26 in all, matching the audit. The whole evaluation contains 21 such steps: per correctness trial the BF16 oracle, the candidate and one FP32 reference run (3 trials, 9 steps), then 2 warmup + 5 timed candidate steps, 1 warmup + 3 timed reference steps, and the audited step (12 steps). The remaining collectives are the evaluator's own: seed sharing (AllGather and Broadcast, 30), 21 int64-max agreement all-reduces after each phase, and 15 float64 sums plus 6 float32 max for the band comparison. So the per-process total spans trials, warmup, timing and audit, not a single decode step. Element counts scale with batch: 7168 and 3584 at B1, 57344 and 28672 at B8.

## Launch accounting (per rank, all four ranks identical)

| Quantity | B8 (this run) | B1 (Step 3) |
|---|---|---|
| Total profiler device events (`gpu_kernel_count`, compared with the budget) | 58948 | 9089 |
| Compute kernel launches | 52933 | 8189 |
| NCCL device kernel launches | 26 | 26 |
| Actual device kernel launches (compute + NCCL device) | 52959 | 8215 |
| `Memcpy` device events (not launches) | 5963 | 848 |
| `nccl:all_reduce` collective records (not launches) | 26 | 26 |
| Verdict | `launch_budget_failed` on all ranks (expected) | same |

The compute kernel and memcpy counts grow about 6.5x from B1 to B8 (the oracle's host-driven routing loops over rows and experts); the NCCL kernel count stays at 26 per step because the collective structure does not depend on batch. The audit was not modified.

## Timing (recorded, not a performance baseline)

| Measurement (slowest rank per rep) | CUDA-event p50 | p95 | Raw reps |
|---|---|---|---|
| Candidate | 1049.7 ms | 1208.8 ms | 1059.9, 1049.7, 1044.6, 1245.7, 1021.9 |
| Eager reference | 1033.1 ms | 1085.4 ms | 1033.1, 1024.5, 1091.2 (1 warmup, 3 reps by evaluator design) |

Same code on both sides; the reported 0.98x is noise. Rep 4 of the candidate series is a 19 percent outlier (1245.7 ms), and the reference series spans 1024 to 1091 ms. First call 994 ms, `build_ms` 2422. No CUDA-graph baseline in distributed mode.

## Memory

| Device | Peak allocated by PyTorch | PyTorch reserved | nvidia-smi sampled peak (10 s) |
|---|---|---|---|
| GPU 0..3, one rank each | 34.30 GB per GPU (B1: 33.96 GB) | not exposed by the evaluator record (`hardware.peak_allocated_bytes` only); the sampled process memory below is an upper bound for reserved plus CUDA context | 37726 MiB per GPU (B1: 37210 MiB) |

Batch 1 to batch 8 adds 0.34 GB allocated and about 0.5 GB of process memory per GPU. The synthetic weights of 8 full-width layers (all 896 experts per layer) dominate; the per-sequence state (MLA caches, KDA states) and activations are small at this batch. Collecting PyTorch reserved memory would need a change to the evaluator's record, which is out of scope for this series.

## Infrastructure

| Check | Result |
|---|---|
| Idle before start | all four GPUs 0 MiB, no compute process; 17 pods on the node, 0 GPU requests; checked by hand immediately before launch and again by the script |
| During | GPU monitor every 10 s: only pids 3836445..3836448 (ours), one per GPU; Kubernetes poll every 30 s: 0 GPU requests at every sample (polling, not continuous) |
| After | all four GPUs 0 MiB, no compute process, 0 GPU requests |
| Cleanup | `cleanup.txt`: no leftover `megabench` processes |
| Shared venv | unmodified; driver 595.71.05; PyTorch 2.14.1+cu130, Triton 3.8.0, bundled NCCL 2.30.7 |
| Caches | dedicated under `cache/` in this directory (880 MB), not committed |
| Classification | completed, not interrupted |

## Deviations from the Step 3 methodology

- Only the case (B8) and the output names changed in the script.
- The runner clone was fast-forwarded to `211ecda` before the run; its raw Step 3 copy was moved outside the repository first.
- `analyze_step4.py` (kept here) adds a decomposition of the collective trace by kind, datatype and reduction op and a run-length view; its first version had NCCL's max and min labels swapped, corrected before this summary was produced.
- IP addresses redacted in the committed copy; `result.jsonl` committed gzip-compressed (219 MB raw: four ranks times 58948 event names).

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt`, `analyze_step4.py`, `env.txt`, `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `nccl-*-38364{45,46,47,48}.log.txt`, `cleanup.txt`, `step4.sh`, `submission.py`.
