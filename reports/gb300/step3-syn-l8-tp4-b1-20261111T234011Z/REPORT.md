# Step 3: TP4 batch 1 (first graded synthetic case) with the oracle as the candidate

| Item | Value |
|---|---|
| Case | `kimi-k3-syn-l8-tp4-b1-s8192` (4 GPUs, TP4, 8 layers, batch 1, context 8192, fresh synthetic weights; graded, suite p1) |
| Candidate | `submission.py` here, identical to Steps 1 and 2 (sha256 `6f9e8a84…c906`): `build()` returns `kimi.reference` |
| Runner commit | `31fed7ab044f630821696adf0cd4d6be89e3fa44` (the runner clone was fast-forwarded before the run; code unchanged since `7f3820e`) |
| Candidate commit | same file at the same commit; sha256 above |
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPUs 0 to 3, host level as `dc`, no sandbox, no agent |
| When (UTC) | 2026-11-11 23:40:20 to 23:41:46 (86 s wall clock) |
| Command | `command.txt`: evaluator defaults (3 trials, 2 warmup, 5 timed reps), `--timeout 1500`, outer `timeout -k 30 1800`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,COLL` to per-process files |
| Exit code | 1: 0 of 1 passed because the status is `non_megakernel`. Expected for the oracle. |
| Evaluator status | `non_megakernel`; all four ranks `pass`; aggregate correctness `pass` |

## Correctness: four ranks, three shared-seed trials

| Check | Verdict | Evidence (`result.jsonl.gz`, `result-summary.txt`) |
|---|---|---|
| Four ranks on four distinct physical GPUs | PASS | rank 0..3 = `cuda:0..3` = `GPU-2305f466…`, `GPU-4db89045…`, `GPU-651ce205…`, `GPU-a7a707cd…` (PCI 0008, 0009, 0018, 0019:06:00.0); `gpu-monitor.log.txt` shows pids 3817872..3817875, one per UUID; NCCL logs name cudaDev 0..3 |
| Seeds shared across ranks | PASS | 230337920, 2378828018, 3298540991 on every rank |
| FP32 band: logits, MLA cache writes, KDA recurrent and conv states | PASS on all ranks | worst normalized ratio 0.481 on every rank (logits, trial 1); kv/pe writes 0.473 to 0.477; states 0.440 to 0.444 |
| `next_token` | PASS | 0 rows outside the tie margin and 0 rows differing from the FP32 argmax, all ranks and trials |
| `expert_ids` | PASS | `sets_differing_from_oracle` 0; shortfall 0.0012 to 0.0034 against allowed 0.0033 to 0.0077 |
| `cache_length` | PASS | `mismatched_elements` 0 |
| Input immutability | PASS | `check_unchanged` raised nothing on any rank |
| Per-rank results | available | `ranks[0..3]` each carry correctness trials, launch audit with all kernel names, timings and peak memory |

## NCCL: initialization and collective sequence

| Check | Result |
|---|---|
| Initialization | NCCL 2.30.7+cuda13.3, `Init COMPLETE` 3 times per process (world group plus the TP and EP groups), 0 `NCCL WARN`, 0 error lines in any of the four logs |
| Collective sequence, from the `COLL` trace | 603 collectives per process: 573 AllReduce, 22 AllGather, 8 Broadcast (the AllGather and Broadcast calls are the evaluator's seed sharing and agreement helpers). The ordered sequence of (kind, element count, datatype, reduction op) is identical across all four processes. |
| Audit-step communication | 26 NCCL device kernels per rank in the audited step (25 `AllReduce_Sum_f32`, 1 `AllReduce_Sum_u64` for the greedy token), the same ordered sequence on all four ranks |
| Transport | NVLink peer memory (`P2P/CUMEM`) between all pairs, NVLS multicast available |

## Launch accounting (per rank, all four ranks identical)

| Quantity | Count |
|---|---|
| Total profiler device events (`gpu_kernel_count`, what the audit compares with the budget) | 9089 |
| Compute kernel launches (`at::native` and similar) | 8189 |
| NCCL device kernel launches (`ncclDevKernel_*`) | 26 |
| Actual device kernel launches (compute + NCCL device) | 8215 |
| `Memcpy` device events (not kernel launches) | 848 |
| `nccl:all_reduce` profiler records (collective-level, not kernel launches) | 26 |
| Verdict | `launch_budget_failed` on all four ranks, budget 1, expected for the oracle |

The split uses the kernel names in the raw record; nothing in the evaluator was changed. See `../ISSUES.md` for the accounting-semantics note.

## Timing (recorded, not a performance baseline)

| Measurement (slowest rank per rep) | CUDA-event p50 | p95 | Reps |
|---|---|---|---|
| Candidate | 162.5 ms | 171.5 ms | 152.3, 159.5, 162.5, 168.3, 172.3 |
| Eager reference | 173.9 ms | 174.4 ms | 173.9, 174.4, 160.4 (1 warmup, 3 reps by evaluator design in distributed mode) |

Candidate and reference run the same code, yet the evaluator reports 1.07x. The candidate is timed with 2 warmups and 5 reps, the reference with 1 warmup and 3 reps, and the reps drift upward during the candidate series (152 to 172 ms). This is timing noise of the host-driven eager oracle, not a speedup; it illustrates why these numbers must not be read as a baseline. No CUDA-graph baseline is attempted in distributed mode.

## Memory

| Device | Peak allocated by PyTorch | nvidia-smi sampled peak (10 s) |
|---|---|---|
| GPU 0..3, one rank each | 33.96 GB per GPU (all four equal) | 37210 MiB per GPU |

## Infrastructure

| Check | Result |
|---|---|
| Idle before start | all four GPUs 0 MiB, no compute process; 17 pods on the node, 0 GPU requests (`occupancy-before.txt`); checked again by hand immediately before launch |
| During | GPU monitor every 10 s: only pids 3817872..3817875 (ours), one per GPU, nothing else (`gpu-monitor.log.txt`); Kubernetes poll every 30 s: 0 GPU requests at every sample (`k8s-monitor.log.txt`, polling, not continuous) |
| After | all four GPUs 0 MiB, no compute process, 0 GPU requests (`occupancy-after.txt`) |
| Cleanup | `cleanup.txt`: no leftover `megabench` processes after the evaluator returned |
| Shared venv | unmodified: `/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv` (PyTorch 2.14.1+cu130, Triton 3.8.0, bundled NCCL 2.30.7); driver 595.71.05 |
| Caches | dedicated under `cache/` in this directory (876 MB), not committed |
| Classification | completed, not interrupted |

## Deviations from the Step 2 methodology

- Case and GPU set (TP4 B1 on GPUs 0 to 3).
- `NCCL_DEBUG_SUBSYS=INIT,ENV,COLL`: every collective is logged per process (logging only; adds host overhead, so timings are even less comparable across steps).
- The runner clone was fast-forwarded to `31fed7a` before the run; its raw copies of the earlier report directories were moved outside the repository to allow the pull.
- IP addresses redacted in the committed copy; `result.jsonl` committed gzip-compressed with `result-summary.txt` (output of `analyze_step3.py`, kept with the report) alongside.

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt`, `env.txt`, `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `nccl-*-38178{72,73,74,75}.log.txt`, `cleanup.txt`, `step3.sh`, `submission.py`.
