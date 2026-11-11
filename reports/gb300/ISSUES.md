# Issue notes for the repository owner (non-blocking)

Observations from the GB300 smoke-test series. None of them blocked the tests, and none of the harness code was changed for them.

## 1. Launch-audit accounting semantics (opened 2026-11-11, Step 2 and Step 3)

`megabench/harness/benchmark.py::_audit_launches` compares `gpu_kernel_count` with the per-rank budget, where `gpu_kernel_count` is the number of profiler events whose `device_type` is CUDA. In the TP4 B1 record each rank's 9089 events split into:

| Kind | Count | Is it a kernel launch? |
|---|---|---|
| compute kernels (`at::native::…`) | 8189 | yes |
| NCCL device kernels (`ncclDevKernel_AllReduce_*`) | 26 | yes |
| `Memcpy` events | 848 | no (copies) |
| `nccl:all_reduce` collective records | 26 | no (collective-level record of the same 26 kernels) |

So each NCCL collective is counted twice, and device copies are counted as launches. For the oracle this changes nothing. For a near-budget megakernel candidate it matters: a single persistent kernel that uses one NCCL collective inside would be counted as 3 events (kernel, NCCL kernel, NCCL record), and any `cudaMemcpy` would add more. The intended semantics of "one launch per rank" (kernels only? kernels plus copies? how NCCL kernels launched by the collective count?) should be fixed in the audit, and the audit should report the split, before a near-budget candidate is graded. Raw evidence: `step2-…/result.jsonl.gz`, `step3-…/result.jsonl.gz` (all event names), split in `result-summary.txt`.

## 2. CUDA-graph reference baseline unavailable on the single-GPU dev case (Step 1)

`graph_baseline_unavailable: cudaErrorStreamCaptureInvalidated` while capturing `kimi.reference` on `kimi-k3-dev-l4-tp1-b8-s8192`. The evaluator falls back to the eager reference. In distributed mode no graph baseline is attempted by design. If the graph baseline is meant to be the "best available baseline" for speedup claims, the oracle needs a capture-safe path.

## 3. Advisory oracle-import hint (Step 1)

`source_hints_advisory.mentions_oracle_module` matches the literal `megabench.tasks.kimi`; `from megabench.tasks import kimi` is not detected. Advisory only; the launch audit and `review_required` still flag such a submission.

## 4. Timing noise in distributed mode (Step 3)

Candidate and eager reference are the same code, yet the evaluator reported 1.07x: the candidate series drifted from 152 to 172 ms over 5 reps while the reference used 1 warmup and 3 reps. The timing of a host-driven eager oracle is noisy; only the candidate's warm CUDA-event latency against the HBM floor should be read as a performance statement, and only with more reps than the smoke-test defaults.

## 5. Evaluator exit code (Step 1)

`megabench evaluate` exits 1 for a correct submission with status `non_megakernel`, the same as for an error. Automation should read `result.jsonl`.
