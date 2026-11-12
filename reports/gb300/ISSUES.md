# Issue notes for the repository owner (non-blocking)

Observations from the GB300 smoke-test series (oracle as the candidate, TP1 dev, TP2 dev, TP4 B1/B8/B64, 2026-11-11/12) and from preparing the negative controls (`negative-controls/`, 2026-11-12). None of them blocked the tests, and none of the harness code was changed for them.

## 1. Launch-audit accounting semantics (Steps 2 to 5)

`megabench/harness/benchmark.py::_audit_launches` compares `gpu_kernel_count` with the per-rank budget, where `gpu_kernel_count` is the number of profiler events whose `device_type` is CUDA. Split of those events by name class, per rank:

| Kind | TP4 B1 (Step 3) | TP4 B8 (Step 4) | TP4 B64 (Step 5) | Is it a kernel launch? |
|---|---|---|---|---|
| compute kernels (`at::native::…`) | 8189 | 52933 | 256893 | yes |
| NCCL device kernels (`ncclDevKernel_AllReduce_*`) | 26 | 26 | 26 | yes |
| `Memcpy` events | 848 | 5963 | 29308 | no (copies) |
| `nccl:all_reduce` collective records | 26 | 26 | 26 | no (collective-level record of the same 26 kernels) |
| total `gpu_kernel_count` | 9089 | 58948 | 286253 | |
| actual device kernel launches | 8215 | 52959 | 256919 | |

So each NCCL collective is counted twice, and device copies are counted as launches. For the oracle this changes nothing. For a near-budget megakernel candidate it matters: one persistent kernel that issues one NCCL collective inside would be counted as 3 events (its own kernel, the NCCL kernel, the NCCL record), and any `cudaMemcpy` adds more. The intended semantics of "one launch per rank" (kernels only? kernels plus copies? how NCCL kernels launched by a collective count?) should be decided and the audit should report the split, before a near-budget candidate is graded. Raw evidence: `step2-…/result.jsonl.gz` through `step5-…/result.jsonl.gz` (all event names); the split is in each `result-summary.txt`, produced by the `analyze_step*.py` kept beside it.

## 2. What the per-process NCCL collective total covers (Steps 3 to 5)

With `NCCL_DEBUG_SUBSYS=COLL`, each process logs 603 collectives for one evaluation of a TP4 case (identical across ranks and identical between B1, B8 and B64 apart from element counts). This is not one decode step. One oracle decode step is 26 collectives (24 float32 sum all-reduces, 1 float32 max for the greedy logit, 1 int64 min for the greedy token), and the evaluation runs 21 such steps: 3 per correctness trial (BF16 oracle, candidate, one FP32 reference run), 7 candidate timing steps (2 warmup + 5 reps), 4 reference timing steps (1 + 3), and the audited step. The other 57 collectives are the evaluator's own (seed AllGather/Broadcast, int64-max agreement after each phase, float64 sums and float32 max in the band comparison). Detail: run-length view in `step4-…/result-summary.txt`.

## 3. CUDA-graph reference baseline unavailable on the single-GPU dev case (Step 1)

`graph_baseline_unavailable: cudaErrorStreamCaptureInvalidated` while capturing `kimi.reference` on `kimi-k3-dev-l4-tp1-b8-s8192`. The evaluator falls back to the eager reference. In distributed mode no graph baseline is attempted by design. If the graph baseline is meant to be the "best available baseline" for speedup claims, the oracle needs a capture-safe path.

## 4. Advisory oracle-import hint (Step 1)

`source_hints_advisory.mentions_oracle_module` matches the literal `megabench.tasks.kimi`; `from megabench.tasks import kimi` is not detected. Advisory only; the launch audit and `review_required` still flag such a submission.

## 5. Timing noise of the eager oracle in distributed mode (Steps 3 to 5)

Candidate and eager reference are the same code, yet the evaluator reported 1.07x (B1), 0.98x (B8) and 1.10x (B64). At B1 the candidate series drifted from 152 to 172 ms over 5 reps; at B8 one rep was a 19 percent outlier (1245.7 ms against a 1050 ms median) and the reference spanned 1024 to 1091 ms. Only a candidate's warm CUDA-event latency against the HBM floor should be read as a performance statement, with more reps than the smoke-test defaults.

## 6. Peak reserved memory is not recorded (Step 4)

The per-rank record has `hardware.peak_allocated_bytes` (`torch.cuda.max_memory_allocated`) but not `max_memory_reserved`. The sampled process memory from nvidia-smi is the only upper bound available without changing the evaluator.

## 7. Near-tie next_token rows grow with batch on synthetic weights (Steps 3 to 5)

`rows_differing_from_fp32_argmax` (BF16 argmax differs from the FP32 argmax, but inside the tie margin) was 0 of 8 at B1, 1 of 8 at B8 and 4 to 9 of 64 at B64, always with `rows_outside_tie_margin` 0. Random synthetic weights produce nearly tied top logits, so the tie-margin rule is exercised often on the synthetic cases. Not a defect; worth knowing when reading synthetic-case grades, and a reason to keep the real-weight cases as the fidelity reference.

## 8. Evaluator exit code (Step 1)

`megabench evaluate` exits 1 for a correct submission with status `non_megakernel`, the same as for an error. Automation should read `result.jsonl`.

## 9. The candidate's stdout and stderr are not preserved (negative-control preparation)

`_run_local_worker` pipes the worker's stdout and stderr and keeps only a 4000-character `stderr_tail` when the worker exits without writing a result; the distributed ranks inherit the same pipes. A candidate therefore has no supported way to leave a diagnostic trace in the evidence. The negative controls work around this by appending JSON lines to a file named by `GOALBOX_NEGCTRL_RECORD`. A `candidate_log` field in the record (or keeping the worker's stderr beside `result.jsonl`) would make reviews of real candidates easier too.

## 10. Host memory of a distributed evaluation, and what it means for real-weight TP8 (Step 5, corrected 2026-11-12)

The Step 5 report attributed the 78 GB growth of host memory during the TP4 B64 run to the immutability snapshots. That was an over-attribution: `snapshot()` copies only tensors of at most 64 MB to the host, which is 2.84 GiB per rank for that case (163 of 218 input tensors; the 55 larger ones, 28.6 GiB, are checksummed on the device), so about 11 GiB for four ranks. The 10 s `free -m` timeline (`step5-…/host-mem-monitor.log.txt`) shows three phases: +22 GB in the first 20 s (four processes starting, CUDA contexts and libraries), about +40 GB spread over the three correctness trials (the per-trial snapshots are part of it; the rest, the eager oracle's host-side work and allocator behaviour, was not measured separately), and +33 GB in the final minute, which is the timing phase and the profiler-based launch audit (286253 events per rank held in host memory until the record is written). Peak 213.5 GB used with 767 GB available; nothing was close to a limit.

Implication for the real-weight TP8 cases (`kimi-k3-tp8-*`), computed from `weight_shapes` and `make_state`: 2510 input tensors per rank, 2208 of them at most 64 MB, so the host snapshot is 36.8 GiB per rank and about 147 GiB per tray at 4 ranks per tray (the other 302 tensors, 171 GiB per rank of weights, stay on the GPU and are checksummed there). The 956 GB Grace hosts can hold that, but the snapshot is taken for every trial and lives next to whatever else runs on the tray, and the profiler phase grows with the candidate's launch count (negligible for a real megakernel, 33 GB for the eager oracle at B64). The real checkpoint itself is loaded straight to the device by `safetensors.load_file(device=…)`, so it does not add host memory. A host-memory monitor should stay part of any real-weight run.

## 11. A failing single-GPU evaluation drops the record's provenance fields (negative-control CPU review)

When `evaluate_case` raises (for example `compare()` rejecting the candidate), `_worker` writes a fresh record with only `case`, `submission`, `status`, `reason` and `traceback` (the runner adds `contract_sha256` and `session`). The `submission_sha256`, `hardware` (hostname, torch, CUDA, GPU) and `timestamp_utc` fields that `evaluate_case` had already computed are lost. For a graded run this means an `incorrect` verdict cannot be tied to the exact submission bytes from the record alone. The Step 6 script records the sha256 of the submission in `env.txt` to compensate. Observed on `negative-controls/cpu-review-<ts>/real-geometry/negctrl_logits/result.jsonl`.
