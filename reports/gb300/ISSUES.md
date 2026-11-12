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

## 11. A failing single-GPU evaluation drops the record's provenance fields (negative-control CPU review, confirmed on the GPU in Step 6)

When `evaluate_case` raises (for example `compare()` rejecting the candidate), `_worker` writes a fresh record with only `case`, `submission`, `status`, `reason` and `traceback` (the runner adds `contract_sha256` and `session`). The `submission_sha256`, `hardware` (hostname, torch, CUDA, GPU) and `timestamp_utc` fields that `evaluate_case` had already computed are lost. For a graded run this means an `incorrect` verdict cannot be tied to the exact submission bytes from the record alone. The Step 6 script records the sha256 of the submission in `env.txt` to compensate. Observed on `negative-controls/cpu-review-<ts>/real-geometry/negctrl_logits/result.jsonl` and on all three Step 6 GPU records (`step6-tp1-negctrl_*/result.jsonl.gz`: keys `case`, `contract_sha256`, `reason`, `session`, `status`, `submission`, `traceback` only).

## 12. The `reason` string prefixes the grader's message with the exception type (Step 6)

`_worker` stores `reason = f"{type(e).__name__}: {e}"`, so a rejection reads `AssertionError: logits: {...}` rather than `logits: {...}`. Harmless for a human reader, but anything that matches on the grader's message (an analyzer, a test of the test, a leaderboard) has to strip the prefix first; the Step 6 analyzer does. Documenting the exact format of `reason` (or storing the grader message in its own field) would avoid that.

## 13. The eager oracle's launch count varies between runs of the same case (Steps 1 and 6)

`gpu_kernel_count` for the oracle on `kimi-k3-dev-l4-tp1-b8-s8192` was 25128 in Step 1 and 24927 in Step 6, same code, same case, other trial seeds. The audited step is host-driven per-row routing, so the number of launches depends on the sampled routes. Irrelevant for the budget of 1, but a launch count is therefore not a stable fingerprint of a candidate, and the CUDA-graph reference baseline was again not capturable on this case (item 3).

## 14. In the distributed path every rank-local failure is classified `incorrect` (Step 7 CPU preflight, confirmed on the GPU in Step 8)

`agree()` (`megabench/harness/distributed.py:25`) raises an AssertionError on every rank whenever any rank reports a local error string, whatever the original exception was. `_rank_worker` then maps AssertionError to `incorrect`. So an input-generation `ValueError`, a `RuntimeError` inside the candidate (for example the routing control's near-tie guard) or an out-of-memory error all come out as `incorrect`, where the single-GPU path would say `error`. Step 8 (GPU) shows the consequence for the healthy ranks: ranks 0, 1 and 3 had no local error, yet their status is `incorrect` with rank 2's message. Observed on the CPU: a `ValueError` from an invalid test geometry gave status `incorrect` on all four ranks with reason `AssertionError: rank 0: input generation failed: ValueError: ...`. Carrying the original exception class into the per-rank status (or raising a non-assertion exception from `agree()` when no rank's error was an assertion) would keep the `error`/`incorrect` distinction that the single-GPU path has.

## 15. No per-rank attribution for corrupted outputs (Step 7 rank-local review; the local-error attribution confirmed on the GPU in Step 8)

Routes are checked for cross-rank identity with an all-reduced checksum before the forced references run, and the band statistics of every output are all-reduced over the TP group, so a corruption on one rank fails every rank with the same message and never reaches the near-top-k routing rule. After `agree()` every rank's status is `incorrect`. The only attribution is the `rank N:` prefix of a rank's own local error (candidate exception or failed `check_unchanged()`); Step 8 confirmed on four GPUs that this prefix is faithful (one prefix, `rank 2:`, when only rank 2 mutates; all four in Step 7 when all four mutate). This is consistent with grading a TP group as one candidate, but it means `ranks[].status`/`ranks[].reason` cannot tell a reviewer which rank produced a wrong tensor. Recording, per rank, the local pre-agreement details (for instance the rank's own `flags` and per-output sums before the all-reduce) would make that possible without changing the verdict.

## 16. Rank-local structural failures desynchronize the ranks (Step 7 rank-local review, from code reading)

`compare()` checks output keys, tensor type, shape, dtype and device per rank and raises before its first collective (`kimi.py::compare`, before the `flags` all-reduce). The ranks reach `compare()` only after an agreement, so a candidate that returns a wrong shape on one rank only makes that rank skip the grading collectives while the others enter them: a collective mismatch that ends with the 10-minute process-group timeout rather than a clean `incorrect`. Moving those structural checks before an agreement (as `check_unchanged()` already is) would make the failure deterministic and fast. None of the prepared controls does this; it was not tested on a GPU on purpose.

## 17. The device checksum for inputs above 64 MB is linear and not adversarial (Step 8 robustness summary, shown on the CPU)

`snapshot()` (`megabench/harness/correctness.py`) keeps, for every contiguous tensor above 64 MB, two exact int64 sums of its int32 words: the plain sum and the sum weighted by `(i mod 65521) + 1` within each 2^26-word chunk. Both are linear in the words, so any mutation whose word deltas sum to zero and whose weighted deltas sum to zero is invisible; the simplest is `+d` at word `i` and `-d` at word `i + 65521` in the same chunk. `negative-controls/checksum_collision_check.py` (CPU, `CUDA_VISIBLE_DEVICES=""`, the evaluator's own `snapshot()`/`check_unchanged()`, evidence `negative-controls/checksum-collision-20261112T202413Z/`): on a 128 MB BF16 tensor a single-word change is detected, three such pairs (deltas 1, 12345 and -3, one of them turning a value of 0.78 into 8e28) leave both sums byte-identical and pass `check_unchanged()`, and the same pair on a 64 MB tensor, which takes the exact host-copy path, is detected. Consequences: the immutability check is a guard against accidental in-place mutation (Steps 6, 7 and 8 show single-element and random corruption detected on both paths), not against a candidate that deliberately edits a large input, which on the real TP8 cases means every weight shard (171 GiB per rank above the limit, item 10). A keyed or non-linear hash (for example a per-run random weight vector, or a cryptographic hash of the words computed on the device) would close this at a similar cost; the 64 MB host-copy path is already exact.
