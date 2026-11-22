# C2 step 8: profiling experiment test matrix (prepared 2026-11-22, not yet authorized to run)

Budget: **30 minutes wall clock, 2 GPU-hours maximum** (one tray, 4 GPUs), inside the unchanged 4-hour C2
ceiling. Synthetic cases only (`kimi-k3-dev-l4-tp1-b8-s8192` for 1 GPU, `kimi-k3-dev-l8-tp2-b8-s8192` for
2 GPUs). No real weights, no agent, no checkpoint conversion. The operator applies the pod; each test is a
broker grade request with controls-only harness options; the build agent reads evidence only.

## Pass levels (unchanged distinctions)

| level | meaning | what this experiment can establish |
|---|---|---|
| TRANSPORT PASS | the verdict channel, isolation boundary, export audit and contamination controls work on real GPUs (C2 steps 0 to 7, 9, 10) | not this step |
| PROFILING FEASIBILITY PASS | the spawner-run profiler delivers the candidate's kernel events, aligns to the evaluator's windows, counts controls correctly, and the supervisor detects the untraced-process control | **the most this step can establish** |
| AUTHORITATIVE AUDIT PASS | the launch count comes from evidence the candidate cannot suppress or forge without reliable detection | **not claimable with any mechanism tested here**: every available profiler produces its events inside the candidate process (PROFILING-FEASIBILITY.md section 3). It needs either the owners' acceptance of the documented residual or a mechanism outside the candidate (option C, on HOLD) |

Census of processes and CUDA contexts (supervisor, NVML, GPU accounting) is evidence of *presence*, never a
kernel-launch count. "Visibility" below means the mechanism reported events; "completeness" means every
launch that happened is in the trace, which no test here can prove for a hostile candidate.

## Common setup for every test

- Harness options (controls only, never in production `GOALBOX_GRADE_ARGS`): `--trials 1 --warmup 1 --reps 5
  --no-scan --reference-reps 0 --timeout 600 --mechanism-test`. Controls return structurally valid,
  numerically wrong outputs, so every control verdict is `incorrect` (exit 4). **Open point (found in the CPU
  check of 2026-11-22):** the evaluator rejects `--trials 0` ("trials, reps, timeout must be positive"), and
  with one correctness trial an incorrect control fails before the timing phase, so no repetition windows
  exist for P2, P3, P5, P6, P7, P8 and P10. A controls-only evaluator flag `--mechanism-test` (skip the
  correctness trials, run warm-up and timed repetitions, grade the timed outputs, mark the record
  `mechanism_test: true`, never a pass, never accepted by the broker's production options, like `--tiny`) is
  the proposed resolution; it is not implemented in this commit because it touches the evaluator and needs
  the owners' GO. Without it, only P1, P4 and P9 (oracle and scan) can run, and counting precision would
  rest on oracle-plus-extra-kernel differences, a weaker test.
- Profiler run by the spawner (uid 1000, pid 1 of the runner) around the candidate process:
  `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o <job>/profile`,
  report exported to SQLite by the grader with `nsys export`.
- Supervisor evidence (independent of the profiler): the spawner's census every 100 ms during the job
  (pids, uids, open `/dev/nvidia*` descriptors, mapped CUDA libraries), the grader's NVML compute-process
  samples (L22) and, if the operator enabled it, GPU accounting. These are presence evidence only.
- Invalid audit outcomes in every test: missing events, profiler failure, a trace without the clock base or
  the "Profiling has stopped" marker, a window without a kernel, kernels from an unexpected process, a census
  process with a CUDA context that is absent from the trace, unexplained GPU activity. An INVALID outcome is
  never a pass and never a feasibility pass.
- Candidates: `megabench/tests/candidates/{oracle_candidate,one_kernel_control,two_kernel_control,
  graph_control,untraced_subprocess_control}.py` (controls added 2026-11-22; CPU-checked for import and scan
  behaviour only).

## The matrix

| # | test | candidate, case, scope | expected observations | required profiler artifacts | independent supervisor evidence | PASS | FAIL | INCONCLUSIVE | establishes |
|---|---|---|---|---|---|---|---|---|---|
| P1 | process-tree profiling, oracle | oracle, 1-GPU case, `--cuda-trace-scope=process-tree`, `--trials 1` | verdict `ok_provisional` (TRANSPORT PASS level); trace contains the candidate process, many kernels per window (the oracle is a multi-kernel reference), clock base present, stop marker after the last window | `profile.nsys-rep`, `profile.sqlite`, `audit_windows` record, evaluator record with windows | census: exactly one candidate CUDA process; NVML: expected process count | events delivered; all 5 windows hold kernels of the expected pid; audit `launch_budget_failed` with the oracle's count (correct for a multi-kernel reference); no invalid reason | no events, missing clock base, kernels in windows from another pid, or `audit_invalid` | profiler crashed or the pod's device plugin denied the runner GPU access (step 1 failure) | visibility, alignment |
| P2 | one-kernel control | `one_kernel_control`, 1-GPU, process-tree | verdict `incorrect`; exactly 1 kernel in each of the 5 windows, 0 graph launches; memsets/memcpys reported, not counted | as P1 | census: one candidate CUDA process | `within_budget` on the audit record with `device_kernels == 1` in every window | any window with 0 or ≥2 kernels; `audit_invalid` | the control's warm-up template computation leaked into a window (visible as hundreds of kernels in window 0 only): rerun with `--warmup 2` | visibility, counting precision |
| P3 | two-kernel control | `two_kernel_control`, 1-GPU, process-tree | exactly 2 kernels per window | as P1 | as P2 | `launch_budget_failed` with `device_kernels == 2` in every window | 1 or ≥3 per window, or `within_budget` | as P2 | counting precision |
| P4 | CUDA graph control, scan on | `graph_control`, 1-GPU, harness options without `--no-scan` | the grader rejects before running: scan `cuda_graph` rule hit (plus `oracle_import` from the controls' shared helper, CPU-checked) | scan record in the verdict | none needed | verdict `scan_failed`, no timing, no trace | the scan passes the control | scan did not run (check `harness_args`) | contract rule D-L7 holds |
| P5 | CUDA graph control, scan off | `graph_control`, 1-GPU, `--no-scan`, process-tree | one `cudaGraphLaunch` per window, graph node events present | as P1 plus `CUDA_GRAPH_NODE_EVENTS` rows | as P2 | audit `launch_budget_failed` with `graph_launches ≥ 1` per window (D-L4) | graph launch invisible (0 graph launches, kernels counted as plain) | `--cuda-graph-trace=node` unsupported by this driver (record the diagnostic) | D-L4 detectability; H1 of DECISIONS.md answered |
| P6 | untraced-subprocess negative control | `untraced_subprocess_control`, 1-GPU, process-tree | trace holds the parent's 1 kernel per window; the child's kernels are absent; census shows a second process of the candidate's uid with `/dev/nvidia*` open; NVML shows one more compute process than expected | as P1 plus census log and NVML samples | **required**: census and NVML both list the extra process | audit `audit_invalid` (reason: census CUDA process absent from the trace) and grade state `CONTAMINATED`/`+CONTAMINATED` (L22) | the extra process is reported by neither census nor NVML, or the audit returns `within_budget` | the child failed to start (check `candidate/<rid>/*.stderr`) | that suppression is detected by presence evidence in this configuration; **not** that all suppression is detectable |
| P7 | system-wide CUDA scope | `one_kernel_control`, 1-GPU, `--cuda-trace-scope=system-wide`, spawner sets `CUDA_INJECTION_SHM_ALLOWED=TRUE` for the candidate, profiler started by the spawner before the candidate | events of the candidate delivered although it is not a child of the profiler command; same counts as P2 | as P1 | as P2 | same result as P2 under system-wide scope | no events (then M4 is dropped, M1 kept) | the version refuses the scope in the pod | visibility under the alternative scope |
| P8 | system-wide scope versus untraced child | `untraced_subprocess_control`, 1-GPU, system-wide | tells whether a same-uid process started without the variable is traced | as P6 | as P6 | documented either way; audit must still be `audit_invalid` if the child is untraced, or `launch_budget_failed`/`within_budget` with the child's kernels attributed to a second pid (expected pids mismatch → `audit_invalid`) | audit not invalid while the child ran untraced | child failed to start | the boundary of system-wide scope |
| P9 | two-GPU NCCL activity | oracle, 2-GPU case, process-tree, `--trials 1` | two candidate processes, one per device; NCCL device kernels present and named; windows per rank aligned by device; verdict `ok_provisional` | as P1 for both ranks, `device_of_rank` from the topology | census: exactly two candidate CUDA processes | both ranks' windows hold kernels of their expected pid on their device; NCCL kernels listed under `nccl_kernels`; no invalid reason | kernels attributed across devices, missing rank, `audit_invalid` | NCCL init failed across the runner's processes (an isolation issue, step 5) | visibility and per-rank attribution |
| P10 | profiler overhead and window alignment | `one_kernel_control` and oracle, 1-GPU, profiled versus unprofiled grades back to back | p50 of `cuda_event_ms` under the profiler within 5 % of the unprofiled grade (D-L5); every kernel start inside its window with ≤ 2 ms slack; no kernel between windows for the one-kernel control | both records, both traces | NVML health counters unchanged (L22) | overhead ≤ 5 %, alignment holds | overhead > 5 % or kernels outside windows | clocks or GPUs throttled during the run (`clocks_event_reasons` in the health record) | alignment; overhead bound |

Order of execution and time boxes: P1 (6 min), P2 and P3 (4 min), P4 and P5 (4 min), P6 (4 min), P7 and P8
(5 min), P9 (4 min), P10 (3 min). The operator stops the step at 30 minutes regardless of progress; whatever
ran is the evidence.

## Verdict of the step

- **PROFILING FEASIBILITY PASS**: P1, P2, P3, P4, P5, P6, P9 and P10 PASS; P7 and P8 are informational.
- **FAIL**: any of P1, P2, P3, P6 FAIL (no visibility, no precision, or suppression undetected).
- **INCONCLUSIVE**: otherwise; the HOLD stays and the next step is a repeat with the recorded fix.
- In no case is AUTHORITATIVE AUDIT PASS claimed: the residual (a candidate attacking its own CUDA driver
  instance can drop records without presence evidence changing) is not detected by any test above.

## Unresolved assumptions (to be confirmed by the experiment or by the operator)

1. The runner container sees the pod's GPUs through the device plugin's environment strategy (C2 step 1).
2. The profiler's injection delivers events when the spawner, not the grader, launches the candidate in the
   pod (the Docker model showed delivery at the same uid; the pod adds seccomp RuntimeDefault and a
   read-only root filesystem; the profiler's runtime directory must be on the runner's writable `/tmp`).
3. `--cuda-graph-trace=node` works with driver 595.71.05.
4. Resolved negatively: `--trials 0` is rejected by the evaluator; the controls-only `--mechanism-test` flag
   (above) is needed for P2, P3, P5 to P8 and P10 and awaits the owners' GO. Controls were CPU-checked for
   import, two calls each, and scan behaviour (`reports/gb300/c2-profiling-prep-20261122/`).
5. GPU accounting mode on rack 10 (operator: `nvidia-smi -q -d ACCOUNTING`); without it NVML sampling is
   the only grader-side presence evidence, and a child shorter than the sampling interval can be missed.
6. Nothing in this matrix changes D-L1 to D-L8, the tolerances, the grading contract or the C2 stop
   conditions.
