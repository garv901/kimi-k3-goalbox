# Decision request: launch audit for GPU-phase grading (B-C2-1)

To Garv and Kevin, from the build agent, 2026-11-22. One page; details in
`sandbox/k8s/PROFILING-FEASIBILITY.md` and `sandbox/k8s/C2-PROFILING-TEST-MATRIX.md`.

## The question

`planning/DECISIONS.md` D-L1 to D-L8 require every timed repetition to be audited for one device kernel
per rank, from a profiler trace, failing outright on CUDA graphs. The candidate now runs in its own
container and uid (the verdict channel of L19). The audit of record must therefore observe the candidate's
kernels from outside its process. The CPU-only investigation found that no available mechanism does that:
Nsight Systems (process-tree and system-wide CUDA scopes), CUPTI and Kineto all produce their event records
inside the traced process. The baseline design before L19 had the same property; its documentation claimed
otherwise and is corrected.

## Residual risk if a profiler trace is used as the audit

A candidate that controls its own process can (a) start a helper process without the profiler's
environment, whose kernels never appear in the trace, or (b) interfere with the CUDA driver or profiler
libraries inside its process so that records are dropped. (a) is detected by presence evidence: the
spawner's process and CUDA-context census (pid 1 of the runner, immune to the candidate), the grader's NVML
samples and, if enabled, GPU accounting; any extra CUDA process makes the audit invalid. (b) is **not
detected** by anything available: a process that drops its own activity records looks like a one-kernel
candidate. The source scan (D-L7) does not forbid raw driver access or `ctypes`, so the residual is open
to a deliberately hostile submission, not to an honest multi-launch one. The timing itself is not affected
(the grader's own events and windows measure it); only the launch count is.

## The three choices

1. **Accept a documented detective audit for this experiment.** The spawner runs the profiler; the grader
   audits with the evaluator's own windows and fail-closed rules (missing or incomplete evidence is never a
   pass); presence evidence must agree with the trace. Verdicts carry `launch_audit.status =
   detective_within_budget`, never "authoritative". Cost: none to the contract; the residual above is
   accepted and recorded with every grade. Optionally narrow the threat model by extending the D-L7 scan
   (forbid `ctypes`, raw driver bindings, environment manipulation, subprocess creation): a contract-adjacent
   change you would approve separately.
2. **Require a stronger external audit mechanism.** None exists in the supported tooling examined; this
   choice means the GPU phase waits for one (vendor feature, or a privileged node-level collector the
   cluster operator would have to run and that still relies on in-process cooperation), with no date.
3. **Approve a submission-contract change (option C).** The candidate delivers a kernel binary and a launch
   plan; the grader launches it in a separate CUDA context that maps only candidate buffers. The count is
   one by construction and the grader's own trace is authoritative. Cost: a new contract for the agent
   (no host-side launch code; Triton/CUTLASS/torch wrappers must export the kernel and its arguments), a
   grader launch subsystem, and grader-established peer and cross-tray fabric mappings for in-kernel TP8
   communication. Substantial engineering and its own validation; not for this goal run's timeline.

## Recommendation of the build agent

Choice 1 for the C2 experiment and, if its matrix passes, for the real-weight validation phase, with the
residual stated in every verdict and in the final report; choice 3 recorded as the path to an authoritative
audit for a later program. The experiment itself (30 minutes, 2 GPU-hours maximum, after C1b passes and
with a separate lease approval) does not depend on this decision; the use of its result does.

## Exact decisions needed

- From Garv and Kevin: choice 1, 2 or 3 above.
- If choice 1: whether to extend the D-L7 scan as described (yes/no).
- From Kevin: the C2 lease approval (tray, start, 4-hour ceiling), and whether GPU accounting mode may be
  enabled on the lease node for the duration (root; presence evidence becomes complete).
- From Garv and Kevin: GO for a controls-only evaluator flag `--mechanism-test` (skips correctness trials for
  the profiling controls, never a pass, never in production options; the evaluator rejects `--trials 0`).
  Without it the matrix shrinks to the oracle and scan tests (P1, P4, P9).
