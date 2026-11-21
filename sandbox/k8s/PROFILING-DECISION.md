# B-C2-1: a trusted launch audit for remotely spawned candidates (decision analysis)

Prepared 2026-11-21, CPU-only (code inspection plus Nsight Systems experiments on rack 3 tray11, inside
Docker containers with their own pid namespaces; no GPU, no cluster change). Decision recorded as L24.

## The problem

DECISIONS.md D-L1 to D-L8 define the launch budget: one device kernel per rank per timed repetition, counted
from an Nsight Systems trace of every timed repetition (D-L5), with CUDA graphs failing outright (D-L4). The
audit of record (`megabench/harness/nsys_audit.py`) traced the evaluator's process tree. Since L19 the
candidate's processes run in the runner container, a different pid namespace and uid, so a trace taken in the
grader contains none of the candidate's kernels, and a trace taken in the runner is produced next to the
candidate. Until this is resolved every grade is a TRANSPORT PASS at most (`launch_audit.status =
advisory_only` in the verdict, the broker says so beside the verdict line).

## What the experiments established (sandbox/k8s/c2/local/, evidence reports/gb300/profiling-decision-20261121/)

| experiment | result |
|---|---|
| E1 clock mapping | trace timestamps are CLOCK_MONOTONIC minus `TARGET_INFO_SESSION_START_TIME.systemClockNs`; agreement within 10 us after library warm-up (first mark 0.2 ms). CLOCK_MONOTONIC_RAW is 23.7 s away on this host. Hence the evaluator records its repetition windows on CLOCK_MONOTONIC (`benchmark.measure_candidate`, field `windows`) and `nsys_audit.audit_windows` aligns any trace to them |
| E2 injection escape | a child started by the traced process with an emptied environment is not traced: none of its events appear, and the session's process list does not contain it (only a `waitpid` of the parent). An untraced process is invisible to the trace, so the trace alone cannot prove that nothing else ran |
| E3b uid separation | with `SETUID`/`SETGID` in the bounding set only, a non-root container gets no effective capabilities (Docker and Kubernetes alike); a file-capability helper (`dropuid`, 0750 root:1000, refuses callers other than uid 1000 and targets at or below it) switches the traced target to uid 1001. The report directory (0700, uid 1000) is then unwritable by the target, the helper is not executable by it, the environment file is unreadable, pid 1 is unsignallable. But the target's events are not delivered: the injection library in a uid-1001 process cannot reach the collector's 0700 runtime directory |
| E3c shared group | keeping gid 1000 for the target and running nsys with umask 007 leaves `/tmp/nsys-agent` at 0700 (the profiler ignores the umask for it); the target still cannot write the report, and can write only the profiler's telemetry directory. The injection library initializes in the uid-1001 target (two initializations in the diagnostics) but no event reaches the collector and the target is absent from the session's process list: delivery needs the 0700 collector directory, which the profiler creates regardless of umask |

## Trust analysis of the options

1. **A: nsys in the runner, same uid as the candidate (today's `nsys_candidate` moved into the runner).**
   The report file is writable by the candidate's uid, and any process left behind can rewrite it after the
   traced process exits. Not authoritative.
2. **B: nsys in the runner, candidate under its own uid (E3b/E3c).** The report file, the environment and the
   spawner are protected; a leftover candidate process cannot touch them. Two limits remain:
   (i) delivery across the uid boundary fails in the version at hand (E3b, E3c: the collector directory is
   0700 to the profiler's uid and not configurable), so B as tested is same-uid only, which forfeits the report
   protection; (ii) more fundamentally, the event
   stream of an injected process is produced inside that process: CUPTI activity buffers live in the
   candidate's address space, so a hostile candidate can drop records (and, per E2, can run kernels in a
   process the trace never sees). B is a sound *detective* control against mistakes and ordinary
   multi-launch submissions, with invalid-on-incomplete rules, but it cannot be called authoritative against
   a candidate that targets the profiler.
3. **C: the grader launches the kernel.** The candidate's `build()` runs in the runner and returns a launch
   plan (kernel binary, entry name, grid, block, shared memory, cooperative flag, argument layout naming the
   input tensors and workspace sizes, peer mappings it needs); the grader loads the module into a *separate
   CUDA context* that maps only the inputs and the candidate's workspace, launches it itself and reads the
   outputs back. One launch per repetition holds by construction; the grader's own trace (its own process,
   nothing injected into a candidate) is the audit of record; device-side launches are found by the grader's
   SASS scan of the binary it was given (D-L6); the reference outputs and the hash key are in another
   context and unreachable by the kernel. Cost: a new submission contract for the GPU phase (the agent still
   writes the kernel and its launch geometry; it no longer writes host code that launches it) and a
   grader-side implementation of the launch path, including peer mappings for multi-rank kernels.
4. **D: accept the advisory audit.** Rejected: D-L8 requires the audit; a full pass would rest on candidate
   process data.
5. **E: a system-wide or driver-side launch counter.** Nsight Systems GPU-metrics sampling is system-wide but
   reports SM activity, not launches, and needs privileges the pods do not have; the kernel driver does not
   see individual launches (user-mode pushbuffers); NVML reports processes and memory, not launches. GPU
   accounting mode (`nvidia-smi -q -d ACCOUNTING`, disabled on rack 3, unknown on rack 10, root to enable)
   would give a complete per-GPU list of processes that ran, which would close E2's gap for process presence
   but still not count launches. Nothing here gives a trusted launch count on its own.

## Recommendation

- **Target design: C** (grader-launched kernel in a separate context). It is the only option in which the
  launch count and the audit of record never pass through candidate-controlled memory, and it also removes
  the remaining export exposure (the kernel sees only what the grader maps). It needs a decision by the
  repository owners because it changes the GPU-phase submission contract; the real-weight validation must
  not start before that decision or before B is accepted as sufficient with its stated limits.
- **Interim for C2: B as a detective control**, run by the spawner for every grade, audited by the grader
  with `audit_windows` (invalid unless complete), with process presence cross-checked against NVML samples
  and, if the operator enables it, GPU accounting. Its result is recorded as `launch_audit.status =
  detective_within_budget` or `audit_invalid`, never as the audit of record, and a grade stays a TRANSPORT
  PASS. This gives C2 the profiler visibility step (plan step 8) and the first measurements for C.
- **C2 experiment to select the mechanism** (plan step 8, 20 minutes, synthetic 1-GPU case, oracle
  candidate), success criteria:
  1. B delivers events across the uid boundary in the pod (if not, record the diagnostic; B is then limited
     to same-uid tracing and loses report protection);
  2. the oracle's kernels fall inside the evaluator's windows with `audit_windows` reporting
     `within_budget`, and a two-kernel control reports `launch_budget_failed`;
  3. a control that starts an untraced subprocess is reported `audit_invalid` through the NVML process count
     (expected processes exceeded) or GPU accounting, not as a pass;
  4. C feasibility probe: the grader creates a second CUDA context, maps one input through the transport's
     IPC path, launches a trivial cubin with `cuLaunchKernel`, and its own `nsys` trace shows exactly that
     kernel, with the reference tensors in the primary context untouched and unmapped.
  Any outcome ends the step at its time box; no step escalates to real weights.

## Association rules for the trusted audit (`nsys_audit.audit_windows`)

- **Phase**: the evaluator's own CLOCK_MONOTONIC window around each timed repetition (sent `run`, received
  the reply), carried in the verdict record (`candidate_timing.windows`). Nothing written by the candidate
  (NVTX ranges included) defines a phase. Warm-up repetitions, correctness trials and the audit pass have no
  windows and are not counted.
- **Job and rank**: one trace per job (the spawner starts the profiler per grade; the report is named by
  the job id); the rank is the device: kernels are attributed to the rank whose device they ran on
  (`device_of_rank`), and the processes seen on that device in the window must be the rank's expected
  candidate process (`expected_pids`, from the spawner's census), else invalid.
- **Process**: kernels from any pid not expected for the rank invalidate the repetition; a trace whose
  process list lacks the candidate's process is invalid (no kernels in the window).
- **NCCL**: NCCL device kernels count as launches (D-L1, D-L3 allows zero); they are named in the record.
- **Profiler-generated and non-kernel events**: memcpy, memset and runtime API records are reported, never
  counted (D-L1); `cudaGraphLaunch*` or graph node events in a window fail (D-L4); the profiler's own
  processes (`nsys`, `nsys-launcher`, `nsys-tee`) launch no kernels and are ignored.
- **Completeness**: the trace must carry the session clock base and a "Profiling has stopped" marker after
  the last window; a window with no kernel is invalid (never a pass). This is the contamination policy
  (L22) applied to the audit: profiler failure, missing events and truncated traces are invalid results.

## What does not change

D-L1 to D-L8 (`planning/DECISIONS.md` unchanged since the baseline, verified by `git diff`), the numerical
tolerances, the trajectory requirements and the sandbox restrictions. The export audit (L21), the verdict
protection (L19) and the network policies are untouched.
