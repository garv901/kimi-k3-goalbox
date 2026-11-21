# Can a trusted external profiler observe every kernel launch of an untrusted candidate? (feasibility, CPU-only)

Prepared 2026-11-21 under the HOLD on B-C2-1: the submission contract and D-L1 to D-L8 in
`planning/DECISIONS.md` are preserved; option C stays a documented alternative, not a recommendation.
Sources: code inspection, the profiler's own documentation (Nsight Systems 2026.3.2 in the project image),
and CPU experiments E1 to E4 on rack 3 tray11 in Docker (own pid namespaces, no GPU, no uid-wide process
controls). Evidence: `reports/gb300/profiling-decision-20261121/`.

## 1. Enforcement versus detection: what the contract requires

- **Enforcement** would make a second launch impossible (only a design in which the trusted side performs
  the launch gives this; that is option C and it changes the contract).
- **Detection** observes the launches that happened and fails the grade when they exceed the budget.
- `planning/DECISIONS.md` requires detection, not enforcement: D-L5 "audit every timed repetition" from the
  profiler trace, D-L4 "a detected CUDA graph launch fails the audit outright", D-L6 device-side launches
  "count as launches" with a static scan as fallback, D-L7 an automated scan instead of review. The
  guarantee the contract actually asks for is therefore: **complete and trustworthy detection with a
  fail-closed rule** (no evidence, incomplete evidence, or evidence of unknown origin = no pass; L22, L24).
- Correction of the record: the audit module's docstring claimed that CUPTI records events "outside the
  candidate process's control". That was never true. CUPTI is a library loaded into the traced process;
  its activity buffers live in that process's address space. The baseline design (candidate process under
  the evaluator, same uid, before L19) had the same property. B-C2-1 changed *where* the candidate runs,
  not the origin of the events. The docstring is corrected in this commit.

## 2. Event taxonomy in a trace (what is counted, what is not)

| event class | trace source | rule |
|---|---|---|
| device kernels (compute) | `CUPTI_ACTIVITY_KIND_KERNEL` | counted (D-L1) |
| NCCL device kernels | same table, names containing `nccl` | counted and named; zero allowed (D-L1, D-L3) |
| CUDA graph launches | runtime API `cudaGraphLaunch*`, `CUDA_GRAPH_*` tables | fail outright (D-L4); also forbidden by the source scan |
| device-side launches (dynamic parallelism) | appear as kernels if visible (H2 open) | counted; SASS/PTX scan fallback (D-L6) |
| memcpy, memset | `_MEMCPY`, `_MEMSET` | reported, not counted (D-L1) |
| CUDA runtime/driver API calls | `_RUNTIME`, `_DRIVER` | reported; used only to detect graph launches |
| NVTX ranges and marks | `NVTX_EVENTS` | **never authoritative**; the candidate's process writes them (L24 uses the evaluator's clock windows instead) |
| profiler-generated events | `nsys`, `nsys-launcher`, `nsys-tee` processes, `PROFILER_OVERHEAD`, `DIAGNOSTIC_EVENT` | ignored for counting; the "Profiling has stopped" diagnostic is required for completeness |
| Kineto `nccl:` records | not in an nsys trace | not counted (D-L2) |

## 3. Mechanisms examined

**M1. Nsight Systems, process-tree scope, same uid as the candidate (spawner launches the candidate under
`nsys profile`).** Works as in E1: events delivered, the clock base aligns to the evaluator's windows within
10 us. Report file and runtime directories are writable by the candidate's uid. Event origin: inside the
candidate process. A child started with an emptied environment is untraced and absent from the trace (E2).

**M2. Nsight Systems with the candidate under its own uid (E3b, E3c).** The report, the environment file,
the spawner and the helper are out of the candidate's reach, but the injection library in the uid-1001
process cannot deliver to the collector's 0700 runtime directory, with or without a shared group and umask.
In addition the helper needs file capabilities, which `allowPrivilegeEscalation: false` (no_new_privs)
disables: confirmed in Docker with `--security-opt no-new-privileges` (`setresuid: Operation not
permitted`). M2 is not available in the pod without a security relaxation, and even then delivers nothing.

**M3. Collector in a separate container, same uid, shared `/tmp`, separate pid namespaces (E4).** The
injection initializes in the runner-side target but the collector in the other pid namespace records no
events and does not list the process. Not available. (Sharing the pid namespace would restore delivery but,
with one uid, also restore the `/proc/<pid>/root` path to the collector's report.)

**M4. Nsight Systems `--cuda-trace-scope=system-wide`.** Documented in this version: traces CUDA activity
of eligible processes launched after collection starts, **same user only**; for a non-root session the
target needs `CUDA_INJECTION_SHM_ALLOWED=TRUE` in its environment before launch. So it is still a
cooperation of the target's CUDA driver instance through shared memory, not a device- or kernel-side
observer: a process started without the variable is not traced (same gap as E2), and the collector must run
as the candidate's uid. What it adds over M1: processes not descended from the profiler's command are
traced if they carry the variable, and the collector is independent of the candidate's process tree. Not
testable without a GPU (needs a CUDA context).

**M5. CUPTI directly (own injection library or Kineto).** Same origin as M1: in-process buffers. No gain.

**M6. Driver- or kernel-side tracing.** Launches on current GPUs go through user-mode pushbuffers; the
kernel module sees channel setup and memory operations, not launches. No supported interface counts
launches from outside the process. NVML reports processes and memory per device, DCGM and the profiler's
GPU-metrics sampling report engine activity over time, not launch counts; both are useful for *presence*
and *interference*, not for the count.

**M7. GPU accounting mode (`nvidia-smi -am 1`, root).** A complete per-device list of processes that ran,
independent of sampling. Disabled on rack 3; unknown on rack 10; the operator would enable it. It closes
the untraced-process gap for presence, not for the count.

**M8. Trusted supervisor in the runner.** The spawner is pid 1 of the runner's namespace, immune to
signals from inside, and under Yama scope 1 no candidate process can ptrace it (it is their ancestor, not a
descendant); its code comes from the read-only image. It can read `/proc/<pid>/{status,cmdline,fd,maps}`
of every process in the namespace (same uid, read mode, not restricted by Yama) and so detect processes,
open `/dev/nvidia*` descriptors (CUDA contexts) and mapped `libcuda`. Its reports can be authenticated to
the grader with a secret handed over at start-up before any candidate runs (a candidate at the same uid
cannot read another process's memory or environment without ptrace). A seccomp user-notification filter
installed before the candidate's exec (allowed under RuntimeDefault with no_new_privs) would make process
creation and device opens synchronous notifications instead of a census; it is an upgrade, not required.

## 4. Separating profiler ownership, process supervision and report ownership

| separation | result |
|---|---|
| profiler uid ≠ candidate uid, same container (M2) | no event delivery; needs file capabilities the pod forbids |
| profiler container ≠ runner container, same uid, shared /tmp (M3) | no event delivery across pid namespaces |
| profiler = spawner (pid 1), same uid as the candidate, same container (M1/M4) | delivery works; report and runtime directories are candidate-writable; the spawner's census and the report hash can be authenticated (M8), and any candidate process alive after the traced process exits invalidates the grade, which closes the file-rewrite window |

Conclusion: ownership of the report file cannot be separated from the candidate's uid without losing
delivery in this profiler version. What can be made trustworthy is the *supervisor* (M8) and the
*completeness checks* (L24: windows, expected pids, stop marker), not the report file's mode bits.

## 5. Detecting untraced subprocesses and additional CUDA contexts from a trusted process

- Spawner census (M8): every pid in the namespace, its open `/dev/nvidia*` descriptors and mapped CUDA
  libraries, sampled at 100 ms during the timed phase; the set must equal the expected candidate processes.
- Grader-side NVML samples (implemented, L22) and, if enabled, GPU accounting (M7): processes that touched
  the device.
- Trace process list versus census: a process in the census with a CUDA context but absent from the trace
  = untraced CUDA process = `audit_invalid`.
- Limit: a hostile candidate that patches its own CUDA driver instance to drop activity records is
  undetectable by any of these. The source scan (D-L7) forbids graph APIs and device launches but not
  `ctypes`, raw driver calls or environment manipulation; extending it is a CPU-side option if the owners
  want to narrow the threat model (not done here: it is a contract-adjacent change).

## 6. Decision matrix

| mechanism | trust boundary of the event origin | completeness | privileges | Kubernetes compatibility | contract impact | status |
|---|---|---|---|---|---|---|
| M1 nsys process-tree, spawner-run, same uid | inside candidate process | process tree only; untraced children invisible (E2) | none | yes | none | viable, detective |
| M4 nsys system-wide CUDA scope, spawner-run | inside candidate process (driver shm) | all same-uid processes carrying the variable | none | yes (untested in a pod) | none | viable, detective; needs GPU test |
| M2 uid-separated target | report protected, origin unchanged | no delivery | file caps, allowPrivilegeEscalation: true | no | none | rejected |
| M3 separate collector container | report protected, origin unchanged | no delivery across pid ns | none | no | none | rejected |
| M5 CUPTI/Kineto in-process | inside candidate process | process only | none | yes | none | no gain over M1 |
| M6 driver/kernel/NVML/DCGM | outside the candidate | presence and activity, no launch count | root for some | partial | none | complement only |
| M7 GPU accounting | outside the candidate | complete process presence, no count | root to enable | operator setting | none | complement |
| M8 spawner census + authenticated reports (+ seccomp notify) | outside the candidate (pid 1, ptrace-immune) | processes, contexts; no launch count | none | yes | none | required complement |
| C grader-launched kernel | outside the candidate (grader's own process) | complete by construction | none | yes | changes the GPU-phase submission contract and the TP8 model (section 7) | documented alternative, on HOLD |

Only C gives event origin outside the candidate. Every detective mechanism (M1, M4) must be combined with
M8 and the L24 completeness rules, and remains forgeable by a candidate that attacks its own CUDA driver
instance. Whether that residual is acceptable for this goal run is the owners' decision.

## 7. Option C: impact on the programming model and on TP8

- **Programming model.** Today `build(case)` returns a Python callable that launches the kernel; the agent
  may use Triton, CUTLASS DSL, torch extensions or raw driver bindings, all of which generate host launch
  code. Under C the candidate would instead deliver a launch plan: kernel binary (cubin or PTX), entry
  name, grid, block, cluster dimensions, dynamic shared memory, cooperative flag, launch attributes, an
  argument layout naming inputs, outputs and workspace buffers, and host-encoded objects such as TMA tensor
  maps (which the grader would encode from a descriptor). Host-side pre-processing in `run(inputs)` (view
  reshapes, scalar computation, stream choice) disappears or must be expressed in the plan. This is a new
  contract for the agent and a new grader subsystem (module loading, argument marshalling, context
  separation, SASS scan of the delivered binary).
- **TP8 communication.** The one-launch rule means the candidate's kernel performs its own communication
  in-kernel. On one tray that needs peer mappings of the other ranks' buffers (cuMemMap or IPC handles);
  across two trays it needs fabric mappings over the rack's NVLink domain (fabric shareable handles and the
  IMEX service) or symmetric heaps in the style of NVSHMEM. Under C the grader, not the candidate, would
  have to establish every mapping the plan asks for in each rank's separate context, including cross-tray
  fabric imports, and verify that the plan maps only candidate buffers. Feasible in principle; a
  substantial addition whose correctness would itself need validation on two trays. Not something to adopt
  during this goal run without the owners' explicit decision; it is recorded here as the only path to true
  enforcement.

## 8. Minimum GPU experiments to validate the remaining plausible approaches (M1, M4, M8)

C2 plan step 8, revised; one tray, 4 GPUs, synthetic cases only, oracle and control candidates from the
test suite; **maximum 30 minutes, 2 GPU-hours** inside the 4-hour C2 ceiling; no automatic continuation.

| # | experiment | success criterion |
|---|---|---|
| G1 | spawner-run `nsys profile --cuda-trace-scope=process-tree` around the oracle on a 1-GPU case; grader runs `audit_windows` with the evaluator's windows and the spawner's expected pids | `within_budget`; every window holds exactly one kernel of the expected pid; overhead versus an unprofiled grade below 5 % of p50 (D-L5) |
| G2 | same with `--cuda-trace-scope=system-wide` and `CUDA_INJECTION_SHM_ALLOWED=TRUE` set by the spawner | events of the candidate's processes delivered without being the profiler's children; otherwise M4 is dropped and M1 kept |
| G3 | two-kernel control and a graph-launch control | `launch_budget_failed` for both; the graph control also fails the scan |
| G4 | untraced-subprocess control (child started with an emptied environment launches the kernel) | the trace lacks the kernel; the spawner census or NVML shows the extra CUDA process; result `audit_invalid`, never a pass |
| G5 | 2-GPU synthetic case with the oracle | NCCL kernels classified and named; windows per rank aligned by device; `within_budget` |

If G1 and G3 to G5 pass, M1 (+M8) is the detective audit for C2 and the owners decide whether its residual
(section 5) is acceptable for real-weight validation. If G1 fails on alignment or delivery, no further
profiler work is useful until the cause is known; the HOLD stays.
