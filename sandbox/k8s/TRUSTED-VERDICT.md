# Trusted verdict channel for GPU jobs (blocker B-C1-1)

Status 2026-11-19: designed, implemented and validated on the CPU (unit tests and a two-container Docker
model with the real grader, spawner and broker files). The GPU-specific parts (CUDA IPC across containers,
GPU visibility in the runner container) are live tests listed at the end. DECISIONS-LOG L19, L20.

## Problem

Until commit `be8905a` the evaluator and the candidate ran in one container as one uid, and the results
half of the control volume was writable by both. A candidate could write a passing verdict, delete another
job's result, replace the evaluator's output, or kill the evaluator. The broker could validate the format
of a result, not its truth (C1-OPERATOR-SAFETY-REVIEW §7).

## Design

Each GPU worker pod has two containers that share only a pod-local emptyDir (`/job`) and the GPUs:

| | `grader` (trusted) | `runner` (candidate side) |
|---|---|---|
| uid / gid | 2000 / 2000 | 1000 / 1000 |
| process | `sandbox/lib/executor.py`: reads requests, runs the evaluator, writes verdicts | `sandbox/lib/spawner.py`: starts what the grader asks for, nothing else |
| runs candidate code | never | always (exec commands, candidate processes) |
| `/ctl-req` (requests) | read-only | not mounted |
| `/ctl-results` (verdicts, logs, done records) | read-write | not mounted |
| `/work` (agent's files) | read-only | read-write |
| `/weights` | read-only | read-only |
| `/job` (emptyDir) | read-write; creates job dirs 0755, `out/` and spawn dirs 0777 | read-write; can create only inside 0777 dirs |
| `/tmp`, `/dev/shm` | own emptyDirs | own emptyDirs |
| records volume | not mounted | not mounted |
| pid namespace | own | own (no `shareProcessNamespace`) |
| capabilities | drop ALL, add IPC_LOCK | drop ALL, add IPC_LOCK |
| root filesystem | read-only | read-only |
| API token | none | none |

Two kinds of request reach the grader (both written only by the controller's broker, DECISIONS-LOG L13):

- **exec**: the agent's command (`gpu-run -- CMD`). The grader copies the script into `/job/<rid>/run.sh`
  (0644), the spawner runs it, writes `out/stdout.log` and `out/exit`. The grader copies the output into
  its log under a `candidate-authored` header and records `trusted: false, reported_exit: N`. The broker
  shows the state as `(candidate-reported)`. Nothing about an exec job is a grade.
- **grade**: `gpu-run --grade CASE [--submission PATH]`. The broker fixes the GPU count from the case
  table, the harness options (`GOALBOX_GRADE_ARGS`) and the device; the agent chooses only the case id
  and a file under `/work`. The grader snapshots the submission's directory into `/job/<rid>/candidate`
  (0755/0644, read-only to the runner; sha256 recorded), runs `python -m megabench evaluate` **in its own
  container** with `GOALBOX_SPAWN_DIR=/job/<rid>/spawn`, and the harness's `CandidateProcess` (remote
  mode, `megabench/harness/isolation.py`) writes a spawn request per rank instead of forking: a 0777
  directory with `cmd.json` and two FIFOs. The spawner starts `python -m megabench.harness.isolation
  --remote <dir>` as uid 1000; the two processes speak the unchanged build/run/profile/stop protocol over
  the FIFOs (`megabench/harness/transport.py`), tensors as raw CUDA IPC handles (GPU) or by value (CPU).
  The evaluator computes the reference, hashes the outputs it received and times them with its own
  events, exactly as before; the verdict is written by the grader to `verdict/<rid>.json` and summarised
  in `done/<rid>.json` with `trusted: true`. The candidate processes' stderr files are copied to
  `candidate/<rid>/` as untrusted evidence.

Outcome classes are decided by the grader alone: the evaluator's status (`ok_provisional`, `incorrect`,
`candidate_error`, `timeout`, …) when it produced a record; `TIMEOUT` (grader's own deadline);
`GRADER_FAILED` (evaluator exited without a record); `INFRA_RUNNER_DOWN` (no spawner heartbeat); `REJECTED`.
The broker refuses a done record that is malformed or older than its dispatch (`MALFORMED_RESULT`, exit 8).

Cleanup: the spawner kills a job's sessions (uid-wide in the pod) when the grader writes `END`; since the
candidate can defeat that, the grader diffs the NVML compute-process list before and after each job and,
if anything is left, writes `/job/RESTART-RUNNER`. The runner's liveness probe (`test ! -e
/job/RESTART-RUNNER`) then fails and kubelet restarts the container, killing every process in it. The
grader clears the flag when it sees a fresh spawner start time. The grader also uses this when the spawner
heartbeat is stale before a job.

## Why each layer is needed (requirement 4)

- **Different uids.** With one uid, a process in the runner can open `/proc/<grader pid>/root/...` and
  `/proc/<pid>/fd/*` (magic links need only `PTRACE_MODE_READ`, which Yama does not restrict for the same
  uid) and so reach every file the grader has mounted or open, in another mount namespace. Different
  uids without `CAP_SYS_PTRACE` deny that, and deny `kill()`.
- **Separate pid namespaces.** Without visibility of the grader's pids there is no magic link to try at
  all, and no signal to send. The runner's pid 1 is the spawner, which the kernel protects from signals
  sent from inside its own namespace (shown in the Docker regression: `kill` returns success, the
  spawner lives).
- **Mount layout.** The runner has no mount of the request spool, the results spool or the records: there
  is no path to forge, not a permission to bypass. The grader has `/work` read-only (kernel-enforced
  even though group 1000 could write), so a grader bug cannot alter the agent's files either.
- **Directory modes in `/job`.** The emptyDir root is writable by both (the runner can create junk there;
  harmless), so the grader works under an unpredictably named root (`/job/grader-<hex>`, 0755) announced
  in `/job/GRADER`; the runner cannot pre-create or alter job directories, `job.json`, `run.sh` or the
  graded snapshot. It writes only in `out/` and in the spawn directories (its own stdout, stderr, exit
  files), none of which the grader treats as a result. A runner that replaces `/job/GRADER` only breaks
  its own service (INFRA_RUNNER_DOWN).
- **Separate `/tmp` and `/dev/shm`.** The evaluator's private result file and rendezvous files live in
  the grader's `/tmp`; NCCL and CUDA IPC bookkeeping do not cross containers through shared memory
  files. Tensor transport uses raw CUDA IPC handles instead of torch's reductions, which would need a
  shared, same-uid `/dev/shm` ref-counter file.
- **Capabilities.** Neither container has `CAP_KILL`, `CAP_SYS_PTRACE`, `CAP_DAC_OVERRIDE` or
  `CAP_CHOWN`; `IPC_LOCK` only. `allowPrivilegeEscalation: false`, seccomp `RuntimeDefault` (no
  `unshare`/`setns`). `check_manifests.py` enforces: two containers named grader/runner, different
  non-root uids, no shared pid namespace, no control mount in the runner, `/work` read-only in the grader,
  separate `/tmp` and `/dev/shm`, `job` the only volume writable from both, liveness probe present.
- **Evaluator configuration.** The harness options and the device are the broker's; the agent's
  `check_candidate` passes nothing through (the former `"${ARGS[@]}"` pass-through, which admitted
  `--no-scan`, is gone). The case's GPU count comes from the case table.

## What the candidate can still do (residual, by design)

- Write anything into its own `out/`, spawn directories, `/work` and `/tmp`; forge its own exec exit code
  and output (labelled candidate-reported); return wrong outputs (graded `incorrect`); hang (graded
  `timeout`); kill its own processes and the spawner's children (graded `candidate_error`); hog the GPU
  after a job (runner restart, state tagged `+RUNNER_RESTART`).
- Read the inputs it is given and write into them (detected: `input_mutation`). Through a CUDA IPC handle
  the runner maps the whole allocation that contains an input tensor; the evaluator must therefore keep
  expected outputs and the hash key out of those allocations (hash-then-free, or dedicated allocation
  segments). This is a GPU-phase hardening item (D-I5), not a verdict-channel hole: it would let a
  candidate copy the reference output, which the launch audit (nsys trace of the timed repetitions) would
  still show as not a single kernel.
- Fill the `job` emptyDir up to its size limit (pod eviction = its own run interrupted; the budget ledger
  records the interruption).

## Evidence

- `megabench/tests/test_sandbox.py::GradeProtocolTests` (broker → grader → spawner → remote candidate on
  the CPU; hostile and killer controls; rejected harness options and paths; runner-down recovery),
  `TransportTests`, updated `SpoolProtocolTests`.
- `sandbox/k8s/c1/local/verdict_test.sh`: the two-container layout on Docker with the real image; every
  hostile attempt denied (results: `reports/gb300/verdict-channel-20261119/`).

## Live tests still required (GPU phase, before any real-weight validation)

1. **GPU visibility in the runner.** The grader container requests `nvidia.com/gpu: 4`; the runner relies
   on `NVIDIA_VISIBLE_DEVICES=all` being honoured for an unprivileged container by this cluster's NVIDIA
   runtime (the device plugin's env-var strategy). If not honoured: request the GPUs on the runner and
   use the env on the grader, or ask the cluster operator for the device plugin's configuration. Either
   way both containers must see the same four GPUs.
2. **CUDA IPC across containers.** `cudaIpcOpenMemHandle` from the runner on a grader allocation and vice
   versa, with the pod's shared IPC namespace and separate pid namespaces. If the driver requires a shared
   pid namespace, `shareProcessNamespace: true` is acceptable *only* because the uids differ (magic links
   and signals stay denied); `check_manifests.py` must then be relaxed for that single field with the uid
   check kept.
3. **Admission.** Kyverno must accept the two-container pod with `IPC_LOCK` and the liveness probe.
4. **Kubelet restart path.** `/job/RESTART-RUNNER` → probe failure → container restart → flag cleared.
5. **Interprocess CUDA events and NVTX** across the boundary (timing fidelity unchanged), and the nsys
   trace of the grader's process tree including the remote candidate processes (`nsys_candidate` must
   trace the runner's processes: to be revisited; the audit of record may need `nsys` launched by the
   spawner on the candidate process and the report handed back as a file the grader re-audits).
