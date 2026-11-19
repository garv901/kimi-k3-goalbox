# C2: GPU smoke test of the trusted verdict channel (plan, awaiting GO)

Prepared 2026-11-19. C2 proves on real GPUs what the CPU model (commit 9dd45f7 and later) cannot. It is a
bounded lease, not the goal run: no checkpoint conversion, no real weights, no agent, no reservation that
outlives its deadline. Each step has a time box and a pass criterion; failing steps are recorded and the
lease ends on schedule.

## Resource budget

| item | value |
|---|---|
| cluster, node | `llm-inference-prod-fin03`, rack 10 tray15 (`pod4-gb300-10-tray15-f3`): 4 GB300, no GPU pods at planning time, driver 595.71.05, device plugin v0.19.2, every GPU node tainted `nvidia.com/gpu` |
| GPUs | 4 on one node (the two-tray TP8 path is C3) |
| pods | one GPU worker pod (`goalbox-gpu-node-a`: grader + runner), the C1 proxy, the controller with the agent disabled |
| lease | maximum 4 hours from the GPU pod's `apply` to its deletion, target 2 hours; the operator deletes the pod at T+4h whatever the state |
| GPU budget | 16 GPU-hours maximum |
| storage, weights | the C1 PVCs only; synthetic cases only |
| who applies | the operator (own Rancher identity); the build agent prepares files and reads evidence, one GO per step (gated-step protocol) |
| preconditions | C1b evidence committed; `get_free` shows tray15 with no GPU pods; GO naming tray and start time |

Tray15 is chosen because it carries no GPU pods, is not used by the other sessions (tray13 wideEP, tray14
kvsim), and has a working venv for a host-side comparison. The pod requests all 4 GPUs, so once Running no
other pod shares the node's GPUs.

## Cluster-changing operations (operator only)

1. apply the rendered C2 worker pod (node-a only, `nodeSelector` on tray15; toleration added 2026-11-19);
2. the C1 controller deployment with `GOALBOX_NO_AGENT=1`;
3. at the end: delete the GPU pod, scale the controller to 0; PVCs stay.

Everything else is `kubectl exec` into a container or a broker request through the controller.

## Steps and pass criteria

| # | step | box | pass |
|---|---|---|---|
| 0 | admission: pod Running 2/2, no policy denial in events | 10 min | Running; else record the policy message, fix on the CPU, next GO |
| 1 | GPU visibility: `nvidia-smi -L` and `torch.cuda.device_count()` in both containers | 10 min | the same 4 UUIDs in both; if the runner sees none, move the resource request to the runner and the env to the grader (manifest change, CPU), re-apply |
| 2 | filesystem and process boundary from the runner: the mount list, uid, visible pids, the probe list of `verdict_test.sh` step 2 | 15 min | identical outcomes to the Docker regression: no control, results or records mount, no grader pid, writes outside `out/` and spawn dirs denied |
| 3 | cross-container CUDA IPC: a 1-GPU synthetic grade through the broker (oracle candidate) | 20 min | verdict `ok_provisional`, trusted, `measurement.valid`, `export_audit[*].checked` true with only input segments listed; if `cudaIpcOpenMemHandle` fails across pid namespaces, record the error code and stop (design change needed, see TRUSTED-VERDICT.md live test 2) |
| 4 | export audit on the GPU: inspect the verdict's `export_audit` records | 10 min | every exported segment belongs to the input pool and holds input blocks only; `exported_bytes` within 2x of `input_bytes` |
| 5 | multi-GPU grade: a 2-GPU synthetic case with the oracle | 20 min | `ok_provisional`, both ranks `measurement.valid`, NCCL across the runner's processes works with separate pid namespaces |
| 6 | failure classification: the hostile and killer control candidates, then a candidate that exceeds the deadline and one that allocates past device memory | 30 min | hostile: verdict from the grader, attempts in candidate stderr only; killer: never a pass, grader alive; deadline: `timeout`, no timing; OOM: `candidate_error`, no timing; all four leave the next oracle grade passing |
| 7 | runner restart: a candidate that leaves a GPU process behind | 15 min | `+RUNNER_RESTART`, liveness probe fails, kubelet restarts the runner, flag cleared, next grade passes |
| 8 | nsys visibility: `nsys profile` launched by the spawner around a remote candidate process (runner container) | 20 min | the report contains the candidate's kernels inside the grader's NVTX rep ranges; this step informs blocker B-C2-1 below and does not gate C2 |
| 9 | contamination controls: GPU health counters recorded before and after each grade; a grade run while a second harmless kernel loop occupies one GPU from the grader container | 15 min | the interfered grade is `CONTAMINATED` (exit 9) with no `candidate_timing`; clean grades carry `gpu_health` and process samples |
| 10 | candidate-authored files cannot set the verdict: compare every verdict file's content with the grader log and the broker's view; confirm the runner wrote nothing under `/ctl-results` | 10 min | the verdict record equals the grader's private result; `candidate/<rid>/` holds only stderr copies |

C2 passes when steps 0 to 7, 9 and 10 pass within the lease. Step 8 is informational. Any failure in 2, 3,
6 or 10 is a blocker for GPU validation of real weights.

## Blocker B-C2-1: launch audit of record in remote mode

`nsys_candidate` and the audit of record trace the evaluator's process tree. In the two-container design
the candidate's kernels run in the runner, so a trace taken in the grader has no candidate kernels, and a
trace taken in the runner is written by a process of the candidate's uid. Options, to be decided after step 8:

- (a) the spawner runs `nsys profile` around the candidate process and the grader re-exports and audits
  the report; a report file in the runner is candidate-writable, so the grader must additionally require
  that no candidate process outlives the traced one (leftover check) and that the trace's NVTX rep ranges
  match its own timing records within tolerance;
- (b) a third container of its own uid runs nsys around the candidate, same caveat about the report file;
- (c) kernel counting from the grader side through a driver-level mechanism, if one exists for this driver.

Until decided, the launch-budget verdict in remote mode is advisory only and the record says so
(`launch_audit.status`). DECISIONS.md D-L4/D-L5 are unchanged.

## What C2 does not do

No checkpoint conversion, no real weights, no Claude agent, no two-tray job, no change to numerical
tolerances or launch budgets, no pod kept beyond the lease.
