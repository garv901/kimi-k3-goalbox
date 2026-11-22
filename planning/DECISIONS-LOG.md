# Decisions log (build agent, from 2026-11-18)

Each entry: what `planning/DECISIONS.md` recommends, what we do instead or in addition, why, and
what we observed on the cluster. The six fixed points of DECISIONS.md section 0 are not changed by
any entry here. Host addresses, user names and tokens are not recorded (section 18); trays are
named by their tray number on rack 10 (Kubernetes) or rack 3 (Slurm, Docker).

## Status of the operator's approvals

- 2026-11-18: briefing (section 0) sent; conditional GO for harness preparation and validation
  only. The 48-hour goal run needs a separate GO after the readiness report.
- 2026-11-19: GO for local implementation and CPU-only validation. No cluster resources may be
  created with the operator's Rancher identity, no GPU pods, no conversion jobs, no GPU tests,
  no registry push, until the operator confirms each.
- 2026-11-19 (later): local implementation and CPU validation done (entries L1 to L12;
  `reports/gb300/cpu-hardening-20261119/`, 31 tests OK). Evaluator contract digest
  `703df854b9c4ffb2e39217164b9d953a9097dba912a24ac0945f94d1fc3b07bd` (identical in the working tree
  and inside the built image). Image `goalbox-k3:dev`, local image id
  `sha256:817c1bdd265457c9a910e65ff99b66931bbe62d29f303b1c3c2994487505c10a` (built 2026-11-19T07:30Z
  on rack 3; the registry digest exists only after a push). Next cluster-changing step, not yet
  authorized: apply namespace, RBAC, network policies, storage and proxy, then the RBAC `can-i` matrix.

## L1. Namespace: a dedicated `goalbox-k3` instead of `inference-wideep`

- Default (section 3, briefing): run in the existing namespace that the shared service account
  can use.
- Decision: a new namespace `goalbox-k3`, created and administered only by the operator's own
  (Rancher) identity, with its own ServiceAccounts, Role and RoleBindings. The shared account has
  no rights there. Approved by the operator on 2026-11-19, subject to RBAC and isolation validation.
- Reason: the Claude Code token is a Kubernetes Secret. In `inference-wideep` any user of the
  shared account can create a pod that mounts a Secret of that namespace, so the token would be
  readable by every user of the shared account. In `goalbox-k3` the shared account can neither
  read Secrets nor create pods (checked read-only with `kubectl auth can-i` on the shared
  account: no rights outside its own namespace; and on the operator's identity: can create
  namespaces, roles, rolebindings, serviceaccounts, secrets, pods, networkpolicies,
  ciliumnetworkpolicies, persistentvolumeclaims in a new namespace).
- Observed: `inference-wideep` holds three Secrets that the shared account can mount
  (registry pull secret and two storage credentials), which confirms the exposure pattern.

## L2. GPU execution: two long-lived worker pods with an in-pod executor, not one pod per request

- Default (section 3): the broker starts GPU pods for each `gpu-run` request.
- Decision: one GPU worker pod per tray (`nvidia.com/gpu: 4`, pinned to tray 14 and tray 15),
  started once and kept for the whole run. Each runs `sandbox/lib/executor.py`, which takes jobs
  from a controller-owned spool volume, runs them with the request's time limit, and writes
  status and output back. The broker (`sandbox/lib/broker.py`, mode `k8s`) runs in the
  controller pod, validates the agent's requests, copies them into the controller-owned spool,
  assigns one or both workers, and copies results back to the agent's spool.
- Reasons: (1) on Kubernetes a pod that holds the GPUs is the only reservation available to a
  non-administrator; a pod per request gives the GPUs back between requests and another user's
  pod can take the tray (section 16 asks for the trays to be reserved for the whole run);
  (2) no `pods/exec` stream is needed for the job lifetime, so a controller restart or an API
  hiccup does not kill a running job; (3) the GPU pods need no ServiceAccount token and no
  network except to each other (NCCL) and to the spool volume.
- Known limit: code running in a GPU pod runs as the pod's user (uid 1000) and can write every
  volume the pod mounts read-write (the agent's workspace, the spool's result directory for its
  own job). The executor records start time, end time, exit code and output itself; after a job
  it kills every remaining process in the pod before it reads the job's result. Records and
  weights are never mounted read-write in a GPU pod.

## L3. Workspace is read-write in GPU jobs

- Default (section 3): "workspace read-only" in GPU pods.
- Decision: `/work` and `/home/agent` are mounted read-write in the GPU worker pods, as in the
  bwrap `gpu` mode of the original sandbox.
- Reason: the agent's own tools (`check_candidate`, `profile_candidate`, `nsys_candidate`,
  `phase_timeline`) and its own test scripts write their reports and compiled binaries under
  `/work`; the agent cannot iterate otherwise. What must be protected from GPU jobs is the
  evaluator state (own process, D-I2), the real weights (read-only hostPath), the records (not
  mounted) and the controller spool (only the job's own result directory is writable).

## L4. Launch audit and graded timing from Nsight Systems, in-process profiler for quick checks

- Default (sections 4, 5): count kernels with the in-process profiler; audit every timed rep.
- Decision: the candidate runs in its own process (D-I2). The evaluator process can no longer
  see the candidate's kernels through the in-process profiler, so the official audit and the
  graded GPU time come from Nsight Systems run by the evaluator over the candidate process
  (CUPTI, not forgeable from inside the candidate process): per timed repetition, the kernels,
  memcpy/memset events and CUDA graph launches inside the evaluator's NVTX range for that rep
  (D-L1, D-L2, D-L4, D-L5). Device-side launches are caught by a static scan of the compiled
  artifacts (D-L6). The quick `check_candidate` path keeps the in-process count as an advisory
  number. The vLLM arms are measured with the same nsys method (section 14).
- Observed: nsys 2026.3.2 with SQLite export is available on rack 3 and goes into the image.

## L5. Image: CUDA 13.4 toolkit on a driver that reports CUDA 13.3

- Observed: rack 10 nodes run driver 595.71.05; rack 3 runs 610.57.04 with CUDA 13.4 installed.
- Decision: base image `nvidia/cuda:13.4.1-devel-ubuntu24.04` (arm64), torch 2.14.1 cu130. CUDA
  minor-version compatibility lets 13.4-built binaries run on every 13.x driver as long as device
  code is compiled to SASS (`-gencode arch=compute_103a,code=sm_103a`), which the environment
  text tells the agent to do. PTX-only binaries would need JIT by a newer driver and are not
  supported; this is recorded in `ENVIRONMENT.md`.

## L6. FP8 MLA cache format (section 13)

- Observed in vLLM (`vllm/models/kimi_k3/nvidia/mla.py`, `ops/fused_mla_key_concat_kv_cache.py`
  on main, 2026-11-19): the Kimi K3 MLA layer stores `[kv_c_normed | k_pe]` (512 + 64 = 576
  values per token) as one `float8_e4m3fn` buffer with one per-tensor scale `_k_scale`; the
  specialized decode runtime accepts only `fp8` / `fp8_e4m3` for MLA layers ("The MLA cores read a
  per-tensor FP8 cache"). The Kimi K3 checkpoint carries no kv scales, so the scale is 1.0.
- Decision: the oracle's `l<L>_kv_cache` and `l<L>_pe_cache` become `float8_e4m3fn` with a
  per-layer FP32 scalar `l<L>_kv_scale` (1.0 unless the converted checkpoint provides one);
  reads dequantize to the activation dtype, the new token's cache write is quantized with the
  same scale. The FP32 reference returns the unquantized writes, so the band compares a
  candidate's FP8 write error with the oracle's. `sol.py` counts 1 byte per cached value.

## L7. Headless goal mode

- Observed: Claude Code 2.1.3xx runs `/goal` non-interactively: `claude -p "/goal <condition>"
  --output-format stream-json --verbose` runs the loop to completion in one invocation and emits
  every message (documentation, "Run non-interactively"). No tmux typing is needed.
- Decision: the agent pod runs that command with `--permission-mode auto`, `--model
  claude-opus-5-5`, `--effort high`, settings that deny WebSearch and WebFetch, and the stream
  written to a volume the sidecar copies into the records. The bwrap launcher keeps the tmux path.

## L8. Container registry

- Observed 2026-11-19: the stored `vccr.io` login on rack 3 returns 401; the GitHub token has no
  package write scope.
- Decision: build locally and record the image ID; wait for the operator to restore registry
  access. No alternate credentials, no other registry.

## L9. Where the Claude credential is exposed (conditional-GO item 4)

- Constraint: Claude Code reads its OAuth token from the environment or its config; there is no
  way to hand it a token the agent process cannot read.
- Decision: the Secret `goalbox-claude-token` is mounted as the file `/goalbox/token` (mode 0400,
  owned by the agent uid) in the agent container only; `agent-entry.sh` exports it into the
  environment of the `claude` process. Exposure boundary: the agent (and any subprocess it starts
  in its pod) can read the token. Mitigations: the pod has no API identity and no other volume
  that leaves the pod (`/records`, `/ctl`, hostPaths are absent; `check_manifests.py` enforces
  this); its only network path is the proxy, whose allowlist is api.anthropic.com, the filtered
  PyPI index/wheels and Hugging Face, so the token cannot be exfiltrated to an arbitrary host;
  GPU pods, candidate code and the records volume never see the Secret (not mounted there); the
  transcript compiler redacts `sk-ant-*` strings and the raw records stay off the public
  repository. The token is created by the operator from a file (manifest 80 is by name only) and
  should be revoked when the run ends. Who else can read the Secret: nobody through a Role in the
  namespace (no Role grants `secrets`), and the shared `dc` account has no binding in
  `goalbox-k3`; the operator verifies with `kubectl auth can-i --as=...` before the Secret exists
  (README step 4).

## L10. Code pinned in the image, not in ConfigMaps

- Decision: the harness (`megabench/`), the bring-up conversion tools, the sandbox scripts, the
  agent tools and the goal text are copied into the image read-only under `/goalbox/`; the image
  digest therefore pins the evaluator contract the agent is graded against. No PVC or ConfigMap
  carries evaluator code (a ConfigMap could be edited after the fact; a PVC is writable by the
  agent). Candidate code lives only on the work volume.

## L11. The 48-hour stop is enforced by Kubernetes

- Fixed point 2 (every rule is an automated check) and section 16 (one run, 48 h, 30-min warning,
  15-min commit window). The controller's Role deliberately has no `delete pods`, so it cannot be
  the enforcer. Decision: the agent pod carries `activeDeadlineSeconds: 173100` (48 h + 5 min);
  the controller writes `/work/STOP-REQUESTED` 30 min before 48 h and the agent's `TASK.md` tells
  it to stop experiments and commit within 15 min. The extra five minutes give the capture
  sidecar its final pass; the stdout stream is captured by the controller independently.

## L12. Launch budget test placement (CPU evidence)

- Observed: the hardened evaluator runs every case through the per-rank path with the candidate in
  its own process; on the CPU with gloo the controls behave as specified (test_harness.py:
  oracle passes TP1/TP2; one mutated byte of an FP8 cache -> `input_mutation` on both ranks; a wrong
  logits shape on rank 1 -> `incorrect` attributed to rank 1 with rank 0 `aborted_by_group`; a
  raising candidate -> `candidate_error` with the traceback in the preserved stderr; a missing
  output key -> `incorrect`; a hanging candidate -> `timeout` at the per-phase deadline; an oracle
  import -> `scan_failed` before anything runs). The sum-preserving checksum collision of ISSUES
  item 17 is now detected by the keyed hash (test_harness.KeyedHashTests).
- Not possible without GPUs (deferred to validation step 2 on one tray): the nsys trace of a real
  candidate, the device-side-launch and CUDA-graph controls on real binaries, CUDA IPC event timing.

## L13. Control spool split: requests read-only in the GPU pods (C1 test 6)

- Observed: candidate code runs in the GPU worker pod as the executor's uid, so anything the executor can
  write, the candidate can write. With one read-write control mount the candidate could inject or alter
  requests (time limits, GPU counts, commands for the other tray).
- Decision: the control PVC is mounted twice in the GPU pods: `req/` read-only (only the controller writes
  it; the kernel enforces the read-only bind mount regardless of uid) and `results/` read-write. The broker
  treats results as untrusted content: it only reports them back to the agent that submitted the job, keeps
  its own list of dispatched jobs, and tolerates malformed done records. Residual: a candidate can forge or
  delay its own job's results (it fools only the agent), and can kill the executor (the container restarts).
  Grading of record and the harness code are outside its reach (image, read-only), as are weights (read-only
  hostPath) and records (not mounted). Shown with `sandbox/k8s/c1/local/mount_test.sh`.

## L14. The trajectory stream does not go through pod stdout

- Observed (read-only, 2026-11-19): the shared cluster account can `list pods --all-namespaces` and `get
  pods/log` in every namespace, including a new one. A trajectory on the agent container's stdout would be
  readable by every user of the shared account (section 18: raw records are private).
- Decision: `agent-entry.sh` writes the stream-json to a pod-local emptyDir (`/stream`, read-write for the
  agent, read-only for the capture sidecar, which copies it into the records with the same append-only
  manifest as the session transcripts). Stdout carries two marker lines. The controller no longer needs
  `pods/log`; its Role is pods get/list/watch plus delete on the single pod `goalbox-agent` (L15).
  Residual: the agent can rewrite its own stream file before the sidecar's next pass (30 s); the sidecar
  records rewrites and keeps both copies, and the session JSONL under `~/.claude` is captured independently,
  so a tampered stream is detectable. Pod specs stay readable by the shared account (no secrets in them).

## L15. Run-time budget: controller ledger, not a wall-clock deadline (C1 test 9)

- Section 16: the run ends at 48 h of *run time*; after an infrastructure outage the run resumes. A pod
  `activeDeadlineSeconds` is wall-clock and restarts with the pod, so it cannot implement the pause policy.
- Decision: `sandbox/lib/budget.py` keeps a ledger on the records volume: a 30 s tick counts only while the
  agent pod is Running, both GPU workers are Ready and the broker is alive; otherwise the interval is an
  interruption (recorded with its reason); a gap between ticks longer than 5 min (controller down) is not
  counted either. At 48 h - 30 min the controller writes `/work/STOP-REQUESTED`; at 48 h it deletes the
  agent pod (Role: delete, resourceNames `goalbox-agent`). The pod's `activeDeadlineSeconds` is 72 h as a
  backstop only. Unit test: `BudgetLedgerTests`.

## L16. Proxy allowlist without wildcards (C1 test 4)

- Observed: `*.huggingface.co` in the allowlist would let the agent make the proxy resolve arbitrary names
  (`<data>.huggingface.co`), a DNS side channel, and would admit any future Hugging Face host.
- Decision: explicit host names only (hub, LFS CDNs, Xet storage, PyPI files, Anthropic API), port 443 only,
  CONNECT only; the only plain-HTTP route is `GET http://pypi.goalbox/simple/<name>/` with the blocked-package
  list. Denied names are logged so a missing legitimate host is visible. The tunnel is opaque TLS to the
  allowed host: those hosts are download-only for an anonymous client (no Hugging Face token, PyPI files
  host), and the Anthropic API is the agent's own channel, so no unrestricted relay exists through them.

## L17. Phase C1 runs the proxy from a public image with a ConfigMap (temporary)

- Observed: the project image cannot be pushed (registry login 401), and the namespace validation should not
  wait for it. The cluster already pulls from docker.io.
- Decision: for C1 only, `sandbox/k8s/c1/40-proxy-c1.yaml` runs `netproxy.py` from a ConfigMap on
  `docker.io/library/python:3.12-slim` with the same Service, labels, ServiceAccount, security context and
  policies. The production proxy manifest (image with the code baked in, L10) replaces it once the image is
  in the registry; `check_manifests.py` allows the ConfigMap only for the C1 overlay.

## L18. Who applies the C1 manifests

- Observed (read-only `kubectl auth can-i`, 2026-11-19): the only non-Rancher credential available to the
  build agent is the shared account's ServiceAccount token, which cannot create namespaces, Roles,
  RoleBindings, NetworkPolicies, PVCs or pods in `goalbox-k3` (and must not be given rights there, L1).
- Decision: the build agent prepares and checks the rendered manifests and the validation script
  (`sandbox/k8s/c1/c1-validate.sh`); the operator runs the script with their own identity and hands the
  evidence directory back for redaction and commit. No new credential is created.

## L19. Trusted verdict channel: grader and runner containers (blocker B-C1-1)

- Observed (C1-OPERATOR-SAFETY-REVIEW §7): with the evaluator and the candidate in one container as one
  uid, and the results spool writable by both, a candidate could forge a pass, alter other jobs' results,
  replace evaluator output or kill the evaluator. The broker checked the format of results, not their truth.
- Decision: each GPU worker pod runs a `grader` container (uid 2000; evaluator, request spool read-only,
  results spool read-write, /work read-only) and a `runner` container (uid 1000; candidate processes;
  /work read-write; no control, results or records mount). They share one pod-local emptyDir and the GPUs;
  separate pid namespaces. The harness's candidate process is started remotely (`GOALBOX_SPAWN_DIR`,
  FIFOs, raw CUDA IPC in `megabench/harness/transport.py`); the protocol, the keyed hashes, the timing
  events, the deadlines, the launch budget and the tolerances are unchanged. Verdicts are written only by
  the grader; exec jobs are labelled candidate-reported. Leftover processes trigger a runner-container
  restart through the liveness probe. Analysis of uid, mounts, pid namespaces, capabilities and shared
  writable paths: `sandbox/k8s/TRUSTED-VERDICT.md`. Validation: CPU unit tests and a two-container Docker
  regression; the GPU-specific parts (GPU visibility in the runner, cross-container CUDA IPC, admission,
  kubelet restart, nsys of remote processes) are live tests before any real-weight validation.

## L20. Grade requests: the agent chooses the case and the file, nothing else

- Observed: the agent's `check_candidate` passed arbitrary arguments to the evaluator (`"${ARGS[@]}"`),
  including controls-only flags such as `--no-scan`; the GPU count came from the agent's request.
- Decision: `gpu-run --grade CASE [--submission PATH]` writes a grade request; the broker validates the
  case against the case table (GPU count from it), requires the submission under /work, and attaches the
  harness options from its own environment (`GOALBOX_GRADE_ARGS`, `GOALBOX_GRADE_DEVICE`). The grader
  snapshots the submission's directory before scanning and grading so what was scanned is what ran. The
  `--tiny` evaluator flag exists for CPU self-tests only; its records carry `tiny_self_test: true` and it
  is never in `GOALBOX_GRADE_ARGS`.

## L21. Export audit of CUDA allocations handed to the candidate (D-I1, D-I5)

- Observed: a CUDA IPC handle maps the whole `cudaMalloc` segment that contains a tensor. With the default
  caching allocator, inputs, reference outputs and the hash key can share one segment, so the candidate
  could read what it must not see.
- Decision: inputs are allocated in a dedicated `torch.cuda.MemPool` (`transport.input_pool`); before the
  candidate runs, `transport.audit_exports` walks the allocator snapshot and requires that every segment
  containing an input belongs to that pool and holds only input blocks; otherwise the trial fails as
  `evaluator_error` (never a pass). Expandable segments are refused (not exportable). The record keeps the
  audit (`export_audit`). Frames received from the runner are decoded with a restricted unpickler (no
  globals), capped in size, and every CUDA tensor description is checked (dtype allow-list, shape/nbytes
  consistency, bounds within the mapped allocation); the GPU grader refuses by-value tensors. The static
  argument is complete for the allocator; its GPU confirmation is C2 step 4.

## L22. Measurement-contamination policy

- Decision: a timing figure is reported only when the rank completed every repetition, the last outputs
  graded correct, the inputs were unchanged, the audit ran, and nothing failed; otherwise the figures are
  kept as `invalid_timing` evidence and the group result carries no `candidate_timing`. The grader records
  GPU health counters before and after each grade and samples the compute-process count during it; an
  uncorrected ECC change, a pending page retirement, a required reset, a missing GPU or more processes than
  the job accounts for turn a passing grade into `CONTAMINATED` (exit 9) and tag any other state
  `+CONTAMINATED`. Candidate crashes, OOM, NCCL failures and timeouts already end the timing phase without a
  figure. Tolerances and budgets are untouched.

## L23. GPU scheduling and the audit of record

- Observed (read-only, 2026-11-19): every GPU node of the cluster carries the `nvidia.com/gpu` taint; the
  GPU worker pods had no toleration and would stay Pending. Added `tolerations` for that taint and a
  manifest check. Device plugin v0.19.2 with a config manager; whether `NVIDIA_VISIBLE_DEVICES=all` grants
  an unprivileged second container the pod's GPUs is C2 step 1.
- Observed: the nsys audit of record traces the evaluator's process tree, which no longer contains the
  candidate. Blocker B-C2-1 (C2-GPU-SMOKE-PLAN.md); until resolved the launch-budget verdict in remote mode
  is advisory and labelled so.

## L24. Launch audit for remote candidates: trusted phase windows, detective profiler, grader-launched target

- Observed (CPU experiments, 2026-11-21): Nsight Systems timestamps align to CLOCK_MONOTONIC through the
  session clock base within 10 us; a child started with an emptied environment is invisible to the trace;
  a uid change under the profiler protects the report file but the injection does not deliver events across
  it in the version at hand; in-process event streams are produced inside the traced process.
- Decision: (1) the evaluator records its own CLOCK_MONOTONIC window around every timed repetition and the
  audit (`nsys_audit.audit_windows`) associates kernels with repetitions through those windows, never through
  NVTX ranges from the candidate process; a window without a kernel, kernels from an unexpected process, a
  missing clock base, a missing rank or a trace that stopped early make the audit `audit_invalid`, never a
  pass (L22 applied to the audit). (2) In remote mode the verdict carries `launch_audit.status` and a
  passing grade is a TRANSPORT PASS until an audit of record exists. (3) A spawner-run profiler with the
  candidate under its own uid is a detective control, not the audit of record. (4) The grader-launched
  kernel in a separate CUDA context is the recommended authoritative design; its adoption changes the
  GPU-phase submission contract and is the repository owners' decision. D-L1 to D-L8 unchanged.

## L25. HOLD on B-C2-1; detection, not enforcement, is what the contract requires

- Decision (operator, 2026-11-21): option C and any change to the submission contract are on HOLD pending
  explicit approval from both repository owners. D-L1 to D-L8 unchanged.
- Finding: the contract requires complete, trustworthy detection with a fail-closed rule, not enforcement;
  every profiler mechanism available (process-tree, system-wide CUDA scope, CUPTI) produces its events inside
  the traced process, as the baseline design already did (the audit docstring's contrary claim is corrected).
  Report-file ownership cannot be separated from the candidate's uid without losing event delivery (E3b,
  E3c, E4), and the uid-drop helper is disabled by no_new_privs. The trustworthy parts are the spawner as
  supervisor (pid 1, ptrace-immune, authenticated census) and the completeness rules of L24.
- Plan: C2 step 8 runs G1 to G5 (30 min, 2 GPU-hours) to validate the spawner-run profiler as a detective
  audit; the owners then decide whether its residual is acceptable. `sandbox/k8s/PROFILING-FEASIBILITY.md`.

## L26. C2 profiling experiment prepared; three pass levels; decision request to the owners

- Prepared 2026-11-22 (not authorized to run): `sandbox/k8s/C2-PROFILING-TEST-MATRIX.md` (tests P1 to P10,
  30 minutes, 2 GPU-hours maximum, controls-only harness options with `--trials 0 --no-scan`, every control
  verdict `incorrect`), control candidates `one_kernel_control`, `two_kernel_control`, `graph_control`,
  `untraced_subprocess_control`, and `reports/gb300/DECISION-REQUEST-B-C2-1.md`.
- Pass levels kept distinct: TRANSPORT PASS (channel and isolation on GPUs), PROFILING FEASIBILITY PASS
  (events delivered, aligned, counted; suppression by an extra process detected by presence evidence),
  AUTHORITATIVE AUDIT PASS (not claimable while candidate-controlled instrumentation can drop records
  without reliable detection). Census of processes or contexts is presence evidence, never a launch count.
- Rules unchanged: D-L1 to D-L8, tolerances, grading contract, C2 stop conditions. Open: the evaluator rejects
  `--trials 0`, so the one-/two-kernel, graph, untraced and overhead controls need a controls-only
  `--mechanism-test` flag (never a pass); proposed, not implemented, awaiting the owners' GO.
