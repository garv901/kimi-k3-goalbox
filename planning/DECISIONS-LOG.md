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
