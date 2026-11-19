# CPU-only validation of the hardened evaluator and the Kubernetes sandbox pieces (2026-11-19)

Branch `gb300-tp8-e2e-decisions`. Everything here ran on rack 3 tray 11 (aarch64 Grace host, no GPU used,
`CUDA_VISIBLE_DEVICES=""`), with the repository's `.venv` (Python 3.12, torch 2.14.1+cu130, gloo backend).
Hostnames, user names and process ids are redacted in the copied log. No cluster resource was created, no
GPU job ran, no registry push, no Secret.

| file | content |
|---|---|
| `unittest-full.log.txt` | `python -m unittest -v megabench.tests.test_kimi megabench.tests.test_harness megabench.tests.test_sandbox`: 31 tests, OK (the `gpuGetDeviceCount failed with code 100` lines are torch's profiler noting there is no GPU; expected on the CPU path). |
| `conversion-sizing-tp8.json` | `tools/conversion_sizing.py --tp 8 --layers 93 --workers 4 --json`: exact output bytes from the fixture's weight shapes, peak disk with temporaries, host RSS estimate per worker. |
| `check_manifests.json` | `sandbox/k8s/check_manifests.py` on the committed manifests: ok, no problems. |
| `nsys-schema-probe.md` | what the Nsight Systems 2026.3.2 SQLite export contains for an NVTX-only CPU run (the schema `nsys_audit.py` is written against). |
| `nvcc-megatimer-check.txt` | `megatimer.cuh` compiles for sm_103a with and without `-DMEGATIMER`; the `megatimer_records` symbol is present only in the development build. |

## What the tests cover (mapped to the GO's validation list)

- Typed error classification: `StatusTests` (every status has a distinct non-zero exit code; phase-based
  classification; group precedence and culprit ranks).
- Distributed agreement and pre-collective failure handling with gloo: `GlooEndToEndTests` on tiny
  geometry, TP1 and TP2, candidate in its own process per rank: oracle passes; one flipped byte of an FP8
  cache on every rank -> `input_mutation` on both ranks; wrong logits shape on rank 1 only -> `incorrect`,
  `ranks_with_status [1]`, rank 0 `aborted_by_group`, phase `structure`; a raising candidate ->
  `candidate_error` with the traceback preserved in the per-rank stderr tail; `build()` raising ->
  `candidate_error` in phase `build`; a missing output key -> `incorrect`; a hanging candidate -> `timeout`
  at the configured per-phase deadline; an oracle import -> `scan_failed` before any process starts.
- Snapshot integrity and the checksum-collision regression: `KeyedHashTests` (the `+d/-d` pair 65521
  words apart that passed the old linear checksums is detected by the keyed hash; a single-word change is
  detected; different keys give different hashes; small and FP8 tensors take the exact byte-compare path).
- Launcher configuration and rank mapping: `TopologyTests` (single host; two trays with
  `GOALBOX_NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`, global rank = node_rank*4 + local, candidate
  rendezvous on port+1, malformed values rejected) and `SpoolProtocolTests` (broker in k8s mode dispatches an
  8-GPU request to both workers with node ranks 0/1 and the first worker as master, a 2-GPU request to one
  worker only, rejects 6 GPUs; the executors run the command, merge the two outputs, kill a stray process
  the job left behind, and report the worst exit code).
- Sandbox and network-policy manifests: `ManifestTests` (the committed manifests pass; the checker catches
  a read-write agent home in the capture sidecar, an added capability, and a RoleBinding to a subject
  outside the namespace).
- Checkpoint conversion logic and deterministic manifests: `SizingTests` plus the ported
  `planning/bringup/{verify_conversion.py,conversion_manifest.py}` (bit-exact comparison against the HF
  source; deterministic manifest), which need the checkpoint and run inside the conversion Job.
- Trajectory capture, integrity and recovery: `CaptureTests` (append-only copies with sha256 manifest;
  a rewrite of already-copied bytes is kept as a separate file and recorded; `verify_records` passes on a
  clean set and fails on a tampered copy; a restarted sidecar continues from its state file) and
  `CompilerTests` (the fallback compiler produces the summary and redacts tokens, addresses, hostnames and
  home paths).
- Oracle: `test_kimi` (7 tests) still passes with the FP8 MLA cache.

## Not possible without Kubernetes or GPUs (deferred; listed for the readiness report)

NetworkPolicy enforcement from inside the pods (`setup/selftest.sh` with `GOALBOX_SELFTEST_K8S=1`); the RBAC
`can-i` matrix against the live cluster; nsys trace of a real candidate and the device-side-launch and
CUDA-graph controls on real binaries; CUDA IPC event timing; the two-tray NCCL/IMEX path; conversion on the
trays and the cross-tray manifest comparison; the real-weight TP8 oracle; the vLLM baselines.

## Incident during the run (no lasting effect)

The first version of the executor's post-job cleanup killed every process of the executing uid, which inside the
GPU pod's private pid namespace is the intended sweep but on the shared test host terminated the test's own
ssh session (only this session's processes were affected; checked with `ps -u` afterwards). The cleanup is now
session-scoped by default and uid-wide only when the pod sets `GOALBOX_EXECUTOR_KILL_ALL_UID=1`.
