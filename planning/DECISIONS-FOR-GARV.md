# Decisions required from the repository owner before optimized submissions are graded

Compiled 2026-11-13 from the GB300 validation campaign (`gb300-tp4-synthetic` at `eac5dd2`) and the Phase B analysis
(`planning/EVALUATOR-HARDENING-PLAN.md`, `planning/TP8-READINESS.md`). Each item names the policy choice, the
evaluator behaviour today, and the options as we see them. None of these has been changed in the code.

## Launch budget (audit: `megabench/harness/benchmark.py::_audit_launches`)

- **D-L1 What is a launch?** Today: every CUDA-side profiler event (compute kernels, NCCL device kernels, `Memcpy`,
  `Memset`, Kineto `nccl:` records). Options: (a) kernels only (compute + NCCL device kernels); (b) kernels plus
  host-issued copies and fills; (c) keep everything. A compliant single-kernel candidate that does one `cudaMemcpyAsync`
  or one `torch.zeros` fails under (b) and (c).
- **D-L2 Report the split and stop counting the `nccl:` records** (pure double count of the NCCL device kernels). Reporting
  change; recommended regardless of D-L1.
- **D-L3 Is any NCCL kernel allowed?** The goal text says the one launch includes all communication; `TASK.md` says NCCL
  kernels are counted. Options: zero NCCL kernels (consistent with the text) or a small allowance.
- **D-L4 CUDA graph replay.** Forbidden by `TASK.md`, detected today only as a hint (`cudagraphlaunch` in CPU event names).
  Decide whether a detected graph launch fails the audit outright, and verify (hypothesis H1) that graph-internal kernels
  are counted individually.
- **D-L5 Audit the timed repetitions.** Today the audit is a separate call after timing; a profiler-aware candidate can
  behave differently under audit. Options: profile every timed repetition (overhead on timing), or take the timing from
  the audited call as well, or accept the gap with manual review.
- **D-L6 Device-side launches** (dynamic parallelism, persistent work queues) inside the one host launch: allowed? Counted
  (hypothesis H2 to test)?
- **D-L7 Manual review.** `review_required` is always true; decide whether the final grade requires a human source and
  trace review (as `goalbox-grade`'s comment says) and what that review checks.
- **D-L8 `max_gpu_launches = 1` on every case** stays as is, or differs per case.

## Input integrity (`megabench/harness/correctness.py::snapshot`)

- **D-I1 Replacement strategy** for tensors above 64 MiB: exact regeneration/re-read (S1), keyed multilinear hash over a
  prime field (S2), cryptographic hash on device (S3), read-only mappings (S4, to be tested), or keep the current linear
  sums and rely on source review. Our recommendation: S1 for synthetic cases and S2 or S3 for checkpoint weights, both
  only meaningful with D-I2.
- **D-I2 Process isolation of the candidate** (strategy S5): is running `build()`/`run()` in a separate process with
  CUDA-IPC-shared inputs in scope? Without it, every in-process check (including the current one) is advisory against a
  hostile candidate.
- **D-I3 Classification of a detected mutation**: today `incorrect` (an assertion). Alternatives: a distinct
  `input_mutation` status, since it is a rule violation rather than a wrong answer.
- **D-I4 Checkpoint weights across trials**: the loaded tensors are shared by all trials; decide whether the digest is
  checked once per trial (as now) or also at the end of timing and audit.

## Distributed failure handling (`megabench/harness/distributed.py`)

- **D-S1 Status taxonomy**: adopt `incorrect | candidate_error | evaluator_error | infrastructure | timeout` with
  per-rank `local_status`/`group_status` (plan P1, P7), replacing the current `incorrect | error` pair in which every
  rank-local error is `incorrect`.
- **D-S2 Structural output checks before the agreement** (plan P2) and guards around `snapshot`, `reference`, timing and
  audit (P3): accepted as evaluator changes?
- **D-S3 Does a `candidate_error` count as a failed case for scoring**, as `incorrect` does? (Recommendation: yes, but
  reported separately.)
- **D-S4 Timeouts**: per-phase deadlines and a shorter collective timeout for grading phases (P4, P5), with values per
  case class (synthetic versus real weights); whether a hanging candidate is a candidate failure.
- **D-S5 Record `max_memory_reserved` and the candidate's stderr** (ISSUES 6, 9) in the per-rank record.

## TP8 and real weights

- **D-T1 Execution path**: (a) a multi-node launcher in the harness for two GB300 trays over MNNVL (evaluator change);
  (b) the x86 8-GPU B300 VM node (needs an x86 image, access, topology check; 9 GiB less memory per GPU); (c) a new
  reduced-depth real-weight TP4 case (for example 48 layers, about 198 GiB per rank) gradable on one tray; (d) wait for a
  DGX B300. Only (a) and (b) grade the existing p0 cases.
- **D-T2 Converted weights**: who runs `convert_checkpoint.sh` (1.64 TB, hours), where the output lives (CephFS, shared
  by other users; or per-tray local RAID copies), and who owns it.
- **D-T3 Case table**: whether the p2 synthetic TP8 cases become gradable on an 8-GPU node, and whether a reduced-depth
  real case is added.
- **D-T4 Validation candidate for TP8**: the oracle as candidate at 93 layers costs about 45 minutes and several hundred
  GB of host memory per tray in the audit phase (estimates); accept that, or validate with fewer layers first.

## Oracle fidelity

- **D-F1 Acceptable independent references**: the vendor's HF modeling code at few layers (exists, B200, 4 layers), a
  production engine (SGLang or vLLM) at full depth, or both; and the thresholds for each row of the comparison matrix.
- **D-F2 Whether the 4-layer B200 verification must be reproduced on GB300** before real-weight grading.

## Isolation and operations

- **D-X1 Sandbox**: ask the administrators to install `bubblewrap` and `socat` on the trays (the AppArmor profile allows
  unprivileged `bwrap`), or build an aarch64 Docker image with the venv and grade through `--docker-image` (already in
  the harness), or both.
- **D-X2 Evidence policy**: any contract-file change changes the contract digest (`4da60e9b…` today). Confirm that the
  baseline evidence stays as the reference for the old digest and that each stage re-establishes its own positive and
  negative controls.
- **D-X3 Evaluator exit code and record fields** (ISSUES 8, 11, 12): keep as is, or make the failure records carry the
  provenance fields and the grader message in its own field.
