# Experiment protocol `dsv41-tp4-rack10-v1`: which scaffolding helps megakernel-writing agents? (TP4 revision)

Revision of protocol `scaffold-search-dsv41` (TP8 on 8x B300) for **one 4x GB300 tray (TP4)**. It is a
**new experiment**, not a replication: the benchmark case, seed kernel, platform and the latency scale
differ, so scores, L and AUC are **not comparable** with TP8 results. Everything else (arms, 16 runs,
K = 3, 2 h budget, randomized order, isolation, frozen evaluator, blind scoring, telemetry) is kept.

Fix everything below before the first run. Any change to a file in the manifest makes the affected
runs invalid; rerun them.

## Question
Given the same model, budget and starting kernel, which additions to an autonomous coding agent's
workspace make it produce a faster correct megakernel?

## Task (identical for every run)
- **Case:** MegaBench `dsv41-flash-dspark-step-k5-ctx2k-b8-tp4`: one DeepSeek-V4.1-Flash + DSpark
  speculative-decoding iteration, batch 8, **TP4 on 4x GB300 (sm_103)**, at most one GPU kernel
  launch per rank. Same workload dimensions as the TP8 case (batch 8, DSpark K = 5, context 2k);
  only the tensor-parallel degree changed (`megabench/cases.py`, `gpus=4, tp=4`; the reference
  `megabench/tasks/dsv41.py` is TP-generic).
- **Start:** the seed kernel in `seeds/dsv41_b8_tp4_nopf/` (the TP8 seed `dsv41_b8_nopf` ported to
  WORLD = 4, `PORT.md`; correct by construction only until its first `./check_candidate`: see
  "Calibration items" below).
- **Goal given to the agent:** make the correct single-launch submission as fast as possible,
  scored only by `./check_candidate` (prompt: `prompts/goal_improve.md`, identical for every arm).
- **Budget:** 2 h of wall clock per run. At T-15 min the harness interrupts the agent and resumes it
  for at most 10 min to write its final NOTES.md summary, then stops it.
- **Agent:** Claude Code 2.1.326, headless `claude -p "/goal ..."`, model `claude-opus-5-5`, web
  tools disabled, one fresh session per run.

## Arms (cumulative overlays on the same plain workspace)
| arm | overlays | what it adds |
|---|---|---|
| **A0S** (control) | S | 9 generic CUDA skills (`overlays/S/.claude/skills`) |
| **A1** tools | S, A1 | `./phase_table` (per-phase, per-rank timeline + critical path for persistent kernels), `./profile_candidate_all_ranks` (ncu across the 4 ranks), `./flag_audit` (compiled-in experiment paths vs NOTES) |
| **A2** knowledge | S, A1, A2 | B300 primitive-cost rows for `./primdb` (measured on B300; GB300 has the same SM, sm_103); skills `blackwell-block-scaled-gemm` (with a tested MXFP4 x MXFP8 tcgen05 GEMM example) and `multi-gpu-sync` |
| **A3** parallel engines | S, A1, A3 | `./capture` + `./task_ab` (run one phase on 1 GPU against captured data), phase-contract template, `engine` subagent, integrator workflow |

Arm-specific instructions live only in the overlays' `TASK.d/<arm>.md`. Comparisons:
**A1 vs A0S** (tools), **A2 vs A1** (knowledge), **A3 vs A1** (parallelism).

## Conditions shared by every run
- One 4x GB300 tray (aarch64, linux/arm64 container, `container/Dockerfile`), nothing else on its
  GPUs; **K = 3 runs at a time**; all GPU use through `gpu_lease` (whole-node = 4-GPU leases are
  exclusive; lease sizes 1, 2, 3, 4; requests are served oldest first across sizes and round-robin
  across runs; at most 3 GPUs per run in small leases; while an experiment run's 4-GPU request has
  waited >= 2 min, 1-3 GPU leases older than 10 min are ended; leases with no GPU process and 0% use
  for 5 min are reaped). Lease waits, preemptions and reaps are recorded per run.
- Isolation: each run is its own Unix user that can read only its workspace and the read-only
  trusted benchmark; no GitHub credentials; web blocked by disallowed tools + an allowlist proxy
  (Anthropic only) + a transcript scan afterwards.
- In-run `./check_candidate` compiles outside the lease and times 1 process; scoring uses the
  unmodified 3-process stock check.
- Every passing in-run check is snapshotted by the harness (root).

## Schedule
16 runs: A0S x4, A1 x4, A2 x4, A3 x4, in the random order produced by
`scaffold_search.schedule --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 --prefix r`
(`scripts/run_main.sh` creates and checks it; the TP8 protocol's order is kept deliberately):

| wave | runs |
|---|---|
| 1 | r01 A2, r02 A2, r03 A0S |
| 2 | r04 A1, r05 A3, r06 A2 |
| 3 | r07 A2, r08 A1, r09 A3 |
| 4 | r10 A3, r11 A0S, r12 A3 |
| 5 | r13 A0S, r14 A0S, r15 A1 |
| 6 | r16 A1 |

The orchestrator starts the next run as soon as one ends (90 s stagger). Each run also starts with
one calibration check of the seed on that node (covariate).

## Outcomes
- **Primary:** score = 28.95 ms / L. **28.95 ms is the TP8 normalizer** (a strong PyTorch baseline
  on 8x B300) kept so the formula is unchanged; it is not a TP4 measurement, and TP4 scores are
  comparable only with TP4 scores. The decision rule below uses ratios of medians (score) and
  differences of medians (AUC), which do not depend on this constant. L is the run's best passing
  snapshot re-measured by `rescore.py`: 3 fresh stock `check_candidate` runs (3 processes each) with
  the trusted evaluator, median of the per-run p50. The snapshot must pass the automated review:
  status `ok_provisional`, 1 launch on each of the **4** ranks, all trials pass on fresh evaluator
  seeds, no tensors loaded from files, no reference/oracle imports. A run with no valid snapshot
  scores its own seed calibration.
- **Co-primary:** AUC = time-average over the 2 h of log(28.95 / best passing L so far).
- **Secondary** (reported, not decision-relevant): final-tree score, time to first improvement,
  GPU-busy fraction, checks run, tokens and $, rate-limit stall minutes, tool use (Skill/Task
  calls), score per $, transcript-scan flags.
- **Covariates:** seed calibration L; lease wait share for whole-node requests (`wait_share_node`);
  preemptions; idle reaps.

## Decision rule
- n = 4 per arm.
- **Margin = max(5%, IQR/median of the A0S scores).**
- **Arm X wins over arm Y** if median score(X) > median score(Y) x (1 + margin) **and** median
  AUC(X) > median AUC(Y). `scaffold_search.report` computes the verdicts.
- Every run is reported. A blind human review of each run's best snapshot can only disqualify it.
- Rate-limit stalls count as lost time; no selective reruns. A run is rerun (under a new id, logged
  in the ledger) only for a harness failure: no result.json, or a node/daemon fault in the logs.

## Calibration items (to be filled from the GPU gates before `run_main.sh start`)
These are not known until the TP4 seed has run on the tray (Gates E-I of the test plan). Filling
them changes `rescore.py`/this file and therefore the manifest; do that once, before the campaign.
| item | TP8 value | TP4 value |
|---|---|---|
| seed p50 (`./check_candidate`, 1 launch/rank) | ~21.2 ms on 8x B300 | **to be measured** (first calibration check) |
| `rescore.SEED_L_FALLBACK_MS` (AUC seed when a run has no calibration) | 21.48 | None -> AUC omitted for such a run until set |
| whole-node check duration (in-run / 3-process rescore) | ~2.5 / ~3.5 min | to be measured |
| PyTorch baseline (score normalizer) | 28.95 ms (B3) | 28.95 kept; optional TP4 baseline = owner decision |

## Manifest (sha256)
`scripts/verify_protocol.py` checks this table; `scripts/run_main.sh start` refuses to start on a
mismatch.

| item | value |
|---|---|
| trusted evaluator | vibe-megakernel 4ac7da6a6607885667c31155f7d55e524b11b1bc + gpu_noise_multirank.patch bb4871ccd876220359d5b635f88380955f62c3d61974967b5c33633018cb2f75 |
| seed dsv41_b8_tp4_nopf (MANIFEST sha256) | c7a262e4781c09e132f6e167774de7b33eaec4d3954cebe93b53bbd8234a913f |
| prompts/goal_improve.md | 0ba36e77480fdd1133858fe15cd786e6e6803c176a2ef16c827bec7f68871ff9 |
| arms/A0S.toml | 88cbc5a9ec5e9cf625c71cc9053d1007a7232e8ebe653b6204c9a08d4167e339 |
| arms/A1.toml | b73c1733d41f9b04b3d6eab7c0d46fe2996b0038444874ff4645de562f09537d |
| arms/A2.toml | 3a75f2d15e9b5eab73a9dcd0012e24bb5c2a083b7bb432d5507633426dd4312a |
| arms/A3.toml | 1f83ee4e82f0c826f1d99e2821d878df2bfade3c925a12f0a3e88fa222cf6e0d |
| overlays/S (tree) | cbbf7024ee35c63de79f4f17cacb93d990b8341d7253d452a19a1bb38fed6d70 |
| overlays/A1 (tree) | 186353ea71cde852b36c9c8324ef07b0dd02b00c165ec611256a3615ffb3af8d |
| overlays/A2 (tree) | f4366b5a119e3e62120fcc7d14b52fd208d798e26bc4dc44ece6ac9bf5c5b2fd |
| overlays/A3 (tree) | 34fe92536f52690725afed319bf5be54818c19227561201d3982b7b8e12ae364 |
| scaffold_search/harness_files (tree) | 2f18b46dddf41e7c7d34419ad7837e041b100e5ed3ca87ae1088ddc0dd33ce59 |
| scaffold_search/gpu_lease.py | 7db8ac6991522135892fc2342f093cd69ccc193e48e7f65f3a9bd45a25e81804 |
| Claude Code / model | 2.1.326 / claude-opus-5-5 |

Tree hash = sha256 over the sorted lines `relpath\0sha256(file)\n`. The benchmark commit is
`886eeb3e` (the TP8 protocol's trusted evaluator) plus one commit adding the b8-tp4 case
(`megabench/cases.py`, `megabench/tests/test_dsv41.py`), on branch `dsv41-dspark-batch8-tp4`.
