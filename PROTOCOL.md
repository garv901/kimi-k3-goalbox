# Experiment protocol: which scaffolding helps megakernel-writing agents?

Fix everything below before the first run. Any change to a file in the manifest makes the affected
runs invalid; rerun them.

## Question
Given the same model, budget and starting kernel, which additions to an autonomous coding agent's
workspace make it produce a faster correct megakernel?

## Task (identical for every run)
- **Case:** MegaBench `dsv41-flash-dspark-step-k5-ctx2k-b8`: one DeepSeek-V4.1-Flash + DSpark
  speculative-decoding iteration, batch 8, TP8 on 8x B300 (sm_103), at most one GPU kernel launch
  per rank.
- **Start:** the seed kernel in `seeds/dsv41_b8_nopf/` (a correct single-launch megakernel,
  ~21.2 ms p50 on 8x B300).
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
| **A1** tools | S, A1 | `./phase_table` (per-phase, per-rank timeline + critical path for persistent kernels), `./profile_candidate_all_ranks` (ncu across 8 ranks), `./flag_audit` (compiled-in experiment paths vs NOTES) |
| **A2** knowledge | S, A1, A2 | B300 primitive-cost rows for `./primdb`; skills `blackwell-block-scaled-gemm` (with a tested MXFP4 x MXFP8 tcgen05 GEMM example) and `multi-gpu-sync` |
| **A3** parallel engines | S, A1, A3 | `./capture` + `./task_ab` (run one phase on 1 GPU against captured data), phase-contract template, `engine` subagent, integrator workflow |

Arm-specific instructions live only in the overlays' `TASK.d/<arm>.md`. Comparisons:
**A1 vs A0S** (tools), **A2 vs A1** (knowledge), **A3 vs A1** (parallelism).

## Conditions shared by every run
- One 8x B300 node; **K = 3 runs at a time**; all GPU use through `gpu_lease` (8-GPU leases are
  exclusive; requests are served oldest first across sizes and round-robin across runs; at most 3
  concurrent 1-GPU leases per run; while an experiment run's 8-GPU request has waited >= 2 min,
  1-3 GPU leases older than 10 min are ended; leases with no GPU process and 0% use for 5 min are
  reaped). Lease waits, preemptions and reaps are recorded per run.
- Isolation: each run is its own Unix user that can read only its workspace and the read-only
  trusted benchmark; no GitHub credentials; web blocked by disallowed tools + an allowlist proxy
  (Anthropic only) + a transcript scan afterwards.
- In-run `./check_candidate` compiles outside the lease and times 1 process; scoring uses the
  unmodified 3-process stock check.
- Every passing in-run check is snapshotted by the harness (root).

## Schedule
16 runs: A0S x4, A1 x4, A2 x4, A3 x4, in the random order produced by
`scaffold_search.schedule --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 --prefix r`
(`scripts/run_main.sh` creates and checks it):

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
- **Primary:** score = 28.95 ms / L (28.95 ms = a strong PyTorch baseline: library kernels +
  torch.compile + CUDA graph + PDL). L is the run's best passing snapshot re-measured by
  `rescore.py`: 3 fresh stock `check_candidate` runs (3 processes each) with the trusted evaluator,
  median of the per-run p50. The snapshot must pass the automated review: status
  `ok_provisional`, 1 launch on each of the 8 ranks, all trials pass on fresh evaluator seeds, no
  tensors loaded from files, no reference/oracle imports. A run with no valid snapshot scores its
  own seed calibration.
- **Co-primary:** AUC = time-average over the 2 h of log(28.95 / best passing L so far).
- **Secondary** (reported, not decision-relevant): final-tree score, time to first improvement,
  GPU-busy fraction, checks run, tokens and $, rate-limit stall minutes, tool use (Skill/Task
  calls), score per $, transcript-scan flags.
- **Covariates:** seed calibration L; lease wait share for 8-GPU requests; preemptions; idle reaps.

## Decision rule
- n = 4 per arm.
- **Margin = max(5%, IQR/median of the A0S scores).**
- **Arm X wins over arm Y** if median score(X) > median score(Y) x (1 + margin) **and** median
  AUC(X) > median AUC(Y). `scaffold_search.report` computes the verdicts.
- Every run is reported. A blind human review of each run's best snapshot can only disqualify it.
- Rate-limit stalls count as lost time; no selective reruns. A run is rerun (under a new id, logged
  in the ledger) only for a harness failure: no result.json, or a node/daemon fault in the logs.

## Manifest (sha256)
`scripts/verify_protocol.py` checks this table; `scripts/run_main.sh start` refuses to start on a
mismatch.

| item | value |
|---|---|
| trusted evaluator | vibe-megakernel 886eeb3e28db071f6c43b31fd0b6a9c26dbeaf27 + gpu_noise_multirank.patch bb4871ccd876220359d5b635f88380955f62c3d61974967b5c33633018cb2f75 |
| seed dsv41_b8_nopf (MANIFEST sha256) | b6e0b89119e2d9be2cd3247cb0d915e839dd7511029c0cdcbeca8a4a811392ee |
| prompts/goal_improve.md | ab730d64a65c38237ca29a4d84c924099142c513129b31b9f6f7994b101ae0c3 |
| arms/A0S.toml | 106e9e674e4da79d98ec4454879958a7078f8044d68e2f50aa329fe9e44f52e7 |
| arms/A1.toml | 343799005ed235196653a26472a638532adb16975ab8fa11e609abd03280bca0 |
| arms/A2.toml | 08238e1cb1271a9a9984d2b3936d3cb0c63e23718471699a14d927a8357f8e85 |
| arms/A3.toml | 6ee23b0614df5cd6b08352f3b921873876ea12980a0c8ccf0eee5857bbf24964 |
| overlays/S (tree) | cbbf7024ee35c63de79f4f17cacb93d990b8341d7253d452a19a1bb38fed6d70 |
| overlays/A1 (tree) | 225808c786438ccb76cadbfa1de69984ab7e1d2467490c7df668ed7311a660bf |
| overlays/A2 (tree) | f4366b5a119e3e62120fcc7d14b52fd208d798e26bc4dc44ece6ac9bf5c5b2fd |
| overlays/A3 (tree) | dfe8f95a46a895ebf9123fc4dbd05bf4f8f135cf6794d60401cb6666c74d9144 |
| scaffold_search/harness_files (tree) | 99154c3af56e751b9c0c1c767da3c8f2b18a5a512068d192ae6c2e05db3baef0 |
| scaffold_search/gpu_lease.py | cb6bf2beaa1b18677dcafb88adf34679af764a21d25ab9612db46e110592e610 |
| Claude Code / model | 2.1.326 / claude-opus-5-5 |

Tree hash = sha256 over the sorted lines `relpath\0sha256(file)\n`.
