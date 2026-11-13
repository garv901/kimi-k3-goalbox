# Baseline freeze: the validated synthetic evaluator campaign (2026-11-11 to 2026-11-12)

This file records what was validated, with which code and which inputs, so that later changes to the evaluator can be
measured against it. The baseline branch `gb300-tp4-synthetic` and everything under `reports/gb300/` on it are frozen:
nothing on that branch is edited by the planning work. All planning lives on `gb300-evaluator-hardening-plan`, branched
from the baseline commit. Every evaluator weakness listed in `reports/gb300/ISSUES.md` and in
`planning/EVALUATOR-HARDENING-PLAN.md` is a finding for review by the repository owner, not an authorization to change a
benchmark rule.

## Baseline commit and benchmark contract

| Item | Value |
|---|---|
| Baseline commit (branch `gb300-tp4-synthetic`) | `eac5dd281f70813cf7817ba0179f1b2be6f2bca5` (Step 8 evidence), accepted by the operator on 2026-11-13 |
| Planning branch | `gb300-evaluator-hardening-plan`, created from `eac5dd2` |
| Benchmark contract digest (`megabench/harness/runner.py::_contract_digest`, sha256 over the 15 contract files) | `4da60e9b9af9214f8c98c7ffb366e024ed3b2d003ffe7f6ea07fc935410e06c5`; recomputed at `eac5dd2` and identical in the `contract_sha256` field of all 11 committed result records (Steps 1 to 8) |
| Evaluator, setup and sandbox code | unchanged since `7f3820e` (`git diff --stat 7f3820e eac5dd2 -- megabench setup sandbox` is empty); the 11 commits between them add only `reports/`, `PROGRESS.md` and `AGENT_TRAJECTORY.md` |
| Upstream of the harness | MegaBench copy from `kamahori/vibe-megakernel` at `7dfc6d42b0e486ef4b99333015472646bf88e20e` (see `NOTICE`), modified for Kimi K3 |
| Baseline's own CPU unit tests | `megabench.tests.test_kimi`: 7 tests OK at `eac5dd2` (rack 3 tray09, 2026-11-13) |

## Runner revisions used by each GPU step

| Step | Runner clone commit | Case | Candidate (sha256) | Result |
|---|---|---|---|---|
| 1 (2026-11-11) | `7f3820e` | `kimi-k3-dev-l4-tp1-b8-s8192` | oracle wrapper `submission.py` `6f9e8a84…c906` | `non_megakernel`, correctness pass |
| 2 | `7f3820e` | `kimi-k3-dev-l8-tp2-b8-s8192` | same | `non_megakernel`, 2 ranks pass |
| 3 | `31fed7a` | `kimi-k3-syn-l8-tp4-b1-s8192` | same | `non_megakernel`, 4 ranks pass |
| 4 (2026-11-12) | `211ecda` | `kimi-k3-syn-l8-tp4-b8-s8192` | same | `non_megakernel`, 4 ranks pass |
| 5 | `76c3e01` | `kimi-k3-syn-l8-tp4-b64-s8192` | same | `non_megakernel`, 4 ranks pass; accepted at `52f8d8c` |
| 6 | `63f6eb8` | `kimi-k3-dev-l4-tp1-b8-s8192` | `positive_control_oracle.py` `6f9e8a84…c906`; `negctrl_logits.py` `b122e3fb…fcaf`; `negctrl_routing.py` `94c98465…6a84`; `negctrl_immutability.py` `c9375eb8…8952` | positive: `non_megakernel`/pass; the three controls `incorrect` with their own grader message |
| 7 | `c7773c1` | `kimi-k3-syn-l8-tp4-b64-s8192` | `negctrl_immutability.py` `c9375eb8…8952` | `incorrect` on 4 ranks, device-checksum path |
| 8 | `f7a1db0` | `kimi-k3-syn-l8-tp4-b1-s8192` | `negctrl_ranklocal_immutability.py` `411c0b0a…4604` | `incorrect` on 4 ranks, reason names rank 2 only |

Full hashes of every control, driver and analyzer are in `reports/gb300/negative-controls/README.md` and in each step's
`env.txt`; the complete list at `eac5dd2`:

| File (`reports/gb300/negative-controls/`) | sha256 |
|---|---|
| `positive_control_oracle.py` | `6f9e8a848ef76fcf234473950f0beedeccdeb163641ce6ee405234c74161c906` |
| `negctrl_logits.py` | `b122e3fb548a80921a0cf66e092460f39a147a0ea149345cc727b12113c0fcaf` |
| `negctrl_routing.py` | `94c984657d7bd67bb1fc2cc1411be13587b08483504ffa6fbb1c9b777f8a6a84` |
| `negctrl_immutability.py` | `c9375eb89f9d2007bddedcd34a189e8146a4248c9a79e72248207d4572cf8952` |
| `negctrl_ranklocal_immutability.py` | `411c0b0a4dcf3a5f796f01677d7e87179dea42652681e4761c89d819b9f04604` |
| `negctrl_ranklocal_routing.py` (CPU only) | `8da9ef5eaafe4824849e928f43d8ece563bf688ae4c650047c6b6e1cb394c4e7` |
| `negctrl_ranklocal_logits.py` (CPU only) | `12174fa5d3bc41a4b7cfb7067561e5fdaa0485249ae1f96267a9174dc94b1a86` |
| `checksum_collision_check.py` (CPU only) | `55e8c51be56afa47c57375b1280b1e390547d5ee8cfa8c74d052fb6197c6f61c` |
| `step6-tp1-control.sh` / `step7-tp4-control.sh` / `step8-tp4-control.sh` | `a943330e…eb5a` / `9f50f6fd…0d42` / `03d1537a…90f4` |
| `step6_analyze.py` / `step7_analyze.py` | `192722d3…46fe` / `8b5ea043…8535` |
| `cpu_review.py` / `cpu_review_driver.sh` / `tp4_preflight.py` / `tp4_preflight_driver.sh` | `5734af73…e72f` / `55c1a4eb…0585` / `3a676330…0691` / `79ce734c…a55f` |

## Completed test matrix

| Dimension | Covered | Not covered |
|---|---|---|
| Cases | TP1 dev (4 layers, B8), TP2 dev (8 layers, B8), TP4 synthetic B1, B8, B64 (graded suite p1) | TP8 synthetic (suite p2), the three real-weight TP8 cases (suite p0) |
| Candidate = oracle | correctness pass on every rank and trial (12 of 12 rank-trials per TP4 case), launch audit `launch_budget_failed` as expected, NCCL traces identical across ranks | any non-oracle candidate; any candidate near the launch budget |
| Negative controls, single GPU | logits band, routing near-top-k rule, input immutability (host-copy path) | next_token tie rule, cache_length, non-finite outputs, structural (shape, dtype, key) failures |
| Negative controls, four GPUs | immutability on all ranks (device-checksum path, 512 MB tensor), immutability on one rank (attribution by `rank N:` prefix) | rank-local routing or logits corruption on a GPU (CPU runs only); structural failure on one rank (predicted to hang; not run on purpose) |
| CPU analyses | evaluator's own `snapshot`/`check_unchanged` detection on real-size tensors; tiny-geometry gloo runs of seven controls; real B64 case on the CPU; checksum-collision demonstration | none of these were run on a GPU except as noted above |
| Infrastructure | rack 10 tray14 host level as `dc`, GPUs idle and no GPU pods checked before each run, 10 s GPU and host-memory monitors, 30 s GPU-pod polling, no interruption in any step | Slurm-scheduled runs, sandboxed runs (`bwrap` absent), two-tray runs |

Hardware and software of the baseline runs: GB300 trays (4 GPUs of 284208 MiB, Grace aarch64, 956 GB host RAM),
driver 595.71.05 (rack 10), PyTorch 2.14.1+cu130 with bundled NCCL 2.30.7, Triton 3.8.0, Python 3.12.3, the venv at
`/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv`.

## Freeze rules

1. `gb300-tp4-synthetic` stays at `eac5dd2` or receives only new evidence directories; no file under `reports/gb300/` is
   edited retroactively (corrections go to new files or to `ISSUES.md` on a later commit with a note).
2. Any change to a contract file changes the contract digest; results produced under the new digest are not comparable
   with the baseline records (the aggregate command refuses mixed digests). The planning branch therefore records, for
   every proposed change, which contract files it touches.
3. The oracle (`tasks/kimi.py::reference`), the band grader (`compare`), the launch audit and the benchmark methodology
   are not modified by the planning work; proposals are written as proposals.
4. Timings in the evidence are diagnostics of the evaluator, not performance baselines.
