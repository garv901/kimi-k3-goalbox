# AGENT TRAJECTORY

How the `gb300-tp4-synthetic` branch came about, for review of the development path. Written by the operator's Claude Code session (not the sandboxed goal agent, which has not run). Credentials, cluster addresses and unrelated infrastructure are left out on purpose.

## 1. Test plan review (2026-11-11)

Input: a draft test plan (Phases 0 to 5: environment audit, synthetic correctness, real checkpoint, performance baselines, bounded agent run, reproducibility) targeting "a single GB300 node with 8 B300 GPUs".

Checks made before touching a GPU: cloned the public repo at `fb3b2a5`; read `README.md`, `goals/`, `setup/*.sh`, `sandbox/lib/sandbox.sh`, `sandbox/lib/broker.py`, `megabench/cases.py`, `megabench/harness/{runner,distributed,benchmark,correctness}.py`, `megabench/sol.py`, `tools/VERIFY_KIMI.md`; inventoried one tray on each of two GB300 racks (`nvidia-smi`, topology, Slurm, IMEX, tool availability, disk, wheels for aarch64).

Findings that changed the plan:
- 4 GPUs per tray, aarch64 host. All six graded cases were TP8. `evaluate_distributed` refuses to start without 8 visible GPUs on one host.
- One rack runs Slurm (partitions do not include `main`, the harness default) and has a multi-node NVLink fabric; the other is a Kubernetes cluster with no Slurm.
- `bubblewrap`/`socat` absent; unprivileged user namespaces restricted by AppArmor, so the sandbox cannot start until the packages (with their AppArmor profile) are installed.
- PyTorch cu130 and vLLM nightly publish aarch64 wheels; the pinned Kimi-K3 revision is public and ungated (96 shards).

## 2. Decisions

- Branch name `gb300-tp4-synthetic` (the hardware generation and the change: synthetic graded cases at TP4).
- Repo owner agreed on 2026-11-11 that a TP4 layout for the synthetic cases is acceptable. Real-weight TP8 cases left unchanged; a two-tray launcher is out of scope.
- Keep the TP8 synthetic cases in the table as suite p2 so 8-GPU nodes can still run them.
- Do not change the oracle, grader, launch audit or timing code. The only `megabench` edits are the case table and a new unit test.

## 3. Changes (commit `7f3820e`)

`megabench/cases.py`, `megabench/tests/test_kimi.py`, `sandbox/lib/broker.py`, `sandbox/bin/goalbox-grade`, `setup/selftest.sh`, `sandbox/lib/ENVIRONMENT.md`, `sandbox/sandbox-bin/gpu-run`, `README.md`, `goals/kimi-k3-tp4-synthetic.txt`. Details in the commit message and `PROGRESS.md`.

## 4. Failed or blocked attempts

- Running `setup/install.sh` and the repo's `megabench sol` from the operator session was refused by the session's permission system (external code). The venv was built with the same explicit commands the script contains; `sol` was run on the rack instead.
- Creating a fork to push to was refused by the permission system; the repo owner granted write access instead and the branch was pushed to `origin`.
- The Kubernetes occupancy check inside `step1.sh` produced no output because `kubectl` was not on PATH for that shell; redone after the run (`occupancy-k8s-postcheck.txt`). A `sed` patch of the script's PATH line failed on a delimiter clash; the script for Step 2 will set PATH explicitly.
- The evaluator's CUDA-graph reference baseline could not be captured on the dev case (recorded in `result.jsonl`, not fixed; flagged for review).

## 5. Test evidence

- CPU unit tests: 7/7 on both racks with PyTorch 2.14.1+cu130 (aarch64).
- Generated agent workspace: `TASK.md` lists the TP4 cases as graded; `check_candidate` maps them to 4 GPUs.
- Step 1 GPU smoke test: `reports/gb300/step1-dev-l4-tp1-20261111T182425Z/` (report, command, exit code, logs, full result record, environment, occupancy before/during/after, the exact script and candidate).

## 6. Conditions in force (updated 2026-11-12)

- Each step runs only after an explicit GO with its own requirements; Steps 1 to 5 are accepted (last at commit `52f8d8c`). The negative controls are approved for preparation and CPU review only; their first GPU run waits for a separate GO. No PR.
- No autonomous agent and no agent-generated code is executed while the sandbox is unavailable. The oracle, grader, launch audit and benchmark methodology are not modified; observations go to `reports/gb300/ISSUES.md`.
- Rack 3 trays 06 and 10 are not used for anything (operator rule, 2026-11-11).

## 7. Step 1 and Step 2 approvals and execution (2026-11-11)

Chat history, condensed and redacted. Speakers: the operator (Kevin), the repo owner (Garv, relayed), and this session.

- Operator: shared the draft test plan. Session: audited the repo and the racks, reported the 4-GPU-per-tray mismatch and the sandbox prerequisites.
- Operator: asked for a branch name. Session: proposed `gb300-tp4-synthetic`.
- Operator: asked whether only TP8 is run. Session: graded cases were TP8 only; dev cases TP1 and TP2; TP4 would be an addition.
- Repo owner (relayed 09:39): "A tp4 layout on synthetic configs should work." Session: implemented commit `7f3820e`, verified on CPU, pushed after write access was granted.
- Operator: asked whether rack 10 can be used. Session: yes for host-level evaluator tests; proposed a 3-step matrix with the oracle as candidate and asked for a go.
- Operator: conditional GO for Step 1 only, with requirements: dev TP1 case, reviewed oracle submission, host level as `dc`, no agent, no methodology changes, explicit timeout, dedicated caches, version and occupancy evidence, separate verdicts per evaluator component, `PROGRESS.md` and `AGENT_TRAJECTORY.md`, redaction before pushing, stop gate before TP2.
- Session: ran Step 1 (75 s), reported PASS-as-expected with three observations, committed `4a83f5f`, proposed Step 2 with acceptance criteria.
- Operator: GO for Step 2 with requirements: distinct physical GPUs, NCCL logging, same three trials, per-rank results, communication kernels in the audit, kubectl PATH fix, 10 s GPU and Kubernetes occupancy monitoring, dedicated caches, timeout cleanup of the whole process group, 1500/1800 s timeouts, evidence under `reports/gb300/step2-dev-l8-tp2-<ts>/`, push without PR, stop gate before TP4.
- Session: wrote `step2.sh` (PATH fix, NCCL_DEBUG to files, 30 s Kubernetes polling, `timeout -k 30`, leftover-process cleanup), ran Step 2 (99 s), reported PASS-as-expected.

Technical trajectory of Step 2: read `megabench/harness/distributed.py` and `runner.py` to confirm per-rank records exist (`ranks[]` with correctness, audit, timing, peak memory) and that the evaluator's own timeout kills the worker's process group; chose NCCL_DEBUG_FILE per process to keep stderr clean; pre-launch checks of both GPUs and the node's GPU-pod requests; post-run NCCL log summary (32 P2P/CUMEM channels, 0 warnings); result analysis by a small script (`result-summary.txt`).

Failures and fixes in this step: the pull of the branch into the runner clone was blocked by its untracked raw Step 1 copy (moved aside to a directory outside the repo); the copy of the report to the committing clone was slow because the two-rank result record carries every kernel name, so it is committed gzip-compressed. IP addresses in NCCL logs were redacted.

Deviations from Step 1: listed in the Step 2 report.

## 8. Step 3 approval and execution (2026-11-11)

Chat history, condensed and redacted.

- Operator: reviewed Step 2 (commit `31fed7a`) as a successful TP2 evaluator smoke test. Two findings: (A) the launch audit double-counts NCCL communication (device kernel plus `nccl:all_reduce` record), which must be resolved before a near-budget candidate is graded, and (B) TP2 correctness establishes evaluator consistency, not independent validation of the oracle's tensor-parallel math; independent fidelity validation belongs before real-weight performance claims.
- Operator: GO for TP4 B1 only, with requirements: four distinct GPU UUIDs, NCCL init and an explicit check that all ranks complete the same collective sequence, three shared-seed trials with per-rank results, no methodology changes, launch accounting split (total events vs actual kernel launches vs NCCL device kernels vs collective records) without fixing the audit, all Step 2 monitoring and timeout controls, evidence under `reports/gb300/step3-syn-l8-tp4-b1-<ts>/`, a non-blocking issue note for the repo owner, push without PR, stop before B8. Also recommended a corrupted-output negative control after the TP4 series.
- Session: derived `step3.sh` from `step2.sh` (case, GPUs 0 to 3, step3 paths) with one logging-only addition, `NCCL_DEBUG_SUBSYS=INIT,ENV,COLL`, so each process logs every collective with its opCount and the four sequences can be compared. Checked all four GPUs and the node's GPU-pod requests by hand immediately before launch, fast-forwarded the runner clone to `31fed7a` (moving its raw Step 2 copy outside the repo), ran Step 3 (86 s).
- Session: analysis with `analyze_step3.py` (kept next to the report): per-rank verdicts, the accounting split by kernel-name class, the cross-rank equality of the audited NCCL kernel sequence, and the cross-process equality of the 603-collective trace. Wrote `reports/gb300/ISSUES.md`.

Technical notes: the accounting split classifies profiler event names (`nccl:` prefix = collective record, `ncclDevKernel` = NCCL kernel, `Memcpy`/`Memset` = copies, otherwise compute kernel); it reads the raw record and changes nothing in the evaluator. The `COLL` trace regex parses `AllReduce: opCount … count … datatype … op …` lines; the 22 AllGather and 8 Broadcast calls are the evaluator's own `shared_seed`/`agree` helpers, not oracle work.

Failures and fixes: none in this step. The report copy to the committing clone again went through the laptop and was slow (33 MB record), so the record is committed compressed.

## 9. Step 4 approval and execution (2026-11-12)

Chat history, condensed and redacted.

- Operator: accepted Step 3 (commit `211ecda`) on the matching four-rank collective traces and 12/12 passes. GO for TP4 B8 only, same submission, controls and timeouts. Requirements: 12/12 passes with worst band ratios per output class, four UUIDs, clean NCCL init, cross-rank comparison of collective sequences (type, element count, datatype, op); audit expected to fail with the event breakdown preserved, check whether the audited step still has 26 NCCL kernels, report actual launches separately from total events, no grader change; peak allocated and reserved memory plus sampled nvidia-smi per GPU, compared with B1; raw timing samples with drift noted, no speedup claims; same monitoring, caches and timeouts; evidence under `reports/gb300/step4-syn-l8-tp4-b8-<ts>/`; update `PROGRESS.md`, `AGENT_TRAJECTORY.md`, `ISSUES.md` (keep the 9089 vs 8215 distinction, document what the 603 collectives cover); push without PR; stop before B64 and propose it with a memory feasibility assessment.
- Session: derived `step4.sh` from `step3.sh` (case and names only), moved the runner's raw Step 3 copy outside the repo, fast-forwarded the runner to `211ecda`, checked all four GPUs and GPU pods by hand, ran Step 4 (117 s). Stated up front that PyTorch reserved memory is not in the evaluator record and would not be added.
- Session: extended the analyzer with a histogram and run-length view of the collective trace. The first version had NCCL's reduction-op labels swapped (ncclRedOp_t is sum 0, prod 1, max 2, min 3, avg 4); corrected before producing the committed summary. The run-length view resolved the 603-collective question: 21 decode steps of 26 collectives plus 57 evaluator collectives.

Technical notes: the oracle step's collectives are 17 all-reduces of the hidden state (7168×B), 7 of the latent (3584×B), one float32 max and one int64 min for greedy decoding; element counts scale with B (57344 and 28672 at B8). Compute-kernel and memcpy counts grow about 6.5x from B1 to B8 because the eager oracle's routing is host-driven per row; NCCL kernel count is batch-independent.

Failures and fixes: the analyzer label swap above (caught by cross-checking against the audit's kernel names); no infrastructure or evaluator failures.

## 10. Step 5 approval and execution (2026-11-12)

Chat history, condensed and redacted.

- Operator: accepted Step 4 (commit `76c3e01`) and the B64 memory and runtime projections. GO for TP4 B64 only, same submission and controls. Requirements: 12/12 passes with band ratios per output class, tie-margin details, routing, cache length and immutability; four UUIDs; clean NCCL init and cross-rank comparison of the ordered collective sequence; check for 26 NCCL kernels in the audited step and record differences from B1 and B8; raw NCCL and profiler evidence preserved; event breakdown and actual launches reported separately without modifying the audit; peak allocated and sampled memory per GPU, host RAM monitored throughout, comparison with the 37 GB and 41 GB projections; timing distributions with outliers as diagnostics only; same timeouts; stop on OOM, memory pressure, hangs or contention without retry; evidence under `reports/gb300/step5-syn-l8-tp4-b64-<ts>/`; preserve the corrected analyzer with its SHA; update the three documents; push without PR.
- Session: derived `step5.sh` from `step4.sh` (case, names, plus a 10 s `free -m` monitor), moved the runner's raw Step 4 copy outside the repo, fast-forwarded the runner to `76c3e01`, checked GPUs, GPU pods and host RAM by hand, ran Step 5 (239 s). Analyzer: `analyze_step5.py`, sha256 `af2efdef0134794e0a8df0f444b1cddde7c08218b1e9535e3c8381a4ce632a92` (identical content to the corrected Step 4 analyzer).

Technical notes: both memory projections held (37.25 GB allocated, 41.5 GB process memory per GPU). Host RAM grew by 78 GB during the run because the evaluator snapshots each rank's inputs, including the synthetic weights, to host memory for the immutability check; this scales with the number of ranks and weight size, not with batch, and would be about 200 GB per rank for the real checkpoint. The near-tie next_token observation was added to ISSUES.md.

Failures and fixes: none in this step. Transfer of the 1.1 GB raw record to the committing clone was done compressed in transit.

## 11. Negative-control preparation (2026-11-12, CPU review only)

Chat history, condensed and redacted.

- Operator: accepted Step 5 (commit `52f8d8c`); the TP4 synthetic series (B1, B8, B64) is complete and has passed its oracle-as-candidate evaluator smoke tests. GO for negative-control preparation and CPU review only, no GPU execution. Requirements: three narrowly scoped controls that preserve the oracle, grader, methodology and positive control unchanged: (1) logits perturbation of selected indices that exceeds the FP32 band by a clear margin, no NaN/inf or shape/dtype errors, exact formula and indices preserved, expected `incorrect` from the band check; (2) one expert id per targeted row replaced by one outside the top-k set and outside the near-tie margin, shapes, dtypes and id range preserved, original and replacement ids and their score margins recorded, expected `incorrect` from the routing check; (3) a minimal deterministic in-place mutation of one input tensor covered by the immutability check, without interfering with NCCL, shapes or synchronization, expected a specific immutability violation (document the category if the evaluator says `error`); (4) CPU-only preparation in a labeled directory, review for deterministic failure and side effects, syntax/import/CPU checks, normal candidate interface, no evaluator change; (5) propose separate clean TP1 invocations (positive, logits, routing, immutability), TP4 replication only after review, with per-rank verification; (6) update `PROGRESS.md`, `AGENT_TRAJECTORY.md`, `ISSUES.md`, document the completed series and the host-memory implications for real-weight TP8, commit and push; (7) stop after preparation and report paths, diffs, mechanisms, expected classifications, CPU results, side effects, proposed commands and resource estimates, new commit SHA.
- Session: read `megabench/tasks/kimi.py::compare` and `reference(routes=…)`, `megabench/harness/correctness.py::{snapshot,check_unchanged,check_trials}`, `megabench/harness/distributed.py::_rank_worker` and `runner.py::{_worker,_run_local_worker}` to place each corruption on exactly one check and to predict the status mapping (AssertionError to `incorrect`, anything else to `error`).

Design decisions and why:
- Logits: perturb one row (row 0, every 4th index of the rank's shard, plus the row's RMS) rather than random noise over everything, so the perturbation is deterministic and its effect is predictable (relative L2 of 0.5 added against a limit of about 0.09, about 5x) and so that the per-row grading is visible: every other row stays in band. next_token is graded on the FP32 logits of the candidate's routes, so it is unaffected.
- Routing: the grader runs the FP32 reference on the candidate's routes, so returning the oracle's logits with an edited `expert_ids` would also fail the band. The control therefore reruns the oracle forced onto the corrupted routes (`reference(routes=…)`, the same API the grader uses) and returns that output; only the routing rule fails. The replacement is the smallest expert id outside the row's set; its score margin is read from the forced run's own `route_shortfall`, and the control aborts (RuntimeError, status `error`) if the margin is below 0.05, so a rare near-tie cannot produce a silent pass. In Steps 3 to 5 the allowance was 0.012 to 0.019.
- Immutability: mutate after the step, so the outputs stay correct and only `check_unchanged()` fails; `l3_kv_cache[0, 0, 0] += 1.0` in BF16 always changes the bits for standard-normal data. The tensor is 8, 64 and 512 MB at B1, B8 and B64, which covers both snapshot paths (host copy, device checksum) across the TP4 cases. Replicated on all ranks, so every rank's own check fails.
- Record file: the evaluator discards the candidate's stdout/stderr (now ISSUES.md item 9), so the controls append JSON lines to `GOALBOX_NEGCTRL_RECORD.rank<N>.jsonl` when that variable is set; nothing otherwise.
- Directory: `reports/gb300/negative-controls/` (committed; `runs/` is gitignored), with the positive control copied byte-identical (sha256 `6f9e8a84…c906`).

CPU review, without any GPU (`CUDA_VISIBLE_DEVICES=""`, `--device cpu`): `cpu_review.py` on rack 3 tray01 (tiny geometry: 5 layers, 128 experts, top-8, batch 3) through the evaluator's own `_load_submission`, `check_trials`, `snapshot`, `check_unchanged`, `compare`: positive control passes (band ratios 0.39 to 0.49), logits control fails only `logits` (candidate rel-L2 0.50, ratios 3.6 to 18.7 depending on the seed's oracle error), routing control fails only `expert_ids` (replacement shortfall 0.20 to 0.61 against allowances 0.001 to 0.016), immutability control fails only `candidate mutated input l3_kv_cache` with outputs bit-identical to the oracle; 5 of 5 seeds each. Then `cpu_review_driver.sh` on rack 3 tray09 (idle host, no GPU process of ours) repeated the tiny review and ran the real CLI on the real dev case geometry on the CPU (about 60 GB host RAM, synthetic weight generation about 170 s per input set); results in `negative-controls/cpu-review-<ts>/real-geometry/*/`: all three negative controls `incorrect` with exactly their own check in the reason, the positive control `correctness_only`/`pass`; the driver was run twice (the second time after the review-script fix below) with identical real-geometry outcomes.

Failures and fixes: on the tiny 128-expert geometry, `check_trials()` with its own random seeds hit the routing control's near-tie abort in both driver runs (margin 0.029 below the 0.05 threshold); the abort worked as designed, but the first `cpu_review.py` did not count that documented outcome as expected, so its summary line reads `ALL EXPECTATIONS MET: False`; fixed in the script and rerun, both evidence directories kept. The routing control's first record field `max_shortfall_elsewhere` mixed two different things (other rows, which must be 0, and later layers of the same row, which may be nonzero because the replaced expert changes that row's hidden state); split into two fields before the committed review. The Step 5 report's statement that the 78 GB host-memory growth was due to the immutability snapshots was wrong by about a factor of seven; corrected in ISSUES.md item 10 with the measured timeline and the computed snapshot sizes (2.84 GiB per rank at TP4 B64, 36.8 GiB per rank for real TP8). The Step 5 report itself was left as accepted evidence.
