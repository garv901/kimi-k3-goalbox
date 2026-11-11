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

## 6. Conditions in force

- Conditional GO was given for Step 1 only; Step 2 (TP2 dev case) and the TP4 synthetic cases wait for review of the Step 1 evidence. No PR until then.
- No autonomous agent and no agent-generated code is executed while the sandbox is unavailable.
