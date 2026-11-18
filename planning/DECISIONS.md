# Decisions by the repository owner (2026-11-18)

These decisions answer `planning/DECISIONS-FOR-GARV.md`. They replace the open options in that file. Build the harness to match them. Where a decision changes evaluator code, the contract digest changes; that is expected (see "Evidence").

## 0. To the Claude agent that reads this file: do this first

Before you run any command that changes something, send the user the briefing below. Write it in Simplified Technical English (ASD-STE100) and keep it as short as it is here.

- You may first run read-only commands (for example `kubectl auth can-i ...`, `kubectl get ...`, `df -h`, `cat /sys/kernel/security/lsm`) to find the real values. Put the values you find in place of the text in angle brackets.
- If a fact on the cluster differs from this file (for example, the account cannot create pods, so the sandbox is the Landlock fallback), change the briefing to the plan you will really follow.
- If you cannot find a value, say so in the briefing and ask for it.
- Then wait for the user to agree before you build or run anything.

Briefing to send (template):

> We will prepare the harness for the Kimi K3 goal run on your cluster. Read this procedure before we start.
>
> **Run.** One Claude goal agent writes a megakernel for one decode step of Kimi K3. The run uses the real weights, 93 layers and TP8 on two GB300 trays. We compare the agent with vLLM PR #61867 on the same trays. No person examines the results. Automatic checks make sure that the agent obeys the rules.
>
> **Sandbox.** The agent operates in a Kubernetes pod on rack 10. The pod has no GPU and no root access. A proxy lets the agent connect only to the Claude API, Hugging Face and the package index. GPU work operates in different pods without external network access. We will not send requests for permissions to the administrators. If your account cannot make pods, we will use Landlock on rack 3.
>
> **Trajectories.** The harness keeps the transcripts of the agent and of each subagent. It also keeps each GPU job, each check result and a code snapshot at each check. These records go to <storage path>. The agent cannot get access to that location. The full records stay on the cluster, not in the public repository.
>
> **Time limit.** The run stops at the first of these two conditions:
>
> - The agent completes the goal.
> - The run time is 48 hours.
>
> Time during an infrastructure failure does not count.
>
> **Work before the run.**
>
> 1. Make the sandbox and do a test of it.
> 2. Convert the checkpoint (1.64 TB) to the local RAID of the two trays.
> 3. Add a launcher for two trays to the evaluator.
> 4. Change the evaluator rules and the KV cache of the oracle to FP8.
> 5. Measure the two vLLM paths at TP8.
> 6. Do the validation steps with the oracle as the candidate.
>
> **Data that we must have.** Supply these items:
>
> - The two trays
> - The Kubernetes namespace
> - The storage path for the records
> - The file with the Claude Code token.
>
> Do you agree with this procedure?

"The agent completes the goal" means the stop condition of section 16: five consecutive optimization attempts each improved latency by less than 2 percent on every graded case. There is no early stop at 1.4x the HBM floor.

## Context for a reader who did not see the earlier work

- `main`: the original harness (sandbox, broker, MegaBench oracle and band grader, vLLM baseline script, goal prompts).
- `gb300-tp4-synthetic` (frozen at `eac5dd2`): the operator's validation of the evaluator on GB300 trays with the oracle as the candidate and with negative controls. Evidence under `reports/gb300/`, open issues in `reports/gb300/ISSUES.md`.
- `gb300-evaluator-hardening-plan` (`bcb672f`, the parent of this branch): CPU-only analysis. `planning/EVALUATOR-HARDENING-PLAN.md` (findings F1 to F9, hypotheses H1 to H5, strategies S1 to S5, failure points D1 to D11, protocol P1 to P7, stages 0 to 6), `planning/TP8-READINESS.md`, `planning/BASELINE-FREEZE.md`, `planning/DECISIONS-FOR-GARV.md` (the questions this file answers) and `planning/evidence/`.
- No goal agent has run on Kimi K3 yet. The sandbox could not start on the trays (`bwrap` and `socat` absent, AppArmor restricts unprivileged user namespaces), and no tray has 8 GPUs.
- vLLM reference: vllm-project/vllm#61867, "Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x)", merged 2026-10-16. Its specialized path runs only at TP8 or TP16 (`specialized_decode/runtime/__init__.py`); at other TP sizes vLLM silently uses the stock path.

## 1. Goal of the experiment

This is a study of the agent, not a kernel race. The question is how far a Claude goal-mode agent gets at writing a megakernel for a model of this size, and how its result compares with vLLM's specialized implementation at the problem shapes that PR #61867 reports. The primary data is the trajectory: what the agent explored, which subagents it spawned and for what, which tools it used, which optimizations it attempted, which it kept, which it could not make work, and why. The latency comparison gives that trajectory a reference point. Sections 13 to 18 define what to record and how to analyze it.

We run one end-to-end experiment: a Claude goal-mode agent writes a megakernel for one Kimi K3 decode step on the real checkpoint, all 93 layers, at TP8. We compare its latency with vLLM's Kimi K3 specialized decode (vllm-project/vllm#61867) and with stock vLLM, both at TP8, on the same GB300 hardware and the same cases.

- Graded cases: the three real-weight cases `kimi-k3-tp8-b{1,8,64}-s8192`.
- Development cases for the agent: the TP4 synthetic cases (one tray) and the p2 synthetic TP8 cases (two trays). The p2 cases become gradable on the two-tray setup and are reported, but the headline result is the real-weight cases.
- Goal prompt: `goals/kimi-k3-tp8.txt`, updated only where the hardware description (two trays, not one 8-GPU node) or the sandbox description changes.
- No person reviews the agent's code or results. Every rule must be enforced by an automated check. If a rule cannot be checked automatically, write that down in the results as a known limit; do not add a manual step.

## 2. Hardware and TP8 execution path (D-T1, D-T3)

- TP8 runs on two GB300 NVL72 trays in the same NVLink domain (MNNVL/IMEX), 4 ranks per tray. This matches the hardware of the vLLM GB300 TP8 numbers. Do not use the x86 8-GPU VM node.
- Add a two-node launcher to the harness (`megabench/harness/distributed.py` or a new module). Ranks 0 to 3 on the first tray, 4 to 7 on the second. Keep the single-host path unchanged for TP1, TP2 and TP4.
- The agent's megakernel must do all communication between ranks, including across the two trays, inside its one kernel launch.
- A 48-layer real-weight TP4 case may be added as a development case only. It is not graded.

## 3. Sandbox: no requests for permissions (D-X1)

- Do not ask administrators for anything: no `sudo`, no package installs, no sysctl or AppArmor changes. Do not work around the AppArmor restriction on unprivileged user namespaces by any other means.
- Primary: Kubernetes on rack 10.
  - Agent pod: CPU only, no GPU; non-root; read-only root filesystem; all capabilities dropped; no service-account token mounted. Workspace on a volume; benchmark read-only; Claude Code OAuth token from a Kubernetes Secret.
  - Proxy pod running `sandbox/lib/netproxy.py` (Claude API, Hugging Face, the filtered package index). A NetworkPolicy permits the agent pod egress only to the proxy and to DNS.
  - GPU work: the broker runs outside the agent pod with the operator's credentials and starts GPU pods for each `gpu-run` request. GPU pods have no external network; for TP8 the two pods of one job can reach only each other (NCCL bootstrap). Workspace read-only, weights read-only at `/weights`. Take the IMEX/MNNVL pod settings from the earlier SGLang Kimi K3 TP8 deployment on rack 10.
  - Grading uses the same GPU pod path, on fresh inputs, after the agent stops.
- Before building, find out with read-only commands whether the operator's account can create pods, secrets and NetworkPolicies, and whether the network plugin enforces NetworkPolicies.
  - If NetworkPolicies are not enforced: keep the pods, run the proxy as a sidecar, and record the network limit as weaker in the results.
  - If pods cannot be created: use the fallback.
- Fallback: host level on rack 3 with Landlock and seccomp (no root needed). Landlock: read-write only the workspace, read-only the venv, CUDA and benchmark, nothing else; TCP connect only to the proxy port. Seccomp: no UDP and no raw sockets. GPU work through the existing Slurm broker. Record the known limits (Landlock filters ports, not hosts; same Unix user as the operator).
- Proxy environment variables alone are not a sandbox.
- Build the container image for aarch64 without cluster privileges (for example `docker buildx --platform linux/arm64` on another machine) and push it to the registry the cluster already pulls from.
- Extend `setup/selftest.sh` with checks that fail if: the agent can read a path outside its mounts, can reach a host not on the allowlist, can reach the Kubernetes API, or a GPU pod can reach the network.
- Keep the bwrap mode in the repository for machines that have it.

## 4. Launch budget (D-L1 to D-L8)

- D-L1: count only device kernels (compute kernels and NCCL device kernels). Do not count memcpy and memset events. Report them in the record.
- D-L2: do not count the Kineto `nccl:` records. Report the split of every event class in the record.
- D-L3: zero NCCL kernels are allowed. The one launch does all communication.
- D-L4: a detected CUDA graph launch fails the audit outright. Before relying on the count, test with a small candidate whether kernels inside a replayed graph appear as separate events (H1) and record the result.
- D-L5: audit every timed repetition, not a separate call after timing. For a one-kernel candidate the profiler overhead is negligible; verify that on a small case and record it.
- D-L6: device-side launches (dynamic parallelism) count as launches. A persistent work queue inside the one kernel is allowed. Test whether device-side launches are visible to the profiler (H2); if they are not, add a static check (for example, scan the compiled SASS/PTX for device-side launch calls).
- D-L7: no human review. `review_required` is removed from the grade. Add an automated source and artifact scan instead: no import of the oracle, no tensors loaded from files, no CUDA graph APIs, no device-side launch APIs, no reads outside the submission directory.
- D-L8: the budget stays one launch per rank on every case.

## 5. Input integrity (D-I1 to D-I4)

- D-I1: synthetic cases: regenerate every input from the trial seed and compare exactly. Real weights: a keyed hash with a new random key per evaluation, computed on the device; the key is never visible to the candidate.
- D-I2: run the candidate in its own process, with inputs shared through CUDA IPC, so it cannot reach the evaluator's state. This is required before the agent run, because there is no human review.
- D-I3: a detected change to an input gets its own status, `input_mutation`. It counts as a failed case.
- D-I4: check the weights after every trial, after the timed repetitions and after the audit.

## 6. Distributed failures (D-S1 to D-S5)

- D-S1: statuses `incorrect`, `candidate_error`, `evaluator_error`, `infrastructure`, `timeout`, `input_mutation`, recorded per rank (local) and for the group.
- D-S2: accepted. Structural output checks run before the agreement; guards around snapshot, reference, timing and audit so that one failing rank cannot leave the others in a collective.
- D-S3: `candidate_error` and `timeout` caused by the candidate count as failed cases. Report them separately from `incorrect`.
- D-S4: a deadline per phase and a shorter collective timeout during grading. Set the values from measured times plus a margin, separately for synthetic and real-weight cases, and two-tray runs. Weight loading gets its own deadline.
- D-S5: record `max_memory_reserved` and the candidate's stderr per rank.

## 7. Weights (D-T2)

- Convert the pinned checkpoint (`moonshotai/Kimi-K3` at `f831ab66…`) once with `setup/convert_checkpoint.sh` at TP8.
- Store each tray's four ranks on that tray's local RAID, not on CephFS. Pick two trays that have local storage with at least 0.9 TB free and can run as a pair; rack 10 tray14 is one candidate. Record which trays and paths were used.
- The operator owns the converted files.

## 8. Oracle fidelity (D-F1, D-F2)

- Repeat the 4-layer comparison with the Hugging Face modeling code (`tools/verify_kimi_vs_hf.py`) on GB300.
- Compare the oracle with vLLM at full depth on the real TP8 cases (next token and logits), using the same band rule as the grader. Do this during the vLLM baseline runs.
- If either comparison fails, stop and report before any agent run.

## 9. vLLM baseline

- Run `baseline/measure_vllm.py` at TP8 on the same two trays, real weights, for both the specialized path (`decode_backend: latency_specialized`) and stock vLLM. Confirm from vLLM's log that the specialized path was actually used.
- Measure in a way comparable with MegaBench: decode-step time on the GPU (for example nsys over CUDA graph replay), not engine-level time. Record both if they differ.
- Write `baseline/BASELINES.json` with both arms.

## 10. Validation order before the agent run

Each stage must pass before the next one starts:

1. Sandbox self-test (section 3).
2. Evaluator changes (sections 4 to 6) on the CPU and on one tray, with the existing positive and negative controls plus new ones for each new check (graph launch, device-side launch, sum-preserving input change, structural failure on one rank).
3. Two-tray launcher with the oracle as candidate on a p2 synthetic TP8 case.
4. 4-layer real-weight fixture on one tray, then the oracle on the real TP8 B1 case across two trays with a reduced trial count.
5. vLLM baseline and oracle fidelity (sections 8 and 9).
6. Agent run.

## 11. Evidence (D-X2, D-X3)

- `gb300-tp4-synthetic` stays frozen as the evidence for contract digest `4da60e9b…`. Do the new work on a new branch.
- Every contract change gets a new digest, and each stage repeats its own positive and negative controls under that digest.
- Failure records keep all provenance fields (submission sha256, hardware, timestamp). The grader message gets its own field. Each status gets its own exit code.
- Keep `PROGRESS.md` and `AGENT_TRAJECTORY.md` updated as before.

## 12. Results to report

For each graded case: the agent's final latency, its ratio to the `sol_info` HBM floor, the vLLM specialized and stock latencies, the speedup against each, the status, and the commit hash. Also report: the agent's wall-clock time, the number of GPU jobs it ran, how long it waited for two free trays, and any limits from section 3 or 4 that applied. The trajectory report of section 17 is the main deliverable; the latency table is part of it.

## 13. Problem shapes to match the vLLM PR

- PR #61867 reports TP8 and TP16, with and without DSpark speculation, with and without decode context parallelism (DCP), at input lengths 8k, 32k and 100k. Its headline numbers are TP16. We use only its complete TP8 row: TP8, no DCP, no speculation, ISL 8k. The graded cases B1, B8 and B64 at context 8192 match that row (PR nsys step times, stock to specialized: B1 8.673 to 5.254 ms, B8 15.490 to 9.745 ms, B64 33.664 to 25.709 ms).
- Out of scope: TP16 (four trays), DSpark verification shapes, DCP, 32k and 100k contexts. Do not add them.
- After the run, also time the agent's final kernel and both vLLM arms at B2, B4, B16 and B32 at 8k (the other columns of the same PR row). These are measurements only, not graded, and the agent does not see them.
- KV cache precision: the PR's numbers use an FP8 KV cache and an FP32 KDA (SSM) state. The oracle uses a BF16 MLA cache. Change the case state and the oracle so that the MLA cache is FP8 in the same format vLLM uses for Kimi K3 (scale handling included), keep the KDA state FP32, and update `sol.py` so that the floor counts FP8 cache bytes. Validate the changed oracle against vLLM as in section 8. If matching vLLM's FP8 format exactly is not possible, stop and report the difference before the agent run.

## 14. Fair measurement against vLLM

- Measure the agent's kernel and both vLLM arms on the same two trays, in the same session where possible, with the same method: GPU time of one decode step with B running requests at context 8192, taken from nsys (for vLLM, the decode CUDA graph replay; for the agent, the one kernel). Report MegaBench's CUDA-event latency for the agent as well.
- Use vLLM's own weights path and FP8 KV cache with no other non-default flags, as in the PR's TP8 runs. Record the vLLM commit and the exact command.
- Confirm from vLLM logs that the specialized arm actually ran the specialized decode path for every measured batch size.
- Report with each comparison that vLLM uses about 40 CuTeDSL kernels inside a CUDA graph while the agent is held to one launch in CUDA C++ and inline PTX. This asymmetry is part of the study; do not remove it.

## 15. What the agent can see

- The agent must not see vLLM's Kimi K3 implementation or other production kernels for this model. Keep vLLM, FlashInfer, CUTLASS/CuTeDSL, SGLang and similar packages blocked in the package index, and keep GitHub off the proxy allowlist. WebSearch is disabled for the same reason (section 16). Hugging Face (including `moonshotai/Kimi-K3` modeling code) stays allowed.
- The agent may see `BASELINES.json` (the vLLM latencies for the graded cases), as the goal prompt says.
- The workspace gets no extra skills or subagent definitions. Scaffold experiments belong to the `scaffold-search` work.
- Profiling and timing tools: the agent gets the tools below, through `gpu-run`, on every case including the two-tray real-weight TP8 cases. Describe each one in `TASK.md` with one working example.
  - **nsys:** `./nsys_candidate --case ID` runs one evaluation of the candidate under Nsight Systems on every rank and returns one report per rank plus a merged timeline (kernel span, NCCL if any, copies, gaps between steps).
  - **ncu:** `./profile_candidate --case ID` (already present) on one rank or all ranks. A cooperative multi-rank kernel can hang under kernel replay; use application replay for multi-rank cases and say so in `TASK.md`. If GPU performance counters are not available to the account inside the GPU pods, do not ask for them to be enabled: make the tool report that clearly, record it as a limit of the run, and keep nsys and the in-kernel timers working.
  - **In-kernel timers (`clock64` and `%globaltimer`):** a header (for example `megatimer.cuh`) that lets the kernel write timestamps at points the agent chooses (per layer, per phase, per SM or block, per rank) into a device buffer, compiled in only when a macro is set; and `./phase_timeline --case ID` that builds the candidate with the macro, runs one step on every rank (two trays for TP8), and converts the buffers into a per-rank, per-layer, per-phase timeline with the critical path and the time each rank waits at each synchronization point. Use `%globaltimer` for wall time and cross-SM order and `clock64` for short intervals on one SM. Align ranks on a common event (for example the first grid-wide synchronization), because `%globaltimer` is not synchronized between GPUs; state that in the output.
  - Graded timing and the launch audit always build the candidate without the timer macro. The automated scan of D-L7 fails a submission whose graded build still writes timer buffers.
- Record every use of these tools in the trajectory (section 17).

## 16. Agent configuration and run conditions

- Pin and record the Claude Code version, the main model (`claude-opus-5-5`) and its effort setting.
- Subagents: allowed. The main model decides whether to spawn subagents, how many, for which tasks, and with which model. Remove `CLAUDE_CODE_SUBAGENT_MODEL=sonnet` from the agent's launch command (`sandbox/bin/goalbox-start` today, and its Kubernetes equivalent); it forces every subagent onto Sonnet. Do not set any default subagent model. Record the model of every subagent the agent spawns.
- Tools: allowed. The agent may use every Claude Code tool except WebSearch, plus every workspace tool of section 15. Launch Claude Code with `--disallowedTools WebSearch`: WebSearch runs on Anthropic's servers, so the proxy cannot filter it, and it can reach vLLM's implementation (section 15). WebFetch and shell downloads stay, limited by the proxy to the allowlist. Record the disabled tool in the results.
- The goal prompt `goals/kimi-k3-tp8.txt` on this branch is already updated for this run: two trays, subagents and tools allowed, the profiling and timing tools named, and the stop rule below. Do not change it, except to correct a tool name if you build a tool under a different name. If you rename a tool, change the goal prompt and `TASK.md` together.
- Start: headless goal mode in a fresh session, empty `/work` except for what the workspace generator writes. No seed kernel.
- Stop rule: do not let the agent stop early at 1.4x the HBM floor. vLLM's specialized arm is already at 1.21x the floor at B64, so that rule can end a run before it reaches the comparison point. Replace the goal's success condition with: continue optimizing until the budget ends or five consecutive attempts each improve latency by less than 2 percent on every graded case. Keep everything else in the goal text.
- Budget: 48 hours of wall clock for the agent. At 30 minutes before the end, interrupt the agent and give it at most 15 minutes to commit its best state and update `PROGRESS.md` with a final summary, then stop it. Also record total tokens and cost.
- No human input during the run. If the run stops because of infrastructure (node failure, broker crash, rate-limit outage), resume the same session once the infrastructure is back and record the gap. Do not resume or restart for any other reason, and do not restart because of the result.
- Number of runs: one run. Report it as a case study and say that one trajectory cannot show variance.
- Reserve the two trays for the agent's GPU jobs for the whole run, so that waiting for hardware does not shape the trajectory. Record any waiting that still happens.

## 17. Trajectory capture and analysis

Record, with timestamps, outside the agent's reach (the agent must not be able to edit or delete these records):

- The full transcript of the main session and of every subagent (Claude Code JSONL), including tool calls and results, model, token usage, and context compactions.
- Every `gpu-run` request: command, requesting agent or subagent, submit, start and end time, queue wait, exit status and output files.
- Every `check_candidate`, `sol_info` and `profile_candidate` call with its full result record.
- A snapshot of `/work` (as a git commit made by the harness, not by the agent) at every `check_candidate` call, including attempts the agent later reverted.
- The agent's own `PROGRESS.md` and `CORRECTNESS.md` history. These are the agent's self-report: useful for its stated reasons, but incomplete and editable by the agent. They are not the record of the trajectory.

How to capture the transcripts so that the agent cannot change them:

- Claude Code writes the main session to `~/.claude/projects/<project>/<session>.jsonl` and each subagent to `~/.claude/projects/<project>/<session>/subagents/agent-*.jsonl`, inside the agent's home, where the agent can write. Do not rely on those files alone.
- Start Claude Code from the harness, outside the sandbox, with `--output-format stream-json`, and save that stream to storage the agent cannot reach.
- A sidecar container (or, in the Landlock fallback, a process outside the Landlock domain) copies the JSONL files every 30 seconds into versioned, append-only storage outside the agent's mounts, with a sha256 per copy.
- At the end, compare the stream, the copies and the final files. Report any difference (a transcript edited or deleted by the agent) in the trajectory report.
- Build the trajectory report from these copies with the conversation compiler (VCC) that the repository owner uses (a Claude Code skill, `conversation-compiler`, with `scripts/VCC.py`), so that the raw run can also be read with `/readchat`. It is not in this repository; ask the user for it. If it is not available, write a small compiler that turns the JSONL into a readable transcript with line references, and keep the raw JSONL.

After the run, produce a trajectory report with:

- A timeline of phases (exploration, reference reading, correctness tooling, first correct kernel, first single-launch kernel, first TP8 run, optimization) with times and GPU hours.
- A table of every subagent: when, why, its task, its model, its tokens, and whether its output was used.
- Tool use counts and how they changed over time (profiler, sol_info, check_candidate, web or Hugging Face reads).
- An optimization table built from the components of the vLLM PR: KDA chain, MLA decode, attention residuals, output projection, MoE routing, routed experts (MXFP4 GEMM), shared experts, split-K and warp-specialized GEMMs, all-reduce scheme, weight prefetch and overlap with communication, FP8 KV handling, plus megakernel-specific items (grid-wide synchronization, persistent scheduling, cross-tray communication inside the kernel). For each: attempted or not, kept or reverted, the measured effect, and the agent's stated reason. List optimizations the agent made that vLLM does not, and vLLM optimizations the agent never attempted.
- Where the agent got stuck: repeated failures, correctness bugs and how long each took, and any rule violations caught by the automated checks.
- Milestones with time stamps: correct multi-kernel version, correct single-launch kernel on TP4 synthetic, on TP8 synthetic, on real weights, faster than vLLM stock, faster than vLLM specialized. Report the milestones reached even if the final kernel never passes the one-launch rule; also report the best correct submission at any launch count.

## 18. Data handling

- The repository is public. Keep raw transcripts, logs and snapshots out of it. Store them on the cluster under the operator's control and back them up.
- Publish to the repository only the trajectory report and redacted summaries: no tokens, host addresses, user names other than tray names, or cluster details beyond those already in the repository.
