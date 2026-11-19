# Scaffold search: which workspace scaffolding helps agents write faster megakernels?

**Protocol revision `dsv41-tp4-rack10-v1`** (fork of `scaffold-search-dsv41`): the same experiment on
one **4x GB300 tray, TP4**. Results are a new experiment, not a replication of the TP8 one.

Autonomous Claude Code agents are each given 2 hours to make a correct single-launch CUDA
megakernel faster (DeepSeek-V4.1-Flash + DSpark speculative iteration, batch 8, TP4 on 4x GB300).
Four arms differ only in what is added to the agent's workspace. You run 16 agent runs
(4 arms x 4 runs) on one 4x GB300 tray, 3 at a time, and the harness scores them.

The full design, metric, decision rule and the sha256 manifest of everything that defines the run
conditions are in **[PROTOCOL.md](PROTOCOL.md)**. Do not change any file listed in its manifest.

| arm | adds to the workspace |
|---|---|
| **A0S** (control) | 9 generic CUDA skills |
| **A1** | A0S + measurement tools: per-phase/per-rank timeline, multi-rank ncu, leftover-flag audit |
| **A2** | A1 + knowledge: B300 primitive costs, block-scaled FP4 GEMM skill + example, multi-GPU sync skill |
| **A3** | A1 + parallel engines: capture + 1-GPU phase harness, engine subagent, integrator workflow |

Score = 28.95 ms / L, where L is the run's best passing kernel re-measured after the run. An arm
wins if its median score beats its comparison arm's by more than max(5%, control IQR/median) and
its median AUC is higher. Comparisons: A1 vs A0S, A2 vs A1, A3 vs A1.

## Requirements
- 1 tray with **4x NVIDIA GB300 (sm_103), NVLink**, aarch64, in a **linux/arm64** container where you are
  **root** (no privileged mode needed; see `container/README.md`). The tray must be reserved for this
  (no other GPU pods/jobs).
- **Internet** from the container: GitHub, PyPI (image build), Anthropic API / claude.ai.
- ~200 GB of disk under `/srv/ss`.
- A **Claude Code account** that can run **3 agents at once for 2 h each** (plus up to 4 engine
  subagents inside each A3 run) with model `claude-opus-5-5`. Rate-limit stalls count as lost
  time and are not rerun, so a plan that throttles will bias scores down.
- **GitHub read access to the benchmark fork** (`scripts/fetch_benchmark.sh`: `kaijunli-infr/vibe-megakernel`,
  branch `dsv41-dspark-batch8-tp4`, pinned commit; `SS_BENCHMARK_URL` overrides the URL). The seed kernel
  is vendored in `seeds/`.
- Optional: a GitHub repo you can push to, for live snapshots of every run (`SS_SNAPSHOT_REPO`).

## Run it
```sh
# 1. build the image (or push it to your registry for Kubernetes: container/README.md)
git clone -b dsv41-tp4-rack10-v1 <harness repo> ss
docker build -t scaffold-search:dsv41-tp4 -f ss/container/Dockerfile ss/container

# 2. start a dedicated container with the tray's 4 GPUs; everything below runs as root inside it
docker run -d --name ss --gpus all --shm-size=64g -v /data/ss:/srv/ss scaffold-search:dsv41-tp4
docker exec -it ss bash
git clone -b dsv41-tp4-rack10-v1 <harness repo> /srv/ss/harness
cd /srv/ss/harness

# 3. logins (details: scripts/login_notes.md)
claude                                                     # /login, then /exit
gh auth login                                              # needs read access to the benchmark fork
export SS_SNAPSHOT_REPO=git@github.com:<you>/<repo>.git   # optional; unset = results stay in /srv/ss

# 4. bootstrap (idempotent; rerun after any container restart): seed, benchmark + patch, trusted
#    snapshot, PROTOCOL manifest check, run users, /dev/nvidia* permissions, lease daemon + proxy
bash scripts/bootstrap_node.sh --check-login

# 5. self-test (~45 min; ends with a 10-min fake-agent run through the whole pipeline)
bash scripts/run_selftest.sh

# 6. the experiment: 16 runs, 3 at a time, ~14 h
bash scripts/run_main.sh start

# 7. monitor
bash scripts/run_main.sh status        # per-run state and lease queue
tail -f /srv/ss/logs/orch-main.log     # orchestrator
tail -f /srv/ss/logs/run-r01.log       # one run
/srv/ss/bin/gpu_lease status           # active leases and queue (JSON)
tmux ls                                # ss-lease, ss-proxy, ss-orch-main, ss-runs-main

# 8. report (works on partial results too)
bash scripts/collect_report.sh /srv/ss/report && cat /srv/ss/report/report.md
```

Recovering runs:
- A run whose agent phase finished but whose rescore was interrupted:
  `bash scripts/run_main.sh rescore rNN`.
- A run that failed because of the harness or node (no `result.json`): rerun it under a new id
  and note it in the ledger, e.g. `cd /srv/ss/harness && python3.12 -m scaffold_search.run_arm
  --arm A2 --run-id r01b` in a tmux window. Do not rerun runs because of their scores.
- `bash scripts/run_main.sh stop` stops everything. `run_main.sh start` again resumes: runs with a
  `result.json` are skipped.

Outputs per run, under `/srv/ss/runs/<run_id>/`: `result.json` (schema in `docs/harness.md`),
`logs/agent.jsonl` (transcript), `snapshots/` (every passing check), `rescore/`, `ws/` (the agent's
workspace). With `SS_SNAPSHOT_REPO` set, the same is pushed to branches `run/<id>-<arm>` and
`results`. Back up `/srv/ss/runs` if you don't use a snapshot repo: a terminated container loses it.

## Time and cost
- Per run: 2 h agent budget + ~15-20 min of node time for calibration (1 check), rescore
  (3 checks) and the final-tree check.
- 16 runs at K = 3: **~14-16 h of the tray** (TP8 estimate; a TP4 in-run check takes ~5.8 min wall / 4.7 min on the lease, Gate E).
- Claude usage: roughly $10-120 per run at API prices; A3 runs use subagents and cost more.
  The self-test's fake-agent run uses no Claude.

## Things to know
- **Run the self-test first.** The package has passed its CPU tests, but its container and
  bootstrap steps have not been exercised on your node.
- **Checks hold the whole node:** an in-run check uses all 4 GPUs (TP4, Gate E: 5.8 min wall, 4.7 min
  on the lease, 1 process; the 3-process rescore check is not yet timed). With 3 runs at once, agents wait for 4-GPU
  leases a large share of the time; this is recorded per run (`lease.wait_share_node`) and is part
  of the protocol. Do not change K or the `GPU_LEASE_*` settings.
- `./check_candidate` first runs `./prebuild` (nvcc) on the CPU outside the lease, so 3 runs
  compile at once; a slow CPU lengthens checks.
- **Web block:** the container needs no CAP_NET_ADMIN. Agents are blocked by disallowed web tools,
  an allowlist proxy (Anthropic hosts only) in their environment, and a transcript scan
  afterwards. An agent that unsets the proxy variables could reach the internet; that is flagged,
  not prevented.
- **Profiling:** ncu kernel replay hangs on this 4-rank cooperative kernel; A1's profiler uses
  application replay. Hardware counters for unprivileged run users may need the host driver
  setting described in `container/README.md`.
- **The harness takes over the container:** `/root` becomes 0700, `/tmp` 1733, `/dev/nvidia*`
  `root:ss-gpu 0660` (other users reach GPUs only through `gpu_lease`). Run users `ss-<run_id>` are
  locked, never deleted. Use a dedicated container.
- A container restart kills running runs (they live in tmux inside it). Rerun
  `bootstrap_node.sh`, then rerun runs without `result.json` under new ids.
- Run ids starting with `smoke`/`test` go to `results/_smoke/` and are ignored by the report;
  lease requests from `builder-*`/`smoke*` ids yield to experiment runs.

## Layout
| path | what |
|---|---|
| `PROTOCOL.md` | design, metrics, decision rule, schedule, sha256 manifest |
| `scaffold_search/` | harness: `gpu_lease`, isolation, workspace build, `run_arm`, telemetry, snapshots, `rescore`/`review`, `schedule`, `orchestrate`, `report`, tests |
| `arms/` | arm definitions (overlay lists, budget) |
| `overlays/` | files each arm adds to the workspace (`S`, `A1`, `A2`, `A3`) |
| `prompts/goal_improve.md` | the agent's goal (identical for every arm) |
| `seeds/dsv41_b8_tp4_nopf/` | the starting kernel, TP4 (vendored, sha256-verified; `PORT.md` = the port) |
| `seeds/dsv41_b8_nopf/` | the original TP8 seed, unchanged (not used by this protocol) |
| `container/` | Dockerfile and how to run it |
| `scripts/` | `bootstrap_node.sh`, `run_selftest.sh`, `run_main.sh`, `collect_report.sh`, login notes |
| `docs/` | harness design and the A1/A2/A3 tool docs |
