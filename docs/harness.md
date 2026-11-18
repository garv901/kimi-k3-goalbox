# Harness (scaffold_search/) — design, usage, verification

Design and operations of the harness. Experiment design: PROTOCOL.md.

## Node layout
| path | mode | what |
|---|---|---|
| `/srv/ss/harness` | 0700 root | checkout of branch `harness` used to RUN experiments (`git pull` to update) |
| `/srv/ss/bin/gpu_lease{,.py}` | 0755 | lease CLI (copy of `scaffold_search/gpu_lease.py`) |
| `/srv/ss/bin/claude` → `claude-2.1.326` | 0755 | pinned Claude Code binary readable by run users |
| `/srv/ss/lease/lease.sock`, `/srv/ss/lease.jsonl` | 0666 / 0600 | lease daemon socket and log |
| `/srv/ss/trusted/vibe-megakernel` | read-only root | benchmark snapshot (git archive of 886eeb3 + patches) |
| `/srv/ss/runs/<run_id>/` | 0711 root | `ws/` (agent, 0700 run user), `home/`, `tmp/`, `snapshots/` + `logs/` (root 0700), `rescore/`, `push/`, `result.json` |
| `/srv/ss/results/repo` | root | clone of branch `results` (pushes are serialized by a file lock) |
| `/srv/ss/schedules/` | root | schedule JSON files |

tmux on the node: `ss-lease` (lease daemon), `ss-proxy` (allowlist proxy; currently started
from `/srv/ss/build/harness`, same code). Restart either with the
command in the "Operations" section; both are needed by every run.

## Components
- **gpu_lease.py** — daemon + CLI. The daemon (root) spawns the command itself as the caller's uid
  (cwd, env and stdio fds passed over the socket), in a new session, with `CUDA_VISIBLE_DEVICES`
  and the supplementary group `ss-gpu`. Policy: 8-GPU leases exclusive; age order across classes
  (the next waiting 8-GPU request blocks only *newer* small requests, older small requests are
  served first, so neither class starves); round-robin by run inside each class; ≤ 3 GPUs per run
  in small leases (sizes 1, 2, 3, 8); default max 20 min (8-GPU) / 60 min (small), non-root capped
  at 30 / 120; expiry, cancel (SIGINT/SIGTERM to the client) or client death kill the session +
  process group (SIGTERM, 10 s, SIGKILL) and stray GPU processes of the same non-root uid. Run id:
  `ss-<run>` users get `<run>` (SS_RUN_ID ignored, no impersonation); root uses `SS_RUN_ID`. Nested
  calls inside a lease run directly. Clients queued during a daemon restart resubmit; a restarted
  daemon adopts GPUs that still run processes (synthetic `orphan` lease) until they exit. A reaper
  thread (5 s) releases leases whose handler died, forces kill-run after 30 s, and kills leases
  whose GPUs had no compute process and 0% util for 300 s (`GPU_LEASE_IDLE_S`; reason
  `idle-reaped` — CPU-only work longer than that inside a lease is killed). Releases log
  `util_mean`. Experiment runs have priority over `builder-*`/`smoke*`/root requests. Log events
  `request/grant/release/abandon/adopt` with `cls` (8 | small), `wait_s`, `hold_s`, `reason`.
- **isolation.py** — run user `ss-<run_id>` (group ss-agents), home with only a copy of the Claude
  credentials, minimal `settings.json` (deny WebFetch/WebSearch, no autoupdate/telemetry) and
  minimal `~/.claude.json` (onboarding done, workspace trusted); node lockdown (see layout; `/tmp`
  1733; `/dev/nvidia*` root:ss-gpu 0660); allowlist CONNECT proxy on 127.0.0.1:3128 (Anthropic /
  Claude hosts only; denied hosts logged with uid to `/srv/ss/proxy.jsonl`); `sync_creds()` copies
  the newest-`expiresAt` OAuth credential file to root and all run homes (every minute from
  run_arm) so token refresh/rotation does not log anyone out.
- **workspace.py** — `make_agent_workspace --agent plain --case <case> --benchmark-root <trusted>`,
  seed sources (sha256-verified), harness files (`./gpu_run`, snapshotting `./check_candidate`
  wrapper, `./profile_candidate` and `./timeline_candidate` on an 8-GPU lease, stock tools in
  `.harness/*.stock`, `TASK.d/harness.md`), then the arm's overlays verbatim; initial git commit;
  chown to the run user.
- **snapshot.py** — the wrapper tars the source tree before the check and leaves
  `.harness/pending/<id>.{tgz,log,json}`; the root monitor moves passing (`ok_provisional`, rc 0)
  tarballs to `snapshots/NNN-<id>.tgz` (+ sha256, `index.jsonl`), logs every check in
  `checks.jsonl` (unfinished ones as `incomplete` at teardown), copies reports, and pushes branch
  `run/<run_id>-<arm>` (ws without .git/build/.so/>50 MB, plus transcript, telemetry, indices).
- **telemetry.py** — 1 Hz nvidia-smi sampler (util/mem per GPU, owning uids, `bypass` = GPU
  process of the run uid without `SS_LEASE_ID`), lease slice, transcript stats (tool counts incl.
  Skill/Task, tokens, cost, rate-limit events, stall minutes from line timestamps), snapshots.
- **rescore.py + review.py** — see "Outcome computation".
- **Conditional small-lease cap**: while an
  experiment run's 8-GPU request has waited ≥ 2 min, small leases older than 10 min are ended
  (SIGTERM, 30 s grace, SIGKILL; `preempt` event + release reason `preempted`; `./gpu_run` prints
  why). Telemetry/result.json: `lease.preempted`, `lease.preempted_min_lost`, `lease.idle_reaped`.
  Bounds the time any run's check waits behind other runs' small leases to ≈ 12.5 min; A3 (most
  small-lease use) bears most preemptions, which are reported per run.
- **Snapshots and run branches** leave out `capture-*` directories (A3 captures, ~0.4 GB), the
  `reports/` dir (tarballs only) and every file > 50 MB.
- **In-run vs rescore checks**: the agent's `./check_candidate` runs the stock check with
  `--processes 1` and first runs `./prebuild` (compile without GPUs; harness-provided for the seed:
  `mk_host.load_ext()`, same digest-keyed .so the evaluator loads), so an in-run check holds the
  node ~2.5 min instead of ~5.3 min. Calibration and rescore use the unmodified stock check
  (3 processes), also after `./prebuild`.
- **run_arm.py** — one run end to end (below). **schedule.py / orchestrate.py** — random mixed
  waves; K concurrent runs in tmux windows, next starts when one finishes; ledger with node-hours
  and $ (72 $/h) pushed to `results/_ledger/`. **report.py** — table + pre-registered verdicts.
- **trusted.py** — builds the trusted snapshot from the clean `/root/vibe-megakernel` checkout and
  applies `scaffold_search/patches/*.patch`.

## Run lifecycle (run_arm.py)
1. node lockdown (idempotent), run user, workspace; pins recorded in `logs/run.json` (Claude
   version, model `claude-opus-5-5`, harness commit, trusted commit + patch sha256, seed sha256,
   prompt sha256, arm file sha256).
2. **Calibration**: the seed checked once (stock check, fresh tree, as the run user, 8-GPU lease)
   *before* the agent clock starts → covariate `calibration.L_ms`.
3. `claude -p "<prompts/goal_improve.md with {DEADLINE}>" --model claude-opus-5-5 --permission-mode
   auto --permission-prompts none --output-format stream-json --verbose --disallowedTools
   WebFetch,WebSearch` as the run user, env from `isolation.agent_env` (HOME, TMPDIR, PYTHONPATH =
   trusted, SS_RUN_ID, proxy vars). DEADLINE = the summary point.
4. Monitor every 5 s: pending checks → snapshots (+ push); creds sync every 60 s; push every 10 min.
5. At T−15 min (T−max(2, budget/4) for budgets < 60): SIGINT to claude (60 s grace, then the group
   is killed); `claude -p --resume <session> "Time is up: write the NOTES.md final summary now; do
   not start new work."` for ≤ 10 min (and not past T).
6. Teardown: kill the group and every process of the user, `gpu_lease kill-run`, verify no GPU
   process of the uid remains (`gpu_clean_after`); final snapshot pass, telemetry, push.
7. `rescore.py` → `result.json`, pushed to `results/<run_id>/` (smoke runs: `results/_smoke/`).

## Outcome computation (rescore.py, review.py)
- Candidates: passing snapshots ordered by the agent-measured p50; the best is re-checked 3× in a
  fresh `make_agent_workspace` tree (stock tools, no build cache) as the run user on an 8-GPU lease;
  L = median of the per-run `fast_p50_ms` (that run only; the gpu_noise history minimum is not
  used). If review fails, the next-best is tried (≤ 3). None valid → **L = this run's calibration of
  the seed** (`source: seed-fallback`).
- Review: all 3 runs `ok_provisional`, launch audit 1 kernel on each of the 8 ranks, every trial of
  every rank passes; the evaluator has no seed option — it draws fresh random seeds per trial
  (`secrets.randbits`), so the 3 rescore runs give 3 fresh seeds per scenario (raw, accept_full,
  accept_mixed) that the agent never saw; recorded as `fresh_seeds_per_scenario`. Sources reachable
  from `submission.py` (imports, quoted local source names, `#include`) must not load tensors from
  files (`torch.load`, `np.load`, safetensors, pickle, fromfile) nor import `megabench`/`reference`
  or mention an oracle.
- Primary score 28.95 / L; AUC = time-average over the budget of log(28.95 / best passing L so far)
  (agent-measured p50 at check end time; seed calibration before the first pass); final tree: one
  check unless its digest equals the chosen snapshot's.
- Transcript scan (flags only): WebFetch/WebSearch calls, network commands in Bash (curl, wget,
  git clone/fetch/pull/push, pip install/download, ssh/scp, http URLs), paths outside the workspace,
  the run home/tmp, the trusted tree, /usr, /opt, /srv/ss/bin.

### result.json (schema `scaffold-search/result/v1`)
`run_id, arm, task, case, budget_minutes, fake_agent, pins{...}, times_utc{agent_start, interrupt,
agent_end, end}, calibration{L_ms,status,minutes,n_processes,probe_errors}, primary{score, L_ms,
source, snapshot, rescore_p50s_ms, b3_ms}, auc, final_tree{L_ms, status, score, same_as?},
candidates_tried[{snapshot, agent_p50_ms, sha256, rescore_p50s, L_ms, review_reports,
review_sources, runs[], digest}], secondary{time_to_first_improvement_min, gpu_busy_fraction,
checks{n,passing}, tokens, cost_usd, rate_limit_stall_min, tool_calls, skill_calls,
subagent_calls, lease{leases, by_gpus, wait_min_total, wait_min_8gpu, max_wait_min, hold_min_total,
gpu_min_held, expired, abandoned}, score_per_usd, abort_flags, agent_exit, gpu_clean_after,
bypass_pids}, transcript_scan{flagged,n,flags}, snapshots[], results_push`.

## M2: gpu_noise multi-rank fix (harness fix, applied for every arm)
`megabench/integrations/gpu_noise.py probe` built the case without `case['execution']`, so on the
8-rank DSv4.1 case both extra timing processes of `check_candidate` crashed (`KeyError:
'execution'`) and the stability verdict rested on 1 process. Patch
`scaffold_search/patches/gpu_noise_multirank.patch` (sha256 in `.ss_patches` of the trusted tree):
for `case.gpus > 1` the probe spawns one process per rank (`torch.multiprocessing.spawn`, NCCL
rendezvous, `tasks.parallel.initialize`), passes the same `execution` dict as `megabench evaluate`,
times `reps` calls per rank and takes the slowest rank per repetition (as `worst_rank_timing`).
Verification: see below. Applied with `python3.12 -m scaffold_search.trusted`.

## Operations
```sh
# lease daemon / proxy (root, on the node)
tmux new -d -s ss-lease "/srv/ss/bin/gpu_lease daemon --ngpu 8 2>&1 | tee -a /srv/ss/logs/ss-lease.log"   # --ngpu 4 on a TP4 slot
tmux new -d -s ss-proxy "cd /srv/ss/harness && python3.12 -m scaffold_search.isolation proxy 2>&1 | tee -a /srv/ss/logs/ss-proxy.log"
/srv/ss/bin/gpu_lease status
# one run
cd /srv/ss/harness && git pull && python3.12 -m scaffold_search.run_arm --arm A0S --run-id r01
# the experiment schedule (see PROTOCOL.md)
python3.12 -m scaffold_search.schedule --name main --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 --prefix r --out /srv/ss/schedules/main.json
tmux new -d -s ss-orch-main "cd /srv/ss/harness && python3.12 -m scaffold_search.orchestrate --schedule /srv/ss/schedules/main.json > /srv/ss/logs/orch-main.log 2>&1"
# report
python3.12 -m scaffold_search.report /srv/ss/results/repo/results
```
Upgrading the lease daemon: `install -m 755 scaffold_search/gpu_lease.py /srv/ss/bin/gpu_lease.py`,
then restart the tmux session; queued clients resubmit, running lease processes are adopted.

## Notes for overlays
- A3 `./task_ab`: pass B-side flags in the `=` form, `--flags-b="-DFOO -DBAR"` (a value starting
  with `-` is otherwise parsed as an option).

## Known limits
- No CAP_NET_ADMIN on the node: the web block is the disallowed tools + the proxy environment +
  the transcript scan. A process that unsets the proxy variables reaches the internet; it is
  flagged afterwards, not prevented.
- No mount namespaces (no CAP_SYS_ADMIN): world-readable files outside the locked directories
  (e.g. `/etc`, `/usr`, files other users leave readable under `/tmp` if they know the name) are
  readable by run users; `/tmp` cannot be listed.
- Workspace wrappers (`check_candidate`, `.harness/*.stock`) live in the agent-writable workspace:
  an agent can edit or bypass them (e.g. call `.harness/check_candidate.stock` through `./gpu_run`),
  which only loses snapshots; scores come from the root-side rescore with the trusted evaluator.
  Agent-reported p50s in `checks.jsonl` feed the AUC and candidate ranking only.
- The snapshot tar is taken just before the check; files edited by a background process during the
  check are not reflected.
- An 8-GPU lease waits for every small lease to end (exclusive); heavy 8-GPU traffic still delays
  small leases by up to the 8-GPU leases queued ahead of them.
- Rescore and calibration run under the run's lease identity; lease covariates are windowed to
  the agent phase.
- Node capacity: one 8-GPU check is ~2.5 min (in-run) / ~3.5 min (3-process rescore check); each
  run also needs 1 calibration + 3 rescore (+1 final unless identical) checks ≈ 15 min of node
  time. A long small lease (up to 60 min, builders) blocks every 8-GPU request until it ends;
  builder-*/smoke* requests wait while any experiment request waits (strict priority).
- Short budgets: with a 10-min fake budget a check queued behind other runs may not finish before
  the T−lead point (logged `incomplete`).
