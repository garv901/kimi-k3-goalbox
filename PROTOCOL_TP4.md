# Experiment protocol (TP4): the scaffold search on 4-GPU slots

Companion to [PROTOCOL.md](PROTOCOL.md) (TP8, branch `scaffold-search-dsv41`). Same question, same arms,
same agent and budget; the case runs on 4 GPUs, and each run gets its own 4-GPU slot. Two rows of the
manifest (TP4 seed, b3) are filled in during preparation; `scripts/run_main_tp4.sh start` refuses
to start while any row is `TBD` or any file differs. After the freeze, any change to a file in the manifest
invalidates the affected runs.

## Questions
- **E4a (replication without contention):** do the TP8 arm effects (A1 vs A0S, A2 vs A1, A3 vs A1) hold
  on a second TP shape when each run has its GPUs to itself? In the TP8 design 3 runs share one node, so
  lease waits and preemptions are a covariate, and A3 (the heaviest small-lease user) bears most of them.
  E4a removes that confound and has n = 6 per arm.
- **E4b (contention ablation):** A1 and A3 with 3 runs sharing one 4-GPU slot (the TP8 sharing pressure).
  Compared with their E4a runs, does contention explain A3 vs A1?

## Task (identical for every run)
- **Case:** MegaBench `dsv41-flash-dspark-step-k5-ctx2k-b8-tp4`: the same DeepSeek-V4.1-Flash + DSpark
  iteration as the TP8 case (batch 8, k5, ctx 2k), TP4 on 4x B300 (sm_103), at most one GPU kernel
  launch per rank. Evaluator: vibe-megakernel `c978283` = the TP8 pin `886eeb3` plus the TP4 case
  entries (no evaluator change), with the same `gpu_noise_multirank.patch`.
- **Start:** seed `seeds/dsv41_b8_tp4_nopf`, the TP4 port of the TP8 seed `dsv41_b8_nopf` (same
  kernel design and flags; only what the rank count requires changed), made once by a builder run
  (preparation step 2) and frozen with `scripts/freeze_seed.py`.
- **Goal prompt:** `prompts/goal_improve_tp4.md` = `prompts/goal_improve.md` with "TP8 on the 8 local
  B300 GPUs" replaced by "TP4 on the 4 local B300 GPUs" (test: `test_tp4_arms_and_prompt`).
- **Budget, agent, interruption:** as in PROTOCOL.md (2 h; T-15 min summary resume; Claude Code 2.1.326,
  `claude-opus-5-5`, headless `/goal`, web tools disabled, fresh session per run).

## Arms
`arms/tp4/{A0S,A1,A2,A3}.toml`: the TP8 arm files with `task = "improve-b8-tp4"` and
`prompt = "goal_improve_tp4.md"`; overlays, budget and caps unchanged. The overlays are the TP8
overlays with the GPU count made neutral: `overlays/A1/TASK.d/A1.md` and `overlays/A3/TASK.d/A3.md`
say "all GPUs" instead of "8 GPUs", and `overlays/A3/capture` defaults `--gpus` to the case's gpus
(was 8). On 8 GPUs every tool behaves as before.

## Conditions shared by every run
- **Slot:** 4 B300 GPUs of one NVLink node in a dedicated container (an 8-GPU node holds two slots,
  GPUs 0-3 and 4-7). Each slot has its own `/srv/ss`, lease daemon (`--ngpu 4`), proxy and run users;
  bootstrapped with `SS_PROTOCOL=PROTOCOL_TP4.md` (`SS_NGPU=4`, written to `/srv/ss/node.env`;
  `run_arm` refuses a task whose gpus differ).
- **Leases:** the TP8 policy with "8-GPU" read as "full slot" (4 GPUs): full leases are exclusive;
  small leases of 1-3 GPUs, at most 3 GPUs per run; same time limits, preemption (a full request
  waiting >= 2 min ends small leases older than 10 min) and idle reaping. In-run `./check_candidate`
  times 1 process; calibration and rescore use the stock 3-process check, all on a full 4-GPU lease.
- **E4a:** K = 1 per slot (no other run shares the GPUs). **E4b:** K = 3 on one slot.
- Isolation, snapshots, rescore, review (1 launch on each of the **4** ranks) and the transcript
  scan as in PROTOCOL.md.

## Schedules
**E4a**: 24 runs, `scaffold_search.schedule --name e4a --arms tp4/A0S=6,tp4/A1=6,tp4/A2=6,tp4/A3=6 --k 4
--seed 20261119 --prefix t --latin`. Each wave holds every arm once; wave position = slot (0-3).
Waves 1-4 form a Latin square (each slot runs each arm once); waves 5-6 are two rows of a second square.
Each slot runs its column one run at a time (`run_main_tp4.sh start e4a --slot I`).

| wave | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---|---|---|---|
| 1 | t01 A3 | t02 A2 | t03 A0S | t04 A1 |
| 2 | t05 A2 | t06 A0S | t07 A1 | t08 A3 |
| 3 | t09 A1 | t10 A3 | t11 A2 | t12 A0S |
| 4 | t13 A0S | t14 A1 | t15 A3 | t16 A2 |
| 5 | t17 A3 | t18 A1 | t19 A0S | t20 A2 |
| 6 | t21 A1 | t22 A3 | t23 A2 | t24 A0S |

**E4b**: 8 runs on one further slot, `--name e4b --arms tp4/A1=4,tp4/A3=4 --k 3 --seed 20261120 --prefix c`
(TP8 mixing rule): c01 A1, c02 A3, c03 A3 | c04 A1, c05 A3, c06 A1 | c07 A3, c08 A1.

E4a and E4b may run at the same time on different slots (5 slots = 3 nodes, or 2 nodes + E4b afterwards).
Each slot also calibrates the seed once per run (covariate; slot = hardware covariate too).

## Outcomes
As in PROTOCOL.md, with b3 = **b3 improve-b8-tp4** (manifest): primary score = b3 / L, co-primary AUC =
time-average of log(b3 / best passing L so far). b3 only scales the score; it does not change any
comparison between arms. Until it is set, rescore reports score = 1/L and flags `b3_unset`.
Secondary metrics and covariates as in PROTOCOL.md; for TP4 the `lease.*_8gpu` fields refer to the full
4-GPU lease (`lease.full_lease_gpus = 4`).

## Decision rule
- **E4a:** n = 6 per arm; margin = max(5%, IQR/median of the E4a A0S scores); arm X wins over Y if median
  score(X) > median score(Y) x (1 + margin) and median AUC(X) > median AUC(Y). Comparisons A1 vs A0S, A2 vs A1,
  A3 vs A1: `scaffold_search.report RESULTS --prefix t`.
- **E4b:** A3 vs A1 with the same rule and the E4a margin (`report RESULTS --prefix c --comparisons A3:A1
  --margin <E4a margin>`). Reported with the contention effect per arm (median score
  E4b / E4a for A1 and for A3) and the E4b lease covariates. A3 vs A1 "explained by contention" = A3 wins
  over A1 in exactly one of E4a and E4b and the E4b covariates show A3's runs losing more lease time.
- Every run is reported; reruns only for harness failures (no result.json, node/daemon fault), under a
  new id, logged in the ledger.

## Preparation (in order; steps 1-4 before the freeze)
1. **Benchmark pin.** `c9782832d264fecd8df4619d5e96ed83aadb77db` must be fetchable from
   `kamahori/vibe-megakernel` (branch `dsv41-dspark-tp4-case`). Bootstrap each slot:
   `SS_PROTOCOL=PROTOCOL_TP4.md bash scripts/bootstrap_node.sh --check-login` (TBD rows allowed here).
2. **Port the seed (builder).** On one slot: `bash scripts/run_main_tp4.sh port [MINUTES]` (default 240):
   `run_arm --arm tp4/PORT --run-id builder-port1 --no-calibrate --no-rescore`; the TP8 seed in a TP4
   workspace with overlays S + A1, prompt `prompts/goal_port_tp4.md`. Done when it has passing snapshots
   and NOTES.md has the "TP4 port" section. A failed attempt: `builder-port2`, ... (logged in the ledger).
3. **Freeze the seed.** `bash scripts/run_main_tp4.sh freeze --dry-run`, inspect the file list, then
   without `--dry-run`. Writes `seeds/dsv41_b8_tp4_nopf/` and prints its manifest row. Review the diff
   against the TP8 seed (it must be a port, not a new optimization), commit the seed (src/ is ignored;
   `src.tar.xz.b64`, `MANIFEST.sha256`, `SEED_SHA256`, `SOURCE.json`).
4. **b3.** Measure the TP4 strong PyTorch baseline the way the TP8 28.95 ms was measured (library kernels
   + torch.compile + CUDA graph + PDL) on one slot, set `TASKS['improve-b8-tp4']['b3_ms']` (and
   `port-b8-tp4`) in `scaffold_search/workspace.py`. If no such baseline is available, set b3 to the
   seed's calibrated L (score 1.0 = the seed) and say so in the report.
5. **Self-test on every slot:** `bash scripts/run_main_tp4.sh selftest` (stage 6 calibrates the TP4 seed and
   rescores it: L within 3% of the calibration).
6. **Freeze:** fill the two TBD rows below, commit, push; every slot `git pull`, rerun bootstrap, and
   `scripts/verify_protocol.py --protocol PROTOCOL_TP4.md --trusted /srv/ss/trusted/vibe-megakernel` must say
   all match.
7. **Run:** slots 0-3 `bash scripts/run_main_tp4.sh start e4a --slot I`; slot 4 `... start e4b`.

## Manifest (sha256)
`scripts/verify_protocol.py --protocol PROTOCOL_TP4.md` checks this table.

| item | value |
|---|---|
| trusted evaluator | vibe-megakernel c9782832d264fecd8df4619d5e96ed83aadb77db (branch dsv41-dspark-tp4-case) + gpu_noise_multirank.patch bb4871ccd876220359d5b635f88380955f62c3d61974967b5c33633018cb2f75 |
| seed dsv41_b8_tp4_nopf (MANIFEST sha256) | TBD (preparation step 3) |
| b3 improve-b8-tp4 (ms) | TBD (preparation step 4) |
| prompts/goal_improve_tp4.md | 481b138c93f473b6088d6d365d28e9af0a893547325f2b1b6fec1691985138f4 |
| prompts/goal_port_tp4.md | 898d12e29d3a7151f6ff0c99a4768c26aaf544532e3866f0f0f28337b99ce67d |
| arms/tp4/A0S.toml | 5081b5c904a8ea85f8473adf40da80cdd87cbf051cd4adb23aaad951ea3bb182 |
| arms/tp4/A1.toml | 97ee28e0dd8e313f06b6a04117f59b1ed07cc35e8871142e07ce136461fa8a7c |
| arms/tp4/A2.toml | 9137166a24b610fe4fe759d1c12f09b013ddbfb23e8e65f5efe9f419d23dce4e |
| arms/tp4/A3.toml | 086bc4b2c9336f34aa970fe4788ec22a20574e0b81fc1a89fd5745b77763b70b |
| arms/tp4/PORT.toml | 3c97f9513fd63697462479568602737f255584bf853d152bf20e2430875c7ff2 |
| overlays/S (tree) | cbbf7024ee35c63de79f4f17cacb93d990b8341d7253d452a19a1bb38fed6d70 |
| overlays/A1 (tree) | 771f126d3a1cf69b94b3e10a6671357e3fbbb25311e0a14a08a8994596e017cf |
| overlays/A2 (tree) | f4366b5a119e3e62120fcc7d14b52fd208d798e26bc4dc44ece6ac9bf5c5b2fd |
| overlays/A3 (tree) | bde3993408509397c467387210dc10145a28e2baf312d6859c219395c7b8e8cf |
| scaffold_search/harness_files (tree) | 15f43f203fbe56436ac0dbf372ac7950b85dbc4153177817afc45a7683293a59 |
| scaffold_search/gpu_lease.py | 8270d51fd19c4ca268a763ba3aa3451aff946cba343c3407e2d9fe40d80fc181 |
| Claude Code / model | 2.1.326 / claude-opus-5-5 |

Tree hash = sha256 over the sorted lines `relpath\0sha256(file)\n`.
