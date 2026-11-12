# Step 7: TP4 B64 immutability negative control on four GPUs (as expected), and the rank-local control review

| Item | Value |
|---|---|
| Case | `kimi-k3-syn-l8-tp4-b64-s8192` (4 GPUs, TP4, 8 layers, batch 64, context 8192, synthetic weights; graded, suite p1) |
| Candidate | `negative-controls/negctrl_immutability.py`, unchanged since its CPU review (sha256 `c9375eb89f9d2007bddedcd34a189e8146a4248c9a79e72248207d4572cf8952`, also in the result record): the oracle's output, then `inputs['l3_kv_cache'][0, 0, 0] += 1.0` on every rank |
| Runner commit | `c7773c13a774d09d208d7838889524991d2316b4` (evaluator unchanged: `git diff --stat 52f8d8c -- megabench setup sandbox` empty; worktree clean apart from the evidence directory) |
| Script | `negative-controls/step7-tp4-control.sh negctrl_immutability b64` (sha256 `9f50f6fd…e30d42`, copy in the directory): Step 5's script plus `GOALBOX_NEGCTRL_RECORD` and a per-rank summary; `CUDA_VISIBLE_DEVICES=0,1,2,3`, NCCL INIT/ENV/COLL logs per process, evaluator `--timeout 1500` inside `timeout -k 30 1800`, dedicated caches |
| Analyzer | `negative-controls/step7_analyze.py` (sha256 `d0d386d6…cec4c`), output `analysis.json` in the evidence directory |
| Where / when | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPUs 0 to 3, host level as `dc`, no sandbox, no agent; 2026-11-12 17:50:34 to 17:51:51 UTC (77 s) |
| Evidence | `step7-tp4-b64-negctrl_immutability-20261112T175029Z/` |
| Exit code | 1 (expected: only `ok_provisional` counts as passed) |

## Pre-checks required by the GO (all done before the GPU run, all on the CPU)

| Check | Result | Evidence |
|---|---|---|
| Snapshot and checksum implementation | `snapshot()` (`megabench/harness/correctness.py:13`) copies a tensor to the host when it is at most 64 MB, non-contiguous or not a multiple of 4 bytes; otherwise it views the tensor as int32 words and keeps two int64 sums on the device: the plain sum and the sum weighted by `(position % 65521) + 1`, in chunks of 2^26 words. `check_unchanged()` recomputes and compares with `torch.equal`. | code reading; `tp4_preflight.py` part A |
| 512 MB `l3_kv_cache` selects the checksum path | B64: `[64, 8192, 512]` BF16 = 536870912 bytes, contiguous, so the snapshot is an int64 pair; B1 (8 MB) and B8 (64 MB, exactly the limit) take the host copy | `tp4-preflight-20261112T173953Z/preflight.json` part A: `snapshot_path: device checksums (two int64)` for B64; the control's record on every rank: `tensor_bytes 536870912`, `snapshot_path: device checksums` |
| Single-element BF16 mutation detectable | A one-element change alters exactly one int32 word, so the plain sum changes by the word delta (nonzero whenever the bits change; adding 1.0 to a standard-normal BF16 value always changes the bits). Measured on the CPU with the evaluator's functions: before `[-2210483810699375, 1518804666980406477]`, after `[-2210483810731866, 1518804666980373986]`, both sums changed by -32491 (the element sits in word 0, weight 1); also detected for an element in the second 256 MB chunk (index 167772160) and for the last element; snapshot deterministic (two snapshots equal); mutate-then-revert passes | `preflight.json` part A, B64 row |
| Mutation after the oracle step and before the immutability check | In the control, `kimi.reference()` returns before the in-place add. In `_rank_worker` (`distributed.py:83-96`) the order is `snapshot` → `reference` → `candidate` → `check_unchanged` → `agree`; the mutation therefore happens between the snapshot and the check and after the outputs exist | code reading; the GPU trace below shows no collective between the candidate's last collective and the agreement all-gather |
| No collective sequence changed | The control adds only a local in-place add; every rank fails `check_unchanged`, so every rank carries a local error into the same `agree()` and none enters `compare()`. NCCL trace: 60 collectives per process, the ordered (kind, count, datatype, op) sequence identical on all four processes | `analysis.json`: `nccl_sequence_identical_across_processes: true` |
| CPU-only preflight | tiny TP4 geometry through the evaluator's own `evaluate_distributed()` on gloo (4 processes): positive control `correctness_only` with 4 ranks `pass`; immutability control 4 ranks `incorrect`, each reason naming all four ranks. Real B64 geometry through the CLI on the CPU (4 gloo processes, 1 trial): 4 ranks `incorrect`, same reason, records on the checksum path, 3 min 22 s, host peak 296 GB used | `tp4-preflight-20261112T173953Z/{preflight.log.txt, preflight.json, real-geometry-b64-negctrl_immutability/}` |

## Four-rank result

| Rank | Device | Status | Reason (identical on all ranks) |
|---|---|---|---|
| 0 | `cuda:0` (`GPU-2305f466…`) | `incorrect` | `AssertionError: rank 0: trial 0: AssertionError: candidate mutated input l3_kv_cache; rank 1: trial 0: AssertionError: candidate mutated input l3_kv_cache; rank 2: trial 0: AssertionError: candidate mutated input l3_kv_cache; rank 3: trial 0: AssertionError: candidate mutated input l3_kv_cache` |
| 1 | `cuda:1` (`GPU-4db89045…`) | `incorrect` | same |
| 2 | `cuda:2` (`GPU-651ce205…`) | `incorrect` | same |
| 3 | `cuda:3` (`GPU-a7a707cd…`) | `incorrect` | same |
| aggregate | | `incorrect` | rank 0's reason (`evaluate_distributed` reports `failures[0]["reason"]`) |

Every rank's own `check_unchanged()` failed (the reason lists all four ranks because `agree()` concatenates every rank's local error), the record file of every rank shows the mutation (0.0393 to 1.0391, 536870912 bytes, checksum path), no trial was recorded (`trials: 0`), and `compare()` was never entered. The result record keeps `submission_sha256`, `hardware` and `timestamp_utc` in the distributed path (the aggregate is merged into `evaluate_case`'s record), unlike the single-GPU failure path (ISSUES.md item 11).

## NCCL and infrastructure

| Check | Result |
|---|---|
| NCCL | 2.30.7, `Init COMPLETE` 3 per process, 0 `NCCL WARN`, 0 error lines, NVLink; 60 collectives per process: seed sharing and agreement (`AG64 AG8 BC64 BC8 AG64 AG8`), the oracle reference (24 float32-sum all-reduces, greedy max, token min), the candidate (the same 26), then the agreement all-gather carrying the four errors (`AG64 AG8`). Identical sequence on all four processes. |
| GPUs | four distinct GPUs by UUID, one pid per GPU, no foreign process; sampled memory 37620 MiB per GPU (Step 5 positive control: 41506 MiB, which includes the FP32 references the control never reached) |
| Host | +28 GB during the run (139 to 167 GB used of 958 GB), 814 GB available at the minimum, swap 0 |
| Kubernetes | 0 GPU-pod requests before, in all 3 polls during, and after |
| Cleanup | no leftover `megabench` processes; all four GPUs at 0 MiB after; stderr empty |
| Stop conditions (unexpected result, NCCL error, hang, interruption) | none |

## Rank-local routing control: review (preparation only, no GPU run)

Read before writing anything: `distributed.py::_rank_worker` and `agree()`, `kimi.py::compare` and `reference(routes=…)`, `parallel.py::initialize`. Three candidates were written and run only on the CPU (tiny TP4 geometry, gloo, `tp4-preflight-20261112T173953Z/`):

| Candidate | What it does on rank 2 only | Evaluator behaviour (predicted from the code, confirmed on the CPU) | Deadlock risk | Attribution to rank 2 |
|---|---|---|---|---|
| `negctrl_ranklocal_routing.py` | replaces one expert id in `expert_ids` (layer 1, row 0) by an id outside the set; no forced rerun | `compare()` first all-reduces `[valid?, checksum, -checksum, differs?]` with MAX over the TP group ("one global decision on the routes before any collective in the forced references"); the checksums disagree, so every rank raises the same `expert_ids: ids out of range, repeated in a row, wrong in a dense layer, or different between ranks` at the same point. The near-top-k shortfall rule is never reached. | none (identical raise on all ranks, before the forced references); tiny run 3.4 s | none: all four ranks report the same error with all four rank prefixes |
| `negctrl_ranklocal_logits.py` | perturbs row 0 of its own vocabulary shard (the TP1 formula) | the per-row relative L2 is built from sums all-reduced over the TP group, so every rank sees the same global number (0.28 for one shard in four, versus a limit of about 0.05) and fails `logits:` identically, after all grading collectives | none (compare collects failures and raises at the end); tiny run 3.4 s | none |
| `negctrl_ranklocal_immutability.py` | mutates `l3_kv_cache[0,0,0]` | only rank 2's `check_unchanged()` fails; `agree()` raises on every rank with exactly `rank 2: trial 0: AssertionError: candidate mutated input l3_kv_cache` | none (no collective added; the agreement happens on all ranks); tiny run 3.6 s | yes: the reason names rank 2 and no other rank, on every rank and in the aggregate |

Limitation of the current evaluator for the requested control: a rank-local routing corruption cannot reach the routing (near-top-k) check, because the grader requires identical routes on all ranks before it runs the forced FP32 references, which are collective. It also cannot be attributed to rank 2 in `ranks[].status` or `ranks[].reason`: every per-rank status after `agree()` is `incorrect`, and output corruptions are graded on all-reduced quantities, so the reason is the same on every rank. The only per-rank attribution the evaluator offers is the "rank N:" prefix of a rank's own local error (a candidate exception or a failed `check_unchanged()`), which the rank-local immutability control exercises. Doing the forced rerun on rank 2 only, to reach the shortfall rule, would add collectives on one rank and is a collective mismatch (hang until the 10-minute process-group timeout), so it was not written.

Proposal: run `negctrl_ranklocal_immutability.py` on TP4 B1 as the rank-local control (expected: 4 ranks `incorrect`, every reason exactly `rank 2: …`, aggregate `incorrect`, 4 records with `corrupted: true` on rank 2 only), and optionally `negctrl_ranklocal_routing.py` on TP4 B1 to exercise the cross-rank route-consistency branch that no run has hit yet (expected: identical `different between ranks` error on all ranks, no attribution). Both are deadlock-free by construction and by the CPU runs. The all-rank `negctrl_routing.py` at TP4 (CPU: shortfall 0.676 against allowed 0.0044 on every rank) would exercise the forced-reference path under NCCL if that is wanted.

## Caveats and observations for ISSUES.md

- Everything that goes through `agree()` becomes `incorrect`: `agree()` raises an AssertionError on every rank for any local error string, including input-generation failures and non-assertion exceptions in the candidate (observed on the CPU: an input-generation `ValueError` from my first, invalid tiny geometry produced status `incorrect` on all ranks). The routing control's near-tie guard (a RuntimeError) would therefore read `incorrect` at TP4, not `error` as at TP1 (item 14).
- Per-rank shape, dtype, device and key checks at the top of `compare()` raise before the first grading collective; a candidate that returns a wrong shape on one rank only would desynchronize the ranks (item 16). None of the prepared controls does this.

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt`, `analysis.json`, `env.txt` (includes case, submission sha256, NCCL version), `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `host-mem-monitor.log.txt`, `nccl-*-290{10,11,12,13}.log.txt` (one IP address redacted as in Step 5), `negctrl-record.rank{0,1,2,3}.jsonl`, `cleanup.txt`, `step7-tp4-control.sh`, `negctrl_immutability.py`. Caches (854 MB) not committed.
