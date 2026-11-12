# Step 6: TP1 negative controls on the single-GPU dev case (all four as expected)

Four separate clean evaluator invocations of `kimi-k3-dev-l4-tp1-b8-s8192` (1 GPU, 4 layers, batch 8, context 8192, synthetic weights; dev case, not graded), one per control, with the unmodified evaluator at commit `63f6eb8`. Evidence: `step6-tp1-<control>-<ts>/` beside this file (one directory per invocation). Analyzer: `negative-controls/step6_analyze.py` (sha256 `192722d3982b3a050b0ff08ce279a9a46d54c0b4850339a75a9b20e21be346fe`), output `analysis.json` in each directory.

| Item | Value |
|---|---|
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPU 0 `GPU-2305f466…`, host level as `dc`, no sandbox, no agent |
| When (UTC) | 2026-11-12 15:17:45 to 15:26:01, four runs of 53 to 73 s wall clock with a few minutes between them |
| Runner commit | `63f6eb8206c730f0f56449179044845f5e36689c` (`git diff --stat 52f8d8c -- megabench setup sandbox` empty: oracle, grader, launch audit and methodology unchanged) |
| Script | `negative-controls/step6-tp1-control.sh` (sha256 `a943330e…eb5a`, copied into each directory): evaluator defaults (3 trials, 2 warmup, 5 timed reps), `--timeout 1500`, outer `timeout -k 30 1800`, `CUDA_VISIBLE_DEVICES=0`, dedicated caches, `GOALBOX_NEGCTRL_RECORD` set, abort if any GPU process or GPU-pod request exists |
| Exit code | 1 for all four (only `ok_provisional` counts as passed, ISSUES.md item 8); the verdicts come from `result.jsonl` |

## Verdicts

| Control (submission sha256) | Expected | Observed status | Observed reason (from `result.jsonl`) | Wall |
|---|---|---|---|---|
| `positive_control_oracle.py` (`6f9e8a84…c906`) | `non_megakernel`, correctness `pass` on 3 trials, launch audit `launch_budget_failed` | `non_megakernel`, correctness `pass`, `launch_budget_failed` | (none) | 73 s |
| `negctrl_logits.py` (`b122e3fb…cfaf`) | `incorrect`, reason `logits: …` | `incorrect` | `AssertionError: logits: {'candidate_rel_l2_max': 0.5002, 'oracle_rel_l2_max': 0.0318, 'worst_ratio': 14.78}` | 61 s |
| `negctrl_routing.py` (`94c98465…6a84`) | `incorrect`, reason `expert_ids: …`; near-tie guard would be `error` (inconclusive) | `incorrect`, guard not triggered | `AssertionError: expert_ids: {'candidate_max_shortfall': 0.3399, 'oracle_max_shortfall': 0.00123, 'allowed': 0.00345, 'sets_differing_from_oracle': 1}` | 58 s |
| `negctrl_immutability.py` (`c9375eb8…8952`) | `incorrect`, reason `candidate mutated input l3_kv_cache` | `incorrect` | `AssertionError: candidate mutated input l3_kv_cache` | 53 s |

The `reason` field is `f"{type(e).__name__}: {e}"`, so the grader's own message follows `AssertionError: ` (ISSUES.md item 12). Each negative control stopped at its first trial, as designed by the evaluator: nothing after the failing check ran, so the three `incorrect` records have no `correctness`, `launch_audit`, timing or `hardware` fields (ISSUES.md item 11); the submission sha256 is in each `env.txt`.

### Positive control, detail

| Check | Verdict | Evidence |
|---|---|---|
| Three seeded trials | PASS | seeds 39328672, 2141071806, 649274595; worst band ratios logits 0.483, kv_write 0.474, pe_write 0.472, recurrent_state 0.425, conv_state 0.428 |
| `next_token` | PASS | 0 rows outside the tie margin (trial 3: 1 of 8 rows differs from the FP32 argmax inside the margin) |
| `expert_ids` | PASS | 0 sets differing; shortfall 0.0017 to 0.0037 against allowed 0.0044 to 0.0085 |
| `cache_length`, input immutability | PASS | 0 mismatched elements; `check_unchanged` raised nothing |
| Launch audit | FAIL, as expected | `launch_budget_failed`: 24927 GPU kernels against a budget of 1 (Step 1, same case, other seeds: 25128), `review_required` true |
| Timing (diagnostic, not a baseline) | recorded | candidate CUDA-event p50 377.4 ms, eager reference p50 365.2 ms, 0.97x; CUDA-graph baseline unavailable again (`cudaErrorStreamCaptureInvalidated`, ISSUES.md item 3); `build_ms` 5.6, first call 361 ms |

### Negative controls, detail

- Logits (`negctrl-record.rank0.jsonl`): row 0, every 4th index of the 163840-entry vocabulary (40960 indices), delta 1.0061 = 1.0 × row RMS, relative L2 change of the row 0.5; finite, no NaN or inf, no shape or dtype change. The grader's `candidate_rel_l2_max` 0.5002 matches the record; the oracle's own error on that trial was 0.0318, so the perturbation exceeded the band by a factor of 14.8. The failure comes from `kimi.compare` via `correctness.compare` (traceback), that is the numerical band check, on trial 0.
- Routing (record): layer 1, row 0, slot 0, original expert 64 replaced by expert 0 (the smallest id outside the row's top-k set); the replacement's score is 0.339 below the k-th best, far above the 0.05 guard, so the near-tie guard did not trigger; `max_shortfall_other_rows` 0.0 (no other row touched), `max_shortfall_same_row_later_layers` 0.0297 (the replaced expert changes row 0's hidden state in later layers, inside the allowance there). The grader reports exactly that shortfall (0.3399) against an allowance of 0.00345 and one set differing; no logits failure, because the control returns the oracle rerun on the corrupted routes. Failure from `kimi.compare`, trial 0.
- Immutability (record): `l3_kv_cache[0, 0, 0]` changed from -0.2246 to 0.7734 (delta 1.0 in BF16), tensor 67108864 bytes = the 64 MB host-copy limit, so the exact host compare path was exercised. The failure comes from `check_unchanged` (`correctness.py:45`, called from `check_trials:117`), before `compare` ran; classification `incorrect`, not `error`. The input tensors live only in the evaluator process; the runner's working tree was unchanged after the run (`git status`), and the next run regenerated its inputs.

## Correctness rejection versus launch-budget rejection

The evaluator has two independent gates. Correctness (`check_trials`: immutability, then the per-output comparison) runs first; an AssertionError there ends the evaluation with status `incorrect` and the grader's message as the reason, which is what all three negative controls produced. Only a candidate that passes correctness reaches timing and the launch audit; a correct candidate that is not a single kernel gets `non_megakernel` with `launch_audit.status = launch_budget_failed` and a `pass` correctness block, which is what the positive control produced here and in Steps 1 to 5. The two outcomes are distinguishable from the structured record (`status`, `correctness`, `launch_audit`) even though both exit with code 1.

## Infrastructure

| Check | Result |
|---|---|
| Idle before each start | GPU 0 (and 1 to 3) 0 MiB, no compute process; 17 pods on the node, 0 GPU requests; checked by hand before the first run and by the script before every run (`occupancy-before.txt`) |
| During | GPU monitor every 10 s: only one pid (ours) on GPU 0, peak sampled memory 63698 MiB (positive control) and 63694 MiB (negatives, same inputs and oracle); Kubernetes poll every 30 s: 0 GPU requests in every sample (2 to 3 samples per run; polling, not continuous); host memory grew 2.9 to 6.3 GB per run and was released after each (149 to 165 GB used of 958 GB, swap 0) |
| After each run | GPU 0 0 MiB, no compute process, 0 GPU requests; `cleanup.txt`: no leftover `megabench` processes |
| stderr | empty for all four runs; no warnings |
| Stop conditions (unexpected result, interruption, unexplained error, near-tie guard) | none triggered; no retry, no seed change |
| Caches | dedicated under `cache/` in each directory (0.8 to 0.9 GB each), not committed |
| Classification | four runs completed, not interrupted |

## Caveats carried from the CPU review

- The routing control's near-tie guard fired on the tiny 128-expert review geometry with random seeds (margin 0.029 below 0.05); on the real geometry it has now been clear by a wide margin on the CPU (0.71, 0.54) and on the GPU (0.34). If it fires in a future run, the result is `error` with the control's own message and must be reported as inconclusive, not as a rejection and not as a pass.
- Timing and launch counts of the negative controls are not recorded by the evaluator (they stop at trial 0) and would be meaningless anyway (the routing control runs the oracle twice).
- Only TP1 has been exercised on a GPU. The distributed path (`_rank_worker`, `agree()`, per-rank `ranks[].status`/`reason`) has only seen the positive control (Steps 2 to 5).

## Files

Per directory: `command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl.gz`, `result-summary.txt`, `analysis.json`, `env.txt` (includes the submission sha256), `occupancy-before.txt`, `occupancy-after.txt`, `gpu-monitor.log.txt`, `k8s-monitor.log.txt`, `host-mem-monitor.log.txt`, `cleanup.txt`, `step6-tp1-control.sh`, `<control>.py`, and `negctrl-record.rank0.jsonl` for the three negative controls.
