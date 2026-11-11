# Step 1: evaluator smoke test with the oracle as the candidate

| Item | Value |
|---|---|
| Case | `kimi-k3-dev-l4-tp1-b8-s8192` (1 GPU, TP1, 4 layers, batch 8, context 8192, synthetic weights) |
| Candidate | `submission.py` in this directory: `build()` returns `kimi.reference` (the BF16 PyTorch oracle) |
| Repository commit | `7f3820ec32eee5cf98913962addf0a71c5dca780` (branch `gb300-tp4-synthetic`) |
| Where | rack 10 tray14 (`pod4-gb300-10-tray14-f3`), GPU 0 `GPU-2305f466-c3f2-22d8-138d-f36d3a9711d5`, host level, no sandbox, no agent |
| When (UTC) | 2026-11-11 18:24:28 to 18:25:43 (75 s wall clock) |
| Command | `command.txt` (evaluator defaults: 3 correctness trials, 2 warmup, 5 timed reps, `--timeout 1200`, outer `timeout 1500`) |
| Exit code | 1 (`exit_code`): the evaluator counts the run as 0 of 1 passed because the status is `non_megakernel`. Expected for the oracle. |
| Evaluator status | `non_megakernel` |

## Verdict per evaluator component

| Component | Verdict | Evidence (`result.jsonl`) |
|---|---|---|
| Submission loading through the normal interface | PASS | `build_ms` 3.4, `first_call_ms_including_jit` 365; `submission_sha256` `6f9e8a84…c906` |
| Fresh inputs per trial | PASS | 3 trials with independent seeds 2345031619, 3331680404, 2376876580 |
| FP32 band on logits, MLA cache writes, KDA states | PASS | candidate distance equals the oracle distance in every trial (same code); `worst_ratio` 0.41 to 0.48 of the band |
| `next_token` | PASS | 0 rows outside the tie margin in all trials; trial 2 has 1 row whose BF16 argmax differs from the FP32 argmax but is inside the margin |
| `expert_ids` near-top-k | PASS | `sets_differing_from_oracle` 0; shortfall 0.0012 to 0.0026 against allowed 0.0033 to 0.0061 |
| `cache_length` | PASS | `mismatched_elements` 0 in all trials |
| Input immutability | PASS | `check_unchanged` raised nothing (a mutation would have ended the run with status `error`) |
| Launch audit | FAIL, as expected | `launch_budget_failed`, 25128 GPU kernels against a budget of 1; mostly `at::native` elementwise and index kernels; `cuda_graph_runtime_hint` false; `review_required` true |
| Candidate timing | recorded, not a baseline | CUDA-event p50 346.1 ms, p95 379.7 ms; raw reps 345.5, 346.1, 350.3, 387.0, 343.7 ms |
| Reference (eager oracle) timing | recorded | CUDA-event p50 345.0 ms, p95 347.4 ms |
| CUDA-graph reference timing | not available | `graph_baseline_unavailable`: `cudaErrorStreamCaptureInvalidated` while capturing the oracle (see observations) |
| Overall | expected smoke-test outcome | correctness passes, the submission fails only on the launch count |

## Infrastructure

| Check | Result |
|---|---|
| Shared venv under `dc` | functional: `/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv` (Python 3.12.3 from `/usr/bin`), unit tests 7/7 before the run |
| Versions (`env.txt`) | driver 595.71.05; nvcc 13.2; PyTorch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24, bundled NCCL 2.30.7); Triton 3.8.0; system libnccl 2.27.7 (not used by this run); kernel 6.14.0-1013-nvidia-64k aarch64 |
| Caches | dedicated, under `cache/` in this directory (CUDA, Triton, Inductor, XDG); 884 MB after the run; not committed |
| GPU occupancy before | all 4 GPUs 0 MiB, no compute processes (`occupancy-before.txt`) |
| GPU occupancy during | `gpu-monitor.log.txt`: one process (ours, pid 3755630) on GPU 0 only, peak 63.7 GB; no other process on any GPU in 7 samples at 10 s |
| GPU occupancy after | all 4 GPUs 0 MiB, no compute processes (`occupancy-after.txt`) |
| Kubernetes scheduler | no pod with a GPU request on the node before or after; the in-script check failed (kubectl not on PATH), the post-run check (`occupancy-k8s-postcheck.txt`, 18:27 UTC) lists 17 pods on the node, all long-running infrastructure daemons started weeks earlier with `gpu=0`. Cluster events are not readable with this service account. |
| Classification | completed, not interrupted |

## Observations for review (not failures of this run)

1. The CUDA-graph reference baseline could not be captured for this case (`cudaErrorStreamCaptureInvalidated`). The evaluator records this and falls back to the eager reference, so `speedup_vs_best_baseline_cuda_event` compares against eager only. Worth checking whether the oracle has a host synchronization (for example `.item()` or data-dependent indexing) that breaks capture, and whether this also happens on B200.
2. `source_hints_advisory.mentions_oracle_module` is false although the submission imports the oracle (`from megabench.tasks import kimi`). The hint matches the literal string `megabench.tasks.kimi`, which this import form does not contain. The hint is advisory; the launch audit and `review_required` still flag the submission.
3. The evaluator's own exit code is 1 for a `non_megakernel` status, so scripts must read `result.jsonl` rather than the exit code to separate "correct but many launches" from an error.

## Files

`command.txt`, `exit_code`, `timing.txt`, `stdout.log.txt`, `stderr.log.txt` (empty), `result.jsonl` (full record including all 25128 kernel names), `env.txt`, `occupancy-before.txt`, `occupancy-after.txt`, `occupancy-k8s-postcheck.txt`, `gpu-monitor.log.txt`, `step1.sh` (the exact script that ran), `submission.py` (the candidate).
