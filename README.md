# kimi-k3-goalbox

This repository is a harness for a Claude goal-mode baseline.
In the baseline, one Claude Code agent writes a megakernel for one Kimi K3 decode step.
The target is one node of 8 NVIDIA B300 GPUs with tensor parallel 8 (TP8).

## The harness

The harness has four parts:

1. **Sandbox.** The agent runs in a bubblewrap (`bwrap`) sandbox. The agent can see only its workspace, its home, the CUDA toolkit and a read-only copy of the benchmark.
2. **Network filter.** A proxy lets the agent reach only the Claude API, Hugging Face and a filtered package index. The index does not supply packages that contain LLM kernels, for example vLLM, FlashInfer and CUTLASS.
3. **GPU broker.** The sandbox has no GPU. The agent sends GPU work with `gpu-run`. The host broker runs the work as a Slurm job or on local GPUs, in a different sandbox with no network.
4. **Benchmark (MegaBench).** A PyTorch oracle gives the math of one Kimi K3 decode step. The benchmark does a check of correctness, counts the GPU kernel launches and measures the latency.

The agent gets a goal prompt (`goals/kimi-k3-tp8.txt`) and works until it gets to the goal.
The goal is one kernel launch for each rank, correct output on all graded cases, and a latency near the HBM floor.

### Graded cases

| Case | Weights | Batch | Layers |
|---|---|---|---|
| `kimi-k3-tp8-b{1,8,64}-s8192` | Real checkpoint | 1, 8, 64 | 93 |
| `kimi-k3-syn-l8-tp8-b{1,8,64}-s8192` | New synthetic weights for each trial | 1, 8, 64 | 8 |

All graded cases use TP8 and a context of 8192 tokens.
The real-weight cases show the latency of the real model.
The synthetic cases use new random weights for each trial.
Thus, a submission that keeps answers for one set of weights does not pass these cases.
Two small cases (`kimi-k3-dev-l4-tp1-b8-s8192` and `kimi-k3-dev-l8-tp2-b8-s8192`) are for development.

### Accuracy check (the oracle band)

The grader runs the BF16 oracle and an FP32 version of the same math.
The expert routes are discrete. Thus the FP32 version uses the expert sets of the oracle, and then the expert sets of the candidate.
The grader then applies these rules:

- For each float output, the distance from the candidate to FP32 must be at most 2 times the BF16 oracle distance, plus 0.002.
- The `next_token` must be the FP32 argmax, or a token in the near-tie margin.
- Each expert set of the candidate must be a near-top-k set under the FP32 router scores.
- The `cache_length` must be equal.
- The candidate must not change its inputs.

## Procedure

1. Install the Python environment (system Python, PyTorch for CUDA 13):

   ```
   setup/install.sh
   ```

2. Download the Kimi K3 checkpoint at the pinned revision (approximately 1.6 TB):

   ```
   setup/download_weights.sh /path/to/hf
   ```

3. Convert the checkpoint to the TP8 layout of the oracle. Each rank gets approximately 200 GB:

   ```
   setup/convert_checkpoint.sh /path/to/hf /path/to/weights 8
   export KIMI_K3_WEIGHTS=/path/to/weights
   ```

4. Do a test of the installation and the sandbox:

   ```
   setup/selftest.sh
   ```

5. Optional: measure the vLLM baseline on the 8 GPUs. This step writes `baseline/BASELINES.json`, and the workspace gets a copy:

   ```
   baseline/install_vllm.sh
   .venv-vllm/bin/python baseline/measure_vllm.py --model /path/to/hf
   ```

6. Make a run and start the agent. The token file holds a Claude Code OAuth token:

   ```
   sandbox/bin/goalbox-new kimi-run1
   sandbox/bin/goalbox-start kimi-run1 /path/to/token-file
   ```

7. When the agent stops, grade the run on new inputs, out of the sandbox:

   ```
   sandbox/bin/goalbox-grade kimi-run1
   ```

8. To stop a run before the end, use `sandbox/bin/goalbox-stop kimi-run1`.

Runs go in `runs/` (or in `GOALBOX_RUNS`).
Set `GOALBOX_GPU_MODE=local` if the node has no Slurm.
With Slurm, set `GOALBOX_PARTITION` (default `main`) and, if necessary, `GOALBOX_SBATCH_ARGS`.

## Repository files

### `setup/`

| File | Contents |
|---|---|
| `install.sh` | Makes `.venv` with PyTorch (CUDA 13), NumPy, safetensors, `huggingface_hub` and Ninja. |
| `download_weights.sh` | Downloads `moonshotai/Kimi-K3` at the pinned revision, with retries, and counts the shards. |
| `convert_checkpoint.sh` | Converts the Hugging Face checkpoint to one file set for each TP rank. |
| `selftest.sh` | Runs the unit tests, the sandbox isolation checks and a GPU check in the sandbox. |

### `sandbox/`

| File | Contents |
|---|---|
| `bin/goalbox-new` | Makes a run directory, the agent workspace and the agent Python environment. |
| `bin/goalbox-start` | Starts the proxy, the broker and Claude Code in goal mode in the agent sandbox. |
| `bin/goalbox-stop` | Stops all the processes of a run. |
| `bin/goalbox-grade` | Grades the committed submission of a run on all graded cases. |
| `lib/sandbox.sh` | The `bwrap` command line for the agent mode and the GPU mode. |
| `lib/netproxy.py` | The allowlist proxy and the filtered package index. |
| `lib/broker.py` | The host broker. It runs `gpu-run` requests as Slurm jobs or on local GPUs. |
| `lib/ENVIRONMENT.md` | The sandbox rules that the agent reads. |
| `sandbox-bin/gpu-run` | The client that the agent uses to send GPU work to the broker. |
| `blocked-packages.txt` | The package names that the filtered index does not supply. |

### `megabench/`

| File | Contents |
|---|---|
| `tasks/kimi.py` | The Kimi K3 oracle: weight shapes, synthetic weights, state, BF16 and FP32 reference, and the band grader. |
| `tasks/kimi_checkpoint.py` | The checkpoint converter and the loader of the converted weights. The docstring has the name map. |
| `tasks/common.py`, `quantization.py` | Normalization helpers and the MXFP4 decode. |
| `tasks/parallel.py`, `distributed.py` | Rank mapping, process groups and the seed helper. |
| `tasks/workloads.py`, `workloads.py` | The task table and the input, reference and compare functions. |
| `tasks/model_specs.json` | Model geometry data. |
| `cases.py` | All the cases and the list of graded cases. |
| `sol.py` | The HBM traffic model. It gives the per-GPU byte count and the latency floor at 8 TB/s. |
| `harness/runner.py` | The command line (`python -m megabench evaluate`, `sol`). |
| `harness/correctness.py` | Correctness trials and the check of the inputs for changes. |
| `harness/distributed.py` | The multi-GPU harness. It starts one process for each rank. |
| `harness/benchmark.py` | Timing with CUDA events and the kernel launch audit. |
| `integrations/make_kimi_workspace.py` | Makes the agent workspace: `TASK.md`, `cases.json`, a copy of the oracle, the submission stub and the tools. |
| `integrations/ncu_profile_case.py`, `ncu_capture.sh` | Nsight Compute capture of one warm step. |
| `tests/test_kimi.py` | CPU unit tests of the oracle and the band grader. |
| `runner.py`, `__main__.py` | Entry points. |

### Other files

| File | Contents |
|---|---|
| `goals/kimi-k3-tp8.txt` | The goal prompt for the agent. |
| `baseline/install_vllm.sh` | Installs vLLM with the Kimi K3 latency-specialized decode path in `.venv-vllm`. |
| `baseline/measure_vllm.py` | Measures the vLLM decode latency for each graded case and writes `BASELINES.json`. |
| `tools/verify_kimi_vs_hf.py` | Compares the oracle with the official Hugging Face model code on 4 real layers. |
| `tools/VERIFY_KIMI.md` | The results of this comparison. |
| `NOTICE` | The source of the copied MegaBench code. |

## Requirements

- One node with 8 B300 GPUs (sm_103a, 288 GB each) and approximately 2 TB of free disk. Each rank holds approximately 200 GB of weights, so the real-weight cases do not fit on B200 (180 GB). The harness was developed and its synthetic cases were tested on B200 (see `tools/VERIFY_KIMI.md`).
- Linux with `bwrap` (bubblewrap) and the CUDA 13 toolkit.
- Slurm, or local GPUs with `GOALBOX_GPU_MODE=local`.
- Claude Code (`claude` on `PATH`, or `GOALBOX_CLAUDE_BIN`) and an OAuth token.

CAUTION: Do not put tokens in the repository. Keep each token in a file outside the repository, with mode 600.
