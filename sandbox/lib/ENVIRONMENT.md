# Environment

You work inside a sandbox. These rules are part of the task.

- **Files.** You can see `/work` (your workspace, read-write), `/home/agent` (your home, read-write), `/tmp`, the system under `/usr` and `/etc`, the CUDA toolkit (`/usr/local/cuda`, with `nvcc`, `ptxas`, `cuobjdump`, `ncu`, `nsys`), and read-only `/goalbox/bench` (the trusted benchmark that the check scripts use). Nothing else from the host machine is present.
- **Python.** `python` is `/home/agent/venv/bin/python`. It has PyTorch, Triton, NumPy, safetensors and Ninja. `pip install` works for most packages. Packages that ship ready-made LLM inference, attention, MoE or quantized-GEMM kernels (for example vllm, sglang, flashinfer, flash-attn, flash-linear-attention, CUTLASS, TensorRT) are refused.
- **Network.** There is no general internet access. Only the Claude API, the filtered package index and Hugging Face (huggingface.co and its file storage) are reachable. Web search and web fetch are disabled. Use `huggingface_hub` (`pip install huggingface_hub`, then `hf download REPO --local-dir DIR`) for model configs, model code and weights. There is no Hugging Face token, so gated repositories are not available. Put large downloads under `/home/agent/hf`, not in `/work`. The GPU jobs see the same files.
- **GPU.** This sandbox has no GPU. To run something on GPUs, use `gpu-run`:

  ```
  gpu-run -t 15 -- python my_test.py               # 1 GPU; wait; prints the output; exit code = command's
  gpu-run -g 4 -t 30 -- ./my_tp4_test.sh             # all 4 GPUs of one node
  id=$(gpu-run -d -g 4 -t 60 -- ./check_candidate --case kimi-k3-syn-l8-tp4-b8-s8192)   # detach
  gpu-run --wait "$id"                                # later: wait and print the output
  gpu-run --status                                    # recent requests and their state
  ```

  Each request runs on the GPUs of one node. On a GB300 NVL72 tray that is 4 NVIDIA GB300 GPUs (B300 silicon, sm_103a, 288 GB HBM3e each) on an aarch64 Grace host; a request for more GPUs than the node has is rejected. Compile device code for sm_103a (for example `nvcc -gencode arch=compute_103a,code=sm_103a`); sm_100a binaries do not run there. It runs in this same filesystem view, with the same paths, a private `/dev/shm`, and no network. Requests run one at a time, and a request can wait in a queue before it starts. Give an honest, short `-t` (minutes, max 120). Compile in the sandbox with `nvcc`, then run on the GPU. Your Bash tool call times out after a few minutes, so use `-d` and `--wait` (or run it in the background) for anything that can queue or run long.
- **Checks.** `./check_candidate --case ID` sends itself through `gpu-run` with the GPU count of the case (set `GPU_MINUTES` to change its time limit). It runs correctness, the launch audit and timing. The real-weight cases load about 200 GB per GPU before the first trial, so use the synthetic and development cases for most iterations. `./sol_info --case ID --json` runs in the sandbox. Final grading is done separately, on fresh inputs, outside this sandbox.
