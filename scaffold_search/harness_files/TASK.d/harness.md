# Harness notes (identical in every run)

- **GPUs are shared** with other jobs on this node through a lease queue. `./check_candidate`,
  `./profile_candidate` and `./timeline_candidate` take an 8-GPU lease themselves (they may wait in
  the queue first). Run every other GPU command through `./gpu_run -g N -- CMD` with N = 1, 2, 3 or
  8, for example `./gpu_run -g 8 -- torchrun --nproc_per_node 8 devtest.py` or
  `./gpu_run -g 1 -- ./mb/mb`. Outside `./gpu_run` the GPUs are not accessible. A lease ends when
  its command exits or after its time limit (8 GPUs: 20 min, smaller leases: 60 min; pass
  `-m MINUTES` to ask for up to 30 / 120), which kills the command.
- A 1-3 GPU lease can be ended after 10 min when another job has been waiting for all 8 GPUs
  (`./gpu_run` then prints why). Keep long jobs under 10 min or checkpoint them. A lease whose
  GPUs stay unused (no GPU process, 0% utilization) for 5 min is also ended.
- Ask for 1 GPU when one is enough: small leases start sooner, and you may hold up to 3 GPUs in
  small leases at once (for example, three 1-GPU microbenchmarks in parallel).
- Compile before you take a lease: a compile error inside a lease wastes queue time.
  `./check_candidate` first runs `./prebuild` (if it exists) without GPUs, so the nvcc build does
  not hold the 8 GPUs; keep `./prebuild` building exactly what `submission.py` loads (the provided
  one calls `mk_host.load_ext()`).
- `./check_candidate` here times 1 evaluator process; the final re-measurement of your best passing
  check uses 3 fresh processes. Correctness, launch audit and timing are the trusted evaluator's.
- Every passing `./check_candidate` is snapshotted by the harness. Your final result is the best
  passing check, re-measured afterwards.
- The machine has no internet access; work with what is in this directory and the installed
  toolchain (CUDA 13.0 at /usr/local/cuda, Nsight Compute, PyTorch, python3.12).
