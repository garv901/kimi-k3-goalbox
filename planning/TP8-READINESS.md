# TP8 and real-checkpoint readiness assessment (2026-11-13, CPU-only; nothing launched)

Verdict: **not ready**. The per-rank fixture of the real-weight cases fits in a GB300's memory with headroom, and both
racks have the pinned checkpoint on disk, but (1) the harness can only start all ranks on one host and no GB300 tray has
8 GPUs, (2) the converted per-rank weight layout does not exist anywhere, (3) there is no sandbox for agent-generated
code, (4) the evaluator issues in `EVALUATOR-HARDENING-PLAN.md` are open, and (5) several quantities below are
estimates that must be measured before a real-weight run. "TP4 synthetic B64 fits" was not used as an argument for
anything here; every number is recomputed for the TP8 cases.

Legend: **M** measured in this campaign (evidence directory given), **D** derived exactly from the evaluator's own code
or from public metadata, **E** estimate (assumption stated), **U** unknown.

## 1. Eight-GPU availability and placement

| Item | Status | Value |
|---|---|---|
| GPUs per GB300 tray | M | 4 (`nvidia-smi`, Slurm `gres gpu:gb300:4`), on both racks |
| Harness launcher | D | `evaluate_distributed` spawns `case.gpus` processes on **one host** (`mp.spawn`, `cuda:first_device+rank`) and returns `unavailable` when fewer GPUs are visible (`distributed.py:148-155`). No multi-node rendezvous, no `torchrun`/Slurm integration. The README states this limitation. |
| Two-tray option | M (fabric) / U (harness) | IMEX is active on both racks with 18 nodes configured; rack 10 exposes `/dev/nvidia-caps-imex-channels/channel0`; other users run 2-node `nvl72` Slurm jobs on rack 3 right now, and a previous SGLang Kimi-K3 TP8 deployment ran on two rack-10 trays with `NCCL_MNNVL_ENABLE=1` (infrastructure handoff notes, 2026-11-08). The harness would need a multi-node launcher (an evaluator change) and the sandbox's `gpu` mode already binds the IMEX channel device. Not tested with this evaluator. |
| Single 8-GPU node option | M (labels) / U (access) | Kubernetes node `k8s-b300-gpu-worker-02-vm-f3`: 8 x `NVIDIA-B300-SXM6-AC`, 275040 MiB each (268.6 GiB, 9168 MiB less than a GB300), compute 10.3, driver 580.126.09, **x86-64 virtual machine**; no GPU pods on it at sampling. Unknown: NVLink topology inside the VM, host RAM, storage, who may use it, and whether an x86 image with CUDA 13 PyTorch exists. The repository's venv is aarch64 and cannot be reused. |
| Rack 3 Slurm occupancy at sampling | M | idle: tray10 (off-limits by operator rule) and tray11; mixed: 01-04, 07-09, 13, 14; allocated: 05, 06 (off-limits), 12; jobs on 15, 16, 18. Two free trays at the same time were not available at sampling. |
| Rack 10 at sampling | M | only tray03 had GPU pods (4); host-level work on other trays is possible but preemptible by the scheduler (seen on 2026-11-02). |
| Reduced-depth real-weight case on one tray | D | at TP4 the real checkpoint is about 4.1 GiB per layer per rank (190.97 GiB x 8 / 93 / 4, plus 1.1 GiB of `embed`/`lm_head`); 48 layers would be about 198 GiB per rank and fit a GB300; 93 layers (382 GiB) do not. Such a case does not exist in `cases.py` (policy decision D-T3). |

## 2. NVLink and NCCL connectivity

| Item | Status | Value |
|---|---|---|
| Intra-tray topology | M | `nvidia-smi topo -m`: every GPU pair `NV18`; GPUs 0,1 on NUMA node 0, GPUs 2,3 on NUMA node 1 (both racks); per-link rate 53.125 GB/s (`nvidia-smi nvlink -s`) |
| NCCL used by the evaluator | M | PyTorch's bundled NCCL 2.30.7 (venv), system NCCL 2.31.2 on rack 3; Steps 2 to 8: `P2P/CUMEM` transport, 0 warnings, identical collective sequences across ranks |
| Collectives per decode step | M | 26 at TP2 and TP4 for 8 layers (24 float32 sums, 1 float32 max, 1 int64 min); for 93 layers: D 3 per MoE layer + 2 for the dense layer + 1 embedding + 2 greedy = about 281 per step (`kimi.py` `sum_value` calls), same count at TP8 |
| Inter-tray (MNNVL) | M elsewhere / U here | works for other users' jobs and the SGLang deployment; never exercised by this evaluator; requires the multi-node launcher |
| Inside the x86 VM node | U | |

## 3. Checkpoint size and shard placement

| Item | Status | Value |
|---|---|---|
| Pinned revision | D | `moonshotai/Kimi-K3` at `f831ab66814297da540d832a5235f8e904f29d06` (`kimi_checkpoint.py`, `model_specs.json`); `main` resolves to the same commit on 2026-11-13 |
| HF copy on CephFS (`/mnt/shared/weights/kimi-k3`, owner `dc`, readable) | M | 96 shards, 1,560,936,091,448 bytes; `config.json` and `model.safetensors.index.json` sha256 equal to the files served by Hugging Face at the pinned revision; shard contents not hashed; 497,220 tensors in the index, metadata total 1,560,860,324,864 bytes |
| HF copy on rack 10 tray14 local RAID (`/opt/verda/weights/moonshotai/Kimi-K3/2496450`, root-owned, readable) | M | same file set and the same two hashes; 7.4 TB free on that RAID |
| Converted per-rank layout (`<out>/tp8/rank<R>/{global,layer000..092}.safetensors`) | M | **not present** anywhere under `/mnt/shared` (depth-4 search); must be produced by `setup/convert_checkpoint.sh HF_DIR OUT_DIR 8` |
| Converted size | D | 190.97 GiB per rank x 8 = 1527.8 GiB = 1.64 TB (`weight_shapes` byte count; the README's "about 1.6 TB" agrees); per layer about 2.05 GiB per rank |
| Conversion cost | E | reads the 1.56 TB HF copy (every shard, since the converter opens tensors by name across shards for each layer) and writes 1.64 TB; with 4 worker processes and CephFS at the measured 431 MB/s single-stream read, hours; from the local RAID (29 GB/s read) the writer side dominates. Not measured; the converter is restartable per layer. |
| Shard placement for a run | E | the loader reads `tp8/rank<R>/*.safetensors` straight to the device (`load_file(device=...)`), 191 GiB per rank. From CephFS at 431 MB/s per stream that is about 8 minutes per rank if streams do not share bandwidth (unknown); the SGLang deployment's full-model load from CephFS took about 20 minutes for 8 ranks on two trays. From a local RAID copy (29 GB/s) under a minute per tray. A local copy of the converted layout per participating tray (1.64 TB, or 0.82 TB per tray for its 4 ranks) is the practical placement; tray14's RAID has room. |

## 4. GPU memory per rank

| Quantity (per rank) | Status | TP8 B1 | TP8 B8 | TP8 B64 |
|---|---|---|---|---|
| Weights (fixture layout, resident for the whole evaluation) | D | 190.97 GiB | 190.97 GiB | 190.97 GiB |
| Per-sequence state (regenerated per trial) | D | 0.26 GiB | 2.11 GiB | 16.89 GiB |
| Evaluator and oracle working set above weights + state | E from M | TP4 B64 synthetic: `peak_allocated_bytes` 34.69 GiB versus 31.46 GiB computed (+3.2 GiB); process memory sampled by `nvidia-smi` 40.5 GiB (+9.1 GiB including context and allocator cache). TP2 dev: +2.7 GiB / +8.7 GiB. Assume +10 GiB at TP8 (the FP32 reference casts one weight at a time, largest 560 MiB; the eager MoE loop decodes one expert at a time). |
| Expected process memory with the oracle as candidate | E | about 201 GiB | about 203 GiB | about 218 GiB |
| GB300 capacity | M | 276.6 GiB (`gpu_memory_bytes` 297,013,411,840) |
| Headroom for the candidate's own buffers on a GB300 | E | about 75 GiB | about 73 GiB | about 58 GiB |
| Headroom on the x86 VM node's B300 (268.6 GiB) | E | about 67 GiB | about 65 GiB | about 50 GiB |
| Current `snapshot()` transients | D | two 512 MiB int64 temporaries plus a 512 MiB weight vector per large tensor call |

Consequence for a candidate: a second, re-laid-out copy of the weights does not fit; a megakernel must consume the fixture
layout in place (MXFP4 blocks `[E, rows, cols/32, 16]` and `[E, rows, cols/32]` scales, BF16 projections as declared).
`peak_allocated_bytes` is recorded per rank only in the distributed path and reserved memory is not recorded (ISSUES 6).

## 5. Host RAM, including startup and profiler overhead

| Quantity (per tray of 4 ranks) | Status | Value |
|---|---|---|
| Installed / available at sampling | M | 956 GiB / 811 GiB (rack 3 tray09); 958 / 823 (rack 10 tray14) |
| Process startup (4 ranks) | M | +22 GB in the first 20 s of Step 5 |
| Snapshot host copies per trial (tensors at most 64 MiB) | D | TP8: 32.2 GiB (B1), 34.1 GiB (B8), 36.8 GiB (B64) per rank, so 129 to 147 GiB per tray, transient per trial (`del originals`), transfer time not measured (NVLink-C2C) |
| Profiler events of the audited call | M at TP4 l8 B64: 286,253 events per rank, +33 GB host for the timing and audit phase of 4 ranks | E for an oracle-as-candidate at 93 layers: about 11.6x, roughly 3.3 million events per rank and a few hundred GB per tray; **negligible for a real megakernel candidate** (a handful of events). An oracle-as-candidate validation at TP8 therefore needs either fewer layers or acceptance of several hundred GB of host memory. |
| Whole-run growth observed | M | Step 5 (TP4 l8 B64, oracle as candidate): +78 GB, peak 213 GB used |

## 6. Input snapshot costs (current implementation)

| Quantity (per rank, per snapshot; two per trial) | Status | TP8 B1 / B8 / B64 |
|---|---|---|
| Host-copy path | D | 2232 / 2232 / 2208 tensors, 32.2 / 34.1 / 36.8 GiB device-to-host |
| Device-checksum path | D | 278 / 278 / 302 tensors, 159.0 / 159.0 / 171.0 GiB read, 42.7 / 42.7 / 45.9 x 10^9 int32 words, about 680 to 730 chunks of 2^26 words, roughly 4 kernels and two 512 MiB temporaries per chunk |
| Time | E | the device pass is a few hundred milliseconds at HBM rates; the host copies dominate and depend on the copy rate (unmeasured) |
| After the replacement (S1 to S3 in the plan) | E | S1 for synthetic cases replaces the host copies by regeneration; S2/S3 keep one device pass; the real-weight digest can be computed once per process at load and checked after every trial |

## 7. Model-loading strategy

| Item | Status | Value |
|---|---|---|
| Loader | D | `kimi_checkpoint.load_weights`: `safetensors.load_file(path, device=cuda:N)` per file (1 global + 93 layer files), stacks six per-layer vectors, validates names, shapes and dtypes against `weight_shapes`; cached per process in `_CHECKPOINT_CACHE` (loaded once, reused by every trial) |
| Placement | D | `KIMI_K3_WEIGHTS` points at the converted root; the sandbox's `gpu` mode binds it read-only at `/weights` |
| Load time | E | see section 3: minutes from CephFS per rank, well under a minute from a local RAID copy |
| Evaluator timeout | D | `--timeout` default 180 s is far below the load time; `goalbox-grade` and `check_candidate` use 3600 s; the process-group timeout of 10 minutes also covers initialization only loosely (ranks that finish loading early wait in `initialize`/`agree`, which happen before and after loading respectively, so a slow rank beyond 10 minutes would time out the others) |
| Development path | D | `convert_checkpoint.sh HF OUT TP --layers N` converts the first N layers; `_case(layers, tp)` in the converter and `kimi.weight_shapes` accept any TP that divides heads, vocabulary and FFN widths (TP1, TP2, TP4, TP8 all valid) |

## 8. Disk and cache requirements

| Item | Status | Value |
|---|---|---|
| Converted TP8 layout | D | 1.64 TB (0.82 TB per tray if each tray stores its own 4 ranks) |
| HF copy | M | 1.56 TB, present twice (CephFS and rack 10 tray14 RAID) |
| CephFS free | M | 22 to 23 TB of 50 TB (shared with other users; a shared-filesystem cleanup deleted other users' models on 2026-11-05 per the handoff notes) |
| Local disks | M | rack 3 tray09 root NVMe 567 GB free (insufficient for a per-tray converted copy); rack 10 tray14 RAID 7.4 TB free, root NVMe 1.3 TB free |
| Evaluator caches | M | dedicated `CUDA_CACHE_PATH`/`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` per run, about 850 MB at TP4 |
| 4-layer development conversions (TP1, TP2, TP4, TP8) | D | about 70 GB each (4 x 2.05 GiB x TP ranks + `embed`/`lm_head` 4.4 GB total) |

## 9. Runtime and timeout feasibility

| Quantity | Status | Value |
|---|---|---|
| Oracle decode step, TP4, 8 layers | M (diagnostic) | B1 162 ms, B8 1050 ms, B64 4264 ms (CUDA-event p50, Steps 3 to 5) |
| Oracle decode step, 93 layers | E (linear in layers; the eager MoE loop is per expert, so a narrower TP8 shard does not shorten it) | about 1.9 s (B1), 12 s (B8), 50 s (B64) |
| Evaluator schedule per case | D | 3 trials x (BF16 oracle + candidate + 1 or 2 FP32 references) + timing (2 warm-up + 5 reps candidate, 1 + 3 reference) + 1 audited call = 21 oracle-equivalent steps when the candidate is the oracle (ISSUES 2) |
| Whole evaluation, oracle as candidate | E | TP4 l8 B64 measured 239 s; at 93 layers about 11.6x: roughly 45 minutes at B64 plus loading; with a fast real candidate about 3 x 2 x 50 s + 4 x 50 s = 500 s of oracle time at B64 plus loading and the candidate |
| Required `--timeout` | E | ≥ 3600 s (as `goalbox-grade`); the hard-coded 10-minute process-group timeout must be considered for the load phase |
| Trial seeds | D | fresh `secrets.randbits(32)` per trial, broadcast from rank 0 |

## 10. Isolation requirements for untrusted agent-generated code

| Requirement | Current state | Status |
|---|---|---|
| Filesystem view limited to the workspace, the venv and a read-only benchmark; no host secrets | `sandbox/lib/sandbox.sh` (bubblewrap) implements it; **`bwrap` and `socat` are not installed on either rack**, and `kernel.apparmor_restrict_unprivileged_userns = 1` (the `bubblewrap` package ships the AppArmor profile that allows it); installation needs an administrator | M |
| No network in GPU jobs; allow-listed proxy for the agent | `netproxy.py` and `sandbox.sh agent`; depend on `bwrap`/`socat` | D |
| GPU jobs through a broker, one at a time, time-limited | `broker.py` (Slurm or local mode); Slurm exists on rack 3 (partitions `front`, `back`, `nvl72`), not on rack 10 | M |
| Alternative isolation for grading only | `megabench evaluate --docker-image IMG --docker-gpus device=…` runs the worker in a container with `--network none --read-only --cap-drop ALL --pids-limit 1024`, the repository mounted read-only, caches on tmpfs (`harness/runner.py:187-223`); Docker 29.1.3 with CDI GPU devices is present on rack 3; **no aarch64 image with this venv exists** (images on the trays are SGLang/vLLM builds) | M |
| Candidate separated from the evaluator's state | not provided by either sandbox: the candidate runs inside the evaluator process (plan strategy S5) | D |
| Weights read-only to the candidate | `/weights` bind is read-only on disk, but the loaded device tensors are writable (plan strategy S4) | D |
| Resource limits | time limit per GPU request (broker), `--pids-limit` in Docker; no GPU-memory or host-memory cgroup in either mode | D |

## 11. What is missing before TP8 feasibility can be stated

1. The execution path decision (D-T1): multi-node launcher on two trays (evaluator change, MNNVL already proven by others),
   the x86 8-GPU VM node (needs an x86 image, access, and a check of its NVLink topology and RAM), or a reduced-depth
   real-weight TP4 case (new case).
2. Measured values that are currently estimates: converted-weight load time from CephFS and from local RAID; snapshot
   host-copy time per trial; the oracle's 93-layer step time; the profiler's host memory at 93 layers; conversion time.
3. The converted layout itself (1.64 TB, hours of conversion, placement decision D-T2), and a 4-layer conversion at
   TP1/TP2/TP4/TP8 for loader and fidelity tests (about 70 GB each).
4. An isolation path (bubblewrap installation or an aarch64 Docker image) before any agent-generated code runs.
5. Two simultaneously free GB300 trays for a two-tray attempt, scheduled with the operator through Slurm on rack 3 (trays
   06 and 10 excluded) or with the Kubernetes scheduler on rack 10.
6. The evaluator changes of Stages 1 to 4 of the hardening plan, or an explicit decision to run TP8 validation with the
   current evaluator first.

## 12. Proposed validation sequence (each step needs its own GO)

1. CPU only: convert the first 4 layers at TP1 and TP4 from the CephFS copy to local disk (about 70 GB each; measures
   conversion throughput); load them through `load_weights` on the CPU; run the oracle on the CPU at TP1 for one step
   (fidelity rows 1, 3, 5 of the plan's matrix become possible with the HF code).
2. One tray, 4 GPUs: the 4-layer TP4 real-weight fixture through the evaluator with the oracle as candidate (a temporary
   case object, not a change to `cases.py`), measuring load time, snapshot time and memory.
3. Per decision D-T1: either the multi-node launcher on a p2 synthetic TP8 case across two trays, or the p2 synthetic
   TP8 cases on the 8-GPU node, with the oracle as candidate; then the real-weight TP8 B1 case with a reduced trial count
   and `--timeout ≥ 3600`, monitors as in Steps 5 to 8.
