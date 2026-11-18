# TP8 bring-up preparation (CPU-only, 2026-11-18)

Branch `gb300-tp8-bringup` from `bcb672f` (Phase B accepted). Scope of this document: the eight items of the bring-up GO.
No GPU was used; `megabench/`, `setup/` and `sandbox/` are unchanged (contract digest `4da60e9b…`); the synthetic
evidence on `gb300-tp4-synthetic` is untouched. Everything new is under `planning/bringup/`. Labels: **M** measured here
(evidence directory given), **D** derived exactly from the evaluator's own code or from public metadata, **E** estimate
(assumption stated), **U** unknown.

Headline: the full 93-layer checkpoint does **not** fit four GB300s at TP4 (371.24 GiB of weights per rank against
276.6 GiB of device memory, D), and neither the host RAM of one tray (956 GB) nor the loader can hold or stream it.
The first GPU milestone is therefore the longest real-layer prefix that fits, **56 layers at TP4** (222.62 GiB per
rank, D), through the unmodified evaluator with the trusted oracle as candidate. Full depth is TP8 only, which needs
either two trays (an evaluator launcher change) or the eight-GPU x86 node (health and access unresolved).

## 1. Loader, conversion requirements, real-checkpoint cases (code reading, `bcb672f`)

- `megabench/tasks/kimi_checkpoint.py` (contract file): `convert()` reads the HF snapshot through the safetensors
  index (`_Reader`, lazy per shard), builds every rank's tensors of one layer at a time (`_layer_tensors`) and writes
  `<out>/tp<TP>/rank<R>/{global,layerNNN}.safetensors` with a process pool (`--workers`, default 4). It is restartable
  (finished layer files are skipped) and accepts `--layers N` for a prefix. Layer types always follow the 93-layer model
  (`_case(93, tp)`), so a prefix conversion is a prefix of the real model.
- `load_weights(case, root, rank, device)` reads `global` plus the first `case.params['layers']` layer files with
  `safetensors.torch.load_file(device=…)`, stacks six per-layer vectors, and validates key set, shapes and dtypes against
  `kimi.weight_shapes(case)`. `kimi.checkpoint_weights()` calls it once per process (`_CHECKPOINT_CACHE`) and reads the
  root from `KIMI_K3_WEIGHTS`.
- The real-weight cases are `kimi-k3-tp8-b{1,8,64}-s8192` (p0, `gpus=tp=8`, `checkpoint=True`). `megabench/cases.py`
  is a contract file and the CLI resolves `--case` by id from it, so **a TP4 real-weight case cannot be added without
  changing the contract digest**. `Case.__post_init__` accepts suites p0 to p3 only. The bring-up therefore constructs
  the case object in a trusted wrapper (`planning/bringup/evaluate_custom_case.py`, suite `p3` = planned, id
  `kimi-k3-real-l<L>-tp4-b<B>-s8192-bringup`) and calls the unmodified `evaluate_case()`; the record carries the
  unchanged contract digest and a `bringup` field marking the case as not in the table. Verified on the CPU (section 7).
- `kimi.weight_shapes()` accepts TP4 for the real geometry (96 heads, 163840 vocabulary, 3072 and 33792 FFN widths all
  divide 4; `3072/4 = 768` is a multiple of 32). `make_inputs()` requires `gpus == tp`, `ep == 1`: satisfied.
- Prefix semantics: the oracle's `full_layer()` rule (`(layer+1) % 4 == 0` or last layer) matches the HF config's
  `full_attn_layers` (1-indexed 4, 8, …, 92, 93; **M** from `config.json`) for every prefix length that is a multiple of 4;
  `load_weights` would reject any other prefix with a shape error. Candidates: 56 or 60 layers.

## 2. Minimal path to the full checkpoint on four GB300s: does not exist; longest fitting prefix instead

| Per rank (D, from `kimi.weight_shapes` and `make_state`) | TP4 L56 | TP4 L60 | TP4 L64 | TP4 L93 | TP8 L93 |
|---|---|---|---|---|---|
| Weights | 222.62 GiB | 238.70 GiB | 254.77 GiB | 371.24 GiB | 190.97 GiB |
| State B1 / B8 | 0.19 / 1.50 GiB | 0.20 / 1.61 GiB | 0.21 / – | – | 0.26 / 2.11 GiB |
| Snapshot host-copy path (≤ 64 MiB tensors) | 1179 tensors, 20.0 GiB | 1263, 21.5 GiB | 1347, 22.9 GiB | – | 2232, 32.2 GiB |
| Snapshot checksum path | 335 tensors, 202.8 GiB | 359, 217.4 GiB | 383, 232.1 GiB | – | 278, 159.0 GiB |
| Device memory (`gpu_memory_bytes`, M) | 276.6 GiB | | | | |
| Headroom after weights + state + about 10 GiB overhead (E, from TP4 synthetic: +3.5 GB allocator, +9 GB process) | about 44 GiB | about 28 GiB | about 12 GiB | negative | about 75 GiB |

- Host offload is not an option: 4 × 371 GiB = 1.45 TiB exceeds the tray's 956 GB (M), and the loader has no streaming
  mode. A streaming or offloading oracle would be new, untrusted code.
- Recommendation: **56 layers** for the first GPU milestones (44 GiB headroom covers the FP32 reference casts, largest
  1.1 GiB for the LM head shard, the snapshot temporaries, 1.5 GiB, and the allocator cache); 60 layers as a second
  step once the measured overhead is known. 64 layers leaves too little margin.
- What the prefix validates: the real loader and converter at scale, the real MXFP4 expert decode, KDA and MLA layers
  with real parameters through the evaluator's distributed path, memory and timing. What it does not validate: model
  fidelity (the LM head applied after layer 56 is not the model's output), hence oracle-versus-oracle correctness only.

## 3. Conversion offline, without changing model semantics (M + D)

- Pure re-layout (D, converter code): `narrow().contiguous()` splits along the documented dimension, `torch.stack` over
  experts and over the three convolution taps, a `view` of `weight_packed [rows, cols/2]` to `[rows, cols/32, 16]`, the
  `[1, h] → [h]` residual projections, `A_log[:96]`. The only dtype casts are `.to(float32)` of tensors that are already
  F32 in the checkpoint (`A_log [128]`, `dt_bias [12288]`, `o_norm [128]`, `{q,k,v}_conv1d [12288,1,4]`,
  `e_score_correction_bias [896]`; **M** from the shard headers), so no value changes. Text-model dtype histogram of the
  source: F32 506, BF16 1954, U8 494592 tensors (M).
- Trial (M, `planning/bringup/evidence/conversion-trial-l4-tp4-20261118T203243Z/`): first 4 layers plus `global` at TP4
  from the CephFS HF copy to `/mnt/shared/home/kevin/weights/kimi-k3-fixture-trial`, rack 3 tray14, CPU only:
  70.75 s wall, 4 workers, peak RSS 34.6 GB per worker, 58,734,856,864 bytes written (20 files), exit 0.
- Verification (M, `verify_conversion.py`): for all 118 fixture tensors of those files (57.6 GB), the concatenation of
  the four rank shards along the documented dimension is bit-identical to the HF source tensor (unsplit tensors identical
  on every rank; all 896 experts of each MoE layer compared), 23 s. The key table of the verifier is written from the
  documented name map, not imported from the converter.
- Manifest (M, `conversion_manifest.py` → `manifest.json`): source directory, pinned revision, sha256 of `config.json`
  (`9710e121…`) and index (`a1c52106…`, both equal to Hugging Face at the pinned revision, Phase B), the 5 shards read with
  sha256, converter sha256 and repository commit, command, 20 output files with sizes and sha256, `DONE.json`, log and
  time output. The HF copies are left unchanged (read-only use; owner `dc`, mode 775).
- Cost of the real conversions (E from the trial: about 57 s per layer job per worker, 35 GB RSS per worker): TP4 L56 =
  53 more layer files, about 13 min with 4 workers (plan 15 to 30 min), 890.5 GiB total output (D) into the same root
  (the trial's 4 layers are reused); TP8 L93 = 94 files, about 22 min with 4 workers, 1527.8 GiB (D). CephFS has 21 TB
  free (M); the converter's output files are created mode 0600 (M) and need `chmod -R a+rX` before a run as `dc`.
- Residual risk: the trial covers layer types dense (0), KDA (1, 2) and MLA (3) and the global tensors; layers 4 to 92
  repeat these types with the same shapes (D). Run the verifier on the full L56 output before milestone 1 (about 56/4 × 23 s
  ≈ 6 min, E).

## 4. Distributed launcher: minimum change for two four-GPU trays, and the fabric facts

Current launcher (`harness/distributed.py:146-183`, D): `mp.spawn(_rank_worker, nprocs=case.gpus)` on one host; rank
`r` uses `cuda:{first_device + r}`; file rendezvous in a local temporary directory; `unavailable` if fewer than
`case.gpus` GPUs are visible; aggregation reads `rank-N.json` from that directory. Nothing in the harness starts a
second host.

Minimum change (proposal, **not applied**; `distributed.py` is a contract file, so the digest changes):

```
evaluate_distributed():
  nnodes    = int(os.environ.get("MEGABENCH_NNODES", "1"))      # 2 for two trays
  node_rank = int(os.environ.get("MEGABENCH_NODE_RANK", "0"))
  local     = case.gpus // nnodes                                # 4
  if device_count < first_device + local: unavailable
  directory = os.environ.get("MEGABENCH_RANK_DIR")  (shared filesystem; required when nnodes > 1) else TemporaryDirectory
  rendezvous = f"tcp://{MEGABENCH_MASTER_ADDR}:{MEGABENCH_MASTER_PORT}" if nnodes > 1 else file URI
  mp.spawn(_rank_worker, nprocs=local, args=(..., rank_offset=node_rank*local, ...))
  if node_rank: return {"status": "node_done", "node_rank": node_rank}
  wait (with a deadline) until rank-0..case.gpus-1 files exist, then aggregate as today
_rank_worker(local_rank, ..., rank_offset):
  rank = rank_offset + local_rank; device = cuda:{first_device + local_rank}
  init_process_group(rank=rank, world_size=case.gpus, init_method=rendezvous)
```

About 25 lines; `initialize()` (groups over `case.gpus`) and the grading code need no change. Both trays run the same
CLI command with different `MEGABENCH_NODE_RANK`; only node 0 produces the record. Before any of that, the fabric can
be validated outside the evaluator with `planning/bringup/nccl_smoke.py` (torchrun, 2 × 4 ranks, exact all-reduce,
bus bandwidth, hostnames gathered, NCCL transport from the INFO log), which is milestone 3.

Fabric and network facts (M, rack 3, trays 09, 11, 14 sampled on 2026-11-18):

| Item | Finding |
|---|---|
| NVLink fabric | every sampled GPU: `Fabric State Completed / Status Success`, health Full bandwidth, **same ClusterUUID `5227df9c…` and CliqueId 3366** on trays 09, 11 and 14 → one NVLink domain across the rack |
| IMEX | `nvidia-imex` active on all sampled trays; connectivity table lists 17 nodes READY (driver 610.57.04); `/dev/nvidia-caps-imex-channels/channel0` present |
| Fabric manager service | inactive on the trays (the NVL72 switch trays manage the fabric) |
| Tray network | one bonded Ethernet VLAN (`bond0.225`) carries the tray addresses; ping tray09 → tray11 0.08 ms; InfiniBand ports down (no IB/RoCE path, so NCCL bootstrap over this interface, data over NVLink) |
| Cross-rack | rack 10 tray14 has a different ClusterUUID (`0ae101be…`, clique 19378): TP8 across racks is impossible; the two trays must be in the same rack |
| Precedent | other users run 2-node `nvl72` Slurm jobs on rack 3 now; an earlier SGLang Kimi-K3 TP8 deployment on two rack-10 trays used `NCCL_MNNVL_ENABLE=1` (infrastructure handoff) |
| Not yet exercised | NCCL between two trays by us; which transport NCCL 2.30.7 (venv) picks; whether `NCCL_MNNVL_ENABLE=1` is needed; `NCCL_SOCKET_IFNAME` (likely `bond0.225`, to confirm from the INFO log) |

Scheduling facts (M): Slurm on rack 3 (`nvl72` partition = all 18 trays, default, `AllowAccounts=ALL`, MaxTime 7 days,
`cons_tres` with `CR_CORE_MEMORY`, slurmdbd accounting) but **kevin has no association** (`srun` → "Invalid account or
account/partition combination"); accounts present: `general` (used by the other users), `exp44`, `root`. Without an
allocation our host-level runs are invisible to Slurm, which can place another user's job on the same tray (a two-node
job of another user was running during sampling; a foreign 68 GB process appeared on tray14 GPU 1 for a few minutes).
The association is an administrator action: `sacctmgr add user kevin account=general` (your call, needs sudo on the Slurm
controller). Trays 06 and 10 stay excluded; tray16 has no account for kevin; tray11 had only 65 GB of host memory
available at sampling (891 GB in use by someone else).

## 5. Two GB300 trays versus the eight-GPU x86 node (M unless marked)

| | Two GB300 trays, rack 3 | `k8s-b300-gpu-worker-02-vm-f3` |
|---|---|---|
| GPUs | 8 × GB300, 276.6 GiB usable each | 8 × B300-SXM6-AC, 275040 MiB = 268.6 GiB each (label) |
| TP8 L93 B1 headroom per rank (E) | about 75 GiB | about 67 GiB |
| GPU interconnect | NV18 inside a tray, NVLink fabric across trays (same clique) | unknown inside the VM (U; needs a pod with `nvidia-smi topo -m`) |
| Host | 2 × (Grace 144 cores, 956 GB) | 240 vCPU, 2105065996 KiB = 1.96 TiB RAM, Q35 VM, Ubuntu 24.04, kernel 6.8 |
| Driver / CUDA | 610.57.04 / torch cu130 venv works (M, Steps 1 to 8) | 580.126.09 / CUDA 13.0 runtime label; an x86 image with torch 2.14.1+cu130 must be built (E: about 1 h with network) |
| Storage | CephFS `/mnt/shared` (21 TB free) holds the HF copy and the converted output; local NVMe 485 to 574 GB free per tray (too small for the per-tray TP8 shards, 764 GiB) | 1.48 TB allocatable ephemeral (too small for the 1.53 TB TP8 layout); CephFS PVCs exist (`csi-cephfs-sc-k8s`), but mounting the existing `/mnt/shared` volume needs a static PV (admin, U) |
| Scheduling / access | host level now (shared by convention), Slurm after the association | taint `nvidia.com/gpu=present:NoSchedule`; our service account can create pods in `inference-wideep` (a debug pod was tried there on 2026-11-02 and is in `StartError` after 19 restarts); `gpu-health-monitor` on the node has 7511 restarts (`RunContainerError`); no GPU pods on it; node `Ready`. Health of GPU containers on this node is unknown at best (U) |
| Evaluator change | yes: the launcher (section 4) | none: the p0 cases run as written on one host |
| Isolation | Docker worker mode exists in the harness; no aarch64 image yet | the pod is the isolation; `--network none` equivalents via NetworkPolicy (U) |
| What can be validated today | fabric facts measured; NCCL smoke test ready | nothing without a pod with GPUs (not run: GPU allocation) |

Assessment: the two-tray path is the one whose prerequisites we can see and test; it costs a reviewed launcher change
and a Slurm association. The x86 node would run the unchanged evaluator but needs an administrator to confirm GPU
container health, an x86 image, and a way to reach the weights; it also has 9 GiB less memory per GPU. Recommendation:
prepare the two-tray path (milestones 3 and 4) and ask the cluster owners about the x86 node in parallel; if it becomes
usable, run the p2 synthetic TP8 cases there first with no evaluator change.

## 6. Staged GPU plan (each milestone needs its own GO; nothing has run)

| # | Milestone | Code | Case | Where | Resources (D/E) | Timeout | Success | Stop conditions |
|---|---|---|---|---|---|---|---|---|
| 1 | TP4 real-weight load-only | `tp4_real_load_only.py` via `m1-tp4-real.sh load 56` | first 56 layers, TP4 | one rack 3 tray (tray14 or 11 by `rack3_free`; trays 06/10 excluded), kevin, host level; or rack 10 tray14 as dc with the same shared paths | 4 GPUs × 222.6 GiB; CephFS read 890 GiB; host: page cache only; load E 8 to 25 min (4 streams at 0.45 GiB/s each measured for 8 GiB; loader efficiency unknown) | `timeout -k 30 3600` | 4 ranks `pass`, `memory_allocated` within 1 GiB of 222.6 GiB, memory released, no foreign process, scheduler monitor empty | GPU busy or > 1 GiB used before start; scheduler job/pod on the node; < 300 GB host memory; any rank error or validation failure; a foreign process appears; load > 45 min |
| 2 | TP4 real-weight single-step correctness | `evaluate_custom_case.py` via `m1-tp4-real.sh evaluate 56 1` (oracle candidate, trials 1, warmup 1, reps 1) | same, B1 | same tray | per rank E about 233 GiB; host: +22 GB startup (M) + 80 GiB snapshot copies per trial (D); oracle step E about 1.1 s (7 × the 8-layer 162 ms); evaluation E 2 min after loading | `timeout -k 30 5400` | status `non_megakernel` with correctness `pass` on 4 ranks (the oracle fails the launch audit by design), band ratios < 1, identical NCCL sequences, contract digest `4da60e9b…` in the record | as 1, plus any rank `incorrect`/`error`, NCCL error or hang, host available < 150 GB |
| 3 | TP8 distributed startup + NCCL smoke | `nccl_smoke.py` with torchrun on two trays | none (256 MiB all-reduce) | two idle rack 3 trays, same clique | 1 GiB per GPU; 2 min | 300 s init timeout | 8 ranks, 2 distinct hosts, exact result, transport in the NCCL log is NVLink/MNNVL (not socket) | init timeout; inexact result; socket transport (report, decide) |
| 4 | TP8 real-weight load and correctness | launcher change (section 4) after review, then the CLI on `kimi-k3-tp8-b1-s8192` with the positive control, trials 1 | p0 B1, 93 layers | two trays (or the x86 node, no change) | TP8 conversion 1.53 TB (E 22 min); per rank E 201 GiB; host snapshot 129 GiB per tray per trial (D); load 191 GiB per rank over 8 concurrent CephFS streams (E 15 to 40 min; the 10-minute process-group timeout applies to the skew between ranks, not to the load itself) | `--timeout ≥ 7200` | `non_megakernel`, correctness pass on 8 ranks | as 2 |

Between 1 and 2 the same weights stay on CephFS; 2 can follow 1 the same day. 3 is independent of 1 and 2 and can run
as soon as two trays are idle. 4 waits for the launcher decision (Garv) and the Slurm association or the x86 node.

## 7. Evaluator contract and trusted code

- `git diff --stat bcb672f -- megabench setup sandbox` is empty; contract digest `4da60e9b9af9214f…` recorded by the CPU
  smoke run of the wrapper (M).
- Trusted code used: the repository's converter, loader, oracle and evaluator; the positive-control candidate
  (`reports/gb300/negative-controls/positive_control_oracle.py`, unchanged, sha256 `6f9e8a84…`); the new
  `planning/bringup/*.py` and the driver, written and reviewed in this session, CPU-tested below. No agent, no sandbox.
- CPU smoke tests (M, same evidence directory): `tp4_real_load_only.py --device cpu` on the 4-layer trial: 4 ranks `pass`
  (shape and dtype validation of 100 tensors per rank, 14.68 GB each; on the CPU safetensors memory-maps the files, so
  this exercises the code path, not the I/O); `evaluate_custom_case.py --device cpu` with the oracle candidate on
  `kimi-k3-real-l4-tp4-b1-s8192-bringup`: status `correctness_only`, correctness `pass` on all 4 ranks (1 trial, seed
  shared), audit `not_applicable_cpu`, 15.75 s wall, peak RSS 19 GB, candidate 488 ms versus reference 528 ms per step on
  the CPU (diagnostic). A first attempt failed because the Case validator rejects suites other than p0 to p3; the wrapper
  now uses `p3`.

## 8. First GPU test (milestone 1): exact commands, resources, estimates, timeouts, stop conditions

Preparation (CPU, no GO needed for the computation itself, but it writes 0.9 TB to the shared filesystem: please confirm):

```
# rack 3, any accessible tray (tray14 was used for the trial), as kevin; about 15 to 30 min, 4 workers, about 140 GB RSS
R=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox; W=/mnt/shared/home/kevin/weights/kimi-k3-fixture-trial
cd $R && CUDA_VISIBLE_DEVICES="" /usr/bin/time -v setup/convert_checkpoint.sh /mnt/shared/weights/kimi-k3 $W 4 4 --layers 56   # (1)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$R .venv/bin/python -s planning/bringup/verify_conversion.py --hf /mnt/shared/weights/kimi-k3 --out $W --tp 4 --layers 56
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$R .venv/bin/python -s planning/bringup/conversion_manifest.py --hf /mnt/shared/weights/kimi-k3 --out $W --tp 4 --layers 56 --repo $R --hash-shards > $W/MANIFEST-tp4.json
chmod -R a+rX $W
```
(1) `convert_checkpoint.sh` passes only HF_DIR OUT_DIR TP WORKERS; for a prefix call the module directly:
`PYTHONPATH=$R .venv/bin/python -s -m megabench.tasks.kimi_checkpoint --src /mnt/shared/weights/kimi-k3 --out $W --tp 4 --layers 56 --workers 4`.

Run (GPU, needs the GO):

```
ssh pod4-gb300-3-trayNN-f3          # NN chosen with rack3_free at GO time (not 06, 10); fallback: rack 10 tray14 as dc with M1_OUT_ROOT in dc's raw area
R=/mnt/shared/home/kevin/workspace/kimi-k3-goalbox
KIMI_K3_WEIGHTS=$W setsid -f nohup $R/planning/bringup/m1-tp4-real.sh load 56 > /dev/null 2>&1 < /dev/null
# monitor: tail -f $R/reports/gb300/m1-tp4-real-load-l56-b1-<ts>/{gpu-monitor,host-mem-monitor,scheduler-monitor}.log.txt
```

The driver records `env.txt` (commit, weights root, `DONE.json`, manifest sha256, script sha256s, GPU inventory,
versions), `occupancy-before/after.txt`, aborts unless all four GPUs are idle (no compute process, ≤ 1 GiB used), no
Slurm job or GPU pod is on the node, and ≥ 300 GB host memory is available; runs the loader under `timeout -k 30 3600`
with 10 s GPU and host-memory monitors and a 30 s scheduler monitor; kills leftovers by pattern; writes
`load/load-rank{0..3}.json` and `load/load-summary.json`.

| Quantity | Value | Label |
|---|---|---|
| Device memory per rank | 222.62 GiB weights + about 1 GiB context | D + E |
| Host memory | page cache of 890 GiB of files (evicted as needed); loader RSS small | E |
| Shared filesystem read | 890.5 GiB | D |
| Load time | 8 min if four concurrent streams keep the measured 0.45 GiB/s each; plan for 10 to 25 min | E (M: 4 × 8 GiB direct reads at 479 to 484 MB/s each, 1.78 GiB/s aggregate; single stream 533 MB/s) |
| Outer timeout | 3600 s (kill after 3630 s) | set |
| Expected result | 4 × `pass`, `memory_allocated_bytes` ≈ 239.0e9, `gpu_free_after_release` ≈ `gpu_free_before` | D |

Stop conditions (abort, record, no retry without a new GO): preflight abort; any rank `error` (CUDA OOM, validation
failure, missing file); a foreign compute process or scheduler job appears on the tray during the run; host available
memory below 100 GB; no progress after 45 min; any `ABORTED` file. Infrastructure interruption ends the step.

## 9. What is still missing

1. The L56 TP4 conversion (0.9 TB on CephFS, about 15 to 30 min of CPU) and its verification and manifest: ready to run on
   confirmation.
2. A Slurm association for kevin on rack 3 (administrator), or acceptance of host-level runs with the collision risk.
3. For TP8: Garv's decision on the launcher change (contract digest changes) or the x86 node; the TP8 conversion (1.53 TB);
   two simultaneously idle trays; the NCCL smoke result.
4. Measured values still estimated: loader throughput from CephFS at scale, oracle step time at 56 and 93 layers, the
   evaluator's device overhead on real weights, snapshot host-copy time.
