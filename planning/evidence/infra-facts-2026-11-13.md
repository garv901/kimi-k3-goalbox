# Infrastructure facts collected for Phase B (read-only commands, 2026-11-13 01:20 to 01:40 UTC)

Collected from rack 3 tray09 (user kevin) and rack 10 tray14 (user dc), plus public Hugging Face metadata. Host names
and IP addresses of the racks are not reproduced here beyond the tray names already used in the reports.

## Trays (both racks)

| Fact | Rack 3 (tray09) | Rack 10 (tray14) |
|---|---|---|
| GPUs per tray | 4 x NVIDIA GB300, 284208 MiB each (`gpu_memory_bytes` 297013411840 in the evaluator records = 276.6 GiB) | same |
| Intra-tray GPU links | `nvidia-smi topo -m`: every GPU pair `NV18`; GPUs 0,1 on NUMA 0, GPUs 2,3 on NUMA 1 | same; `nvidia-smi nvlink -s -i 0`: links at 53.125 GB/s each |
| Driver / toolkit | 610.57.04; `/usr/local/cuda-13.1`, `cuda-13.4` (`cuda` symlink) | 595.71.05; nvcc 13.2 (from the Step 8 `env.txt`) |
| Python environment used by the evaluator | `/mnt/shared/home/kevin/workspace/kimi-k3-goalbox/.venv`: Python 3.12.3, torch 2.14.1+cu130, CUDA 13.0, bundled NCCL 2.30.7, Triton 3.8.0 (aarch64) | the same venv over CephFS |
| Host RAM | 956 GiB total (`free -g`), 811 GiB available at sampling | 958 GiB total, 823 GiB available |
| IMEX | `nvidia-imex` active, 18 entries in `/etc/nvidia-imex/nodes_config.cfg` | active, 18 entries, `/dev/nvidia-caps-imex-channels/channel0` present |
| Scheduler | Slurm (partitions `front`, `back`, `nvl72`); at sampling: idle tray10 and tray11; mixed 01-04, 07-09, 13, 14; allocated 05, 06, 12; user `fi` runs several 1- and 2-node `nvl72` jobs (for example one job on trays 13+14, one on 01+07) | Kubernetes; `kubectl get nodes`: trays 01-17 with 4 allocatable GPUs each, plus `k8s-b300-gpu-worker-02-vm-f3` with 8; at sampling only tray03 had GPU pods (4) |
| `bwrap` / `socat` | not installed; `kernel.apparmor_restrict_unprivileged_userns = 1` | not installed; same sysctl |
| Docker | 29.1.3 with CDI `nvidia.com/gpu=N` devices; images present include SGLang and vLLM builds, no image with this repository's venv | not checked (Kubernetes node) |
| Local disk | root NVMe 1.8 TB, 567 GB free | root NVMe 1.8 TB, 1.3 TB free; `/opt/verda` RAID (`/dev/md0`) 11 TB, 7.4 TB free |
| Shared disk | CephFS `/mnt/shared` 50 TB, 23 TB free | same filesystem, 22 TB free |

## 8-GPU Kubernetes node `k8s-b300-gpu-worker-02-vm-f3` (labels only; not accessed)

`kubernetes.io/arch=amd64`, `nvidia.com/gpu.product=NVIDIA-B300-SXM6-AC`, `nvidia.com/gpu.count=8`,
`nvidia.com/gpu.memory=275040` (MiB, = 268.6 GiB, less than a GB300's 284208), `nvidia.com/gpu.compute.major=10`,
`minor=3`, `nvidia.com/cuda.driver-version.full=580.126.09`, `nvidia.com/gpu.machine=Ubuntu-24.04-PC-Q35-ICH9-2009`
(a virtual machine). No GPU pods on it at sampling. Not known: NVLink/NVSwitch topology inside the VM, host RAM, storage,
who may use it, whether a pod image with CUDA 13 PyTorch for x86 exists.

## Kimi-K3 checkpoint copies found (Hugging Face layout, not the converted per-rank layout)

| Copy | Contents | Revision check |
|---|---|---|
| `/mnt/shared/weights/kimi-k3` (CephFS, owner `dc`, mode 775, files dated 2026-07-27/28) | 96 shards `model-00001-of-000096.safetensors` ..., 1560936091448 bytes of shards; `model.safetensors.index.json`: 497220 tensors, metadata total_size 1560860324864, 247296 `weight_packed` (MXFP4) tensors, 165 vision tensors; `config.json`, modeling/configuration files, README, LICENSE | sha256 of `config.json` = `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` and of `model.safetensors.index.json` = `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`, both equal to the files served by Hugging Face at the pinned revision `f831ab66814297da540d832a5235f8e904f29d06` (fetched 2026-11-13; `x-repo-commit` of `main` is the same revision). Shard contents were not hashed. |
| `/opt/verda/weights/moonshotai/Kimi-K3/2496450` (rack 10 tray14 local RAID, owner root, mode 755) | same file set (96 shards, same index metadata) | same two sha256 values |
| Other directories matched by name (`kimi-k3-dspark`, `Kimi-K3-NVFP4`, `kern/kimi-k3`, `rain/profiles/kimi-k3-48L`) | different models or configs only | not relevant |
| Converted per-rank layout (`tp8/rank<R>/*.safetensors`, `DONE.json`) | none found under `/mnt/shared` (depth 4) | would have to be produced by `setup/convert_checkpoint.sh` |

Single-stream direct reads of one shard (`dd bs=16M count=64 iflag=direct`, one sample each): CephFS copy 431 MB/s
(rack 3 tray09); rack 10 local RAID copy 29.1 GB/s.

## Pinned model configuration (from `megabench/tasks/model_specs.json`, revision `f831ab66…`)

`num_hidden_layers` 93, `hidden_size` 7168, `num_experts` 896, `vocab_size` 163840, `kv_lora_rank` 512, `q_lora_rank` 1536,
`moe_intermediate_size` 3072, `intermediate_size` 33792, `num_attention_heads` 96, `transformers_version` 4.56.2,
quantization `compressed-tensors` / `mxfp4-pack-quantized` (group 32, 4-bit float, uint8 scales) for the routed experts;
attention, shared experts, dense MLP, lm_head and vision excluded from quantization.

## Baseline's own CPU unit tests at `eac5dd2`

`python -m unittest megabench.tests.test_kimi` on rack 3 tray09 (`CUDA_VISIBLE_DEVICES=""`): 7 tests, OK, 4.4 s.
