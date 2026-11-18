# Seed `dsv41_b8_tp4_nopf`: the TP8 seed `dsv41_b8_nopf` ported to TP4 (2026-11-18)

Source: `seeds/dsv41_b8_nopf` (SOURCE_COMMIT 6cc15c5, SEED_SHA256 b6e0b891...), the attempt-9
single-launch megakernel for `dsv41-flash-dspark-step-k5-ctx2k-b8` (TP8, 8x B300). Target:
`dsv41-flash-dspark-step-k5-ctx2k-b8-tp4` (`gpus=4, tp=4`) on one 4x GB300 tray. The reference
(`megabench/tasks/dsv41.py`) is TP-generic; its rank shards at tp=4 are what the kernel now expects.

| quantity (per rank) | TP8 | TP4 | kernel symbol |
|---|---|---|---|
| query heads | 8 | 16 | `NHL = 64 / WORLD` |
| indexer heads | 4 | 8 | `IHL = 32 / WORLD` |
| vocabulary rows (embed, LM head, logits) | 16160 | 32320 | `VL = 129280 / WORLD` |
| o-lora groups / rows (wo_a rows = wo_b K) | 1 / 1024 | 2 / 2048 | `NGRP = 8 / WORLD`, `OL = NGRP * ORK` |
| wo_a K per group (heads/groups * D) | 4096 | 4096 | `GW` |
| routed experts (backbone / draft) | 48 / 16 | 96 / 32 | `EL = E / WORLD`, `ELMAX`, `NUMAX 160` |
| shared-expert INTER chunks of 64 | 4 or 5 | 9 | `sc0/sc1 = 36 * rank / WORLD` (unchanged code) |
| Engram rows | `part` from host | `part` from host | unchanged (host passes the shard's rows) |
| comm buffers / peers | 8 | 4 | `comm0..3`, `Ctx.comm[WORLD]` |

## Kernel changes (`mk_head.cu`, `mk_tail.cu`)
- Constants above; scratch regions sized with `NHL/IHL/OL` (`S_HW`, `S_LOW`, `S_LOWF`, `S_UCNT`).
- `make_bw`: `wo_a` is `[OL, GW]`, `wo_b` is `[H, OL]` (scales follow: `(rows/32) x (cols/32)`).
- Phase D/E (indexer head weights): tiles of 48 (row, head) pairs, `IHL` tiles, `S_HW[r*IHL+hh]`.
- Phase F (indexer scores): queries staged as bf16 `[48][IHL][128]` (values are `bfr(fp4 * pow2)`,
  exact in bf16); dot products unpack 16 bf16 per lane; head weights `hw_s[r*IHL+h]`.
- H1 (scores on tensor cores): loop `hb = 0, 8` over two 8-head passes; `hm_s[WPB][NHL]`; the
  per-row max reduction covers 16 heads.
- H2 (P.V): A fragments rows g and g+8 are heads g and g+8 (`static_assert(NHL == 16)`); the 4
  accumulators of each n-tile are stored for both heads; the finalize loop runs `NHL*D / (NT*8) = 2`
  iterations per thread.
- Phase I (wo_a): `NGRP * (ORK/64) * 8` split-K tiles; tile `t` of group `gi` stages the attention
  output columns `[gi*GW, +GW)` and accumulates into `S_LOWF[r*OL + gi*ORK + n]`.
- Phase J (wo_b): K = OL in two 1024-wide `tile48k` slices accumulated in registers (`accv[JE]`),
  output bf16 to the comm buffer as before (the all-reduce over `WORLD` peers is unchanged).
- MoE: `Shared.ecnt[ELMAX]`, `estart[ELMAX+1]`, `u_*[NUMAX]`, `S_UCNT` zeroed for `NUMAX` units.

## Host / dev files
- `mk_host.py`: `WORLD = 4`, `comm{0..3}` params (NPARAM 280), `VL = vocab // WORLD`, asserts the
  case's `world_size == 4`. `mk_params.h` regenerated.
- `devtest.py`: `WORLD_SIZE` from the environment, `torchrun --nproc_per_node 4`.
- `mb/`, `mb48/`: TP8 dev microbenchmark sources kept; x86-64 binaries (`mb/mb*`, `mb48/{mb,mb2,l2}`)
  removed (the node is aarch64).

## Status
- CPU: `mk_host.load_ext()` builds with nvcc (13.4 on rack-3 tray12, `-gencode arch=compute_103a,
  code=sm_103a -DNO_PF`): `mk_kernel` 128 registers, 896 B stack, 0 spills, 21,696 B static smem
  (TP8 seed, same toolchain: 128 regs, 1056 B stack, 20,704 B smem).
- GPU: **not run**. First GPU test = `./check_candidate` of this seed (harness calibration).
