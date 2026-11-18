# Phase contract table — filled example for the toy kernel (`a3tools/toy/toy.cu`)

Contract version: v1 (toy.cu as shipped, capture from `./task_ab --selftest`)
Kernel-wide budget: 256 threads/block, 148 blocks (1/SM, cooperative), 68 regs/thread (`cuobjdump -res-usage`), 0 spills,
0 dynamic + 0 static shared memory, no TMEM.

| phase | task | capture in → out | inputs | outputs | smem / regs / TMEM / blocks | sync in → sync out | in-kernel window (µs) | task_ab time (µs) | owner / status |
|---|---|---|---|---|---|---|---|---|---|
| gemv | `gemv` | `gemv_in` → `gemv_out`, layers 0,2 | `x`: bf16 [8][1024] row-major (written by `act` of the previous layer); `W[l]`: bf16 [4096][1024] row-major (input tensor) | `y`: f32 [8][4096] row-major, one warp per output column, fp32 accumulation, no rounding | 0 B / 64 (task build) / – / all 148 | grid barrier → grid barrier (`gsync`) | not measured (use `./phase_table`) | 30.7–32.7 (selftest, layer 2) | engine-1 / open |
| act | – (not extracted) | `gemv_out` → next layer's `gemv_in` | `y`: f32 [8][4096], only columns < 1024 read | `x`: bf16 [8][1024] = bf16(silu(0.05·y)) | 0 B / – / – / all | grid barrier → grid barrier | not measured | – | – |

## Merge log
| date | phase | variant | task_ab A→B (µs) | check before → after | kept? |
|---|---|---|---|---|---|
| (example) | gemv | `variants/gemv/vec8` | 32.7 → (B) | n/a (toy) | yes / no + reason |
