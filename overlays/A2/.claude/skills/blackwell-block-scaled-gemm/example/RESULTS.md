# bs_gemm on B300 (sm_103a, CUDA 13.0, driver 580.173.02), measured 2026-11-17

Status: **correct (tests pass); a starting point for performance, below the 4 TB/s target at ~19 MB.**

## Correctness (`./bs_gemm test`, full output in `test_output_b300.log`)
9 shapes (M = 1, 8, 16, 33, 48, 64; N = 128..4096; K = 256..7168; split-K 1..8, uneven splits, PDL on/off),
each launched 3x against a double-precision CPU reference of the dequantized operands. Tolerance:
|err| <= |ref| * 2^-8 * 1.01 + 1e-5 * max|ref| (one BF16 rounding of the output). Max error = half a BF16
ulp of the largest outputs. Self-check: changing one weight scale byte on the device makes the test
fail (`"mutation","caught":true`).

## Bandwidth, N=4096 (M = tokens; weight bytes = FP4 + UE8M0 scales; L2-cold: rotating copies > 4x L2)
`us_graph` = average per launch of back-to-back PDL launches in a CUDA graph. `stream` = the same
launches issued from the host one by one.

| K | weight MB | M | splits | us_graph | TB/s graph | us_stream | TB/s stream | us, L2-hot graph |
|---|---|---|---|---|---|---|---|---|
| 7168 | 15.60 | 8 | 8 | 5.60 | 2.78 | 5.79 | 2.69 | 5.29 |
| 7168 | 15.60 | 48 | 8 | 7.49 | 2.08 | 7.68 | 2.03 | 7.04 |
| 8704 | 18.94 | 8 | 8 | 6.12 | **3.10** | 6.29 | 3.01 | 5.70 |
| 8704 | 18.94 | 48 | 8 | 8.01 | 2.36 | 8.23 | 2.30 | 7.46 |
| 14336 | 31.2 | 8 | 8 | 7.67 | 4.07 | | | |
| 28672 | 62.4 | 8 | 8 | 12.28 | 5.08 | | | |

Without PDL (same kernel): 15.6 MB, M=8: 7.32 us graph / 9.95 us stream.

**Marginal streaming rate** (difference between sizes, M=8): ~7 TB/s, so the main loop runs near HBM
speed; the **fixed cost per standalone launch is ~3.4 us** at M=8 (~4 us at M=48): setup + TMEM alloc,
first-TMA latency, split-K cluster reduction, and the launch gap. A K=256 launch (0.56 MB) takes 2.35 us.

All shapes and M = 1..64 are in primdb: `./primdb q gemm.bs_mxfp4 gpu=B300` (rows with `l2=cold|warm`).
For comparison, the best existing B300 primbench engines at N=4096, K=14336, M=8 (L2 cold):
BF16 `mma_sync` 25.9 us, INT8-weight `mma_sync` 22.3 us, BF16 `tcgen05_swapab` 31.6 us; this MXFP4 kernel 7.67 us.

## Optimization history (N=4096, K=7168, M=8, us per launch)
| step | stream | graph |
|---|---|---|
| v1: 1 CTA/SM, 5 x 32 KB stages, split-K 4 with FP32 partials in global + atomic ticket | 11.3 | |
| + PDL (weights before `griddepcontrol.wait`) and DSMEM cluster reduction | 6.4 | |
| 128-K stages, <= 112 KB smem so 2 CTAs/SM, split-K 8 | 5.8 | 5.6 |
| rejected: split-K partials pushed with scalar `st.async` + mbarrier | 6.8 | 6.3 (much worse at M >= 32) |
| rejected: <= 74 KB smem (3 CTAs/SM) | 5.9 | 5.6 (M=48: 8.4 vs 7.3) |

## Not done / limits
- No per-shape tuning (stages, splits, CTA shape); fixed costs not attacked further (see SKILL.md).
- Activation quantization (BF16 -> MXFP8) is not part of the example; inputs are generated in MXFP8.
- N must be a multiple of 128, K of 128, M <= 64; split-K <= 8 (cluster size).
