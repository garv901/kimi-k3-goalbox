---
name: blackwell-block-scaled-gemm
description: Recipe and a tested standalone example for skinny (M <= 64 token rows) GEMMs with FP4/FP8 block-scaled weights on Blackwell (sm_100a/sm_103a, B200/B300) using the native tensor-core path - tcgen05.mma kind::mxf8f6f4.block_scale (MXFP4 weights x MXFP8 activations, UE8M0 scales), TMA loads of packed FP4 (16U4_ALIGN16B), TMEM accumulators and scales, swap-AB, a multi-stage smem ring, split-K over a cluster, PDL. Use instead of software dequantization (cvt fp4->f16 + mma.sync) when a weight-streaming phase is far below HBM bandwidth.
---

# Block-scaled skinny GEMM on Blackwell (MXFP4 x MXFP8, tcgen05)

`example/bs_gemm.cu` is a complete, tested kernel (one file, CUDA C++ + inline PTX, no CUTLASS):
`Y[M][N] (BF16) = X[M][K] (MXFP8) * W[N][K]^T (MXFP4)`, generic shapes (N % 128 == 0, K % 128 == 0, M <= 64).
Read it next to this recipe. Build and check it first (one GPU):
```
cd .claude/skills/blackwell-block-scaled-gemm/example
nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a bs_gemm.cu -o bs_gemm -lcuda
../../../../gpu_run -g 1 -- ./bs_gemm test        # 9 shapes vs a CPU reference + a mutation self-check
../../../../gpu_run -g 1 -- ./bs_gemm bench 4096 7168   # us and TB/s for M = 8..64 (args: N K [splits])
```
(`./gpu_run` is at the workspace root; adjust the relative path.) Measured numbers are in
`example/RESULTS.md` and as `./primdb q gemm.bs_mxfp4 gpu=B300` rows; it is a **starting point**, not a
tuned kernel (see "Where the time goes").

## Why
Software FP4 decode (`cvt.rn.f16x2.e2m1x2` + scale multiply + `mma.sync`) costs several ALU
instructions per weight byte; at M <= 64 the ALU pipe, not HBM, then limits a weight stream to a fraction of
HBM bandwidth. The tensor core decodes block-scaled FP4 itself: no per-element instructions remain, and the
kernel becomes a pure TMA -> smem -> MMA stream. FP8 `mma.sync` is emulated on sm_100a/sm_103a (see
`./primdb q mma_sync`), and `tcgen05.mma kind::i8` does not exist on sm_103a.

## The recipe
1. **Swap-AB.** MMA M must be 64 or 128, but you have <= 64 tokens. Put the *weight* in operand A
   (128 weight rows = MMA M) and the *tokens* in operand B (MMA N = tokens padded to a multiple of 8
   (8 works) or 16). D[weight row][token] lands in TMEM: lane = weight row, column = token.
2. **Instruction**: `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [d_tmem], a_desc, b_desc, idesc, [sfa_tmem], [sfb_tmem], p;`
   A = E2M1 (FP4), B = E4M3 (FP8), K = 32 per instruction, FP32 accumulate, one UE8M0 scale per 32 K
   per row for each operand. Instruction descriptor (both operands K-major):
   ```
   idesc = sf_id << 4 | 5 << 7 (A=E2M1) | 0 << 10 (B=E4M3) | (N>>3) << 17 | 1 << 23 (UE8M0) | (128>>4) << 24 | sf_id << 29
   ```
   `sf_id` (0..3) picks which byte of the 32-bit TMEM scale column holds this K-block's scale (below).
   For FP4 x FP4 use `kind::mxf4` (K = 64, both packed) instead; for FP8 x FP8 `kind::mxf8f6f4` with A=B=E4M3.
3. **FP4 weights in smem**: `kind::mxf8f6f4` reads FP4 as 16 values (8 bytes) per 16-byte slot. TMA
   produces exactly that from packed HBM data with `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`: view the packed
   `[N][K/2]` bytes as a 3-D tensor `{128 values, N rows, K/128 chunks}` (strides `K/2` and 64 bytes), box
   `{128, 128, 1}`, `CU_TENSOR_MAP_SWIZZLE_128B`. Each box gives a 128-row x 128-byte K-major SW128 tile =
   128 rows x 128 FP4 values. **The mbarrier tx count is the packed size** (8 KB per box, not the 16 KB of
   smem it fills). Packing: low nibble = even k.
4. **Smem descriptors** (K-major, SW128): `start>>4 | 1<<16 | (1024>>4)<<32 | 1<<46 | 2<<61`
   (SBO = 1024 B between 8-row groups, bit 46 = descriptor version on sm_100+). Advance K by 32 values
   = 32 bytes inside the swizzled row by adding 2 to the descriptor (16-byte units). Tiles must be 1024 B
   aligned. The FP8 activation tile `[tokens][128 B]` uses the same descriptor (TMA UINT8, box `{128, Npad}`, SW128).
5. **Scales**: block-scaled MMA reads scales from TMEM. Store them in HBM already in the "128x4 atom"
   layout: one 512-byte atom = 128 rows x 4 K-blocks (128 K values), `byte(r, kb) = (r%32)*16 + (r/32)*4 + kb`.
   Copy an atom smem -> TMEM with `tcgen05.cp.cta_group::1.32x128b.warpx4 [tmem_col], desc` where
   `desc = start>>4 | (128>>4)<<32 | 1<<46` (no swizzle). It fills 4 TMEM columns; the 4 MMAs of that 128-K
   chunk use the same column address with `sf_id = 0..3`. Use the same atom layout for the activation
   scales (rows = tokens, padded to 128 with zeros). Load atoms with plain `cp.async.bulk` into the same
   stage (and the same mbarrier) as the operands. Issue `tcgen05.cp` from the MMA thread right before the
   MMAs: tcgen05 ops from one thread execute in order, and the stage's `tcgen05.commit` covers both.
6. **Warp roles** (192 threads): warp 4 = TMA producer (one elected lane), warp 5 = MMA issuer (one
   elected lane), warps 0-3 = epilogue (warp w may only access TMEM lanes 32w..32w+31). TMEM is allocated
   by one warp (`tcgen05.alloc` + `relinquish_alloc_permit`), freed by the same warp at the end.
7. **Ring**: STAGES x {W tile, X tile, SFA atom, SFB atom}; `full[s]` (producer `arrive.expect_tx`, TMA
   `complete_tx`), `empty[s]` (signalled by `tcgen05.commit ... mbarrier::arrive::one` after the MMAs of
   stage s), one `acc` barrier committed after the last MMA. Producer waits `empty` with the inverted
   phase (fresh barriers pass parity 1). Every wait has a bounded spin + `trap` so bugs fault, not hang.
8. **Fill the GPU**: N/128 output tiles is far below the SM count for typical N; split K across a
   **cluster** `(1, S, 1)` (S <= 8; the example picks the largest S with tiles x S <= 2 x SMs). Each CTA writes its FP32 partial `[tokens][128]` to its own (now idle)
   ring smem, `barrier.cluster` (release/acquire), then each CTA sums a 1/S slice of the outputs from all S
   peers with `ld.shared::cluster.v4.f32` (via `mapa`) and stores BF16; a final relaxed cluster barrier keeps
   smem alive until peers are done reading. (Alternative for S > 8: FP32 partials to global + atomic
   ticket, last arriver reduces; slower.)
9. **Hide the head and the tail with PDL**: launch with `cudaLaunchAttributeProgrammaticStreamSerialization`;
   `griddepcontrol.launch_dependents` at kernel start; the producer issues the first STAGES weight and
   weight-scale loads, *then* `griddepcontrol.wait`, then the activation loads (activations come from the
   previous kernel; weights never do). Keep smem <= ~112 KB and TMEM <= 128 columns per CTA so the next
   grid's CTA can co-reside with this grid's tail.
10. **Cache hints**: weights `EVICT_FIRST` (read once), activations/scales `EVICT_LAST` (re-read by every
   N tile).

## Measuring bandwidth honestly
- Rotate through enough weight copies that the total exceeds ~4x L2 (B300 L2 is large: a 15-20 MB weight
  read twice in a row is an L2 hit). The example also prints an L2-hot number for contrast.
- Report per-launch time from back-to-back launches *and* from a CUDA graph; standalone kernel time
  includes launch/ramp/tail, which a fused or PDL-chained pipeline partly hides.
- `%globaltimer` stamps per CTA (build with `-DBS_TIMING`, `./bs_gemm stamps N K M`) split the kernel into
  setup, first-data latency, streaming, and tail. Stamps that store to global memory right before a
  release-ordered barrier delay that barrier, so trust A/B timings of variants more than stamp deltas.

## Where the time goes (example, B300; full table in `example/RESULTS.md`)
- **Steady-state stream is near HBM rate**: at M=8 the extra time from 15.6 -> 31.2 -> 62.4 MB of weight
  is ~2.1 us and ~4.6 us, i.e. ~7 TB/s marginal. At M=48 the marginal rate is ~5 TB/s.
- **Each standalone launch has ~3.4 us of fixed cost** (M=8; ~4 us at M=48): CTA setup + TMEM alloc
  (~0.4 us), first TMA data (~0.6-1 us after start, mostly HBM latency + descriptor fetch), split-K
  reduction and cluster barriers (~1 us), and the launch gap that PDL only partly hides. So per-launch
  bandwidth depends on size: ~2.8 TB/s at 15.6 MB, ~3 TB/s at 18.9 MB, ~4 TB/s at 31 MB, ~5 TB/s at 62 MB.
  The 4 TB/s target for an ~19 MB weight is **not** reached by this standalone kernel.
- What moved the numbers (graph time, N=4096 K=7168 M=8): first version (1 CTA/SM, 5 x 32 KB stages,
  global split-K with atomic ticket) 10.3 us -> + PDL 6.8 -> DSMEM cluster reduction with float4
  loads 6.4 -> smaller stages so 2 CTAs fit per SM, split-K 8: 5.6 us. Tried and rejected: pushing
  partials with scalar `st.async` into a dedicated receive buffer (slower: one 4 B push per token per
  thread, and the buffer costs ring stages); 3 CTAs/SM (fewer stages, slower at M=48).
- **In a real kernel the fixed costs are what to attack**: run the GEMM inside a persistent kernel or
  PDL-chain it after its producer so weights stream before the activations exist (prologue before
  `griddepcontrol.wait`); batch several matrices (experts) into one launch with a work list so CTAs stream
  continuously; reduce split-K partials without two cluster barriers (e.g. `red.global.add.f32` into a
  pre-zeroed FP32 output, or a vectorized DSMEM push).

## Numerics
- MXFP8 activations: quantize each 32-value block of a row with scale `2^ceil(log2(amax/448))` (UE8M0 =
  exponent + 127) and E4M3 values (saturating). Check the tolerance your checker allows; FP8 activations
  change results versus BF16 activations (relative error ~2^-4 per element, mostly averaging out in the sum).
- Accumulation is FP32 in TMEM; split-K partial sums in FP32; output rounded once to BF16.

## Pitfalls seen while building the example
- Max dynamic smem per CTA is 227 KB; the ring plus barriers must fit (the first version did not and the
  launch failed with `invalid argument`).
- Using `cudaGetDriverEntryPoint` is deprecated in CUDA 13; link `-lcuda` and call `cuTensorMapEncodeTiled`.
- `kind::i8` is rejected by ptxas for sm_103a.
- `asm volatile(... ::: "memory")` on every DSMEM load serializes them; issue all peer loads first.
- A global split-K reduction with partials in `[n][token]` order made each thread's stores strided; 1-3 us
  of tail. Write partials in the order the reducer reads them, or reduce through DSMEM.

## Source and license
Patterns (instruction descriptor bits, 16U4_ALIGN16B weight map, 128x4 scale atoms with
`tcgen05.cp 32x128b.warpx4`, warp-specialized mbarrier ring, PDL weight prologue) were adapted from
sgl-project/sgl-nvidia-mega (Apache-2.0), `sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gemm1.cu`, `gemm2.cu` and
`routed_expert_gemm.py`; see `NOTICE`. No shapes or layouts of that model are used here; the kernel was
written for this example.
