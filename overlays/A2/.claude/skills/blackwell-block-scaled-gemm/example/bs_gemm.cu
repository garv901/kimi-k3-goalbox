// Skinny block-scaled GEMM on Blackwell (sm_100a / sm_103a):  Y[M][N] = X[M][K] * W[N][K]^T
//   W: MXFP4 weights (E2M1, 2 per byte along K, low nibble = even k) + UE8M0 scale per 32 K
//   X: MXFP8 activations (E4M3) + UE8M0 scale per 32 K,   Y: BF16,   M (tokens) <= 64.
//
// Recipe (see ../SKILL.md):
//   * swap-AB: the weight is tcgen05 operand A (128 weight rows = MMA M), the tokens are
//     operand B (MMA N = M padded to a multiple of 8/16), the FP32 accumulator lives in TMEM
//     (lane = weight row, column = token).
//   * one instruction: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale, A = E2M1, B = E4M3,
//     K = 32 per MMA, scale factors read from TMEM (tcgen05.cp from smem).
//   * TMA loads packed FP4 with CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B: every 16 FP4 values
//     (8 bytes in HBM) land in a 16-byte smem slot, the layout the f8f6f4 MMA expects for FP4;
//     the smem tile is then a plain K-major SWIZZLE_128B tile (128 B per row = 128 FP4 values).
//   * warp specialization: warp 4 = TMA producer, warp 5 = MMA issuer, warps 0-3 = epilogue;
//     a STAGES-deep smem ring with full/empty mbarriers (empty is signalled by tcgen05.commit).
//   * split-K across CTAs to fill the SMs (N/128 tiles is far below 148 SMs); the last CTA of
//     each N tile (atomic ticket, self-resetting) sums the FP32 partials and writes BF16.
// Scale layout (generic choice, same "128x4 atom" CUTLASS uses for block-scaled MMA):
//   one 512-byte atom holds scales of 128 rows x 4 K-blocks (128 K values):
//   byte(r, kb) = (r % 32) * 16 + (r / 32) * 4 + kb.
//   SFA: [N/128][K/128] atoms.  SFB: [K/128] atoms, rows >= M zero.
// Patterns adapted from sgl-nvidia-mega (Apache-2.0, see ../NOTICE): instruction descriptor,
// 16U4_ALIGN16B weight map, tcgen05.cp 32x128b.warpx4 scale copies, mbarrier ring.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a bs_gemm.cu -o bs_gemm -lcuda
// Run:   ./bs_gemm test            # correctness on several shapes (exit 1 on mismatch)
//        ./bs_gemm bench [N K]     # bandwidth for M in {8,16,32,48,64}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = 0; cuGetErrorString(r_, &s_); \
  fprintf(stderr, "CU error %s at %s:%d\n", s_ ? s_ : "?", __FILE__, __LINE__); exit(2); } } while (0)

constexpr int BM = 128;          // weight rows per CTA (MMA M)
constexpr int BK = 128;          // K per pipeline stage = one SW128 row of FP4-in-16B-slots
constexpr int A_STAGE = BM * BK; // smem bytes per A stage (FP4 in 16B-aligned slots = 1 B per value)
constexpr int A_TX = BM * BK / 2;// bytes TMA moves from HBM per A stage (the tx count is the packed size)
constexpr int SF_STAGE = 512;    // one 512 B scale atom (128 rows x 4 K-blocks) per stage, for A and for B
#ifndef BS_SMEM_KB
#define BS_SMEM_KB 112
#endif
constexpr int SMEM_BUDGET = BS_SMEM_KB * 1024;  // <= half an SM: two CTAs per SM (split-K CTAs, or this grid's tail + the next grid via PDL)
constexpr int THREADS = 192;     // warps 0-3 epilogue, 4 TMA, 5 MMA

template <int MP> struct Cfg {
  static constexpr int B_STAGE = MP * BK;
  static constexpr int STAGES_FIT = (SMEM_BUDGET - 1280) / (A_STAGE + B_STAGE + 2 * SF_STAGE);
  static constexpr int STAGES = STAGES_FIT > 8 ? 8 : STAGES_FIT;
  static constexpr int OFF_A = 0;
  static constexpr int OFF_B = OFF_A + STAGES * A_STAGE;
  static constexpr int OFF_SFA = OFF_B + STAGES * B_STAGE;
  static constexpr int OFF_SFB = OFF_SFA + STAGES * SF_STAGE;
  static constexpr int OFF_BAR = OFF_SFB + STAGES * SF_STAGE;
  static constexpr int SMEM = OFF_BAR + 256 + 1024;  // + barriers, + 1 KB alignment slack
  // TMEM columns: accumulator [0, MP), then per-stage scale slots (8 columns each = 2 atoms)
  static constexpr int COL_SFA = 64;
  static constexpr int COL_SFB = COL_SFA + 4 * STAGES;
  static constexpr int TMEM_COLS = 128;  // two co-resident CTAs fit in the SM's 512 columns
  static_assert(COL_SFB + 4 * STAGES <= TMEM_COLS, "TMEM columns");
  static_assert(STAGES >= 2 && MP * BM * 4 <= STAGES * A_STAGE, "stages / reduction buffer");
  static_assert(MP % 8 == 0 && MP >= 8 && MP <= 64, "MP");
};

// ------------------------------------------------------------------ device helpers (PTX)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n.reg .pred P;\nelect.sync _|P, 0xffffffff;\n@P mov.u32 %0, 1;\n}\n" : "+r"(pred));
  return pred;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
// Spin on a phase parity. Bounded: traps after ~2^26 polls so a broken pipeline faults
// instead of hanging the GPU (see multi-gpu-sync skill: every spin needs a deadline).
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0, n = 0;
  while (true) {
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}\n"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return;
    if (++n > (1u << 26)) { printf("mbar_wait timeout block %d thread %d bar %u\n", blockIdx.x, threadIdx.x, bar); asm volatile("trap;"); }
  }
}
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ull;  // CUTLASS TMA cache-hint encodings
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ull;
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%2, %3, %4}], [%5], %6;"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(hint) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "l"(hint) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t hint) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(hint) : "memory");
}
// K-major SWIZZLE_128B smem descriptor: 8-row groups 1024 B apart (SBO), version 1 (bit 46).
__device__ __forceinline__ uint64_t sdesc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (uint64_t)1 << 16 | (uint64_t)(1024 >> 4) << 32 |
         (uint64_t)1 << 46 | (uint64_t)2 << 61;
}
// Descriptor for tcgen05.cp of one 512 B scale atom (no swizzle, 8-row groups 128 B apart).
__device__ __forceinline__ uint64_t sdesc_sf(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (uint64_t)(128 >> 4) << 32 | (uint64_t)1 << 46;
}
// Instruction descriptor, kind::mxf8f6f4.block_scale (bit layout as in sgl-nvidia-mega gemm1/2):
//   [4,6) SFB byte id, [7,10) A type (5 = E2M1), [10,13) B type (0 = E4M3), [17,23) N>>3,
//   bit 23 scale type UE8M0, [24,29) M>>4, [29,31) SFA byte id. A and B K-major.
__device__ __forceinline__ uint32_t idesc_mxf4x8(int n, int sf_id) {
  return (uint32_t)sf_id << 4 | 5u << 7 | 0u << 10 | (uint32_t)(n >> 3) << 17 | 1u << 23 |
         (uint32_t)(BM >> 4) << 24 | (uint32_t)sf_id << 29;
}
__device__ __forceinline__ void mma_bs(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc,
                                       uint32_t sfa_tmem, uint32_t sfb_tmem, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n}\n"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(accum));
}
__device__ __forceinline__ void tc_cp_sf(uint32_t tmem, uint32_t smem) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem), "l"(sdesc_sf(smem)) : "memory");
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}

#ifdef BS_TIMING
__device__ unsigned long long g_stamps[1024][8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define STAMP(i) g_stamps[blockIdx.x + gridDim.x * blockIdx.y][i] = gtime()
#else
#define STAMP(i)
#endif
struct Args {
  const uint8_t* sfa;   // [N/128][K/128][512]
  const uint8_t* sfb;   // [K/128][512]
  __nv_bfloat16* y;     // [M][N]
  int M, N, K, splits;  // splits = cluster size along grid y (1..8)
};

__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

template <int MP>
__global__ void __launch_bounds__(THREADS, 2)
bs_gemm_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap, Args args) {
  using C = Cfg<MP>;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);  // SW128 needs 1 KB alignment
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar_full = sbase + C::OFF_BAR, bar_empty = bar_full + 8 * C::STAGES;
  const uint32_t bar_acc = bar_empty + 8 * C::STAGES;
  uint32_t* tmem_slot = (uint32_t*)(smem + C::OFF_BAR + 8 * (2 * C::STAGES + 1));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int ntile = blockIdx.x, split = blockIdx.y;  // cluster (1, splits, 1): split == rank in cluster
  const int ksteps = args.K / BK;
  const int s0 = (int)((long long)split * ksteps / args.splits);
  const int s1 = (int)((long long)(split + 1) * ksteps / args.splits);
  const int nsteps = s1 - s0;

  if (threadIdx.x == 0) {
    STAMP(0);
    // PDL: let the next kernel in the stream get scheduled now; it only touches its own weights
    // until its griddepcontrol.wait, which waits for this whole grid to finish.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  if (warp == 4 && lane == 0) {
    for (int s = 0; s < C::STAGES; ++s) { mbar_init(bar_full + 8 * s, 1); mbar_init(bar_empty + 8 * s, 1); }
    mbar_init(bar_acc, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();  // barriers initialised; TMEM is allocated below, off the producer's path

  if (warp == 4) {
    // ---------------- TMA producer: starts streaming weights before TMEM alloc and before the PDL wait
    if (elect_one()) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&wmap) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
      const size_t KC = args.K / 128;
      auto load_w = [&](int i) {  // weights + weight scales: independent of earlier kernels
        const int s = i % C::STAGES, ks = s0 + i;
        const uint32_t full = bar_full + 8 * s;
        mbar_expect_tx(full, A_TX + MP * 128 + 2 * SF_STAGE);
        // W: box {128 values, 128 rows, 1 chunk} -> smem [row][128 B], SW128
        tma_3d(sbase + C::OFF_A + s * A_STAGE, &wmap, 0, ntile * BM, ks, full, EVICT_FIRST);
        bulk_g2s(sbase + C::OFF_SFA + s * SF_STAGE, args.sfa + ((size_t)ntile * KC + ks) * 512, SF_STAGE, full, EVICT_FIRST);
      };
      auto load_x = [&](int i) {  // activations + their scales: produced by the previous kernel
        const int s = i % C::STAGES, ks = s0 + i;
        const uint32_t full = bar_full + 8 * s;
        tma_2d(sbase + C::OFF_B + s * C::B_STAGE, &xmap, ks * BK, 0, full, EVICT_LAST);
        bulk_g2s(sbase + C::OFF_SFB + s * SF_STAGE, args.sfb + (size_t)ks * 512, SF_STAGE, full, EVICT_LAST);
      };
      const int npre = nsteps < C::STAGES ? nsteps : C::STAGES;
      for (int i = 0; i < npre; ++i) load_w(i);           // fill the ring with weights
      asm volatile("griddepcontrol.wait;" ::: "memory");  // X / scales of X may come from the previous kernel
      for (int i = 0; i < npre; ++i) load_x(i);
      for (int i = npre; i < nsteps; ++i) {
        mbar_wait(bar_empty + 8 * (i % C::STAGES), ((i / C::STAGES) & 1) ^ 1);
        load_w(i); load_x(i);
      }
    }
    __syncwarp();
  } else {
    if (warp == 0) {  // TMEM allocation is warp-wide
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(tmem_slot)), "r"(C::TMEM_COLS));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("bar.sync 1, 160;" ::: "memory");  // warps 0-3 and 5
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = *tmem_slot;
    if (threadIdx.x == 0) STAMP(1);
    if (warp == 5) {
      // ---------------- MMA issuer (one thread issues everything; tcgen05 ops run in issue order)
      if (elect_one()) {
        for (int i = 0; i < nsteps; ++i) {
          const int s = i % C::STAGES;
          mbar_wait(bar_full + 8 * s, (i / C::STAGES) & 1);
          if (i == 0) STAMP(2);
          if (i == nsteps - 1) STAMP(3);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          const uint32_t sfa_col = tmem + C::COL_SFA + 4 * s, sfb_col = tmem + C::COL_SFB + 4 * s;
          // scales smem -> TMEM: one 512 B atom = 128 rows x 4 K-blocks -> 4 columns x 128 lanes
          tc_cp_sf(sfa_col, sbase + C::OFF_SFA + s * SF_STAGE);
          tc_cp_sf(sfb_col, sbase + C::OFF_SFB + s * SF_STAGE);
          const uint64_t a = sdesc_sw128(sbase + C::OFF_A + s * A_STAGE);
          const uint64_t b = sdesc_sw128(sbase + C::OFF_B + s * C::B_STAGE);
          for (int j = 0; j < 4; ++j)  // K = 32 per MMA = 32 B along the SW128 row (desc unit 16 B);
            mma_bs(tmem, a + 2 * j, b + 2 * j, idesc_mxf4x8(MP, j), sfa_col, sfb_col, (i | j) != 0);  // byte j = K-block j
          tc_commit(bar_empty + 8 * s);  // frees the smem stage when these MMAs (and the cps) finish
        }
        tc_commit(bar_acc);
      }
      __syncwarp();
    } else {
      // ---------------- epilogue: warps 0-3 own TMEM lanes 32*warp .. +31 (= weight rows)
      mbar_wait(bar_acc, 0);
      if (threadIdx.x == 0) STAMP(4);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const int row = warp * 32 + lane;
      float* red = (float*)(smem + C::OFF_A);  // ring is idle now: [MP][128] FP32 partial of this CTA
#pragma unroll
      for (int c = 0; c < MP; c += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(tmem + ((uint32_t)(warp * 32) << 16) + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int q = 0; q < 8; ++q) red[(c + q) * BM + row] = __uint_as_float(r[q]);
      }
      if (threadIdx.x == 0) STAMP(6);
#ifdef BS_EPI_GDC_WAIT
      asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
      // No griddepcontrol.wait needed here before writing Y: this point is causally after the
      // producer's griddepcontrol.wait (its X loads fed the MMAs that signalled bar_acc).
    }
  }
  // ---------------- split-K reduction through DSMEM. Work = M x 32 groups of 4 rows; CTA `split`
  // takes a contiguous 1/S of them, reads the 4-float group from every peer's smem (float4 DSMEM
  // loads, all issued before use), sums in FP32 and writes 4 BF16 (coalesced along N).
  if (args.splits > 1) cluster_sync_all();  // every CTA's partial is in its smem
  if (threadIdx.x == 0) STAMP(7);
  if (warp < 4) {
    const float* red = (const float*)(smem + C::OFF_A);
    const int G = args.M * (BM / 4);
    const int g0 = split * G / args.splits, g1 = (split + 1) * G / args.splits;
    for (int g = g0 + threadIdx.x; g < g1; g += 128) {
      const int m = g / (BM / 4), r4 = (g % (BM / 4)) * 4;
      const uint32_t local = smem_u32(red + m * BM + r4);
      float4 v[8];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        if (p < args.splits) {
          uint32_t remote;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(p));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(v[p].x), "=f"(v[p].y), "=f"(v[p].z), "=f"(v[p].w) : "r"(remote));
        }
      }
      float4 t = v[0];
#pragma unroll
      for (int p = 1; p < 8; ++p)
        if (p < args.splits) { t.x += v[p].x; t.y += v[p].y; t.z += v[p].z; t.w += v[p].w; }
      __nv_bfloat162 lo = __floats2bfloat162_rn(t.x, t.y), hi = __floats2bfloat162_rn(t.z, t.w);
      uint2 out; out.x = *(uint32_t*)&lo; out.y = *(uint32_t*)&hi;
      *(uint2*)(args.y + (size_t)m * args.N + ntile * BM + r4) = out;
    }
  }
#ifndef BS_REL2_RELEASE
  // keep smem alive until every peer has read it; relaxed: nothing of ours needs publishing here
  if (args.splits > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
#else
  if (args.splits > 1) cluster_sync_all();
#endif
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();  // all TMEM reads are done before warp 0 frees TMEM
  if (threadIdx.x == 0) STAMP(5);
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(*tmem_slot), "r"(C::TMEM_COLS));
  }
}

// ------------------------------------------------------------------ host side
static float e2m1_val(int code) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (code & 8 ? -1.f : 1.f) * mag[code & 7];
}
static float e4m3_val(uint8_t v) {
  const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
  const float f = e == 0 ? ldexpf((float)m / 8.f, -6) : ldexpf(1.f + (float)m / 8.f, e - 7);
  return s ? -f : f;
}
static float ue8m0_val(uint8_t e) { return ldexpf(1.f, (int)e - 127); }
static size_t sf_off(int r, int kb, int KC) {  // offset inside one operand's atom array, rows tiled by 128
  return ((size_t)(r / 128) * KC + kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}

// W packed [N][K/2] bytes viewed as 3-D {128 values, N rows, K/128 chunks}; box {128, 128, 1}.
static CUtensorMap make_wmap(const void* w, int N, int K) {
  CUtensorMap m;
  cuuint64_t dims[3] = {128, (cuuint64_t)N, (cuuint64_t)(K / 128)};
  cuuint64_t strides[2] = {(cuuint64_t)K / 2, 64};  // bytes, dims 1..2
  cuuint32_t box[3] = {128, BM, 1}, es[3] = {1, 1, 1};
  CKD(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, (void*)w, dims, strides, box, es,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
// X [MP][K] bytes (rows >= M zero); box {128 B, MP rows}.
static CUtensorMap make_xmap(const void* x, int MP, int K) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)MP};
  cuuint64_t strides[1] = {(cuuint64_t)K};
  cuuint32_t box[2] = {128, (cuuint32_t)MP}, es[2] = {1, 1};
  CKD(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)x, dims, strides, box, es,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

static bool g_pdl = true;  // launch with programmatic dependent launch (PDL)

struct Problem {
  int M, MP, N, K;
  std::vector<uint8_t> w, sfa, x, sfb;  // host copies in device layout
};
static Problem make_problem(int M, int N, int K, uint32_t seed) {
  Problem p; p.M = M; p.N = N; p.K = K; p.MP = M <= 8 ? 8 : (M + 15) / 16 * 16;
  std::mt19937 rng(seed);
  const int KC = K / 128;
  p.w.resize((size_t)N * K / 2); p.sfa.assign((size_t)(N / 128) * KC * 512, 0);
  p.x.assign((size_t)p.MP * K, 0); p.sfb.assign((size_t)KC * 512, 0);
  for (auto& b : p.w) b = (uint8_t)rng();
  for (int r = 0; r < N; ++r) for (int kb = 0; kb < K / 32; ++kb) p.sfa[sf_off(r, kb, KC)] = (uint8_t)(124 + rng() % 7);
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) {
    uint8_t v; do { v = (uint8_t)rng(); } while ((v & 0x7F) == 0x7F || ((v >> 3) & 15) > 9);  // |x| <= ~2^2, no NaN
    p.x[(size_t)m * K + k] = v;
  }
  for (int m = 0; m < M; ++m) for (int kb = 0; kb < K / 32; ++kb) p.sfb[sf_off(m, kb, KC)] = (uint8_t)(122 + rng() % 9);
  return p;
}
static void reference(const Problem& p, std::vector<double>& y) {
  const int K = p.K, KC = K / 128;
  std::vector<float> xd((size_t)p.M * K);
  for (int m = 0; m < p.M; ++m) for (int k = 0; k < K; ++k)
    xd[(size_t)m * K + k] = e4m3_val(p.x[(size_t)m * K + k]) * ue8m0_val(p.sfb[sf_off(m, k / 32, KC)]);
  y.assign((size_t)p.M * p.N, 0.0);
  std::vector<float> wd(K);
  for (int n = 0; n < p.N; ++n) {
    for (int k = 0; k < K; ++k) {
      const uint8_t b = p.w[(size_t)n * K / 2 + k / 2];
      wd[k] = e2m1_val(k & 1 ? b >> 4 : b & 15) * ue8m0_val(p.sfa[sf_off(n, k / 32, KC)]);
    }
    for (int m = 0; m < p.M; ++m) {
      double s = 0; const float* xr = &xd[(size_t)m * K];
      for (int k = 0; k < K; ++k) s += (double)wd[k] * xr[k];
      y[(size_t)m * p.N + n] = s;
    }
  }
}

struct Dev {
  int copies = 1;
  std::vector<uint8_t*> w, sfa; uint8_t *x = 0, *sfb = 0; __nv_bfloat16* y = 0;
  std::vector<CUtensorMap> wmap; CUtensorMap xmap;
};
static int pick_splits(int N, int K) {
  int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int tiles = N / BM, ksteps = K / BK;
  int best = 1;
  for (int s = 1; s <= std::min(ksteps, 8); ++s) if (tiles * s <= 2 * sms) best = s;  // cluster <= 8, <= 2 CTAs/SM
  return best;
}
static void upload(const Problem& p, Dev& d, int copies, int splits) {
  d.copies = copies;
  for (int c = 0; c < copies; ++c) {
    uint8_t *w, *sfa;
    CK(cudaMalloc(&w, p.w.size())); CK(cudaMalloc(&sfa, p.sfa.size()));
    CK(cudaMemcpy(w, p.w.data(), p.w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sfa, p.sfa.data(), p.sfa.size(), cudaMemcpyHostToDevice));
    d.w.push_back(w); d.sfa.push_back(sfa); d.wmap.push_back(make_wmap(w, p.N, p.K));
  }
  CK(cudaMalloc(&d.x, p.x.size())); CK(cudaMemcpy(d.x, p.x.data(), p.x.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.sfb, p.sfb.size())); CK(cudaMemcpy(d.sfb, p.sfb.data(), p.sfb.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.y, (size_t)p.M * p.N * 2)); CK(cudaMemset(d.y, 0xFF, (size_t)p.M * p.N * 2));  // poison
  d.xmap = make_xmap(d.x, p.MP, p.K);
}
static void release(Dev& d) {
  for (auto q : d.w) cudaFree(q);
  for (auto q : d.sfa) cudaFree(q);
  cudaFree(d.x); cudaFree(d.sfb); cudaFree(d.y);
}
template <int MP>
static void launch_t(const Problem& p, Dev& d, int copy, int splits, cudaStream_t st) {
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(bs_gemm_kernel<MP>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<MP>::SMEM)); attr = true; }
  Args a{d.sfa[copy], d.sfb, d.y, p.M, p.N, p.K, splits};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.N / BM, splits); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = Cfg<MP>::SMEM; cfg.stream = st;
  cudaLaunchAttribute at[2]; int na = 0;
  at[na].id = cudaLaunchAttributeClusterDimension; at[na].val.clusterDim.x = 1; at[na].val.clusterDim.y = splits; at[na].val.clusterDim.z = 1; ++na;
  if (g_pdl) { at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[na].val.programmaticStreamSerializationAllowed = 1; ++na; }
  cfg.attrs = at; cfg.numAttrs = na;
  CK(cudaLaunchKernelEx(&cfg, bs_gemm_kernel<MP>, d.wmap[copy], d.xmap, a));
}
static void launch(const Problem& p, Dev& d, int copy, int splits, cudaStream_t st = 0) {
  switch (p.MP) {
    case 8: launch_t<8>(p, d, copy, splits, st); break;
    case 16: launch_t<16>(p, d, copy, splits, st); break;
    case 32: launch_t<32>(p, d, copy, splits, st); break;
    case 48: launch_t<48>(p, d, copy, splits, st); break;
    case 64: launch_t<64>(p, d, copy, splits, st); break;
    default: fprintf(stderr, "unsupported MP %d\n", p.MP); exit(2);
  }
}

static bool run_test(int M, int N, int K, int splits_override, bool mutate = false) {
  Problem p = make_problem(M, N, K, 1234 + M * 7 + N + K);
  const int splits = splits_override > 0 ? splits_override : pick_splits(N, K);
  Dev d; upload(p, d, 1, splits);
  if (mutate) {  // self-check of the test: change one weight scale on the device only; must be caught
    const uint8_t v = p.sfa[sf_off(77, 5, K / 128)] + 3;
    CK(cudaMemcpy(d.sfa[0] + sf_off(77, 5, K / 128), &v, 1, cudaMemcpyHostToDevice));
  }
  bool ok = true; double max_err = 0, max_ref = 0;
  std::vector<double> ref; reference(p, ref);
  for (double v : ref) max_ref = std::max(max_ref, std::fabs(v));
  for (int rep = 0; rep < 3 && ok; ++rep) {  // repeat: launches must not depend on leftover state
    CK(cudaMemset(d.y, 0xFF, (size_t)M * N * 2));
    launch(p, d, 0, splits); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> y((size_t)M * N);
    CK(cudaMemcpy(y.data(), d.y, y.size() * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      const double g = __bfloat162float(y[i]), r = ref[i], err = std::fabs(g - r);
      max_err = std::max(max_err, err);
      // BF16 output: 2^-8 relative rounding, plus FP32-accumulation slack against the row scale
      if (!(err <= std::fabs(r) * (1.0 / 256) * 1.01 + max_ref * 1e-5)) {
        if (bad++ < 5 && !mutate) fprintf(stderr, "  mismatch m=%zu n=%zu got %.6g ref %.6g\n", i / N, i % N, g, r);
      }
    }
    if (bad) { ok = false; if (!mutate) fprintf(stderr, "  %d mismatches\n", bad); }
  }
  printf("{\"test\":\"bs_gemm\",\"M\":%d,\"N\":%d,\"K\":%d,\"splits\":%d,\"pdl\":%d,\"mutated\":%d,\"max_abs_err\":%.4g,\"max_abs_ref\":%.4g,\"pass\":%s}\n",
         M, N, K, splits, (int)g_pdl, (int)mutate, max_err, max_ref, ok ? "true" : "false");
  release(d);
  return ok;
}

static int stages_for(int MP) {
  return MP == 8 ? Cfg<8>::STAGES : MP == 16 ? Cfg<16>::STAGES : MP == 32 ? Cfg<32>::STAGES : MP == 48 ? Cfg<48>::STAGES : Cfg<64>::STAGES;
}
// Median per-launch time (us) of `iters` back-to-back launches; graph: captured in one CUDA graph.
static double time_launches(const Problem& p, Dev& d, int splits, cudaStream_t st, int iters, bool rotate, bool graph) {
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  cudaGraphExec_t gexec = nullptr;
  if (graph) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) launch(p, d, rotate ? i % d.copies : 0, splits, st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&gexec, g, 0)); CK(cudaGraphDestroy(g));
    CK(cudaGraphLaunch(gexec, st)); CK(cudaStreamSynchronize(st));
  }
  std::vector<double> per;
  for (int rep = 0; rep < 5; ++rep) {
    CK(cudaEventRecord(e0, st));
    if (graph) CK(cudaGraphLaunch(gexec, st));
    else for (int i = 0; i < iters; ++i) launch(p, d, rotate ? i % d.copies : 0, splits, st);
    CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); per.push_back(ms * 1e3 / iters);
  }
  if (gexec) CK(cudaGraphExecDestroy(gexec));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  std::sort(per.begin(), per.end());
  return per[per.size() / 2];
}

// bench: one JSON line per (M, PDL on/off). rows: primdb-style rows (p./m./c. keys) for `./primdb add`.
static void run_bench(int M, int N, int K, int splits_override, bool rows) {
  for (int pdl = 1; pdl >= (rows ? 1 : 0); --pdl) {
    g_pdl = pdl;
    Problem p = make_problem(M, N, K, 99);
    const int splits = splits_override > 0 ? splits_override : pick_splits(N, K);
    const double wbytes = (double)p.w.size() + p.sfa.size();
    int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const int copies = std::max(2, (int)std::ceil(4.0 * l2 / wbytes));  // rotate so W never sits in L2
    Dev d; upload(p, d, copies, splits);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    for (int i = 0; i < 2 * copies; ++i) launch(p, d, i % copies, splits, st);
    CK(cudaStreamSynchronize(st));
    const int iters = std::max(200, 4 * copies);
    const double us_stream = time_launches(p, d, splits, st, iters, true, false);
    const double us_graph = time_launches(p, d, splits, st, iters, true, true);
    const double us_hot = time_launches(p, d, splits, st, iters, false, true);
    if (!rows) {
      printf("{\"bench\":\"bs_gemm\",\"M\":%d,\"N\":%d,\"K\":%d,\"splits\":%d,\"ctas\":%d,\"pdl\":%d,\"weight_mb\":%.2f,"
             "\"us_stream\":%.3f,\"tb_s_stream\":%.3f,\"us_graph\":%.3f,\"tb_s_graph\":%.3f,\"us_graph_l2_hot\":%.3f,\"copies\":%d}\n",
             M, N, K, splits, N / BM * splits, pdl, wbytes / 1e6, us_stream, wbytes / (us_stream * 1e-6) / 1e12,
             us_graph, wbytes / (us_graph * 1e-6) / 1e12, us_hot, copies);
    } else {
      for (int warm = 0; warm < 2; ++warm) {
        const double us = warm ? us_hot : us_graph;
        printf("{\"family\":\"gemm\",\"primitive\":\"gemm.bs_mxfp4\",\"p.M\":%d,\"p.N\":%d,\"p.K\":%d,"
               "\"p.engine\":\"tcgen05_mxf8f6f4_blockscale\",\"p.w_dtype\":\"mxfp4\",\"p.x_dtype\":\"mxfp8\","
               "\"p.config\":\"swapAB M128xN%d kstage=128 stages=%d splitK=%d cluster(1,%d) dsmem_pull pdl\","
               "\"p.tuned\":false,\"c.gpu\":\"B300\",\"c.arch\":\"sm_103a\",\"c.l2\":\"%s\",\"c.ctas\":%d,"
               "\"m.us\":%.4g,\"m.hbm_gbps\":%.6g,\"m.tflops\":%.6g,\"verified\":true,\"n_reps\":5,"
               "\"s.bench\":\".claude/skills/blackwell-block-scaled-gemm/example/bs_gemm.cu\","
               "\"note\":\"Y[M][N] bf16 = X[M][K] mxfp8 * W[N][K]^T mxfp4 (UE8M0 per 32), native block-scaled tcgen05; "
               "us = CUDA-graph average over back-to-back PDL launches (includes the inter-kernel gap); "
               "hbm_gbps counts W + W-scale bytes; l2 cold = rotating copies > 4x L2. Example kernel from the skill, "
               "a starting point, not tuned per shape.\"}\n",
               M, N, K, p.MP, stages_for(p.MP), splits, splits, warm ? "warm" : "cold", N / BM * splits,
               us, wbytes / (us * 1e-6) / 1e9, 2.0 * M * N * K / (us * 1e-6) / 1e12);
      }
    }
    CK(cudaStreamDestroy(st));
    release(d);
  }
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "test";
  int N = argc > 2 ? atoi(argv[2]) : 4096, K = argc > 3 ? atoi(argv[3]) : 7168;
  int splits = argc > 4 ? atoi(argv[4]) : 0;
  if (N % BM || K % BK) { fprintf(stderr, "N must be a multiple of %d and K of %d\n", BM, BK); return 2; }
  if (!strcmp(mode, "test")) {
    bool ok = true;
    if (argc > 2) {
      for (int M : {8, 48}) ok &= run_test(M, N, K, splits);
    } else {
      ok &= run_test(8, 256, 512, 1);    // single CTA per tile, 2 k-steps
      ok &= run_test(48, 256, 2048, 3);  // uneven split
      ok &= run_test(1, 128, 256, 1);
      ok &= run_test(33, 384, 1024, 2);
      ok &= run_test(64, 512, 1792, 0);
      ok &= run_test(8, 4096, 7168, 0);  // the benchmark shapes
      ok &= run_test(48, 4096, 7168, 0);
      g_pdl = false;
      ok &= run_test(16, 1024, 2048, 0);
      g_pdl = true;
      const bool caught = !run_test(8, 512, 1024, 0, true);  // a corrupted scale must fail the check
      printf("{\"test\":\"mutation\",\"caught\":%s}\n", caught ? "true" : "false");
      ok &= caught;
    }
    printf(ok ? "ALL PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
  }
#ifdef BS_TIMING
  if (!strcmp(mode, "stamps")) {
    int M = argc > 4 ? atoi(argv[4]) : 8;
    Problem p = make_problem(M, N, K, 5); const int splits = pick_splits(N, K);
    Dev d; upload(p, d, 2, splits); g_pdl = false;  // stamps of one grid without overlap
    for (int i = 0; i < 20; ++i) launch(p, d, i % 2, splits);
    CK(cudaDeviceSynchronize());
    static unsigned long long st[1024][8]; CK(cudaMemcpyFromSymbol(st, g_stamps, sizeof st));
    const int n = N / BM * splits; unsigned long long t0 = ~0ull;
    for (int b = 0; b < n; ++b) t0 = std::min(t0, st[b][0]);
    double mx[8] = {0}, av[8] = {0};
    for (int b = 0; b < n; ++b) for (int i = 0; i < 8; ++i) { double v = (st[b][i] - t0) / 1e3; mx[i] = std::max(mx[i], v); av[i] += v / n; }
    const char* nm[8] = {"start", "setup", "first_full", "last_full", "acc_done", "end", "tmem_read", "cl_sync1"};
    for (int i = 0; i < 8; ++i) printf("%-10s avg %.3f us  max %.3f us\n", nm[i], av[i], mx[i]);
    return 0;
  }
#endif
  if (!strcmp(mode, "bench")) {
    for (int M : {8, 16, 32, 48, 64}) run_bench(M, N, K, splits, false);
    return 0;
  }
  if (!strcmp(mode, "rows")) {  // shapes of the primbench gemm rows plus the example's own
    const int shapes[6][2] = {{4096, 4096}, {4096, 7168}, {4096, 8704}, {4096, 14336}, {8192, 8192}, {28672, 4096}};
    for (auto& sh : shapes) for (int M : {1, 2, 4, 8, 16, 32, 64}) run_bench(M, sh[0], sh[1], 0, true);
    return 0;
  }
  fprintf(stderr, "usage: %s test|bench [N K [splits]]\n", argv[0]);
  return 2;
}
