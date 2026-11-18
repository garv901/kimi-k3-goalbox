// phase_table.cuh: per-block, per-phase %globaltimer stamps for persistent (multi-rank) kernels.
//
// Everything below compiles to nothing unless the build defines PHASE_TABLE (-DPHASE_TABLE), so the
// scored build can keep the stamps in the source. `./phase_table` runs the instrumented build on all
// ranks and prints the per-phase, per-rank table. See TASK.d/A1.md for the full recipe.
//
// Device side (one writer per block: thread (0,0,0); stamps from other threads are ignored):
//   PT_INIT();                  first statement of the kernel (every thread may execute it)
//   PT_FINISH();                last statement of the kernel
//   PT_ARRIVE(id); PT_RELEASE(id);   inside your grid barrier: ARRIVE right after the block-level
//                               __syncthreads() that precedes the grid-wide wait, RELEASE right after
//                               the wait. One pair per barrier crossing; id = call-site id (e.g. the
//                               caller's __LINE__ passed down as an argument). Phases are the
//                               intervals between consecutive barriers.
//   PHASE_BEGIN(id); PHASE_END(id);  optional sub-phase interval inside a barrier interval (any id).
//   PT_MARK(id);                optional context marker (e.g. layer index); later phases are
//                               attributed to the context of the last mark.
//   PT_SPIN(stmt);              optional: runs stmt (a wait loop) and adds its duration (writer
//                               thread only) to the next barrier record as "spin".
// Host side (in the same .cu file as the kernel):
//   phase_table_bind(dev_ptr, words) -> 0 when instrumented, -1 when PHASE_TABLE is off.
//   PHASE_TABLE_PYBIND(m);      inside PYBIND11_MODULE: exposes m.phase_table_bind(ptr, words).
//
// Buffer (uint64 words, any int64/uint64 CUDA tensor; size it with phase_table_words(blocks, slots)):
//   [0] magic 'PHASETB1'  [1] gridDim.x  [2] words per block region  [3..15] reserved
//   region of block b at 16 + b * R: [0] record count, [1] pending spin ns, [2] kernel start ns,
//   [3] kernel end ns, [4] record slots, [5..7] reserved, then records of 4 words:
//   [t_a, t_b, kind << 56 | (uint32)id, spin ns].  Grids up to PHASE_TABLE_MAX_BLOCKS (4096) blocks.
//   kind 1 = barrier (t_a arrive, t_b release), 2 = phase begin, 3 = phase end, 4 = mark.
//   Records past the region end are counted but dropped (the table reports them).
#pragma once

#include <stdint.h>

#define PHASE_TABLE_MAGIC 0x3142545345534850ull /* "PHSESTB1" little endian */
#define PHASE_TABLE_HEADER_WORDS 16

static inline uint64_t phase_table_words(uint64_t blocks, uint64_t slots_per_block) {
  return PHASE_TABLE_HEADER_WORDS + blocks * (8 + 4 * slots_per_block);
}

#ifdef PHASE_TABLE

#include <cuda_runtime.h>

#ifndef PHASE_TABLE_MAX_BLOCKS
#define PHASE_TABLE_MAX_BLOCKS 4096
#endif
static __device__ uint64_t* pt_buffer_ = nullptr;
static __device__ uint64_t pt_words_ = 0;
static __device__ uint64_t* pt_reg_[PHASE_TABLE_MAX_BLOCKS];  // per-block region, set by PT_INIT each launch

__device__ __forceinline__ uint64_t pt_now_() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ bool pt_writer_() { return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0; }
__device__ __forceinline__ uint32_t pt_bid_() { return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); }
// region of this block, or nullptr (not writer / unbound / too many blocks)
__device__ __forceinline__ uint64_t* pt_region_() {
  const uint32_t b = pt_bid_();
  return (pt_writer_() && b < PHASE_TABLE_MAX_BLOCKS) ? pt_reg_[b] : nullptr;
}
__device__ __forceinline__ void pt_init_() {
  const uint32_t b = pt_bid_();
  if (!pt_writer_() || b >= PHASE_TABLE_MAX_BLOCKS) return;
  uint64_t* buf = pt_buffer_;
  if (buf == nullptr) { pt_reg_[b] = nullptr; return; }
  const uint64_t nb = (uint64_t)gridDim.x * gridDim.y * gridDim.z;
  const uint64_t R = (pt_words_ - PHASE_TABLE_HEADER_WORDS) / nb;
  if (R < 12) { pt_reg_[b] = nullptr; return; }
  if (b == 0) {
    buf[0] = PHASE_TABLE_MAGIC;
    buf[1] = nb;
    buf[2] = R;
  }
  uint64_t* r = buf + PHASE_TABLE_HEADER_WORDS + (uint64_t)b * R;
  r[0] = 0;
  r[1] = 0;
  r[3] = 0;
  r[4] = (R - 8) / 4;  // slots
  r[2] = pt_now_();
  pt_reg_[b] = r;
}
__device__ __forceinline__ void pt_finish_() {
  uint64_t* r = pt_region_();
  if (r) r[3] = pt_now_();
}
// kind 1: arrive opens the record, release closes it and advances the count
__device__ __forceinline__ void pt_arrive_(uint32_t id) {
  uint64_t* r = pt_region_();
  if (!r) return;
  const uint64_t t = pt_now_();
  const uint64_t n = r[0];
  if (n < r[4]) {
    uint64_t* q = r + 8 + 4 * n;
    q[0] = t;
    q[2] = (1ull << 56) | id;
  }
}
__device__ __forceinline__ void pt_release_(uint32_t id) {
  uint64_t* r = pt_region_();
  if (!r) return;
  const uint64_t t = pt_now_();
  const uint64_t n = r[0];
  if (n < r[4]) {
    uint64_t* q = r + 8 + 4 * n;
    q[1] = t;
    q[2] = (1ull << 56) | id;
    q[3] = r[1];
  }
  r[1] = 0;
  r[0] = n + 1;
}
__device__ __forceinline__ void pt_point_(uint32_t kind, uint32_t id) {
  uint64_t* r = pt_region_();
  if (!r) return;
  const uint64_t t = pt_now_();
  const uint64_t n = r[0];
  if (n < r[4]) {
    uint64_t* q = r + 8 + 4 * n;
    q[0] = t;
    q[1] = t;
    q[2] = ((uint64_t)kind << 56) | id;
    q[3] = 0;
  }
  r[0] = n + 1;
}

#define PT_INIT() pt_init_()
#define PT_FINISH() pt_finish_()
#define PT_ARRIVE(id) pt_arrive_((uint32_t)(id))
#define PT_RELEASE(id) pt_release_((uint32_t)(id))
#define PHASE_BEGIN(id) pt_point_(2u, (uint32_t)(id))
#define PHASE_END(id) pt_point_(3u, (uint32_t)(id))
#define PT_MARK(id) pt_point_(4u, (uint32_t)(id))
#define PT_SPIN(...)                                                                   \
  do {                                                                                 \
    const uint64_t pt_spin_t0_ = pt_now_();                                            \
    __VA_ARGS__;                                                                       \
    uint64_t* pt_r_ = pt_region_();                                                    \
    if (pt_r_) pt_r_[1] += pt_now_() - pt_spin_t0_;                                    \
  } while (0)

static inline int phase_table_bind(uint64_t dev_ptr, uint64_t words) {
  uint64_t* p = (uint64_t*)dev_ptr;
  if (words < PHASE_TABLE_HEADER_WORDS + 8) return -2;
  if (cudaMemcpyToSymbol(pt_buffer_, &p, sizeof(p)) != cudaSuccess) return -3;
  if (cudaMemcpyToSymbol(pt_words_, &words, sizeof(words)) != cudaSuccess) return -3;
  return 0;
}

#else  // PHASE_TABLE off: no code, no state

#define PT_INIT() do {} while (0)
#define PT_FINISH() do {} while (0)
#define PT_ARRIVE(id) do { (void)sizeof(id); } while (0)
#define PT_RELEASE(id) do { (void)sizeof(id); } while (0)
#define PHASE_BEGIN(id) do { (void)sizeof(id); } while (0)
#define PHASE_END(id) do { (void)sizeof(id); } while (0)
#define PT_MARK(id) do { (void)sizeof(id); } while (0)
#define PT_SPIN(...) do { __VA_ARGS__; } while (0)

static inline int phase_table_bind(uint64_t, uint64_t) { return -1; }

#endif  // PHASE_TABLE

#define PHASE_TABLE_PYBIND(m) \
  (m).def("phase_table_bind", [](uint64_t ptr, uint64_t words) { return phase_table_bind(ptr, words); })
