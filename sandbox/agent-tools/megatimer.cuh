// megatimer.cuh: in-kernel phase timing for the development build (DECISIONS section 15).
//
// Compile the development build with -DMEGATIMER and call MEGATIMER_MARK(phase_id) from one
// thread per block at the boundaries you care about; each mark stores (phase_id, block, %globaltimer
// in ns, clock64) into a device buffer `megatimer_records` that `phase_timeline` reads back and
// prints as a per-phase timeline. The graded build must be compiled WITHOUT -DMEGATIMER: the
// evaluator's source/artifact scan rejects submissions whose build flags or binaries contain the
// macro or the `megatimer_records` symbol (scan.py rules timer_macro).
//
// Usage inside a kernel (one thread per block, e.g. threadIdx.x == 0):
//     MEGATIMER_MARK(3);
// Host side, before the launch:  megatimer_reset();  after:  megatimer_dump("timeline.bin");
#pragma once
#include <cstdint>
#include <cstdio>

#ifdef MEGATIMER
#define MEGATIMER_CAPACITY (1u << 20)
struct megatimer_record { uint32_t phase; uint32_t block; uint64_t globaltimer_ns; uint64_t clock64; };
__device__ megatimer_record megatimer_records[MEGATIMER_CAPACITY];
__device__ unsigned int megatimer_count;

__device__ __forceinline__ uint64_t megatimer_globaltimer() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

#define MEGATIMER_MARK(phase_id) do {                                                 \
    unsigned int _i = atomicAdd(&megatimer_count, 1u);                                \
    if (_i < MEGATIMER_CAPACITY) {                                                    \
      megatimer_records[_i].phase = (phase_id);                                       \
      megatimer_records[_i].block = blockIdx.x + gridDim.x * blockIdx.y;              \
      megatimer_records[_i].globaltimer_ns = megatimer_globaltimer();                 \
      megatimer_records[_i].clock64 = clock64();                                      \
    }                                                                                 \
  } while (0)

static inline void megatimer_reset() {
  unsigned int zero = 0;
  cudaMemcpyToSymbol(megatimer_count, &zero, sizeof zero);
}

// Writes: uint32 count, then count records of 24 bytes each (little endian).
static inline void megatimer_dump(const char* path) {
  unsigned int count = 0;
  cudaMemcpyFromSymbol(&count, megatimer_count, sizeof count);
  if (count > MEGATIMER_CAPACITY) count = MEGATIMER_CAPACITY;
  megatimer_record* host = new megatimer_record[count ? count : 1];
  cudaMemcpyFromSymbol(host, megatimer_records, count * sizeof(megatimer_record));
  FILE* f = fopen(path, "wb");
  if (f) { fwrite(&count, sizeof count, 1, f); fwrite(host, sizeof(megatimer_record), count, f); fclose(f); }
  delete[] host;
}
#else
#define MEGATIMER_MARK(phase_id) do {} while (0)
static inline void megatimer_reset() {}
static inline void megatimer_dump(const char*) {}
#endif
