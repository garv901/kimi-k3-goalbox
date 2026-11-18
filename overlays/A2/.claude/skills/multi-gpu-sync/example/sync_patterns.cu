// Tested single-GPU versions of the multi-gpu-sync patterns (blocks stand in for ranks/GPUs).
//   1. epoch counters: producers publish rows with red.release, consumers wait with ld.acquire,
//      deadline + backoff; counters are never reset (3 launches reuse them via epochs).
//   2. Lamport mailbox: every "rank" (block) pushes 16 B vectors into every other rank's mailbox,
//      receivers poll LOCAL slots for a non-sentinel value, re-arm the sentinel; 2 generations.
//   3. grid barrier vs counters: cost of a full grid barrier per round vs per-row counters.
// Same code works across GPUs with .sys scope and peer pointers (P2P / symmetric memory).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a sync_patterns.cu -o sync_patterns
// Run:   ./sync_patterns          (prints PASS lines and timings; exit 1 on failure)
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// ---- pattern 1: release/acquire counters with deadline
__device__ __forceinline__ void signal(unsigned* ctr) {
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(ctr) : "memory");
}
__device__ __forceinline__ void wait_geq(const unsigned* ctr, unsigned target, unsigned long long deadline_ns) {
  if (threadIdx.x == 0) {
    unsigned v, n = 0; unsigned long long t0 = 0;
    while (true) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory");
      if ((int)(v - target) >= 0) break;
      if ((++n & 1023) == 0) {
        const unsigned long long now = gtime();
        if (!t0) t0 = now; else if (now - t0 > deadline_ns) { printf("timeout ctr %p v %u target %u\n", ctr, v, target); __trap(); }
      }
      if (n > 64) __nanosleep(64);
    }
  }
  __syncthreads();
}
// rows [0,R): producer block p writes row p (only after waiting for row p-1: a dependency chain),
// consumer blocks wait for each row and check it. ctr[r] counts publications of row r (1 per launch).
__global__ void counters_kernel(float* rows, unsigned* ctr, int R, int W, unsigned epoch, int* errors) {
  const int b = blockIdx.x;
  if (b < R) {  // producer of row b
    if (b > 0) wait_geq(&ctr[(b - 1) * 32], epoch + 1, 2000000000ull);
    for (int i = threadIdx.x; i < W; i += blockDim.x)
      rows[(size_t)b * W + i] = (b > 0 ? rows[(size_t)(b - 1) * W + i] : 0.f) + 1.f + epoch;
    signal(&ctr[b * 32]);  // one 128 B line per counter
  } else {      // consumer: verify every row
    for (int r = 0; r < R; ++r) {
      wait_geq(&ctr[r * 32], epoch + 1, 2000000000ull);
      for (int i = threadIdx.x; i < W; i += blockDim.x)
        if (rows[(size_t)r * W + i] != (float)(r + 1) * (1.f + epoch)) atomicAdd(errors, 1);
    }
  }
}

// ---- pattern 2: Lamport mailbox, sentinel = 0xFFFFFFFF words (never produced by senders)
constexpr unsigned EMPTY = 0xFFFFFFFFu;
// box[gen][dst][src][vec] uint4. Every block is a rank: pushes `vecs` vectors to every rank, then sums.
__global__ void mailbox_kernel(uint4* box, int ranks, int vecs, unsigned gen, unsigned round, unsigned* out, int* errors) {
  const int me = blockIdx.x;
  auto slot = [&](int dst, int src, int v) { return box + ((((size_t)gen * ranks + dst) * ranks + src) * vecs + v); };
  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    const unsigned val = round * 1000 + me * 10 + (v & 7);  // never EMPTY
    const uint4 x = make_uint4(val, val + 1, val + 2, val + 3);
    for (int dst = 0; dst < ranks; ++dst)
      asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(slot(dst, me, v)), "r"(x.x), "r"(x.y), "r"(x.z), "r"(x.w) : "memory");
  }
  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    unsigned sum = 0;
    for (int src = 0; src < ranks; ++src) {
      uint4 x; unsigned n = 0; unsigned long long t0 = 0;
      do {
        asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(slot(me, src, v)) : "memory");
        if ((++n & 1023) == 0) { const unsigned long long now = gtime(); if (!t0) t0 = now; else if (now - t0 > 2000000000ull) __trap(); }
      } while (x.x == EMPTY || x.y == EMPTY || x.z == EMPTY || x.w == EMPTY);
      const unsigned want = round * 1000 + src * 10 + (v & 7);
      if (x.x != want || x.w != want + 3) atomicAdd(errors, 1);
      sum += x.x;
      *slot(me, src, v) = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);  // re-arm this generation
    }
    out[me * vecs + v] = sum;
  }
}

// ---- pattern 3: R rounds; each round every block does a tiny bit of work, then either a full grid
// barrier or a wait on only its single dependency (block b waits for block b-1 of the same round).
__global__ void rounds_grid_kernel(int R, float* sink) {
  cg::grid_group g = cg::this_grid();
  float x = threadIdx.x;
  for (int r = 0; r < R; ++r) { x = x * 1.0001f + 1.f; g.sync(); }
  if (x == 12345.f) sink[0] = x;
}
__global__ void rounds_counter_kernel(int R, unsigned* ctr, unsigned base, float* sink) {
  float x = threadIdx.x;
  for (int r = 0; r < R; ++r) {
    x = x * 1.0001f + 1.f;
    signal(&ctr[blockIdx.x * 32]);
    const int dep = (blockIdx.x + gridDim.x - 1) % gridDim.x;
    wait_geq(&ctr[dep * 32], base + r + 1, 2000000000ull);
  }
  if (x == 12345.f) sink[0] = x;
}

int main() {
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  bool ok = true;
  {  // 1
    const int R = 64, W = 4096, C = 16;  // 64 producer blocks + 16 consumers; all co-resident (80 <= SMs)
    float* rows; unsigned* ctr; int* err;
    CK(cudaMalloc(&rows, (size_t)R * W * 4)); CK(cudaMalloc(&ctr, R * 32 * 4)); CK(cudaMalloc(&err, 4));
    CK(cudaMemset(ctr, 0, R * 32 * 4)); CK(cudaMemset(err, 0, 4));
    for (unsigned epoch = 0; epoch < 3; ++epoch) counters_kernel<<<R + C, 256>>>(rows, ctr, R, W, epoch, err);
    CK(cudaDeviceSynchronize());
    int e; CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    printf("{\"test\":\"epoch_counters\",\"launches\":3,\"errors\":%d,\"pass\":%s}\n", e, e ? "false" : "true");
    ok &= e == 0;
  }
  {  // 2
    const int ranks = 8, vecs = 1024;
    uint4* box; unsigned* out; int* err;
    const size_t n = (size_t)2 * ranks * ranks * vecs;
    CK(cudaMalloc(&box, n * 16)); CK(cudaMemset(box, 0xFF, n * 16)); CK(cudaMalloc(&out, ranks * vecs * 4));
    CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4));
    for (unsigned round = 0; round < 6; ++round) mailbox_kernel<<<ranks, 256>>>(box, ranks, vecs, round & 1, round, out, err);
    CK(cudaDeviceSynchronize());
    int e; CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    printf("{\"test\":\"lamport_mailbox\",\"rounds\":6,\"generations\":2,\"errors\":%d,\"pass\":%s}\n", e, e ? "false" : "true");
    ok &= e == 0;
  }
  {  // 3
    const int R = 2000; float* sink; unsigned* ctr; CK(cudaMalloc(&sink, 4));
    CK(cudaMalloc(&ctr, sms * 32 * 4)); CK(cudaMemset(ctr, 0, sms * 32 * 4));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int Rv = R; void* args[] = {&Rv, &sink};
    CK(cudaLaunchCooperativeKernel((void*)rounds_grid_kernel, sms, 256, args, 0, 0));  // warm-up
    CK(cudaEventRecord(a));
    CK(cudaLaunchCooperativeKernel((void*)rounds_grid_kernel, sms, 256, args, 0, 0));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms_grid; CK(cudaEventElapsedTime(&ms_grid, a, b));
    rounds_counter_kernel<<<sms, 256>>>(R, ctr, 0, sink);  // warm-up (epoch 0..R)
    CK(cudaEventRecord(a));
    rounds_counter_kernel<<<sms, 256>>>(R, ctr, R, sink);  // epochs continue, no reset
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms_ctr; CK(cudaEventElapsedTime(&ms_ctr, a, b));
    printf("{\"bench\":\"grid_sync_vs_neighbor_counter\",\"blocks\":%d,\"rounds\":%d,\"us_per_grid_sync\":%.3f,"
           "\"us_per_counter_round\":%.3f}\n", sms, R, ms_grid * 1e3 / R, ms_ctr * 1e3 / R);
  }
  printf(ok ? "ALL PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
