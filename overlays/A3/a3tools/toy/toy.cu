// Toy persistent kernel for the A3 self-test and the phases.md example: two phases per layer.
//   gemv: y[r][n] = sum_k x[r][k] * W[l][n][k]          (fp32 out, scratch)
//   act : x[r][n'] = bf16(silu(y[r][n']))   for n' < K   (feeds the next layer)
// grid barrier between phases. One launch runs NL layers.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define NT 256
#define R 8
#define K 1024
#define N 4096
#define NL 4
struct TP { uint64_t v[4]; };   // x (bf16 [R][K], updated in place), W (bf16 [NL][N][K]), y (f32 [R][N]), bar
#include "ss_task.cuh"

__device__ void gsync(unsigned* bar) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* g = bar + 1;
    unsigned gen = *g;
    __threadfence();
    if (atomicAdd(bar, 1u) == gridDim.x - 1) { atomicExch(bar, 0u); __threadfence(); atomicAdd(bar + 1, 1u); }
    else while (*g == gen) {}
    __threadfence();
  }
  __syncthreads();
}

__device__ void phase_gemv(const TP& p, int l) {
  const __nv_bfloat16* x = (const __nv_bfloat16*)p.v[0];
  const __nv_bfloat16* W = (const __nv_bfloat16*)p.v[1] + (size_t)l * N * K;
  float* y = (float*)p.v[2];
  const int lane = threadIdx.x & 31, wpb = NT / 32;
  for (int n = blockIdx.x * wpb + threadIdx.x / 32; n < N; n += gridDim.x * wpb) {
    float acc[R] = {0};
    for (int k = lane; k < K; k += 32) {
      const float w = __bfloat162float(W[(size_t)n * K + k]);
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] += w * __bfloat162float(__ldcg(&x[r * K + k]));
    }
#ifdef TOY_SLOW   // deliberately slower variant for the self-test: redo the dot product
    for (int k = lane; k < K; k += 32) {
      const float w = __bfloat162float(W[(size_t)n * K + k]);
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] += 1e-30f * w * __bfloat162float(__ldcg(&x[r * K + k]));
    }
#endif
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[r];
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) y[r * N + n] = v;
    }
  }
}

__device__ void phase_act(const TP& p) {
  __nv_bfloat16* x = (__nv_bfloat16*)p.v[0];
  const float* y = (const float*)p.v[2];
  for (int i = blockIdx.x * NT + threadIdx.x; i < R * K; i += gridDim.x * NT) {
    const int r = i / K, n = i % K;
    const float v = __ldcg(&y[r * N + n]) * 0.05f;
    x[i] = __float2bfloat16(v / (1.f + expf(-v)));
  }
}

__global__ void __launch_bounds__(NT, 1) toy_kernel(const __grid_constant__ TP p) {
  SS_CAP_PARAMS(&p, sizeof(p));
  unsigned* bar = (unsigned*)p.v[3];
  for (int l = 0; l < NL; ++l) {
    SS_CAP("gemv_in", l, SS_REG("x", p.v[0], R * K, SS_BF16));
    phase_gemv(p, l);
    gsync(bar);
    SS_CAP("gemv_out", l, SS_REG("y", p.v[2], R * N, SS_F32));
    phase_act(p);
    gsync(bar);
  }
}

#ifdef SS_TASK
__global__ void __launch_bounds__(NT, 1) task_gemv(const __grid_constant__ TP p) { phase_gemv(p, ss_task_layer()); }
SS_TASK_REGISTER(gemv, task_gemv, TP, NT, 0)
#endif

extern "C" __attribute__((visibility("default"))) int toy_launch(const uint64_t* params, int nblocks, void* stream) {
  TP tp;
  for (int i = 0; i < 4; ++i) tp.v[i] = params[i];
  void* args[] = {&tp};
  return (int)cudaLaunchCooperativeKernel((void*)toy_kernel, dim3(nblocks), dim3(NT), args, 0, (cudaStream_t)stream);
}
