// ss_task.cuh -- phase capture points and single-phase task entry points (A3 tools).
//
// Include once in the kernel translation unit, AFTER the kernel's parameter struct is defined.
// Everything compiles to nothing unless the build defines SS_CAPTURE (used by ./capture) or
// SS_TASK (used by ./task_ab), so the macros can stay in the shipped source.
//
//   SS_CAP_PARAMS(&p, sizeof(p));                       // once, at the top of the persistent kernel
//   SS_CAP("moe_in", layer, SS_REG("scratch", scr, scr_bytes, SS_U8),
//                           SS_REG("x2q", x2q_ptr, 48 * 5120, SS_F16));   // at a phase boundary
//
// SS_CAP is a grid-wide collective: every block of the (cooperative) launch must reach it. In a
// capture build it waits for all blocks (grid barrier), copies the listed regions into the capture
// buffer, and waits again. It only does this on the armed launch and for the selected layers; on
// every other call it returns immediately without a barrier.
//
//   #ifdef SS_TASK
//   __global__ void __launch_bounds__(512, 1) task_my_phase(const __grid_constant__ KP p) {
//     ... same context setup as the persistent kernel ...
//     my_phase(ctx, ..., ss_task_layer());
//   }
//   SS_TASK_REGISTER(my_phase, task_my_phase, KP, 512, 160 * 1024)
//   #endif
//
// ./task_ab launches a registered task kernel cooperatively on one GPU (grid = SM count unless
// --blocks), with the parameter block of the captured launch whose pointers were re-pointed to the
// restored state on that GPU. ss_task_layer() returns --layer; ss_task_aux() returns --aux.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

enum { SS_U8 = 0, SS_I8 = 1, SS_U16 = 2, SS_I16 = 3, SS_U32 = 4, SS_I32 = 5, SS_U64 = 6, SS_I64 = 7,
       SS_F16 = 8, SS_BF16 = 9, SS_F32 = 10, SS_F64 = 11 };

struct SSReg {
  const char* name;
  const void* ptr;
  unsigned long long count;   // elements
  int dtype;                  // SS_* above
};
#define SS_REG(NAME, PTR, COUNT, DTYPE) SSReg{NAME, (const void*)(PTR), (unsigned long long)(COUNT), (DTYPE)}

#define SS_EXPORT extern "C" __attribute__((visibility("default")))

__host__ __device__ __forceinline__ int ss_dtype_size(int t) {
  return t <= SS_I8 ? 1 : t <= SS_I16 ? 2 : t <= SS_I32 ? 4 : t <= SS_I64 ? 8 : t <= SS_BF16 ? 2 : t == SS_F32 ? 4 : 8;
}

#if defined(SS_CAPTURE) || defined(SS_TASK)
__device__ unsigned g_ss_bar[2];
// Sense-reversing grid barrier (needs a cooperative launch: all blocks co-resident).
__device__ __noinline__ void ss_grid_barrier() {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* vgen = &g_ss_bar[1];
    const unsigned gen = *vgen;
    __threadfence();
    const unsigned prev = atomicAdd(&g_ss_bar[0], 1u);
    if (prev == gridDim.x - 1) {
      atomicExch(&g_ss_bar[0], 0u);
      __threadfence();
      atomicAdd(&g_ss_bar[1], 1u);
    } else {
      while (*vgen == gen) {}
    }
    __threadfence();
  }
  __syncthreads();
}
#endif

// ------------------------------------------------------------------------------------- capture
#ifdef SS_CAPTURE
#define SS_MAXL 32
#define SS_MAXREG 32
#define SS_MAGIC 0x53534350u   // "PCSS"
struct SSCapCfg {
  unsigned char* buf;          // record area
  unsigned long long cap;      // bytes
  unsigned char* pbuf;         // parameter block: [u64 bytes][bytes...]
  unsigned long long pcap;
  int armed;
  int nlayers;
  int layers[SS_MAXL];
};
__device__ SSCapCfg g_sscap;
__device__ unsigned long long g_sscap_used;
__device__ unsigned long long g_sscap_dropped;
__device__ unsigned g_sscap_seq;

// Record layout (all offsets relative to the record start, 256-byte aligned data):
//   header 128 B: u32 magic, u32 nreg, i32 layer, u32 seq, u64 total_bytes, char name[48], pad
//   nreg x 128 B: char name[48], u64 addr, u64 bytes, i32 dtype, i32 pad, u64 data_off, pad
//   data
__device__ __forceinline__ unsigned long long ss_align(unsigned long long v) { return (v + 255ull) & ~255ull; }

__device__ __noinline__ void ss_cap_point(const char* name, int layer, const SSReg* regs, int n) {
  if (!g_sscap.armed) return;
  bool sel = false;
  for (int i = 0; i < g_sscap.nlayers; ++i) sel |= (g_sscap.layers[i] == layer);
  if (!sel) return;
  if (n > SS_MAXREG) n = SS_MAXREG;
  ss_grid_barrier();
  const unsigned long long base = *(volatile unsigned long long*)&g_sscap_used;
  unsigned long long off = ss_align(128ull + 128ull * n);
  unsigned long long offs[SS_MAXREG];
  for (int i = 0; i < n; ++i) {
    offs[i] = off;
    off = ss_align(off + regs[i].count * (unsigned long long)ss_dtype_size(regs[i].dtype));
  }
  const unsigned long long total = off;
  if (base + total > g_sscap.cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_sscap_dropped, 1ull);
    ss_grid_barrier();
    return;
  }
  unsigned char* rec = g_sscap.buf + base;
  const unsigned long long gt = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long gn = (unsigned long long)gridDim.x * blockDim.x;
  for (int i = 0; i < n; ++i) {
    const unsigned long long bytes = regs[i].count * (unsigned long long)ss_dtype_size(regs[i].dtype);
    const unsigned char* src = (const unsigned char*)regs[i].ptr;
    unsigned char* dst = rec + offs[i];
    if ((((uintptr_t)src) & 15) == 0 && (bytes & 15) == 0) {
      for (unsigned long long j = gt; j < bytes / 16; j += gn)
        ((uint4*)dst)[j] = __ldcg((const uint4*)src + j);
    } else {
      for (unsigned long long j = gt; j < bytes; j += gn) dst[j] = *(volatile const unsigned char*)(src + j);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    uint32_t* h = (uint32_t*)rec;
    h[0] = SS_MAGIC;
    h[1] = (uint32_t)n;
    ((int32_t*)h)[2] = layer;
    h[3] = atomicAdd(&g_sscap_seq, 1u);
    *(unsigned long long*)(rec + 16) = total;
    char* nm = (char*)(rec + 24);
    int k = 0;
    for (; k < 47 && name[k]; ++k) nm[k] = name[k];
    for (; k < 48; ++k) nm[k] = 0;
    for (int i = 0; i < n; ++i) {
      unsigned char* rh = rec + 128 + 128 * i;
      int q = 0;
      for (; q < 47 && regs[i].name[q]; ++q) rh[q] = regs[i].name[q];
      for (; q < 48; ++q) rh[q] = 0;
      *(unsigned long long*)(rh + 48) = (unsigned long long)(uintptr_t)regs[i].ptr;
      *(unsigned long long*)(rh + 56) = regs[i].count * (unsigned long long)ss_dtype_size(regs[i].dtype);
      *(int*)(rh + 64) = regs[i].dtype;
      *(unsigned long long*)(rh + 72) = offs[i];
    }
    __threadfence();
  }
  ss_grid_barrier();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *(volatile unsigned long long*)&g_sscap_used = base + total;
    __threadfence();
  }
}

__device__ __forceinline__ void ss_cap_params(const void* p, unsigned long long bytes) {
  if (!g_sscap.armed || blockIdx.x != 0 || bytes + 8 > g_sscap.pcap) return;
  for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
    g_sscap.pbuf[8 + i] = ((const unsigned char*)p)[i];
  if (threadIdx.x == 0) *(unsigned long long*)g_sscap.pbuf = bytes;
}

#define SS_CAP(NAME, LAYER, ...)                                                   \
  do {                                                                             \
    const SSReg _ss_regs[] = {__VA_ARGS__};                                        \
    ss_cap_point(NAME, LAYER, _ss_regs, (int)(sizeof(_ss_regs) / sizeof(SSReg)));  \
  } while (0)
#define SS_CAP_PARAMS(PTR, BYTES) ss_cap_params((const void*)(PTR), (unsigned long long)(BYTES))

SS_EXPORT int ss_capture_config(int dev, void* buf, unsigned long long cap, void* pbuf, unsigned long long pcap,
                                int armed, const int* layers, int nlayers) {
  cudaSetDevice(dev);
  SSCapCfg c;
  memset(&c, 0, sizeof(c));
  c.buf = (unsigned char*)buf;
  c.cap = cap;
  c.pbuf = (unsigned char*)pbuf;
  c.pcap = pcap;
  c.armed = armed;
  c.nlayers = nlayers < SS_MAXL ? nlayers : SS_MAXL;
  for (int i = 0; i < c.nlayers; ++i) c.layers[i] = layers[i];
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return (int)e;
  e = cudaMemcpyToSymbol(g_sscap, &c, sizeof(c));
  if (e != cudaSuccess) return (int)e;
  if (armed) {
    const unsigned long long z = 0;
    const unsigned zu = 0;
    cudaMemcpyToSymbol(g_sscap_used, &z, sizeof(z));
    cudaMemcpyToSymbol(g_sscap_dropped, &z, sizeof(z));
    cudaMemcpyToSymbol(g_sscap_seq, &zu, sizeof(zu));
  }
  return (int)cudaDeviceSynchronize();
}

SS_EXPORT int ss_capture_status(int dev, unsigned long long* used, unsigned long long* dropped) {
  cudaSetDevice(dev);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return (int)e;
  cudaMemcpyFromSymbol(used, g_sscap_used, sizeof(*used));
  return (int)cudaMemcpyFromSymbol(dropped, g_sscap_dropped, sizeof(*dropped));
}
#else
#define SS_CAP(NAME, LAYER, ...) do {} while (0)
#define SS_CAP_PARAMS(PTR, BYTES) do {} while (0)
#endif

// ------------------------------------------------------------------------------------- tasks
#ifdef SS_TASK
struct SSTaskInfo { int layer; int aux; int pad0; int pad1; };
__device__ SSTaskInfo g_sstask;
__device__ __forceinline__ int ss_task_layer() { return g_sstask.layer; }
__device__ __forceinline__ int ss_task_aux() { return g_sstask.aux; }

typedef int (*SSTaskFn)(const void* kp, unsigned long long kp_bytes, int blocks, void* stream);
struct SSTaskEntry { const char* name; SSTaskFn fn; };
static SSTaskEntry g_ss_tasks[64];
static int g_ss_ntasks = 0;
struct SSTaskReg {
  SSTaskReg(const char* n, SSTaskFn f) { if (g_ss_ntasks < 64) g_ss_tasks[g_ss_ntasks++] = SSTaskEntry{n, f}; }
};

#define SS_TASK_REGISTER(NAME, KERNEL, KPT, THREADS, SMEM)                                                 \
  static int ss_task_fn_##NAME(const void* kp, unsigned long long kpb, int blocks, void* st) {             \
    if (kpb != sizeof(KPT)) return -100;                                                                    \
    KPT k;                                                                                                  \
    memcpy((void*)&k, kp, sizeof(KPT));                                                                     \
    cudaError_t e = cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SMEM)); \
    if (e != cudaSuccess) return (int)e;                                                                    \
    if (blocks <= 0) {                                                                                      \
      int dev = 0, nsm = 0, per = 0;                                                                        \
      cudaGetDevice(&dev);                                                                                  \
      cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);                                    \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, KERNEL, (THREADS), (size_t)(SMEM));               \
      blocks = nsm * (per > 0 ? per : 1);                                                                   \
    }                                                                                                       \
    void* args[] = {(void*)&k};                                                                             \
    return (int)cudaLaunchCooperativeKernel((void*)KERNEL, dim3(blocks), dim3(THREADS), args,              \
                                            (size_t)(SMEM), (cudaStream_t)st);                              \
  }                                                                                                         \
  static SSTaskReg ss_task_reg_##NAME(#NAME, ss_task_fn_##NAME);

SS_EXPORT int ss_task_count() { return g_ss_ntasks; }
SS_EXPORT const char* ss_task_name(int i) { return (i >= 0 && i < g_ss_ntasks) ? g_ss_tasks[i].name : nullptr; }
// Returns 0 on success, -1 unknown task, -100 parameter-size mismatch, else a cudaError_t.
SS_EXPORT int ss_task_launch(int dev, const char* name, const void* kp, unsigned long long kp_bytes, int layer,
                             int aux, int blocks, void* stream) {
  cudaSetDevice(dev);
  for (int i = 0; i < g_ss_ntasks; ++i) {
    if (strcmp(g_ss_tasks[i].name, name) != 0) continue;
    SSTaskInfo info{layer, aux, 0, 0};
    cudaError_t e = cudaMemcpyToSymbolAsync(g_sstask, &info, sizeof(info), 0, cudaMemcpyHostToDevice,
                                            (cudaStream_t)stream);
    if (e != cudaSuccess) return (int)e;
    return g_ss_tasks[i].fn(kp, kp_bytes, blocks, stream);
  }
  return -1;
}
#endif
