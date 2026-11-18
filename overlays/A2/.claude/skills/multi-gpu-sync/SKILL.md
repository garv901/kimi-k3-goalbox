---
name: multi-gpu-sync
description: Synchronization patterns for persistent / multi-GPU kernels on Hopper/Blackwell - grid barriers vs per-item counters (epochs, release/acquire, spin with backoff and deadlines), NVLink mailbox and multicast push all-reduce (Lamport sentinels), chunked communication overlap, splitting work across GPUs and blocks for balance (incl. slicing expert/FFN intermediate dims), PDL kernel chaining, and deadlock hazards of cooperative kernels. Use when a kernel spends time in barriers, cross-GPU handshakes, all-reduces or load imbalance.
---

# Multi-GPU and grid-level synchronization

Tested example: `example/sync_patterns.cu` (epoch counters with deadlines, a Lamport mailbox with 2
generations, grid barrier vs counter timing; one GPU, blocks stand in for ranks). Build:
`nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a sync_patterns.cu -o sync_patterns`, run under
`./gpu_run -g 1 --`.

Generic patterns with small CUDA/PTX snippets. Numbers marked *B300* come from `./primdb q sync gpu=B300`,
`./primdb q mem gpu=B300` (rows measured on this node, see `TASK.d/A2.md`); everything else cites its source.
Measure your own case before you act on any estimate here.

## 0. First: price the sync before you change it
For each barrier/handshake kind, estimate `count x (latency + skew)` per launch:
- *latency* = last arrival -> release (a timestamp per block at arrive and at release; `%globaltimer`
  ticks are 32 ns on B300).
- *skew* = mean busy time -> slowest block's busy time before the barrier; that is waiting caused by
  imbalance, and a faster barrier does not remove it.
Fix the biggest product first. Removing a barrier saves latency + the part of skew that other work
can absorb; balancing work saves skew.

## 1. Grid barriers vs per-item counters
A grid barrier makes *every* block wait for the *slowest* block. A counter (flag) per produced item
lets each consumer wait only for the producers it actually reads.

**Release/acquire counter** (producer: write data, then `red.release`; consumer: `ld.acquire` spin):
```cuda
__device__ __forceinline__ void signal(unsigned* ctr) {   // after this block's data writes
  __syncthreads();                                         // all threads' writes precede thread 0's release
  if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(ctr) : "memory");
}
__device__ __forceinline__ void wait_geq(const unsigned* ctr, unsigned target, unsigned long long deadline_ns) {
  if (threadIdx.x == 0) {
    unsigned v, n = 0; unsigned long long t0 = 0, now;
    while (true) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory");
      if ((int)(v - target) >= 0) break;                   // wrap-safe compare (epoch counters)
      if ((++n & 1023) == 0) {                              // deadline check every 1024 polls
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (!t0) t0 = now; else if (now - t0 > deadline_ns) { printf("timeout ctr %p\n", ctr); __trap(); }
      }
      __nanosleep(n < 64 ? 0 : 64);                         // backoff only after a short hot spin
    }
  }
  __syncthreads();                                          // other threads ordered after thread 0's acquire
}
```
- Use `.sys` scope (and `ld.acquire.sys`) when the producer is another GPU over NVLink (P2P/IPC or
  symmetric memory); `.gpu` within one GPU; `.cluster` within a cluster.
- **Epoch counters** avoid resets: never zero a counter; the k-th use waits for `base + k * producers`.
  Keep `base` in a kernel argument or read it once at kernel start; compare with wrap-safe signed
  difference. Resetting counters inside the kernel needs its own barrier and is a classic race.
- Monotonic per-row/per-tile counters let consumers start on row r as soon as row r is done
  (dataflow), instead of after the whole phase.
- A hierarchical barrier (warp -> block -> `red` to one counter -> one poller per block) is the fast
  generic grid barrier; `cooperative_groups::grid.sync()` is correct but not cheaper. Measure both:
  latency grows with the number of arriving blocks and with L2 distance.
- *Cost hints (B300)*: `__syncthreads` 12 ns at 192 threads, named `bar.sync` 18 ns, mbarrier
  arrive->wait 26 ns, cluster-remote mbarrier hand-off ~220 ns (`sync.mbarrier_remote_arrive`), L2
  pointer-chase latency in `mem l2.chase_latency`. A grid-wide barrier involves an L2 round trip by every
  block plus a poll. Measured on B300 with `example/sync_patterns.cu` (148 blocks x 256 threads, no work):
  `cooperative_groups` grid.sync 1.18 us per barrier; a release/acquire wait on one neighbour's counter
  0.99 us per round. The counter's win comes from waiting on *fewer, earlier* producers (skew), not from
  a cheaper primitive. Counting barriers x ~1.2-1.7 us is a quick sanity check.

**Spin hygiene.** Spin on one thread (or one lane per warp) and broadcast via `__syncthreads`/`bar.sync`;
128 threads polling the same L2 line slow down the producer. Put each counter on its own 128 B line.
Every spin needs a deadline that faults (`__trap()` or a store to address 0, as SGLang's
`fault_at_null`): a hung multi-GPU kernel otherwise hangs the node until someone kills it, while a
fault surfaces as an error immediately.

## 2. Cross-GPU exchange: pull with flags vs push into mailboxes
*Pull (handshake)*: each rank writes its partial locally, raises a flag, peers poll the flag remotely
and then read the data remotely: two NVLink round trips plus the flag barrier.

*Push into a mailbox*: the producer writes its data straight into the *consumer's* memory and the
consumer polls **local** memory. One one-way NVLink latency, no remote polling.

**Lamport-style sentinel (no separate flag)**: the data itself says it has arrived.
- Mailbox words start as a value producers never write (SGLang uses BF16 `-0.0` = `0x8000`, or `+0.0`
  as an "empty marker"; producers map that value to its other-signed twin before sending, which is
  numerically identical).
- Producer: `st.relaxed.sys.global.v4.b32` (or `multimem.st`, below) of 16 B vectors.
  Each 16 B vector is written atomically as a unit by the vector store, so the consumer sees all of a
  vector or none.
- Consumer: polls `ld.relaxed.sys.global.v4` (or `ld.volatile`) of each vector until no word is the
  sentinel, with a deadline; then uses the data and **writes the sentinel back** for reuse.
- **Generations**: rotate 2-3 mailbox buffers (a per-call generation/epoch picks the buffer), so a fast
  rank's next round can never overwrite a slot a slow rank has not consumed yet, and no reset barrier is
  needed. SGLang: 3 generations in `all_reduce_mhc.py` (`LAMPORT_GENERATIONS = 3`); a 2-way epoch flip
  per row cluster in `two_shot_mhc.cu`.
```cuda
// consumer: wait for one 16 B vector from rank `src` in generation g (local memory)
uint4 v; unsigned n = 0;
do {
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(&box[g][src][i]));
  deadline_tick(++n);
} while (v.x == EMPTY || v.y == EMPTY || v.z == EMPTY || v.w == EMPTY);
box[g][src][i] = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);   // re-arm for generation g's next use
```
- **NVLink multicast (NVLS)**: with a multicast object (CUDA `cuMulticastCreate`/`cuMulticastBindMem`, or
  `torch.distributed._symmetric_memory` `rendezvous(...).multicast_ptr`), one
  `multimem.st.relaxed.sys.global.v4.bf16x2 [mc_ptr], {...}` writes the vector into *every* rank's copy;
  `multimem.ld_reduce.add` reads the sum of all ranks' copies. One-shot all-reduce = every rank multicasts
  its partial into its own slot of every rank's mailbox, then each rank reduces locally. Check that
  multicast is supported on the node first (`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED)`);
  plain P2P stores (`cudaIpcOpenMemHandle` / symmetric memory peer pointers) work everywhere NVLink P2P does.
- **Two-shot (reduce-scatter + all-gather)** for larger payloads: row t is owned by rank `t % world`;
  each rank pushes row t only to its owner, the owner sums in a fixed rank order (deterministic), pushes
  the result to everyone. Bytes per rank ~2x payload instead of world x payload (`two_shot_mhc.cu`).
- **Fewer bytes**: send BF16 partials instead of FP32 when the numerics allow (sum in FP32 on receipt).
- **Fuse**: the producer's epilogue can store directly into the mailbox, and the consumer can apply the
  next op (residual add, norm, quantization) right after the reduction: no extra pass over memory
  (SGLang `wo_b_publish.py`, `all_reduce_mhc.py`).

## 3. Overlap communication with compute (chunking)
- Split the producer's output into chunks (rows or column blocks); publish each chunk as soon as it is
  done (per-chunk counter or sentinel). Consumers start on chunk 0 while chunk 1 is computed.
- Inside a persistent kernel, dedicate a few blocks (or one warp per block) to communication so it does not
  wait for the slowest compute block; or let each block push its own tile's partial right after its
  epilogue.
- While a block waits for a remote chunk, prefetch *exactly the bytes it will consume next* (e.g. TMA its
  next weight tile into shared memory). Avoid broad L2 prefetches: they add DRAM traffic and can evict
  data the critical path needs.
- In a chain of normal kernels, **PDL** (programmatic dependent launch) overlaps kernel boundaries:
  launch with `cudaLaunchAttributeProgrammaticStreamSerialization`, call
  `griddepcontrol.launch_dependents` early in the producer, and in the consumer load everything that does
  not depend on the producer (weights, constants) *before* `griddepcontrol.wait`. See the
  `blackwell-block-scaled-gemm` example (on B300, PDL took a 15.6 MB skinny GEMM from 10.3 to 6.8 us per
  launch, before other changes).
  Under CUDA-graph capture across streams, a PDL edge may be programmatic rather than full: data
  read before `griddepcontrol.wait` must not come from the previous kernel (SGLang commits ca37213, 98f42c6).

## 4. Balance work across GPUs and blocks
Imbalance cost = slowest - mean, paid at every barrier. Fix the layout before tuning the barrier.
- **Slice instead of owning whole units.** If units (experts, heads, tiles) have data-dependent load,
  giving each GPU whole units makes the per-GPU load depend on the routing. Slicing *every* unit along a
  dimension that does not change with routing (for an FFN/expert: the intermediate dimension, so each GPU
  holds `I / world` columns of up/gate and rows of down) makes per-GPU bytes and FLOPs identical for any
  routing. The partial outputs then need a sum across GPUs, which a tensor-parallel layer typically
  already does. Cost: smaller per-unit tiles (more tiles; check that tiles still fill the MMA M=128 or
  use swap-AB), and each GPU now touches every routed unit. A hybrid (TP inside groups of GPUs, unit
  ownership across groups) bounds imbalance to between-group variation (SGLang TP4 x EP2, `spec.py`).
- **Within a GPU: dynamic scheduling.** Replace a static `tile = blockIdx.x + k * gridDim.x` assignment
  with an atomic work counter (`t = atomicAdd(&next, 1)`) or Blackwell cluster launch control
  (`clusterlaunchcontrol.try_cancel`, SGLang `gemm1.cu`); order the work list longest-first. Reset the
  counter with an epoch, not inside the kernel.
- **Split-K / split-N for skinny work**: if there are fewer output tiles than SMs, split K across
  blocks and reduce (cluster DSMEM reduction or last-arriver reduction with an atomic ticket).
- Measure per-GPU and per-block busy time, not just totals: `slowest - mean` per phase is the number to
  attack.

## 5. Deadlock hazards (cooperative and persistent kernels)
- **Co-residency**: a grid barrier or any spin-wait on another block deadlocks unless *all* blocks are
  resident. Launch with `cudaLaunchCooperativeKernel` (fails instead of hanging if the grid does not fit)
  or size the grid from `cudaOccupancyMaxActiveBlocksPerMultiprocessor x SMs`. Anything that lowers
  occupancy (more smem, registers, `__launch_bounds__` change, a debug build, MPS/green contexts, another
  process on the GPU) can turn a working kernel into a hang.
- **Profilers**: ncu kernel replay re-runs one kernel alone; a kernel that waits on peers (other GPUs or
  a previous kernel's flags) then waits forever. Use application replay for all ranks together, or
  measure with in-kernel timestamps.
- **Multi-GPU launch skew**: ranks start at different times (host jitter, 10s-100s of us); a spin
  deadline must cover the worst skew; early ranks' time includes waiting for late ranks.
- **Producer must not depend on consumer progress** (e.g. a ring buffer whose consumer is a block that
  is not yet scheduled). Order: only wait on work that is guaranteed to be resident or already done.
- **Side streams**: a collective (all-reduce) on a side stream with no deadline that holds a CTA on every
  SM can block the main stream; SGLang removed such a slot for that reason (commit 7b8b296).
- **Memory model**: data must be written before the flag with release semantics (or a `__threadfence()` /
  `fence.acq_rel` + relaxed flag), and read after the flag with acquire. Writes by TMA or other async
  proxies need `fence.proxy.async` before generic-proxy readers. A missing fence often "works" in tests and
  fails under load.
- Counters/flags reused across launches without epochs: a fast rank's next launch can see the previous
  launch's flag.

## Sources
- PTX ISA 8.7+/9.0: memory consistency model (release/acquire, scopes), `mbarrier`, `barrier.cluster`,
  `multimem`, `griddepcontrol`, `clusterlaunchcontrol`.
- sgl-project/sgl-nvidia-mega (Apache-2.0) at 1195222: `deepseek_v41/sm103/all_reduce_mhc.py` (Lamport
  mailbox, 3 generations, deadlines; adapted from FlashInfer's MNNVL all-reduce), `sm103/src/two_shot_mhc.cu`
  (two-shot, empty markers, epoch flip, `fault_at_null`), `sm103/wo_b_publish.py` (multicast publish in a GEMM
  epilogue), `sm103/moe/src/gemm1.cu` (CLC work stealing), `spec.py` (TP x EP layout); PR #1 on PDL chains.
- FlashInfer `comm/trtllm_mnnvl_ar` (one-shot Lamport all-reduce on NVLink multicast).
- `torch.distributed._symmetric_memory` (symmetric buffers, `multicast_ptr`).
- B300 primitive rows in this workspace: `./primdb q sync gpu=B300`, `./primdb q mem gpu=B300`.
