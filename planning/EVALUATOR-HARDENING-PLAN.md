# Evaluator hardening plan (Phase B, CPU-only analysis, 2026-11-13)

Scope: the MegaBench evaluator of this repository at the frozen baseline `eac5dd2` (contract digest `4da60e9b…`;
`planning/BASELINE-FREEZE.md`). This document analyses four areas, states what is confirmed and what is a hypothesis,
proposes designs, and ends with a staged implementation and regression-test plan. Nothing in the evaluator, the oracle,
the grader or the launch audit was changed for it, and no GPU was used. Line numbers refer to `eac5dd2`. Companion
documents: `planning/TP8-READINESS.md`, `planning/DECISIONS-FOR-GARV.md`, raw facts in `planning/evidence/`.

## 1. Confirmed findings versus hypotheses

| # | Statement | Status | Evidence |
|---|---|---|---|
| F1 | Inputs above 64 MiB are protected only by two linear (mod 2^64) sums; a `+d`/`-d` word pair 65521 words apart is invisible | confirmed | `megabench/harness/correctness.py:13-39`; `reports/gb300/negative-controls/checksum-collision-20261112T202413Z/` (ISSUES 17) |
| F2 | `compare()` recomputes the FP32 references from the inputs after the candidate ran, so an undetected input mutation changes the grading band | confirmed by code reading, not yet demonstrated | `megabench/tasks/kimi.py:410-411`, `425`; `harness/distributed.py:84-99` |
| F3 | Any rank-local error becomes `incorrect` on every rank; healthy ranks carry the failing rank's message | confirmed on GPUs | Steps 7 and 8 (ISSUES 14, 15) |
| F4 | A per-rank structural failure of the candidate's output (keys, shape, dtype, device) raises before the first grading collective and desynchronizes the ranks | confirmed by code reading, deliberately not run | `kimi.py:385-394` versus `:406`; `distributed.py:97-102` |
| F5 | `snapshot()`, `reference()` and both `_measure()` calls run outside the per-rank try blocks: an exception there on one rank leaves the others in a collective until the 10-minute process-group timeout | confirmed by code reading | `distributed.py:83-84, 107-113, 118-120` |
| F6 | The launch audit counts every CUDA-side profiler event: compute kernels, NCCL device kernels, `Memcpy`/`Memset` events and Kineto's `nccl:` collective records (NCCL therefore counted twice) | confirmed | Steps 1 to 6 records (ISSUES 1); tables in section 4 |
| F7 | The audit is a separate call after the timed repetitions; the timed calls are never audited | confirmed by code reading | `distributed.py:107-113`; `harness/runner.py:130-148`; `harness/benchmark.py:49-72` |
| F8 | The candidate runs inside the evaluator process and can read or alter evaluator state (the snapshot dictionary, the profiler) through ordinary Python | structural property; exploitation not demonstrated | `distributed.py:73, 88`; `harness/runner.py:53-69` |
| F9 | The oracle has been compared with the vendor's Hugging Face code on 4 real layers and with `fla` kernels, on a B200, by the repository author; no comparison at 93 layers, at TP > 1, or on GB300 exists | confirmed (what exists), not reproduced here | `tools/VERIFY_KIMI.md` |
| H1 | Kernels launched by a replayed CUDA graph appear as individual CUPTI kernel events in the audit count | hypothesis, to test | none |
| H2 | Device-side launches (dynamic parallelism) inside one host launch appear as extra kernel events | hypothesis, to test | none |
| H3 | A candidate can detect the profiler and take a one-kernel path only during the audit | hypothesis (mechanism obvious, not demonstrated) | `torch.autograd.profiler` state is readable in-process |
| H4 | `cuMemSetAccess` read-only mappings (`CU_MEM_ACCESS_FLAGS_PROT_READ`) work for device memory on GB300 and would turn mutation into a fault | hypothesis, documentation ambiguous | none |
| H5 | Re-reading the converted checkpoint from local NVMe/RAID is fast enough for an exact per-trial comparison (about 29 GB/s single stream measured on rack 10's RAID; 431 MB/s on CephFS) | partly measured (one `dd` sample each), extrapolated | `planning/evidence/infra-facts-2026-11-13.md` |

## 2. Large-tensor input integrity

### 2.1 Current implementation (`megabench/harness/correctness.py:13-39`)

- Threshold: a tensor takes the **host-copy path** when `numel * element_size <= 1 << 26` bytes (64 MiB = 67,108,864
  bytes, inclusive), or when it is non-contiguous, or when its byte size is not a multiple of 4. The copy is exact
  (`value.to("cpu", copy=True)`, later `torch.equal`).
- Otherwise (**device-checksum path**) the tensor is viewed as int32 words and reduced in chunks of 2^26 words into two
  int64 accumulators: `total[0] += chunk.sum()` and `total[1] += (chunk * weight[:n]).sum()` with
  `weight = (arange(2^26) % 65521) + 1`. PyTorch int64 arithmetic wraps, so both are sums modulo 2^64. Only the two
  int64 values go to the host. `check_unchanged()` recomputes and compares with `torch.equal`.
- Cost per snapshot: one `.to(int64)` temporary (512 MiB) and one product temporary (512 MiB) per chunk, about four
  kernels per chunk, one `arange` weight vector (512 MiB) per tensor call, one device-to-host copy of 16 bytes per
  tensor. Not measured in isolation; the whole Step 7 B64 run (two snapshots of 28.6 GiB per rank plus everything
  else) took 77 s.

### 2.2 The two functions and the collision mechanism

Both sums are linear over the words `w_i` of a tensor (modulo 2^64): `S0 = Σ w_i`, `S1 = Σ ((i mod 2^26) mod 65521 + 1)·w_i`.
Any change `δ` with `Σ δ_i ≡ 0` and `Σ weight_i·δ_i ≡ 0` is invisible. The simplest: `+d` at word `i` and `-d` at word
`i + 65521` inside one chunk (equal weights). Verified with the evaluator's own functions on the CPU
(`checksum_collision_check.py`): on a 128 MiB BF16 tensor one word change is detected; three such pairs (deltas 1, 12345
and -3; one of them turns a value of 0.78 into 8e28) leave both sums byte-identical and pass; the same pair on a 64 MiB
tensor (host path) is detected. The collision space has dimension `n - 2` per chunk; cross-chunk pairs with equal
`(i mod 65521)` also work. No cryptographic skill is needed.

### 2.3 Which tensors take which path (from `kimi.weight_shapes` and `make_state`; `planning/evidence/case_sizes.json`)

| Case (per rank) | Weight tensors / bytes | State tensors / bytes | Host-copy path | Device-checksum path | On the checksum path |
|---|---|---|---|---|---|
| `kimi-k3-syn-l8-tp4-b1` | 200 / 29.75 GiB | 18 / 27 MiB | 171 tensors, 2.72 GiB | 47 tensors, 27.06 GiB | 21 MXFP4 expert block tensors (1176 MiB each) and 21 scale tensors (73.5 MiB), `embed` and `lm_head` shards (560 MiB), dense `gate`/`up`/`down` (115.5 MiB) |
| `…-b8` | same | 18 / 214 MiB | 171 / 2.91 GiB | 47 / 27.06 GiB | same (the 64 MiB `l3_kv_cache` is exactly at the limit and stays on the host path) |
| `…-b64` | same | 18 / 1.71 GiB | 163 / 2.84 GiB | 55 / 28.62 GiB | plus `l3_kv_cache`, `l7_kv_cache` (512 MiB) and six `recurrent_state` (96 MiB) |
| `kimi-k3-tp8-b1` (real) | 2322 / 190.97 GiB | 188 / 264 MiB | 2232 / 32.21 GiB | 278 / 159.03 GiB | all 276 expert block tensors (588 MiB) of the 92 MoE layers plus `embed` and `lm_head` (280 MiB); expert scales (38.5 MB), MLA/KDA projections, latent projections, routers and shared experts are below 64 MiB and are exact |
| `kimi-k3-tp8-b8` | same | 188 / 2.11 GiB | 2232 / 34.05 GiB | 278 / 159.03 GiB | same |
| `kimi-k3-tp8-b64` | same | 188 / 16.89 GiB | 2208 / 36.83 GiB | 302 / 171.03 GiB | plus the 24 MLA `kv_cache` tensors (256 MiB) |
| `kimi-k3-dev-l4-tp1-b8` | 100 / 53.63 GiB | 10 / 218 MiB | 61 / 0.66 GiB | 49 / 53.18 GiB | expert blocks 4704 MiB, scales 294 MiB, `embed`/`lm_head` 2240 MiB, dense FFN 462 MiB |

So on the graded cases every expert weight (the bulk of the model) and the LM head are on the weak path; the per-sequence
state is exact except the B64 MLA caches and KDA recurrent states.

### 2.4 Implications

- **Grading band inflation (synthetic and real cases).** `compare()` runs `truth_o = reference(values, fp32=True, routes=oracle)`
  after the candidate returned (`kimi.py:410`). `expected` was computed from the pristine inputs (`distributed.py:84`), so a
  sum-preserving mutation of, say, an expert block tensor makes `truth_o` drift from `expected`, which raises `oracle_err`
  and with it `limit = 2·oracle_err + 0.002` for every output and the `next_token` margin (`2·max|oracle - truth_o|`).
  A candidate can therefore widen its own tolerance after producing its outputs, on any tensor above 64 MiB. Mutating
  words the FP32 reference actually reads (E2M1 codes of experts chosen by the oracle routes) is sufficient; two words
  65521 apart inside one expert's block row set already changes that expert's output. Not demonstrated yet; proposed as
  the first Stage 0 control.
- **Real checkpoint persistence.** `checkpoint_weights()` keeps the loaded tensors in `_CHECKPOINT_CACHE` and returns a
  shallow `dict(...)` (`kimi.py:150-161`), so a mutation of a weight persists into later trials, the timing phase and the
  audit; both the oracle and the candidate then see the mutated weights. On synthetic cases the weights are regenerated
  per trial, so only the within-trial band inflation matters.
- **Structural limit.** The snapshot dictionary lives in the candidate's process. A hostile candidate can replace it
  (frame introspection) rather than defeat the checksum. Every in-process integrity check is a guard against accidental
  or unsophisticated mutation; the real controls today are fresh inputs on a grading run outside the agent's reach and
  the human source review that `goalbox-grade` already requires. Any keyed or cryptographic scheme is only meaningful
  together with process isolation of the candidate (strategy S5).

### 2.5 Replacement strategies (not implemented)

| Strategy | Mechanism | Memory overhead (per rank) | GPU synchronization | Kernel launches | Host transfers | Adversarial robustness | Cost estimate |
|---|---|---|---|---|---|---|---|
| **S1 Exact comparison against the source** | Synthetic weights: regenerate each tensor from `(seed, stream)` with the same generator calls and compare chunk-wise (`torch.equal` per chunk, or an `all` reduction). Checkpoint weights: re-read the rank's safetensors files tensor by tensor and compare on device. State tensors: keep an exact second copy (B64 TP8: 16.9 GiB, or pinned host). | one tensor temporary (TP4: 1176 MiB; TP8: 588 MiB) plus the state copy | one per tensor (or one at the end with a running flag) | about 3 per chunk | none for synthetic; the full re-read (191 GiB per rank) for the checkpoint | exact: no collision, no key; still in-process unless combined with S5 | synthetic: about the cost of `make_inputs` (not measured separately; Step 5 first call at B64 was 4.4 s including the candidate); checkpoint: 191 GiB at 29 GB/s local RAID about 7 s per check, at CephFS 431 MB/s about 8 min (unusable per trial) |
| **S2 Keyed multilinear hash over a prime field** | `H = Σ r_i·w_i mod (2^61 - 1)` with per-trial random multipliers `r_i` (device PRNG from a secret seed), 128-bit intermediate via two 64-bit halves or a small custom kernel | constant (no copies); multipliers generated on the fly per chunk (512 MiB temporaries like today) | one 8-byte copy per tensor | about 6 per chunk (TP8: about 700 chunks per rank) | 8 bytes per tensor | collision probability ≤ 1/(2^61 - 1) per tensor per trial for any adversary **who does not know the seed**; against a key-reading candidate it is as weak as today, so it requires S5. Not collision-proof in the cryptographic sense: an adversary with the key finds collisions trivially. Modulus 2^64 would be weak (powers of two cancel); a prime field is required | one or two passes over 171 GiB: order 50 to 150 ms per snapshot at HBM rates (estimate), two snapshots per trial |
| **S3 Cryptographic hash on device** | SHA-256 or BLAKE3 tree hash per tensor in a custom CUDA kernel; digest kept outside the candidate process | constant | one 32-byte copy per tensor | one or two per tensor | 32 bytes per tensor | collision resistance without a secret (computational); the stored digest must still be out of the candidate's reach (S5) | GPU SHA-256 throughput of tens of GB/s is reported in the literature, not measured here: about 5 to 20 s per snapshot at TP8 (estimate) |
| **S4 Prevent instead of detect: read-only mappings** | Map the weight allocations read-only for the candidate (`cuMemSetAccess` with `CU_MEM_ACCESS_FLAGS_PROT_READ`); a write faults and the trial ends as a candidate execution error | none | none | none | none | mutation impossible, not merely detected; the evaluator keeps a writable mapping of the same physical memory | unknown: H4 must be tested on GB300 first (documentation ambiguous); requires a custom allocator for the fixture tensors |
| **S5 Process isolation of the candidate (enabler)** | Run `build()`/`run()` in a child process; share the input tensors through CUDA IPC (`torch.multiprocessing`); keep snapshots, keys and digests in the evaluator process | a second CUDA context (about 0.5 to 1 GiB) | IPC event per call | none | handles only | makes S2/S3 digests and the profiler state unreachable from the candidate; the candidate can still mutate shared memory, which S1/S2/S3 then detect | engineering: medium; adds one process per rank; the launch audit must then profile the child (CUPTI in the child or `nsys`-style tracing) |

Recommendation (for decision, not implemented): S1 for the synthetic cases (exact, no secret, no extra resident copy)
together with S5; for the real-checkpoint cases S2 or S3 computed once at load and re-verified after every trial, with
the digest held by the evaluator process; S4 investigated on a GPU because it removes the per-trial cost entirely. Do
not treat any non-cryptographic hash as collision-proof; the S2 bound is information-theoretic only while the seed is
secret.

## 3. Distributed failure handling

### 3.1 Call graph at `eac5dd2`

```
python -m megabench evaluate                      harness/runner.py:498-521 main
 └ _run_one (260-304)                             local worker in its own session; killpg on --timeout (default 180 s)
    └ _worker (170-184) → evaluate_case (72-167)  record: hardware, submission_sha256, source hints
       └ evaluate_distributed (distributed.py:146-183) for gpus > 1
          └ mp.spawn(_rank_worker, nprocs=gpus, join=True) (154)
             _rank_worker (49-125), per rank:
               61  init_process_group(nccl|gloo, timeout=10 min)
               66  initialize(case)                  dist.new_group x (ep + tp), collective
               73  _load_submission → build(case)    in try → agree (76)
               per trial (80-104):
                 81  shared_seed()                   broadcast_object_list
                 82  fresh_inputs()                  make_inputs in try → agree (45)
                 83  snapshot(inputs)                NOT in a try
                 84  reference(case, inputs)         NOT in a try; 26 collectives per step
                 86-96 try: candidate(inputs); synchronize; check_unchanged  → agree (96)
                 97-102 try: compare(...)            → agree (102)
                        kimi.compare: 385-394 structural checks (no collective)
                                      406 all_reduce(flags)   ← first grading collective
                                      410-411 FP32 references (collectives)
                                      422, 436, 441 band sums, next_token stats
                                      461 raise at the end
               105 fresh_inputs (agree) → 106 barrier → 107 _measure(candidate)  NOT in a try
               108 barrier → 110 _measure(reference) NOT in a try → 112 barrier → 113 _audit_launches (own try)
               118-120 except: status incorrect (AssertionError) | error (anything else)
               122 write rank-N.json; destroy_process_group
          156 read rank-N.json for every rank (FileNotFoundError if a rank died without writing)
          157-160 aggregate: incorrect if any rank incorrect, else error; reason = failures[0]
```

### 3.2 Points where one rank can raise or stop before the others reach a matching collective

| Id | Point | Trigger | What happens today | Visible result |
|---|---|---|---|---|
| D1 | `init_process_group` (61) | CUDA init failure or busy device on one rank | that rank writes `error`; the others wait for the 10-minute timeout | `error` after about 10 minutes |
| D2 | `build()` (73) | hangs (compilation), or issues collectives on some ranks only | no per-phase deadline; outer `--timeout` kills everything | `timeout`, no rank diagnostics (temp dir removed) |
| D3 | `snapshot()` (83) | OOM (two 512 MiB temporaries) or any exception on one rank | rank exits with `error`; the others enter `reference()` collectives and wait 10 minutes | `error` after 10 minutes; healthy ranks report `DistBackendError` |
| D4 | `reference()` (84) | OOM or CUDA error on one rank mid-sequence | same as D3 | same |
| D5 | candidate call (88) | a candidate exception | handled: `agree()` → `incorrect` on all ranks (F3) | `incorrect` even for an OOM |
| D6 | candidate call (88) | the candidate hangs, or participates in collectives inconsistently across ranks | no per-call deadline; outer timeout | `timeout`, no rank or phase information |
| D7 | `compare()` structural checks (`kimi.py:385-394`) | wrong keys, type, shape, dtype or device on one rank | that rank raises before the `flags` all-reduce, is caught at 100, and enters `agree()` (all_gather) while the others sit in `all_reduce(flags)`: mismatched collectives | hang until the 10-minute timeout (F4, ISSUES 16) |
| D8 | inside `compare()` after the first collective | OOM on one rank in an FP32 reference | mismatched collectives | hang, 10 minutes |
| D9 | `_measure(candidate)` / `_measure(reference)` (107, 110) | any exception on one rank | rank exits with `error`; the others wait at `barrier()` | `error` after 10 minutes |
| D10 | rank process dies (OOM killer, segfault in an extension, `os._exit`) | no `rank-N.json` | `mp.spawn` terminates the other ranks and raises; `evaluate_case` raises; `_worker` writes `error` | `error` with a parent traceback, no per-rank records |
| D11 | parent `--timeout` | anything that takes too long | `killpg`; the temporary rank directory is deleted | `timeout` with only `worker exceeded N s` |

### 3.3 Proposed fail-fast protocol (preserves rank-specific diagnostics; no new grading collective)

- **P1 Typed agreement.** Replace the error string in `agree()` by a small record `{phase, kind, exc_type, message}` with
  `kind ∈ {incorrect, candidate_error, evaluator_error, infrastructure}`. Every rank derives the same group decision from
  the gathered records, raises a typed exception accordingly, and keeps its **own** local record in `rank-N.json`
  (`local_status`, `local_phase`, `local_reason`) next to the group fields (`group_status`, `group_reason`, `ranks_failed`).
  This fixes F3 (healthy ranks no longer read as `incorrect`) and gives the attribution Step 8 showed, without adding
  collectives (same all_gather, richer payload).
- **P2 Structural validation before agreement.** Perform the output key/type/shape/dtype/device checks inside the per-rank
  try block right after `check_unchanged()`, so a structural failure is a local error that the existing `agree()` at line
  96 resolves before any grading collective. `compare()` keeps its checks (they then never fire in the distributed path).
  Fixes F4 with no new collective.
- **P3 No unguarded phase.** Wrap `snapshot()`, `reference()`, both `_measure()` calls and the audit in the same try/agree
  pattern with `kind = evaluator_error` or `infrastructure` (CUDA/NCCL errors). Cost: about five more small all_gathers per
  evaluation.
- **P4 Phase heartbeats and per-phase deadlines.** Each rank writes `rank-N.phase` (phase name, timestamp) before every
  phase. The parent uses `mp.spawn(..., join=False)` and polls: on a per-phase deadline (configurable; the candidate call
  and `compare()` separately from the real-weight load) it kills the process group and reports `timeout` with every rank's
  last phase. Diagnostics survive because the parent copies the rank directory into the output location before deleting it
  (P6).
- **P5 Shorter collective timeout for the grading phases.** Keep the 10-minute process-group timeout for initialization
  and weight loading, but pass a shorter configurable timeout (for example 120 s on synthetic cases) to the grading
  collectives (a second process group, or `dist.monitored_barrier` on a gloo side group before each grading phase, which
  names the missing rank). PyTorch's NCCL watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING`, on by default) then turns a peer's
  death into a `DistBackendError` on the healthy ranks, classified `infrastructure` by P1. Optional:
  `TORCH_NCCL_DUMP_ON_TIMEOUT` flight-recorder dumps into the output directory.
- **P6 Keep per-rank files on every exit path** (`--rank-dir`, default beside the result file).
- **P7 Status taxonomy in the structured result** (replaces the current `incorrect | error` pair for failures):

| Category | Source | Aggregate `status` | Per-rank fields | Counts against the candidate? |
|---|---|---|---|---|
| Candidate incorrectness | `AssertionError` from `compare()`, `check_unchanged()`, structural checks | `incorrect` | failing ranks: `local_status=incorrect`, reason; others: `local_status=pass`, `group_status=incorrect` | yes |
| Candidate execution error | any other exception raised inside `build()` or the candidate call (including a CUDA OOM or illegal address raised there) | `candidate_error` | same pattern, `exc_type` kept | yes, but reported separately (policy decision D-S3) |
| Evaluator internal error | exceptions in `make_inputs`, `snapshot`, `reference`, `compare` internals, `_measure`, the audit | `evaluator_error` | phase and traceback of the failing rank | no; the case must be rerun |
| Infrastructure interruption | `DistBackendError`/NCCL errors, device lost, rank process killed, missing rank file, GPU preemption | `infrastructure` | last phase per rank, exit codes, stderr tail | no; rerun |
| Timeout | per-phase deadline (P4) or outer `--timeout` | `timeout` | last phase per rank, elapsed per phase | policy decision (a hanging candidate is a candidate failure; a hung load is infrastructure) |

`exit code 1` for every non-`ok_provisional` status stays (ISSUES 8); automation reads the status field.

### 3.4 Validation on the CPU first

Tiny-geometry gloo tests (as in `tp4_preflight.py`) with one control per hazard: D3 (monkeypatched `snapshot` raising on
rank 2), D7 (wrong shape on rank 2: today a hang, after P2 `incorrect` on all ranks with `rank 2:` attribution), D9 (an
exception in the timing phase on one rank), D10 (`os._exit(3)` on one rank), D6 (a sleeping candidate on one rank: after P4
a `timeout` naming rank and phase). Each test must finish within the configured deadline with the expected status; the
current behaviour (hangs) is recorded first with a short process-group timeout so that the suite stays bounded. GPU
replication at TP4 under NCCL afterwards, under its own GO.

## 4. Launch-budget semantics

### 4.1 Current accounting (`megabench/harness/benchmark.py:49-72`)

One call of the candidate on the performance inputs, after the warm-up and timed repetitions, under
`torch.profiler.profile(activities=[CPU, CUDA])`. `gpu_kernel_count` is the number of profiler events whose `device_type`
is CUDA. `within_budget` requires `1 <= count <= max_gpu_launches` (1 on every case). `cuda_graph_runtime_hint` is set when
any CPU-side event name contains `cudagraphlaunch`. `review_required` is always true. Per rank, measured with the oracle
as the candidate (names in each step's `result.jsonl.gz`):

| Event class | TP1 dev l4 B8 (Step 6) | TP2 dev l8 B8 (Step 2) | TP4 B1 (Step 3) | TP4 B8 (Step 4) | TP4 B64 (Step 5) | Is it a kernel launch by the candidate? |
|---|---|---|---|---|---|---|
| compute kernels (`at::native::…`, cuBLAS, elementwise) | 22420 | 52560 | 8189 | 52933 | 256893 | yes |
| NCCL device kernels (`ncclDevKernel_AllReduce_…`) | 0 | 26 | 26 | 26 | 26 | a kernel, launched by the NCCL library as a consequence of the candidate's collective call |
| `Memcpy` events (DtoH, HtoD, DtoD) | 2506 | 5921 | 848 | 5963 | 29308 | no: DMA copies issued from the host (`cudaMemcpyAsync`), not kernels, but host-issued GPU work |
| `Memset` events | 1 | 0 | 0 | 0 | 0 | no: fills issued from the host (`torch.zeros`), same remark |
| `nccl:all_reduce` collective records | 0 | 26 | 26 | 26 | 26 | no: Kineto's high-level record of the same 26 collectives (double count) |
| **`gpu_kernel_count`** | 24927 | 58533 | 9089 | 58948 | 286253 | |
| actual kernel launches (compute + NCCL device) | 22420 | 52586 | 8215 | 52959 | 256919 | |

### 4.2 Does the audit implement the intended one-launch constraint?

The goal text says: "each rank runs exactly one persistent kernel launch per step, and that launch includes all
communication between ranks"; `TASK.md` adds "the launch audit counts NCCL kernels too", forbids replaying a multi-kernel
CUDA graph and forbids moving model work to the CPU. Against that intent the current audit is:

- **Stricter than stated** on copies: a true single-kernel candidate that performs one `cudaMemcpyAsync` (for example to
  read `token` or to publish `cache_length`) or allocates an output with `torch.zeros` (one `Memset`) exceeds the budget
  of 1, because copies and fills are counted as launches. Whether that is intended is a policy question (D-L1).
- **Double-counting NCCL**: each collective contributes a device kernel and a Kineto record. For the intent ("includes all
  communication") any NCCL kernel already means more than one launch, so the double count changes nothing for a
  compliant candidate; it still misstates the number (D-L2).
- **Not covering the timed calls** (F7): the audited call is separate from the seven timed calls. A candidate that
  detects the profiler (H3) can present one kernel to the audit and many to the clock. The audit should count during the
  timed repetitions, or the timing should be taken from the audited call as well (D-L5).
- **Graph replay**: forbidden by `TASK.md`, detected only as a hint. Whether CUPTI reports graph-internal kernels one by
  one (H1) decides if the count alone catches it; to be tested with a synthetic candidate (D-L4).
- **Device-side launches** (dynamic parallelism) and persistent work-queue designs are one host launch; whether child
  kernels appear as events (H2) and whether they are allowed is a policy question (D-L6).
- **Process scope**: CUPTI sees all threads of the process; a helper process launching kernels would not be seen
  (hypothesis; a `--docker`/`bwrap` sandbox with a pids limit does not prevent it). Candidates in a separate process (S5)
  need the profiler attached to that process.
- **Count ≥ 1** correctly rejects a candidate that does no GPU work at all.

Conclusion: the audit measures "number of CUDA-side profiler events of one unaudited-for-timing call", which is a
proxy for, not an implementation of, "one kernel launch per rank for the timed step". The gap matters only once a
near-budget candidate exists; for the oracle both readings give `launch_budget_failed`.

### 4.3 Decision request to the repository owner (summary; full list in `planning/DECISIONS-FOR-GARV.md`)

D-L1 what counts as a launch (kernels only; kernels plus memcpy/memset; everything); D-L2 report the split and stop
double-counting NCCL records (reporting change, no policy); D-L3 whether any NCCL kernel is allowed at all; D-L4 how a
replayed graph is detected and judged; D-L5 audit the timed repetitions; D-L6 device-side launches; D-L7 the role of the
manual `review_required` step in the final grade; D-L8 whether `max_gpu_launches = 1` stays for every case.

## 5. Independent oracle fidelity

### 5.1 What exists (not reproduced in this campaign)

`tools/VERIFY_KIMI.md`: on one B200 on 2026-11-10 the repository author compared (a) the oracle's KDA step, short
convolution and gated norm with `flash-linear-attention` 0.6.0 kernels (rel-L2 about 1.6e-3 on BF16 outputs, 7.3e-8 on
the FP32 state, convolution state equal); (b) the first 4 real layers (dense, KDA, KDA, MLA) through the converted-weight
loader against `KimiLinearForCausalLM` (transformers 4.56.2) for 2 decode steps of 4 sequences: logits rel-L2 5.9e-3 and
8.1e-3, top-1 agreement 4 of 4. The script needs a private environment (transformers 4.56.2, `fla-core`, einops), the HF
source files, HF shards 1 to 4 and 94, and 4-layer TP1 converted weights. Nothing at 93 layers, nothing at TP > 1 against
an external reference, nothing on GB300, and no statement about MXFP4 decode equality on real tensors beyond what the
4-layer logits imply. Independent fidelity is therefore **not established**; the 4-layer result is the only external
anchor.

### 5.2 What "independent" can mean here

1. The vendor's modeling code (`modeling_kimi_linear.py`, Hugging Face): authoritative semantics, same BF16 noise class,
   runnable at a few layers on one GPU or slowly on the CPU.
2. Production engines serving the real checkpoint: SGLang images for Kimi K3 exist on rack 3 (`docker images`), and the
   vLLM latency path is the repository's own baseline; both give full-depth greedy tokens and top-k logprobs through an
   API. They are independent implementations but expose only the sampling-level outputs.
3. The oracle against itself at different precisions and TP degrees (already in the unit tests): not independent, but
   the only way to check TP communication and the greedy tie rule exactly.

### 5.3 Minimum comparison matrix

| Component | Reference | Inputs | Metric and threshold (proposed) | Geometry / resources |
|---|---|---|---|---|
| MLA attention and cache | HF MLA layer forward (layer 3 real weights) and `fla`-free PyTorch; oracle `full_layer` branch | (a) fresh rows (`reset` true), (b) 8192 random latents in the cache, (c) 3 consecutive steps carrying `kv_write`/`pe_write` | `kv_write`, `pe_write` BF16 rel-L2 ≤ 1e-3 (expected near exact); attention output rel-L2 ≤ 5e-3; absorbed versus explicit formulation agree | 1 GPU or CPU; 4-layer TP1 converted weights (about 70 GB) |
| KDA recurrent and convolution states | `fla` `fused_recurrent_kda` and `chunk_kda` (consistency between the two), HF `KimiLinearAttention` KDA branch | random inputs, ≥ 8 steps with and without `reset`; real `A_log` (first 96 of 128), `dt_bias`, conv taps of layers 1 and 2 | FP32 recurrent state rel-L2 ≤ 1e-6 per step; outputs ≤ 2e-3; conv state equal | 1 GPU; the existing `verify_kimi_vs_hf.py` part 1 extended to multi-step and reset |
| MoE routing | HF `block_sparse_moe` gate (sigmoid + `e_score_correction_bias`, top-16, raw-score renormalization), compressed-tensors MXFP4 dequantization | ≥ 10^4 random hidden states against real routers of ≥ 3 layers; a sample of real expert tensors | selected sets identical (ties reported separately); routing weights rel-L2 ≤ 1e-6; decoded MXFP4 matrices bit-identical | CPU feasible |
| Tensor-parallel communication | the oracle at TP1 against TP2, TP4 (TP8 when available) on the same global weights (4-layer checkpoint converted at each TP) | same tokens, same state | `logits` rel-L2 ≤ 1e-3 (reduction order only), `expert_ids` and `cache_length` identical, `next_token` identical outside the tie margin; 26 collectives per step | 2 and 4 GPUs of one tray; conversions at TP1/TP2/TP4 of 4 layers |
| Next-token selection | HF greedy on the same logits; the oracle's shard-wise `_greedy` against a single-device argmax | ≥ 10^3 random states; constructed ties across shard boundaries | identical outside the tie margin; lowest index on exact ties | CPU |
| Full depth, real weights (later) | SGLang or vLLM serving the pinned checkpoint | ≥ 50 text prompts, greedy continuation of ≥ 32 tokens | next-token agreement per step reported (divergence after the first disagreement is expected); chosen-token logprob difference ≤ a few 1e-2 | 8 GPUs for the oracle at TP8 (see `TP8-READINESS.md`); the engine needs two trays as well |

Only the first five rows are feasible before a TP8 path exists; the first, third and fifth are CPU-feasible.

## 6. Risk register

| Id | Risk | Severity | Mitigation | Validation criterion | Estimated cost |
|---|---|---|---|---|---|
| R1 | Band inflation through sum-preserving mutation of inputs above 64 MiB (F1, F2) | high (grading integrity) | S1/S2 with S5; interim: human review of candidate source for in-place writes to inputs | the Stage 0 control passes today and is rejected after the change; all existing immutability controls still rejected; positive controls pass | CPU hours now; GPU: 2 TP4 runs of about 2 min each later |
| R2 | In-process candidate can alter any evaluator state (F8) | high (structural) | S5; grading on fresh inputs outside the agent's reach (already the policy) | a control that tampers with `originals` is caught after S5 | engineering: medium |
| R3 | Launch audit semantics (F6, F7, H1 to H3) | high for scoring a near-budget candidate; none for the oracle | decisions D-L1 to D-L8; report the split; audit the timed calls | synthetic one-kernel and graph-replay candidates classified as decided | CPU: analyzer; GPU: a few TP1 runs |
| R4 | Distributed hangs on evaluator-internal or structural failures (F4, F5) | medium (10 minutes lost, diagnostics lost) | P1 to P7 | hazard suite finishes within deadlines with expected statuses | CPU hours; one TP4 GPU replication |
| R5 | Healthy ranks reported `incorrect`; infrastructure failures reported as candidate failures (F3) | medium (misattribution) | P1, P7 | Step 8 control shows `local_status=pass` on ranks 0, 1, 3 | included in R4 |
| R6 | No sandbox on either rack (`bwrap`, `socat` absent; user namespaces restricted) | high for running agents | administrator installs `bubblewrap` and `socat`, or use the harness's existing Docker worker mode with an aarch64 image | `setup/selftest.sh` sandbox section passes | administrator time; image build about 1 hour |
| R7 | Oracle fidelity unverified beyond 4 layers (F9) | medium | matrix in 5.3 | thresholds met | CPU hours plus 1 to 4 GPUs for the TP rows |
| R8 | TP8 not runnable with the current harness on GB300 trays | high for schedule | see `TP8-READINESS.md` and decisions D-T1 to D-T4 | an 8-rank evaluation of a p2 synthetic case completes | depends on the chosen path |
| R9 | Host memory and timeouts of an oracle-as-candidate TP8 validation (snapshots 129 to 147 GiB per tray per trial, profiler events about 11x Step 5) | medium | smaller validation case (fewer layers) or a non-oracle candidate; `--timeout ≥ 3600`; monitors | run stays below 80 percent of host RAM | none beyond planning |

## 7. Staged implementation and regression-test plan (proposal; nothing started)

| Stage | Content | Contract files touched | Regression tests (must pass before and after unless stated) | Resources | Gate |
|---|---|---|---|---|---|
| 0 Controls and measurements (no evaluator change) | (a) sum-preserving mutation control (`negctrl_pair_mutation.py`) showing F2 on the dev case on the CPU: today `check_unchanged` passes and `oracle_rel_l2_max` grows; (b) hazard suite D3, D6, D7, D9, D10 on tiny gloo geometry recording today's behaviour with a 60 s process-group timeout; (c) measure `snapshot()` and `make_inputs()` time and memory in isolation on the CPU (real geometry) and later on a GPU | none | the 7 unit tests; Step 6 CPU-review and Step 7 preflight scripts rerun unchanged | CPU: about 2 hours of tray time; GPU for (c): one TP4 B64 run (GO) | operator GO for the GPU part |
| 1 Typed agreement, structural check relocation, statuses (P1, P2, P3, P7) | `harness/distributed.py` only | hazard suite: D7 becomes `incorrect` with `rank 2:` attribution, D3/D9 `evaluator_error`, D10 `infrastructure`, all within the deadline; Step 7/8 controls give `local_status=pass` on healthy ranks; positive control unchanged | CPU; one TP4 replication of Steps 7 and 8 (GO) | decisions D-S1 to D-S3 |
| 2 Snapshot replacement (S1 synthetic, S2 or S3 checkpoint; S4 investigated) | `harness/correctness.py` (+ a kernel if S3) | Stage 0(a) control rejected; `negctrl_immutability`, `negctrl_ranklocal_immutability` rejected; positive controls pass; cost delta recorded against Stage 0(c) | CPU; TP4 GPU runs (GO) | decision D-I1 to D-I3 |
| 3 Launch audit reporting and policy (D-L1 to D-L8) | `harness/benchmark.py` | oracle counts per class equal the tables in 4.1; a one-kernel dummy candidate counts 1; a graph-replay dummy is classified as decided; audit-during-timing agrees with the separate audit for the oracle | one TP1 and one TP4 GPU run (GO) | decisions D-L1 to D-L8 |
| 4 Heartbeats, per-phase deadlines, rank-dir preservation (P4 to P6) | `harness/distributed.py`, `harness/runner.py` | D6 control ends as `timeout` naming rank and phase; rank files present after a parent timeout | CPU; one TP4 GPU hang control (GO) | none |
| 5 Isolation | none (`sandbox/`, `setup/`, an image) | `setup/selftest.sh`; the Step 6 controls through the Docker worker (`--docker-image`) give the same records | administrator or image build | decision D-X1 |
| 6 TP8 path | per decision (multi-node launcher touches `harness/distributed.py`; an 8-GPU node touches nothing) | 4-layer checkpoint conversion at TP1/TP2/TP4/TP8 loads on the CPU through `load_weights`; a p2 synthetic TP8 case with the oracle; then one real-weight case | see `TP8-READINESS.md` | decisions D-T1 to D-T4 |

Order of implementation recommended: 0 → 1 → 2 → 3 → 4, with 5 and 6 in parallel once their decisions exist. Every stage
that touches a contract file changes the digest; the baseline evidence remains valid for digest `4da60e9b…` only.
