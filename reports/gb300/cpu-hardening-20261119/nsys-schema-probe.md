# Nsight Systems 2026.3.2 SQLite export, CPU-only probe (rack 3, 2026-11-19)

`nsys profile -t nvtx,cuda --cuda-graph-trace=node -o probe python -c '<two NVTX ranges megabench:rank0:rep0/rep1>'`
followed by `nsys export --type sqlite -o probe.sqlite probe.nsys-rep`. Tables present without a GPU:
ANALYSIS_DETAILS, ANALYSIS_FILE, DIAGNOSTIC_EVENT, ENUM_* (incl. ENUM_NSYS_EVENT_TYPE), META_DATA_CAPTURE,
META_DATA_EXPORT, NVTX_EVENTS, PROCESSES, PROFILER_OVERHEAD, ProcessStreams, StringIds, TARGET_INFO_GPU,
TARGET_INFO_NIC_INFO, TARGET_INFO_SESSION_START_TIME, TARGET_INFO_SYSTEM_ENV, ThreadNames.

`NVTX_EVENTS` columns used by `nsys_audit.py`: `start`, `end` (ns), `eventType` (59 = NvtxPushPopRange,
60 = NvtxStartEndRange), `text` (non-registered string) or `textId` -> `StringIds.value`, `globalTid`.
The two probe ranges came back as `(59, 'megabench:rank0:rep0', start, end)` and `rep1`.

The CUDA tables (`CUPTI_ACTIVITY_KIND_KERNEL` with `deviceId`, `shortName`/`demangledName` -> StringIds,
`CUPTI_ACTIVITY_KIND_MEMCPY`, `CUPTI_ACTIVITY_KIND_MEMSET`, `CUPTI_ACTIVITY_KIND_RUNTIME` with `nameId`,
`CUDA_GRAPH_NODE_EVENTS`) only appear when a GPU was traced; `nsys_audit.py` treats a missing table as empty
and its unit test builds those tables by hand with the documented columns. The first GPU validation step
confirms the column names against a real trace.
