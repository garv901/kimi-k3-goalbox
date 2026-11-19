"""Timing of the candidate (through its isolated process) and of the in-process reference."""

from __future__ import annotations

import statistics
import time
from typing import Callable


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50_ms": None, "p95_ms": None}
    ordered = sorted(values)
    at = (len(ordered) - 1) * 0.95
    lo = int(at)
    p95 = ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)
    return {"p50_ms": statistics.median(values), "p95_ms": p95}


def summarize(host_ms: list[float], event_ms: list[float | None]) -> dict:
    events = [v for v in event_ms if v is not None]
    host = _percentiles(host_ms)
    event = _percentiles(events)
    return {"reps": len(host_ms), "host_ms": host_ms, "host_p50_ms": host["p50_ms"], "host_p95_ms": host["p95_ms"],
            "cuda_event_ms": events, "cuda_event_p50_ms": event["p50_ms"], "cuda_event_p95_ms": event["p95_ms"]}


def _measure(fn: Callable, inputs: dict, device: str, warmup: int, reps: int) -> dict:
    """Time an in-process callable (the reference). Host time and CUDA events around each call."""
    import torch

    cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn(inputs)
    if cuda:
        torch.cuda.synchronize(device)
    host_ms: list[float] = []
    event_ms: list[float] = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        fn(inputs)
        if cuda:
            end_event.record()
            end_event.synchronize()
            event_ms.append(start_event.elapsed_time(end_event))
        host_ms.append((time.perf_counter() - start) * 1000)
    return summarize(host_ms, event_ms)


def measure_candidate(candidate, inputs: dict, warmup: int, reps: int, *, rep_timeout_s: float,
                      nvtx_prefix: str | None) -> tuple[dict, dict | None]:
    """Time the isolated candidate. Each timed repetition is wrapped in an NVTX range
    `<prefix>:rep<N>` (D-L5) and the outputs of the last repetition are returned for grading."""
    outputs = None
    for _ in range(warmup):
        outputs, _ = candidate.run(inputs, rep_timeout_s)
    host_ms: list[float] = []
    event_ms: list[float | None] = []
    for index in range(reps):
        outputs, timing = candidate.run(inputs, rep_timeout_s, timed=True,
                                        nvtx=f"{nvtx_prefix}:rep{index}" if nvtx_prefix else None)
        host_ms.append(timing["host_ms"])
        event_ms.append(timing["cuda_event_ms"])
    record = summarize(host_ms, event_ms)
    record["method"] = ("GPU time between the parent's events around the candidate's own interprocess events; "
                        "host time includes the pipe round trip to the candidate process")
    return record, outputs


def _audit_launches(fn: Callable, inputs: dict, device: str, launch_budget: int) -> dict:
    """In-process launch count of a callable (kept for tooling; the graded audit is nsys_audit.py)."""
    import torch

    if torch.device(device).type != "cuda":
        return {"status": "not_applicable_cpu"}
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn(inputs)
            torch.cuda.synchronize(device)
        return classify_events([(e.name, e.device_type == torch.autograd.DeviceType.CUDA) for e in prof.events()],
                               launch_budget)
    except Exception as exc:
        return {"status": "unverified", "reason": f"{type(exc).__name__}: {exc}"}


def classify_events(events: list[tuple[str, bool]], launch_budget: int) -> dict:
    """Split profiler events into device kernels, NCCL kernels, copies/memsets and collective
    records (D-L1, D-L2) and compare the kernel count with the budget. Pure function, CPU-testable."""
    kernels, nccl_kernels, copies, memsets, records, other = [], [], [], [], [], []
    graph_hint = False
    for name, on_device in events:
        lower = name.lower()
        if not on_device:
            if "cudagraphlaunch" in lower or "graph_launch" in lower:
                graph_hint = True
            continue
        if lower.startswith("memcpy") or "memcpy" in lower:
            copies.append(name)
        elif lower.startswith("memset") or "memset" in lower:
            memsets.append(name)
        elif lower.startswith("nccl:") or lower.startswith("c10d::") or lower.startswith("record_param_comms"):
            records.append(name)
        elif "ncclkernel" in lower or lower.startswith("ncclDevKernel".lower()):
            nccl_kernels.append(name)
        else:
            kernels.append(name)
    count = len(kernels) + len(nccl_kernels)
    status = ("launch_budget_failed" if graph_hint or not 1 <= count <= launch_budget else "within_budget")
    return {"status": status, "device_kernel_count": count, "compute_kernels": len(kernels),
            "nccl_kernels": len(nccl_kernels), "memcpy_events": len(copies), "memset_events": len(memsets),
            "collective_records": len(records), "max_gpu_launches": launch_budget,
            "cuda_graph_launch_detected": graph_hint, "kernel_names": sorted(set(kernels + nccl_kernels))[:64],
            "counting_rule": "device kernels (compute + NCCL) count; memcpy, memset and collective records do not; "
                             "a CUDA graph launch fails outright"}
