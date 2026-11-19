"""Launch audit and GPU step time from an Nsight Systems trace of the evaluator process tree.

The evaluator wraps every timed repetition of rank R in the NVTX range `megabench:rankR:repN`
(isolation.py, benchmark.py). When the whole evaluation runs under
`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`, the exported SQLite file has those
ranges (NVTX_EVENTS, from the evaluator process) and every device event of the candidate
process (CUPTI_ACTIVITY_KIND_KERNEL, _MEMCPY, _MEMSET, _RUNTIME). CUPTI records them in the
driver, outside the candidate process's control, so this is the audit of record (D-L4, D-L5).

For each rank and repetition this module counts, on the rank's device and inside the range:
  device kernels (compute and NCCL device kernels; D-L1, D-L3), memcpy and memset events
  (reported, not counted; D-L1), CUDA graph launches (runtime API `cudaGraphLaunch*`; one fails
  the audit outright, D-L4), and the span from the first kernel start to the last kernel end
  (the GPU step time that section 14 compares with vLLM).
The audit passes when every repetition has between 1 and `budget` kernels and no graph launch.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path

RANGE_RE = re.compile(r"^megabench:rank(\d+):rep(\d+)$")


def export_sqlite(report: Path, sqlite_path: Path | None = None, nsys: str = "nsys") -> Path:
    sqlite_path = sqlite_path or report.with_suffix(".sqlite")
    subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(sqlite_path), str(report)],
                   check=True, capture_output=True, text=True, timeout=3600)
    return sqlite_path


def _tables(conn) -> set[str]:
    return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


def _columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"pragma table_info({table})")]


def _strings(conn) -> dict[int, str]:
    return dict(conn.execute("select id, value from StringIds")) if "StringIds" in _tables(conn) else {}


def read_ranges(conn) -> dict[tuple[int, int], tuple[int, int]]:
    """(rank, rep) -> (start_ns, end_ns) for every megabench rep range."""
    strings = _strings(conn)
    out = {}
    columns = _columns(conn, "NVTX_EVENTS")
    for start, end, text, text_id in conn.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"):
        label = text if text is not None else strings.get(text_id)
        if not label:
            continue
        match = RANGE_RE.match(label)
        if match:
            out[(int(match.group(1)), int(match.group(2)))] = (int(start), int(end))
    del columns
    return out


def _events(conn, table: str, name_columns: tuple[str, ...] = ()) -> list[dict]:
    if table not in _tables(conn):
        return []
    columns = set(_columns(conn, table))
    strings = _strings(conn) if name_columns else {}
    wanted = ["start", "end"] + [c for c in ("deviceId", "globalPid", "globalTid", "streamId", "bytes", "copyKind") if c in columns]
    name_col = next((c for c in name_columns if c in columns), None)
    if name_col:
        wanted.append(name_col)
    rows = []
    for row in conn.execute(f"select {', '.join(wanted)} from {table}"):
        item = dict(zip(wanted, row))
        if name_col:
            item["name"] = strings.get(item.pop(name_col), str(item.get(name_col)))
        rows.append(item)
    return rows


def audit_trace(sqlite_path: Path, *, budget: int, device_of_rank: dict[int, int], ranks: list[int] | None = None) -> dict:
    """Per-rank, per-rep audit. `device_of_rank` maps global rank -> deviceId as CUPTI reports it on this host."""
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
        if "NVTX_EVENTS" not in tables:
            return {"status": "unverified", "reason": "no NVTX ranges in the trace"}
        ranges = read_ranges(conn)
        kernels = _events(conn, "CUPTI_ACTIVITY_KIND_KERNEL", ("shortName", "demangledName", "mangledName"))
        copies = _events(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
        memsets = _events(conn, "CUPTI_ACTIVITY_KIND_MEMSET")
        runtime = _events(conn, "CUPTI_ACTIVITY_KIND_RUNTIME", ("nameId",))
        graph_nodes = _events(conn, "CUDA_GRAPH_NODE_EVENTS") + _events(conn, "CUDA_GRAPH_EVENTS")
    finally:
        conn.close()
    graph_launch_calls = [r for r in runtime if "graphlaunch" in (r.get("name") or "").lower()]
    result = {"status": "within_budget", "max_gpu_launches_per_rank": budget, "ranks": {},
              "tables": sorted(tables), "rep_ranges_found": len(ranges),
              "counting_rule": "device kernels (compute + NCCL) inside the rep range on the rank's device; "
                               "memcpy/memset reported, not counted; any cudaGraphLaunch in the window fails"}
    if not ranges:
        return result | {"status": "unverified", "reason": "no megabench:rank*:rep* ranges in the trace"}
    wanted_ranks = ranks if ranks is not None else sorted({rank for rank, _ in ranges})
    failed = False
    for rank in wanted_ranks:
        device = device_of_rank.get(rank)
        reps = sorted(rep for (r, rep) in ranges if r == rank)
        rank_record = {"device_id": device, "reps": []}
        for rep in reps:
            start, end = ranges[(rank, rep)]

            def inside(event, same_device=True):
                if same_device and device is not None and event.get("deviceId") is not None and event["deviceId"] != device:
                    return False
                return event["start"] >= start and (event["end"] or event["start"]) <= end

            in_kernels = [k for k in kernels if inside(k)]
            other = sum(1 for k in kernels if inside(k, same_device=False)) - len(in_kernels)
            in_copies = [c for c in copies if inside(c)]
            in_memsets = [m for m in memsets if inside(m)]
            in_graphs = [g for g in graph_launch_calls if g["start"] >= start and g["start"] <= end]
            in_nodes = [g for g in graph_nodes if inside(g)]
            nccl = [k for k in in_kernels if "nccl" in (k.get("name") or "").lower()]
            span_ms = ((max(k["end"] for k in in_kernels) - min(k["start"] for k in in_kernels)) / 1e6) if in_kernels else None
            busy_ms = sum((k["end"] - k["start"]) for k in in_kernels) / 1e6 if in_kernels else None
            count = len(in_kernels)
            ok = 1 <= count <= budget and not in_graphs and not in_nodes
            failed |= not ok
            rank_record["reps"].append({
                "rep": rep, "range_ms": (end - start) / 1e6, "device_kernels": count, "nccl_kernels": len(nccl),
                "memcpy_events": len(in_copies), "memset_events": len(in_memsets), "graph_launches": len(in_graphs),
                "graph_node_events": len(in_nodes), "kernels_on_other_devices_in_window": other,
                "gpu_span_ms": span_ms, "gpu_busy_ms": busy_ms, "within_budget": ok,
                "kernel_names": sorted({k.get("name") or "?" for k in in_kernels})[:16]})
        spans = [r["gpu_span_ms"] for r in rank_record["reps"] if r["gpu_span_ms"] is not None]
        rank_record["gpu_span_p50_ms"] = sorted(spans)[len(spans) // 2] if spans else None
        rank_record["within_budget"] = all(r["within_budget"] for r in rank_record["reps"]) and bool(rank_record["reps"])
        result["ranks"][str(rank)] = rank_record
    result["status"] = "launch_budget_failed" if failed else "within_budget"
    spans = [r["gpu_span_p50_ms"] for r in result["ranks"].values() if r["gpu_span_p50_ms"] is not None]
    result["gpu_span_p50_ms_slowest_rank"] = max(spans) if spans else None
    return result


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="audit a megabench nsys trace")
    parser.add_argument("report", type=Path, help=".nsys-rep or .sqlite")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--device-of-rank", default="", help="rank:device pairs, e.g. 0:0,1:1 (default: rank i -> device i)")
    args = parser.parse_args(argv)
    sqlite_path = args.report if args.report.suffix == ".sqlite" else export_sqlite(args.report)
    mapping = {int(a): int(b) for a, b in (pair.split(":") for pair in args.device_of_rank.split(",") if pair)}
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    ranks = sorted({rank for rank, _ in read_ranges(conn)})
    conn.close()
    mapping = mapping or {rank: rank for rank in ranks}
    print(json.dumps(audit_trace(sqlite_path, budget=args.budget, device_of_rank=mapping), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
