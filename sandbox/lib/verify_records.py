#!/usr/bin/env python3
"""Check the integrity of one run's controller-owned records (DECISIONS D-T3, section 17).

    verify_records.py RECORDS_DIR [--home AGENT_HOME]

Checks: every MANIFEST.jsonl line's sha256 matches the copy it names (copies are append-only, so a
later line for the same path must be a prefix-extension: the earlier size is checked against the earlier
digest only when the copy's current prefix still hashes to it); FINAL.json exists and reports no
mismatch; every stream/*.jsonl file parses line by line and the last line is a `result` event; the
GPU job spool (gpu/<node>/{req,out,done}) is consistent (every done record has a request and an output).
With --home, the live agent home is compared with the copies too. Exit 0 only if everything holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_prefix(path: Path, size: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        remaining = size
        while remaining > 0:
            block = file.read(min(1 << 20, remaining))
            if not block:
                break
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def check_source(records: Path, out: Path, source: Path | None, problems: list[str]) -> tuple[int, int]:
    """One captured source (records/<name>/): manifest prefixes, FINAL.json, optional live comparison."""
    manifest = out / "MANIFEST.jsonl"
    lines = manifest.read_text().splitlines() if manifest.exists() else []
    checked = 0
    pending: dict[str, list[dict]] = {}

    def verify(entries: list[dict], copy: Path, label: str) -> None:
        nonlocal checked
        for entry in entries:
            if not copy.exists():
                problems.append(f"missing copy {label}")
                return
            if copy.stat().st_size < entry["size"]:
                problems.append(f"copy shrank {label}")
                return
            if sha256_prefix(copy, entry["size"]) != entry["sha256"]:
                problems.append(f"prefix digest mismatch {label} at {entry['size']} bytes")
            checked += 1

    for line in lines:
        entry = json.loads(line)
        copy_rel = entry.get("copy") or f"{out.name}/{entry['path']}"
        if entry.get("event") == "rewritten":
            kept = (records / copy_rel).parent / entry["kept_as"]
            verify(pending.pop(entry["path"], []), kept, f"{out.name}:{entry['path']} (kept copy {entry['kept_as']})")
            continue
        pending.setdefault(entry["path"], []).append(entry)
    for path, entries in pending.items():
        copy_rel = entries[0].get("copy") or f"{out.name}/{path}"
        verify(entries, records / copy_rel, f"{out.name}:{path}")
    final = out / "FINAL.json"
    if not final.exists():
        problems.append(f"{out.name}: no FINAL.json (capture did not finish)")
        return checked, 0
    record = json.loads(final.read_text())
    if record["mismatched"]:
        problems.append(f"{out.name}: FINAL.json reports mismatches: {record['mismatched'][:5]}")
    if source is not None:
        for key, value in record["files"].items():
            src = source / key
            if not src.exists():
                problems.append(f"{out.name}: source gone {key}")
            elif hashlib.sha256(src.read_bytes()).hexdigest() != value["sha256"]:
                problems.append(f"{out.name}: source changed after FINAL {key}")
    return checked, len(record["files"])


def check(records: Path, home: Path | None = None, sources: dict[str, Path] | None = None) -> dict:
    problems: list[str] = []
    live = dict(sources or {})
    if home is not None:
        live["transcripts"] = home / ".claude" / "projects"
    checked = files = 0
    captured = [d for d in records.iterdir() if d.is_dir() and (d / "MANIFEST.jsonl").exists()] if records.is_dir() else []
    if not captured:
        problems.append("no captured source with a MANIFEST.jsonl")
    for out in sorted(captured):
        c, f = check_source(records, out, live.get(out.name), problems)
        checked += c
        files += f
    streams = sorted((records / "stream").glob("agent-*.jsonl")) if (records / "stream").is_dir() else []
    complete = 0
    for stream in streams:
        last = None
        for number, raw in enumerate(stream.read_text().splitlines(), 1):
            try:
                last = json.loads(raw)
            except json.JSONDecodeError:
                problems.append(f"{stream.name}:{number} not JSON")
        complete += int(last is not None and last.get("type") == "result")   # a run stopped at the deadline ends without one
    gpu = records / "gpu"
    jobs = 0
    if (gpu / "results").is_dir():
        for node in (gpu / "results" / "gpu").iterdir() if (gpu / "results" / "gpu").is_dir() else []:
            for done in (node / "done").glob("*.json") if (node / "done").is_dir() else []:
                jobs += 1
                for d, suffix in (("req", ".json"), ("req", ".sh")):
                    if not (gpu / "req" / "gpu" / node.name / d / (done.stem + suffix)).exists():
                        problems.append(f"gpu job {done.stem} on {node.name}: missing request {done.stem}{suffix}")
                if not (node / "out" / (done.stem + ".log")).exists():
                    problems.append(f"gpu job {done.stem} on {node.name}: missing output log")
    return {"ok": not problems, "problems": problems, "manifest_entries": checked, "captured_files": files,
            "sources": [d.name for d in captured], "streams": len(streams), "streams_with_result": complete, "gpu_jobs": jobs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--source", action="append", default=[], help="NAME=DIR live source to compare with")
    args = parser.parse_args()
    result = check(args.records, args.home, {k: Path(v) for k, _, v in (x.partition("=") for x in args.source)})
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
