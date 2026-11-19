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


def check(records: Path, home: Path | None = None) -> dict:
    problems: list[str] = []
    transcripts = records / "transcripts"
    manifest = transcripts / "MANIFEST.jsonl"
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
        relative = Path(entry["path"]).relative_to(".claude/projects")
        if entry.get("event") == "rewritten":
            verify(pending.pop(entry["path"], []), transcripts / relative.parent / entry["kept_as"], f"{entry['path']} (kept copy {entry['kept_as']})")
            continue
        pending.setdefault(entry["path"], []).append(entry)
    for path, entries in pending.items():
        verify(entries, transcripts / Path(path).relative_to(".claude/projects"), path)
    final = transcripts / "FINAL.json"
    if not final.exists():
        problems.append("no FINAL.json (capture did not finish)")
    else:
        record = json.loads(final.read_text())
        if record["mismatched"]:
            problems.append(f"FINAL.json reports mismatches: {record['mismatched'][:5]}")
        if home is not None:
            for key, value in record["files"].items():
                src = home / key
                if not src.exists():
                    problems.append(f"source gone {key}")
                elif hashlib.sha256(src.read_bytes()).hexdigest() != value["sha256"]:
                    problems.append(f"source changed after FINAL {key}")
    streams = sorted((records / "stream").glob("*.jsonl")) if (records / "stream").is_dir() else []
    for stream in streams:
        last = None
        for number, raw in enumerate(stream.read_text().splitlines(), 1):
            try:
                last = json.loads(raw)
            except json.JSONDecodeError:
                problems.append(f"{stream.name}:{number} not JSON")
        if last is None or last.get("type") != "result":
            problems.append(f"{stream.name}: last event is {None if last is None else last.get('type')!r}, not 'result'")
    gpu = records / "gpu"
    jobs = 0
    if gpu.is_dir():
        for node in gpu.iterdir():
            for done in (node / "done").glob("*.json") if (node / "done").is_dir() else []:
                jobs += 1
                for d, suffix in (("req", ".json"), ("req", ".sh"), ("out", ".log")):
                    if not (node / d / (done.stem + suffix)).exists():
                        problems.append(f"gpu job {done.stem} on {node.name}: missing {d}/{done.stem}{suffix}")
    return {"ok": not problems, "problems": problems, "manifest_entries": checked, "streams": len(streams), "gpu_jobs": jobs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    result = check(args.records, args.home)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
