#!/usr/bin/env python3
"""Trajectory capture sidecar (DECISIONS section 17, D-T*). Runs as a separate user (uid 2000) next to
the agent container, with the agent's home mounted read-only and the records volume read-write; the
agent has no path to the records volume.

Every INTERVAL seconds it copies, into RECORDS/transcripts/, every file under the agent's
~/.claude/projects/** (the main-session and subagent JSONL transcripts; D-T2) that is new or has grown,
appends only the new bytes (transcripts are append-only), and writes RECORDS/transcripts/MANIFEST.jsonl
with one line per copy: path, size, sha256 of the whole copy, UTC time. The harness's own stream-json
output (D-T1) is written by the launcher straight into RECORDS/stream/ and is not handled here.

At the end (`--final`) it does one more pass and writes RECORDS/transcripts/FINAL.json with the sha256 of
every source file as it is at that moment, so verify_records.py can show that the copy equals the source
(D-T3). A source that shrank or changed bytes it had already copied is recorded as "rewritten" and the
original copy is kept under a new name; nothing is ever deleted from RECORDS.

    capture.py --home /agent-home --records /records [--interval 30] [--once] [--final]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sources(home: Path) -> list[Path]:
    root = home / ".claude" / "projects"
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


class Capture:
    def __init__(self, home: Path, records: Path):
        self.home, self.records = home, records
        self.out = records / "transcripts"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "STATE.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.manifest_path = self.out / "MANIFEST.jsonl"

    def target(self, src: Path) -> Path:
        return self.out / src.relative_to(self.home / ".claude" / "projects")

    def pass_once(self) -> dict:
        copied = rewritten = unchanged = 0
        for src in sources(self.home):
            key = str(src.relative_to(self.home))
            size = src.stat().st_size
            known = self.state.get(key, {"copied": 0, "head_sha": None})
            dst = self.target(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as file:
                rewritten_now = size < known["copied"] or (known["copied"] and self._head_changed(file, known))
                if size == known["copied"] and dst.exists() and not rewritten_now:
                    unchanged += 1
                    continue
                if rewritten_now:
                    # Not append-only any more: keep the old copy under a new name, start a fresh one, and
                    # tell the verifier (earlier manifest lines for this path now refer to the kept copy).
                    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                    kept = dst.with_name(dst.name + f".rewritten-{stamp}-{os.getpid()}")
                    if dst.exists():
                        dst.rename(kept)
                    with self.manifest_path.open("a") as manifest:
                        manifest.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": key,
                                                   "event": "rewritten", "kept_as": kept.name}) + "\n")
                    known = {"copied": 0, "head_sha": None}
                    rewritten += 1
                file.seek(known["copied"])
                with dst.open("ab") as out:
                    while block := file.read(1 << 20):
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
            copied += 1
            digest = sha256_file(dst)
            head = self._head_sha(src)
            self.state[key] = {"copied": size, "head_sha": head}
            with self.manifest_path.open("a") as manifest:
                manifest.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": key,
                                           "size": size, "sha256": digest}) + "\n")
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.state_path)
        return {"copied": copied, "rewritten": rewritten, "unchanged": unchanged}

    @staticmethod
    def _head_sha(src: Path, limit: int = 1 << 16) -> str:
        with src.open("rb") as file:
            return hashlib.sha256(file.read(limit)).hexdigest()

    def _head_changed(self, file, known) -> bool:
        pos = file.tell()
        file.seek(0)
        head = hashlib.sha256(file.read(min(known["copied"], 1 << 16))).hexdigest()
        file.seek(pos)
        return known["head_sha"] is not None and head != known["head_sha"]

    def final(self) -> dict:
        summary = self.pass_once()
        entries = {str(src.relative_to(self.home)): {"size": src.stat().st_size, "sha256": sha256_file(src),
                                                     "copy_sha256": sha256_file(self.target(src))}
                   for src in sources(self.home)}
        mismatched = [k for k, v in entries.items() if v["sha256"] != v["copy_sha256"]]
        record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": entries, "mismatched": mismatched,
                  "summary": summary}
        (self.out / "FINAL.json").write_text(json.dumps(record, indent=1))
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--stop-file", type=Path, help="exit (after a final pass) when this file appears")
    args = parser.parse_args()
    capture = Capture(args.home, args.records)
    if args.final:
        record = capture.final()
        print(json.dumps({"final": True, "files": len(record["files"]), "mismatched": record["mismatched"]}))
        return 1 if record["mismatched"] else 0
    if args.once:
        print(json.dumps(capture.pass_once()))
        return 0
    while True:
        capture.pass_once()
        (capture.out / "HEARTBEAT").write_text(str(time.time()))
        if args.stop_file and args.stop_file.exists():
            capture.final()
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
