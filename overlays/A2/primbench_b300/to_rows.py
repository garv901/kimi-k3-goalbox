#!/usr/bin/env python3
"""Turn a run_seed.sh output directory into primdb rows (one JSON object per line on stdout).

Mirrors megabench.primdb.importers.primbench(), plus conditions c.gpu / c.arch so B300 rows
never collide with (or get mistaken for) the B200 seed rows, and adds facts found while porting.
Self-contained: python3 only (no megabench import needed).
"""
import hashlib
import json
import sys
from pathlib import Path

UNTUNED = {"gemm_small_tc"}
UNTUNED_NOTE = ("a lower bound on what tcgen05 reaches here; ",
                "its time is an upper bound on what a tuned tcgen05 kernel takes at this shape; ")


def row_id(row):  # same as megabench.primdb.row_id
    key = json.dumps([row.get("primitive"), row.get("params", {}), row.get("cond", {}),
                      row.get("source", {}).get("file"), row.get("source", {}).get("job")],
                     sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def normalize(flat, **defaults):  # same as megabench.primdb.normalize
    row = {"kind": "measure", "params": {}, "metrics": {}, "cond": {}, "source": {}}
    for key, value in (defaults | flat).items():
        prefix, _, name = key.partition(".")
        target = {"p": "params", "m": "metrics", "c": "cond", "s": "source"}.get(prefix)
        if target and name:
            row[target][name] = value
        elif isinstance(value, dict) and key in row:
            row[key] |= value
        else:
            row[key] = value
    row.setdefault("id", row_id(row))
    return row


def gpu_name(out: Path) -> str:
    txt = (out / "gpu.txt").read_text() if (out / "gpu.txt").is_file() else ""
    return "B300" if "B300" in txt else ("B200" if "B200" in txt else (txt.split(",")[0].strip() or "?"))


def main(out: Path):
    arch = (out / "arch.txt").read_text().strip() if (out / "arch.txt").is_file() else "sm_103a"
    gpu = gpu_name(out)
    tag = {"c.gpu": gpu, "c.arch": arch}
    rows = []
    for path in sorted(out.glob("*.jsonl")):
        bench = {"tc_overhead": "tc_mma"}.get(path.stem, path.stem)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            flat = json.loads(line)
            if path.stem in UNTUNED:
                flat = {**flat, "p.tuned": False,
                        "note": flat.get("note", "").replace(*UNTUNED_NOTE)}
            m = flat.get("metrics") or {k: v for k, v in flat.items() if k.startswith("m.")}
            if flat.get("kind", "measure") == "measure" and not m:
                continue
            rows.append(normalize(flat, **tag, **{
                "s.bench": f"primbench/{bench}.cu (A2 overlay copy, {arch})",
                "s.file": f"primbench-{gpu.lower()}/{path.name}", "s.job": "a2-seed",
                "s.timer": "clock64"}))
    if arch == "sm_103a":
        rows.append(normalize({
            "kind": "fact", "family": "tc_mma", "primitive": "tcgen05.mma", "p.kind": "i8",
            "p.arch": "sm_103a", "available": False,
            "note": "tcgen05.mma kind::i8 does not assemble for sm_103a (B300): ptxas 'Feature "
                    ".kind::i8 not supported on .target sm_103a'. Use f16/tf32/f8f6f4/mxf8f6f4/"
                    "mxf4/mxf4nvf4 kinds.",
        }, **tag, **{"s.bench": "ptxas (CUDA 13.0) on primbench/tc_mma.cu"}))
    for r in rows:
        print(json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
