#!/usr/bin/env python3
"""Peak disk and host-memory estimate for the TP8 checkpoint conversion (conditional-GO item 6).

Exact output bytes come from the fixture's weight_shapes() for the real 93-layer geometry: every
rank's safetensors files are the per-rank shards plus the replicated tensors, so the output is
sum over ranks of the per-rank bytes (plus safetensors headers, a few KB per file). The converter
(megabench.tasks.kimi_checkpoint.convert) runs one layer per worker process: a worker holds the
layer's source tensors (read whole, then split) and the TP shards it is about to save, so its peak RSS
is about 2x the layer's source bytes plus the reader (mmap, not RSS) and the interpreter. The MoE
layers dominate (896 MXFP4 experts). In-flight temporaries on disk are one `.tmp` file per rank per
worker. Measured on 2026-11-13 (TP4, 4 layers): about 35 GB RSS per worker, consistent with the
2x-layer estimate below.

    CUDA_VISIBLE_DEVICES= python -s tools/conversion_sizing.py [--tp 8] [--layers 93] [--workers 4]
                                   [--out-dir /opt/verda/cache/goalbox] [--src-dir HF_DIR] [--json]
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from megabench.cases import select_cases
from megabench.tasks.kimi import weight_shapes

ELEMENT = {torch.bfloat16: 2, torch.float32: 4, torch.float16: 2, torch.uint8: 1, torch.int8: 1, torch.int64: 8,
           torch.float8_e4m3fn: 1, torch.float8_e8m0fnu: 1}


def nbytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * ELEMENT[dtype]


def sizing(tp: int, layers: int, workers: int) -> dict:
    base = select_cases("all", ["kimi-k3-tp8-b1-s8192"])[0]
    case = replace(base, gpus=tp, tp=tp, params=base.params | {"layers": layers})
    shapes = weight_shapes(case)
    per_rank = sum(nbytes(shape, dtype) for shape, dtype, _, _ in shapes.values())
    by_layer: dict[int, int] = {}
    global_bytes = 0
    for key, (shape, dtype, _, _) in shapes.items():
        if key.startswith("l") and "_" in key and key[1:key.index("_")].isdigit():
            by_layer[int(key[1:key.index("_")])] = by_layer.get(int(key[1:key.index("_")]), 0) + nbytes(shape, dtype)
        else:
            global_bytes += nbytes(shape, dtype)
    # stacked per-layer norms/res-proj live under un-prefixed keys with a leading layer dim: already in global_bytes
    biggest_layer_rank = max(by_layer.values())
    biggest_layer_all = biggest_layer_rank * tp     # the source layer = all shards together
    out_total = per_rank * tp
    peak_tmp = workers * biggest_layer_all           # one .tmp per rank per worker, worst case all MoE layers
    peak_rss_per_worker = 2 * biggest_layer_all + (3 << 30)   # two copies of the layer plus the interpreter
    return {"tp": tp, "layers": layers, "workers": workers,
            "output_bytes_total": out_total, "output_bytes_per_rank": per_rank,
            "global_tensors_bytes_per_rank": global_bytes,
            "biggest_layer_bytes_all_ranks": biggest_layer_all, "biggest_layer_bytes_per_rank": biggest_layer_rank,
            "peak_disk_bytes": out_total + peak_tmp,
            "peak_host_rss_bytes_per_worker_estimate": peak_rss_per_worker,
            "peak_host_rss_bytes_total_estimate": workers * peak_rss_per_worker,
            "layer_bytes_per_rank_histogram": sorted(set(by_layer.values()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--layers", type=int, default=93)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--src-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = sizing(args.tp, args.layers, args.workers)
    if args.out_dir and args.out_dir.exists():
        usage = shutil.disk_usage(args.out_dir)
        report["out_dir"] = {"path": str(args.out_dir), "free_bytes": usage.free,
                             "fits_with_0.9TB_margin": usage.free - report["peak_disk_bytes"] >= 0.9e12}
    if args.src_dir and args.src_dir.exists():
        report["src_bytes"] = sum(p.stat().st_size for p in args.src_dir.glob("*.safetensors"))
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        gb = lambda b: f"{b / 1e9:,.1f} GB"
        print(f"TP{report['tp']} x {report['layers']} layers, {report['workers']} workers")
        print(f"  output total {gb(report['output_bytes_total'])}  ({gb(report['output_bytes_per_rank'])} per rank)")
        print(f"  biggest layer {gb(report['biggest_layer_bytes_all_ranks'])} source; peak disk incl. temporaries {gb(report['peak_disk_bytes'])}")
        print(f"  host RSS estimate {gb(report['peak_host_rss_bytes_per_worker_estimate'])} per worker, "
              f"{gb(report['peak_host_rss_bytes_total_estimate'])} total")
        if "out_dir" in report:
            print(f"  {report['out_dir']['path']}: {gb(report['out_dir']['free_bytes'])} free; fits with 0.9 TB margin: {report['out_dir']['fits_with_0.9TB_margin']}")
        if "src_bytes" in report:
            print(f"  source checkpoint {gb(report['src_bytes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
