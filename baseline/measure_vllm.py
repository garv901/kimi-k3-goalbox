"""Measure the vLLM baseline decode-step latency for each graded Kimi K3 case.

Method (the same for every batch size B, context 8192, TP8):
  vllm bench latency --input-len 8192 --batch-size B, once with --output-len 1 and once
  with --output-len 1+N. decode_step_ms = (latency(1+N) - latency(1)) / N.
Both runs use FULL decode CUDA graphs, FP8 KV cache, greedy sampling, and
--additional-config '{"decode_backend": "latency_specialized"}' (PR 61867). Use
--arm stock to measure stock vLLM without the specialized path.

  .venv-vllm/bin/python baseline/measure_vllm.py --model HF_DIR [--arm specialized|stock] [--steps 64]
Writes baseline/BASELINES.json (merged with existing arms).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = {"kimi-k3-tp8-b1-s8192": 1, "kimi-k3-tp8-b8-s8192": 8, "kimi-k3-tp8-b64-s8192": 64}


def latency(args, batch: int, output_len: int) -> float:
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        cmd = [str(Path(sys.executable).parent / "vllm"), "bench", "latency",
               "--model", args.model, "--trust-remote-code", "--tensor-parallel-size", "8",
               "--input-len", "8192", "--output-len", str(output_len), "--batch-size", str(batch),
               "--max-model-len", str(8192 + args.steps + 8), "--kv-cache-dtype", "fp8",
               "--num-iters-warmup", "3", "--num-iters", str(args.iters), "--output-json", out.name]
        if args.arm == "specialized":
            cmd += ["--additional-config", json.dumps({"decode_backend": "latency_specialized"})]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(Path(out.name).read_text())["avg_latency"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="HF snapshot directory of moonshotai/Kimi-K3")
    parser.add_argument("--arm", choices=("specialized", "stock"), default="specialized")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args()
    import torch
    import vllm
    path = HERE / "BASELINES.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for case in args.cases:
        batch = CASES[case]
        prefill = latency(args, batch, 1)
        full = latency(args, batch, 1 + args.steps)
        step_ms = (full - prefill) / args.steps * 1e3
        data.setdefault(case, {})[f"vllm_{args.arm}"] = {
            "decode_step_ms": round(step_ms, 4), "batch": batch, "context": 8192, "tp": 8,
            "method": f"(latency(out=1+{args.steps}) - latency(out=1)) / {args.steps}, vllm bench latency, FP8 KV",
            "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
            "measured_utc": datetime.now(timezone.utc).isoformat()}
        print(case, f"{step_ms:.3f} ms per decode step", flush=True)
        path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
