"""Create the agent workspace for the Kimi K3 megakernel task.

The workspace holds TASK.md, cases.json, a read-only snapshot of the PyTorch
oracle, a submission stub and three tools that call the trusted benchmark
(mounted read-only at /goalbox/bench inside the sandbox):
  check_candidate [--case ID]   correctness (oracle band), launch audit, timing
  profile_candidate [--case ID] Nsight Compute capture of one rank
  sol_info [--case ID] --json   per-GPU HBM floor and byte breakdown
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ..cases import CASES, GRADED

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ("kimi.py", "common.py", "quantization.py", "distributed.py", "parallel.py")
DEV_CASE = "kimi-k3-dev-l4-tp1-b8-s8192"

TASK = """# Kimi K3 decode-step megakernel

Write one persistent GPU kernel per rank that runs one complete Kimi K3 text decode
step for a batch of B sequences, tensor parallel over TP GPUs. The cases are in
`cases.json`. The graded cases are:

{graded}

The exact math, the input layouts and the output layouts are in the PyTorch oracle
`reference/megabench/tasks/kimi.py` (with its small helpers). `weight_shapes()` lists
every per-rank weight tensor. `make_state()` lists the per-sequence state.

## Interface

`submission.py` exports `build(case: dict) -> run`. `run(inputs: dict) -> dict` gets this
rank's tensors and returns this rank's outputs: {outputs}.
`case['execution']` (multi-GPU cases) gives rank, world_size, tp_rank, device, and the live
torch.distributed `tp_group`. Do all communication inside the one launch per rank. Use the
same `build` for every case: read the geometry from `case['params']`.

## Correctness: the oracle band

The grader runs the BF16 oracle and an FP32 version of the same math. Expert routing is
discrete, so the FP32 version takes the expert sets as given: once the oracle's sets, and
once your sets. For logits (each batch row), the MLA cache writes and the KDA states, your
relative L2 distance to the FP32 result on your routes must be at most {band} x the BF16
oracle's distance to the FP32 result on its routes + {floor}. Your `next_token` must be the
FP32 argmax on your routes, or within the near-tie margin of it. Your `expert_ids` sets (any
order inside a row, the same on all ranks) must be near-top-k sets: under the FP32 router
scores, the weakest expert you chose may be at most {band} x the oracle's worst shortfall
+ {slack} below the k-th best score. `cache_length` must be exact. You may use other precisions (for example MXFP8
activations) when the result stays in the band.

## Rules

- At most one GPU kernel launch per rank for the whole step (the launch audit counts
  NCCL kernels too).
- Compute from the runtime inputs on every call. Weights (synthetic cases), tokens,
  reset flags and states change between trials. Do not mutate inputs, cache answers,
  call the oracle, replay a multi-kernel CUDA graph, or move model work to the CPU.
- The real-weight cases load the converted checkpoint. The synthetic cases make fresh
  random weights for every trial, so a submission that depends on fixed weights fails.

## Tools

- `./check_candidate --case ID` runs the trusted harness on a GPU through `gpu-run`
  (the default case is `{dev}`, one GPU). Multi-GPU cases request all their GPUs.
- `./sol_info --case ID --json` prints the per-GPU HBM floor and its byte breakdown.
- `./profile_candidate --case ID [ncu options]` captures one warm step with Nsight Compute
  (all ranks) into `reports/`.
- `BASELINES.json` has the measured latency of the best known implementation
  (vLLM with the Kimi K3 latency-specialized decode path) for each graded case.
"""


def make_workspace(output: Path, *, bench: str = "/goalbox/bench") -> Path:
    project = output.resolve()
    project.mkdir(parents=True, exist_ok=False)
    (project/".gitignore").write_text("reports/\n__pycache__/\n*.log\n*.ncu-rep\n")
    (project/"reports").mkdir()
    (project/"submission.py").write_text("def build(case: dict):\n    raise NotImplementedError(case['id'])\n")
    target = project/"reference/megabench"
    (target/"tasks").mkdir(parents=True)
    shutil.copy2(ROOT/"megabench/__init__.py", target/"__init__.py")
    shutil.copy2(ROOT/"megabench/tasks/__init__.py", target/"tasks/__init__.py")
    for name in REFERENCE:
        shutil.copy2(ROOT/"megabench/tasks"/name, target/"tasks"/name)
    shutil.copy2(ROOT/"megabench/cases.py", target/"cases.py")
    (project/"cases.json").write_text(json.dumps([c.to_dict() for c in CASES], indent=2)+"\n")
    baselines = ROOT/"baseline/BASELINES.json"
    if baselines.exists():
        shutil.copy2(baselines, project/"BASELINES.json")
    from ..tasks.kimi import BAND, FLOOR, ROUTE_FLOOR
    outputs = ("logits", "next_token", "kv_write", "pe_write", "recurrent_state", "conv_state",
               "expert_ids", "cache_length")
    by_id = {c.id: c for c in CASES}
    graded = "\n".join(f"- `{i}`: B={by_id[i].params['batch']}, {by_id[i].params['layers']} layers, "
                       f"TP{by_id[i].tp}, context {by_id[i].params['context']}, "
                       f"{'real checkpoint' if by_id[i].params.get('checkpoint') else 'fresh synthetic'} weights"
                       for i in GRADED)
    (project/"TASK.md").write_text(TASK.format(graded=graded, outputs=", ".join(f"`{o}`" for o in outputs),
                                               band=BAND, floor=FLOOR, slack=ROUTE_FLOOR, dev=DEV_CASE))
    gpus = {c.id: c.gpus for c in CASES}
    common = ("#!/bin/bash\nset -eu\n"
              f"export PYTHONPATH={bench}\ncd /work\n"
              f"CASE={DEV_CASE}\nARGS=()\n"
              "while [ $# -gt 0 ]; do case $1 in --case) CASE=$2; shift 2 ;; *) ARGS+=(\"$1\"); shift ;; esac; done\n"
              f"declare -A GPUS=({' '.join(f'[{k}]={v}' for k, v in gpus.items())})\n"
              "[ -n \"${GPUS[$CASE]:-}\" ] || { echo \"unknown case $CASE\"; exit 2; }\n")
    tools = {
        "check_candidate": common +
        "[ -e /dev/nvidiactl ] || exec gpu-run -g \"${GPUS[$CASE]}\" -t \"${GPU_MINUTES:-30}\" -- ./check_candidate --case \"$CASE\" \"${ARGS[@]}\"\n"
        "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        "exec python -m megabench evaluate --case \"$CASE\" --submission /work/submission.py "
        "--method local-check --session-id \"local-$run_id\" --device cuda:0 --trials 2 --warmup 2 --reps 10 "
        "--timeout 3600 --output \"reports/check-$CASE-$run_id.jsonl\" \"${ARGS[@]}\"\n",
        "profile_candidate": common +
        "[ -e /dev/nvidiactl ] || exec gpu-run -g \"${GPUS[$CASE]}\" -t \"${GPU_MINUTES:-30}\" -- ./profile_candidate --case \"$CASE\" \"${ARGS[@]}\"\n"
        "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        "# extra arguments go to ncu, for example: --set full -k regex:my_kernel\n"
        "exec ncu --profile-from-start off --target-processes all -o \"reports/profile-$CASE-$run_id\" \"${ARGS[@]}\" "
        "python -m megabench.integrations.ncu_profile_case --case \"$CASE\" --submission /work/submission.py\n",
        "sol_info": common + "exec python -m megabench sol --case \"$CASE\" \"${ARGS[@]}\"\n",
    }
    for name, text in tools.items():
        (project/name).write_text(text)
        (project/name).chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=goalbox", "-c", "user.email=goalbox@invalid", "commit", "-q",
                    "-m", "Kimi K3 megakernel workspace"], cwd=project, check=True)
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench", default="/goalbox/bench", help="trusted benchmark path as seen by the tools")
    args = parser.parse_args()
    print(make_workspace(args.output, bench=args.bench))


if __name__ == "__main__":
    main()
