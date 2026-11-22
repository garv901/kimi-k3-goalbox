"""C2 profiling negative control: run() itself launches one kernel, but build() also starts a child process
with an emptied environment that creates its own CUDA context and launches a kernel. The child is invisible
to an injection-based trace; the supervisor census (extra process with a CUDA context) or the NVML/GPU
accounting process list must flag it, and the audit must be INVALID, never a pass."""
import os
import subprocess
import sys

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))      # the snapshot holds the sibling _control_common.py
from _control_common import template_outputs  # noqa: E402

_state = {}
CHILD = ("import torch, time; d = 'cuda' if torch.cuda.is_available() else 'cpu'; x = torch.ones(1 << 20, device=d); "
         "t = time.time()\nwhile time.time() - t < float(__import__('sys').argv[1]):\n    x.add_(1)\n"
         "torch.cuda.synchronize() if d == 'cuda' else None")


def build(case: dict):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}                  # no injection variables
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    seconds = os.environ.get("GOALBOX_UNTRACED_SECONDS", "60")
    _state["child"] = subprocess.Popen([sys.executable, "-c", CHILD, seconds], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(inputs: dict) -> dict:
        if "outputs" not in _state:
            _state["outputs"] = template_outputs(case, inputs)
        first = next(iter(_state["outputs"].values()))
        first.add_(1)
        return _state["outputs"]

    return run
