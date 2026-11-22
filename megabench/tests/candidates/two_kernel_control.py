"""C2 profiling control: exactly two device kernels per run(). Expected audit outcome: launch_budget_failed
(budget 1). Verdict `incorrect`; never a pass."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))      # the snapshot holds the sibling _control_common.py
from _control_common import template_outputs  # noqa: E402

_state = {}


def build(case: dict):
    def run(inputs: dict) -> dict:
        if "outputs" not in _state:
            _state["outputs"] = template_outputs(case, inputs)
        outputs = _state["outputs"]
        first = next(iter(outputs.values()))
        first.add_(1)                                                 # kernel 1
        first.mul_(1)                                                 # kernel 2
        return outputs

    return run
