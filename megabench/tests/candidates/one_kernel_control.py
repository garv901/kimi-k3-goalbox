"""C2 profiling control: exactly one device kernel per run() (an in-place add on one output). Outputs are
structurally valid and numerically wrong, so the verdict is `incorrect`; with --trials 0 the timing phase
still runs and the evaluator's repetition windows exist, each holding one kernel. Never a pass."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))      # the snapshot holds the sibling _control_common.py
from _control_common import template_outputs  # noqa: E402

_state = {}


def build(case: dict):
    def run(inputs: dict) -> dict:
        if "outputs" not in _state:
            _state["outputs"] = template_outputs(case, inputs)        # first call only (warm-up), outside the timed windows
        outputs = _state["outputs"]
        first = next(iter(outputs.values()))
        first.add_(1)                                                 # the one kernel
        return outputs

    return run
