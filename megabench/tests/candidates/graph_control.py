"""C2 profiling control: replays a captured CUDA graph per run(). Expected: the source scan fails on the
cuda_graph rule when scanning is on; with --no-scan the audit must fail outright on the graph launch (D-L4)
and the trace must show the graph launch and its node events. Falls back to a plain kernel on the CPU."""
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))      # the snapshot holds the sibling _control_common.py
from _control_common import template_outputs  # noqa: E402

_state = {}


def build(case: dict):
    def run(inputs: dict) -> dict:
        if "outputs" not in _state:
            _state["outputs"] = template_outputs(case, inputs)
            first = next(iter(_state["outputs"].values()))
            if first.is_cuda:
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    first.add_(1)
                    torch.cuda.current_stream().synchronize()
                    with torch.cuda.graph(graph, stream=stream):
                        first.add_(1)
                _state["graph"] = graph
        first = next(iter(_state["outputs"].values()))
        if "graph" in _state:
            _state["graph"].replay()                                  # one cudaGraphLaunch, one node
        else:
            first.add_(1)
        return _state["outputs"]

    return run
