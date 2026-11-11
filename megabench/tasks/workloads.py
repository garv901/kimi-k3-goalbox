"""Trusted input generator and PyTorch oracle for ready whole-model cases."""

from __future__ import annotations

import importlib

import torch

from ..cases import Case


Inputs = dict[str, torch.Tensor]
Outputs = dict[str, torch.Tensor]

TASK_MODULES = {"kimi_k3_step": "kimi"}


def make_inputs(case: Case, seed: int, device: str = "cpu") -> Inputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family not in TASK_MODULES:
        raise NotImplementedError(f"no input generator for {case.id}")
    module = importlib.import_module(f"{__package__}.{TASK_MODULES[case.family]}")
    return module.make_inputs(case, seed, device)


def reference(case: Case, t: Inputs) -> Outputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family not in TASK_MODULES:
        raise NotImplementedError(f"no PyTorch oracle for {case.id}")
    module = importlib.import_module(f"{__package__}.{TASK_MODULES[case.family]}")
    return module.reference(case, t)


def compare_outputs(case: Case, t: Inputs, expected: Outputs, actual, *, serial: bool = False):
    """Task-specific grading (for example the Kimi oracle band), or None for exact-tolerance grading."""
    module = importlib.import_module(f"{__package__}.{TASK_MODULES[case.family]}")
    compare = getattr(module, "compare", None)
    return None if compare is None else compare(case, t, expected, actual, serial=serial)
