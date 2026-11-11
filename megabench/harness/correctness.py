"""Validate submission outputs against a task reference."""

from __future__ import annotations

import secrets
import time
from typing import Callable

from ..cases import Case
from ..tasks.workloads import compare_outputs, make_inputs, reference


def snapshot(inputs: dict) -> dict:
    """Detect input mutation without a host copy of large weights.

    Tensors up to 64 MB are copied to the host. Larger tensors get two 64-bit
    checksums on the device (plain and position-weighted sums of 32-bit words).
    """
    import torch

    result = {}
    for name, value in inputs.items():
        value = value.detach()
        if value.numel()*value.element_size() <= 1 << 26 or not value.is_contiguous() or value.numel()*value.element_size() % 4:
            result[name] = value.to("cpu", copy=True)
            continue
        words = value.reshape(-1).view(torch.uint8).view(torch.int32)
        total = torch.zeros(2, dtype=torch.int64, device=value.device)
        step = 1 << 26
        weight = (torch.arange(step, device=value.device, dtype=torch.int64) % 65521) + 1
        for start in range(0, words.numel(), step):
            chunk = words[start:start+step].to(torch.int64)
            total[0] += chunk.sum()
            total[1] += (chunk*weight[:chunk.numel()]).sum()
        result[name] = total.cpu()
    return result


def check_unchanged(inputs: dict, before: dict) -> None:
    import torch

    after = snapshot(inputs)
    for name in inputs:
        if not torch.equal(after[name], before[name]):
            raise AssertionError(f"candidate mutated input {name}")


def compare(case: Case, inputs: dict, expected: dict, actual, device: str, *, serial: bool = False) -> dict:
    details = compare_outputs(case, inputs, expected, actual, serial=serial)
    return _compare(expected, actual, case, device) if details is None else details


def _compare(expected: dict, actual: dict, case: Case, device: str) -> dict:
    import torch

    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AssertionError(f"output keys: expected {sorted(expected)}, got "
                             f"{sorted(actual) if isinstance(actual, dict) else type(actual)}")
    details = {}
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{name}: output must be a Tensor")
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: expected {tuple(want.shape)} {want.dtype}, "
                                 f"got {tuple(got.shape)} {got.dtype}")
        if got.device != want.device:
            raise AssertionError(f"{name}: output is on {got.device}, expected {device}")
        if want.is_floating_point():
            delta = (got.float() - want.float()).abs()
            rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
            close = torch.isclose(got.float(), want.float(),
                                  atol=case.atol, rtol=rtol, equal_nan=False)
            details[name] = {"max_abs_error": float(delta.max().item()),
                             "mismatched_elements": int((~close).sum().item())}
            if not bool(close.all()):
                raise AssertionError(f"{name}: {details[name]}")
        else:
            mismatch = int((got != want).sum().item())
            details[name] = {"mismatched_elements": mismatch}
            if mismatch:
                raise AssertionError(f"{name}: {mismatch} integer mismatches")
    return details


def check_trials(case: Case, candidate: Callable, device: str,
                 trials: int) -> tuple[dict, float | None]:
    """Check fresh inputs, input preservation, and special acceptance scenarios."""
    import torch

    result = {"status": "pass", "trials": []}
    first_call_ms: float | None = None
    trial_seeds = [secrets.randbits(32) for _ in range(trials)]
    for trial_index, seed in enumerate(trial_seeds):
        inputs = make_inputs(case, seed, device)
        scenario = "seeded"
        if case.family in ("spec_target_step", "spec_full_iteration"):
            scenario = ("eagle3_raw", "accept_full", "accept_one")[trial_index % 3]
            if scenario != "eagle3_raw":
                if case.family == "spec_target_step":
                    from ..tasks.eagle3 import set_acceptance_scenario
                else:
                    from ..tasks.speculative import set_acceptance_scenario
                accepted_prefix = (case.params["draft_depth"] if
                                   scenario == "accept_full" else 1)
                set_acceptance_scenario(case, inputs, accepted_prefix)
        # Full MoE weights exceed 60 GB. Keep mutation snapshots on host
        # so correctness does not need two full copies on one GPU.
        originals = snapshot(inputs)
        expected = reference(case, inputs)
        first_start = time.perf_counter() if trial_index == 0 else None
        actual = candidate(inputs)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        if first_start is not None:
            first_call_ms = (time.perf_counter() - first_start) * 1000
        check_unchanged(inputs, originals)
        details = compare(case, inputs, expected, actual, device,
                          serial=case.gpus == 1)
        trial_record = {"seed": seed, "scenario": scenario, "outputs": details}
        if case.family in ("spec_target_step", "spec_full_iteration"):
            trial_record["spec_accounting"] = {
                "proposed_draft_tokens": case.params["draft_depth"],
                "accepted_draft_tokens": int(expected["accepted_count"].item()),
                "committed_tokens": int(expected["committed_count"].item()),
            }
        result["trials"].append(trial_record)
        del inputs, originals, expected, actual
    return result, first_call_ms
