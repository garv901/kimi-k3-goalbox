"""Shared helper for the C2 profiling controls (tests only; these import the oracle, so they run with
--no-scan and are never a pass). build() computes one reference step to learn the output structure, so the
timed run() returns correctly structured tensors while launching a known number of device kernels."""
from megabench.cases import Case
from megabench.tasks import kimi


def template_outputs(case: dict, inputs: dict) -> dict:
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})
    expected = kimi.reference(contract, inputs)
    return {name: value.clone() for name, value in expected.items()}
