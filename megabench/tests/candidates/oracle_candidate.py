"""Positive control: the oracle as the candidate (never scanned; tests only)."""
from megabench.cases import Case
from megabench.tasks import kimi


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})

    def run(inputs: dict) -> dict:
        return kimi.reference(contract, inputs)

    return run
