"""Negative control: oracle outputs with one output key missing on every rank."""
from megabench.cases import Case
from megabench.tasks import kimi


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        del out["cache_length"]
        return out

    return run
