"""Harness check only: the PyTorch oracle itself as the candidate.
Expected result: the correctness band passes, the launch audit fails (many kernels), timing = oracle timing."""
from megabench.cases import Case
from megabench.tasks import kimi


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})

    def run(inputs: dict) -> dict:
        return kimi.reference(contract, inputs)

    return run
