"""Negative control: rank 1 returns logits with a wrong shape; the other ranks return the oracle's outputs."""
from megabench.cases import Case
from megabench.tasks import kimi


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})
    rank = int((case.get("execution") or {}).get("rank", 0))

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        if rank == 1:
            out["logits"] = out["logits"][:, :-1].contiguous()
        return out

    return run
