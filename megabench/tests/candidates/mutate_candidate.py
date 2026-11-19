"""Negative control: oracle outputs, then one byte of the first MLA layer's FP8 latent cache flipped on every rank."""
import torch
from megabench.cases import Case
from megabench.tasks import kimi


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != "execution"})
    name = f"l{min(kimi.mla_layers(contract))}_kv_cache"

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        inputs[name].view(torch.uint8).view(-1)[0] ^= 1
        return out

    return run
