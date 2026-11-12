"""Negative control 3 (input immutability): the oracle's output, then one input element changed in place.

Mechanism: after the oracle step has produced its outputs, the candidate adds DELTA = 1.0 to one element of one
input tensor: inputs[NAME][0, 0, 0], where NAME is the MLA latent cache of the first MLA layer
(f'l{layer}_kv_cache' for the lowest layer with (layer+1) % 4 == 0: `l3_kv_cache` in the 4- and 8-layer cases).
The outputs are exact oracle outputs; the mutation happens after they were computed, so nothing else changes.
The element is BF16 with magnitude below 8 (standard normal data), so adding 1.0 always changes its bits.
No shape, dtype or device changes; no collective is added or removed, so the four ranks stay in step.

Why this tensor: the evaluator's snapshot() (megabench/harness/correctness.py) copies tensors of at most 64 MB to
the host and compares them exactly, and checksums larger tensors on the device. `l3_kv_cache` is
[B, 8192, 512] BF16: 8 MB at batch 1, 64 MB at batch 8 (exactly the host-copy limit) and 512 MB at batch 64
(checksum path), so the TP4 series would exercise both detection paths. The same mutation is made on every rank
(the cache is replicated, not sharded), so every rank's own check_unchanged() must fail.

Expected evaluator verdict: AssertionError "candidate mutated input l3_kv_cache" from check_unchanged(), which
runs BEFORE compare(). The harness maps AssertionError to status `incorrect` (not `error`): single GPU, the
exception propagates out of check_trials; distributed, each rank records "trial 0: AssertionError: candidate
mutated input l3_kv_cache" as its local error and agree() raises an AssertionError listing every rank.

Record: with GOALBOX_NEGCTRL_RECORD set, one JSON line per call to <path>.rank<N>.jsonl holds the tensor name,
index, the value before and after, and the tensor's byte size (which snapshot path it took).
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

INDEX = (0, 0, 0)
DELTA = 1.0
RECORD_ENV = 'GOALBOX_NEGCTRL_RECORD'
HOST_COPY_LIMIT = 1 << 26


def _record(case: dict, payload: dict) -> None:
    path = os.environ.get(RECORD_ENV)
    if not path:
        return
    rank = (case.get('execution') or {}).get('rank', 0)
    with open(f'{path}.rank{rank}.jsonl', 'a', encoding='utf-8') as file:
        file.write(json.dumps(payload) + '\n')


def target_name(contract: Case) -> str:
    layer = min(kimi.mla_layers(contract))
    return f'l{layer}_kv_cache'


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != 'execution'})
    name = target_name(contract)

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        target = inputs[name]
        before = float(target[INDEX])
        target[INDEX] += DELTA
        after = float(target[INDEX])
        size = target.numel() * target.element_size()
        _record(case, {'control': 'immutability', 'tensor': name, 'index': list(INDEX), 'delta': DELTA,
                       'value_before': before, 'value_after': after, 'changed': before != after,
                       'tensor_bytes': size,
                       'snapshot_path': 'host copy, exact compare' if size <= HOST_COPY_LIMIT else 'device checksums'})
        return out

    return run
