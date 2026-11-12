"""Rank-local negative control (input immutability on one rank): the oracle's output on every rank, then one
input element changed in place on TARGET_RANK only.

Mechanism: identical to negctrl_immutability.py (inputs['l3_kv_cache'][0, 0, 0] += 1.0 after the oracle step), but
only the process whose case['execution']['rank'] == TARGET_RANK mutates. Every rank runs exactly the oracle's
collectives; the mutation is a local in-place add and takes part in no collective, so the ranks stay in step.

Why this is the rank-attributable control: in megabench/harness/distributed.py the per-rank check_unchanged() runs
inside the rank's own try block; its message becomes that rank's local_error and agree() all-gathers the local
errors and raises on every rank with "rank N: ..." for the ranks that had one. So the expected reason on all four
ranks is "AssertionError: rank 2: trial 0: AssertionError: candidate mutated input l3_kv_cache": it names rank 2
and only rank 2, while the status is `incorrect` on every rank (agree() raises an AssertionError everywhere; the
evaluator has no "pass on this rank, fail on that one" state) and the aggregate is `incorrect`.

Record: with GOALBOX_NEGCTRL_RECORD set, one JSON line per call to <path>.rank<N>.jsonl on every rank, with
'corrupted': true only on the target rank.
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

TARGET_RANK = 2
INDEX = (0, 0, 0)
DELTA = 1.0
RECORD_ENV = 'GOALBOX_NEGCTRL_RECORD'


def _rank(case: dict) -> int:
    return int((case.get('execution') or {}).get('rank', 0))


def _record(case: dict, payload: dict) -> None:
    path = os.environ.get(RECORD_ENV)
    if path:
        with open(f'{path}.rank{_rank(case)}.jsonl', 'a', encoding='utf-8') as file:
            file.write(json.dumps(payload) + '\n')


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != 'execution'})
    name = f'l{min(kimi.mla_layers(contract))}_kv_cache'
    rank = _rank(case)

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        payload = {'control': 'ranklocal_immutability', 'rank': rank, 'target_rank': TARGET_RANK, 'tensor': name,
                   'corrupted': rank == TARGET_RANK}
        if rank == TARGET_RANK:
            target = inputs[name]
            before = float(target[INDEX])
            target[INDEX] += DELTA
            after = float(target[INDEX])
            payload.update(index=list(INDEX), delta=DELTA, value_before=before, value_after=after, changed=before != after,
                           tensor_bytes=target.numel() * target.element_size())
        _record(case, payload)
        return out

    return run
