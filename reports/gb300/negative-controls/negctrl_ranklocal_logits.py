"""Rank-local negative control (output-only logits corruption on one rank): the oracle's output on every rank; on
TARGET_RANK only, row 0 of that rank's vocabulary shard of `logits` is perturbed as in negctrl_logits.py
(every STRIDE-th index gets + SCALE * RMS(row)). Only the returned tensor changes; no collective is added.

What the evaluator does with it (megabench/tasks/kimi.py::compare): the per-row relative L2 of the logits is
computed from sums that are all-reduced over the TP group, so the band check sees one global number per row.
With one of four shards perturbed by a relative 0.5 on that shard, the row's global relative L2 is about 0.25
(sqrt(1/16)), far above the band limit (about 0.07 in Steps 3 to 6). Every rank therefore fails `logits:` with
the same numbers, after all the grading collectives (compare() collects failures and raises at the end), so there
is no deadlock. Like the rank-local routing control, the evaluator cannot attribute the failure to rank 2: the
reason is identical on all ranks ("rank 0: trial 0: AssertionError: logits: {...}; rank 1: ...; ..."). It does
show that a corruption of one shard is caught by the global grading. Aggregate `incorrect`.

Record: one JSON line per call per rank; the target rank's line carries the shard size, perturbed indices, delta
and the row's relative L2 change on that shard.
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

TARGET_RANK = 2
ROW = 0
STRIDE = 4
SCALE = 1.0
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
    rank = _rank(case)

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        payload = {'control': 'ranklocal_logits', 'rank': rank, 'target_rank': TARGET_RANK, 'corrupted': rank == TARGET_RANK}
        if rank == TARGET_RANK:
            logits = out['logits'].clone()
            before = logits[ROW].clone()
            row = logits[ROW]
            delta = SCALE * row.square().mean().sqrt()
            row[::STRIDE] += delta
            out['logits'] = logits
            payload.update(row=ROW, stride=STRIDE, scale=SCALE, shard_vocab=int(logits.shape[-1]),
                           perturbed_indices=int(len(range(0, logits.shape[-1], STRIDE))), delta=float(delta),
                           rel_l2_change_of_row_on_this_shard=float((row - before).norm() / before.norm()))
        _record(case, payload)
        return out

    return run
