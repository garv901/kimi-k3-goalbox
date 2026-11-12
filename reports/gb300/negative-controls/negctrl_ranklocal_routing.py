"""Rank-local negative control (expert routing on one rank): the oracle's output on every rank; on TARGET_RANK only,
one expert id in `expert_ids` is replaced by an id outside that row's set. No forced rerun on any rank.

Why there is no forced rerun: negctrl_routing.py reruns the oracle forced onto the corrupted routes so that only the
routing rule fails. That rerun is a distributed computation (TP all-reduces in every layer). Doing it on one rank
only would give that rank more collectives than the others, which is a collective mismatch (hang until the
process-group timeout, then `error`). A rank-local control must therefore leave the collective sequence exactly
the oracle's and change only the returned tensor.

What the evaluator does with it (megabench/tasks/kimi.py::compare, read before writing this): before any forced
reference, every rank computes a checksum of its candidate routes and all-reduces [valid?, checksum, -checksum,
differs-from-oracle?] with MAX. If the checksums differ between ranks, every rank raises the same
AssertionError('expert_ids: ids out of range, repeated in a row, wrong in a dense layer, or different between
ranks') at the same point, so there is no deadlock. Consequences, by design of the grader: (1) the near-top-k
shortfall rule is never reached for a rank-local corruption, because routes must agree across ranks before the
forced references can run; (2) the reason is identical on all ranks and does not say which rank diverged
(agree() then prefixes every rank's own identical error with its own rank number). Expected on all four ranks:
status `incorrect`, reason "AssertionError: rank 0: trial 0: AssertionError: expert_ids: ... different between
ranks; rank 1: ...; rank 2: ...; rank 3: ...". Aggregate `incorrect`. Ranks 0, 1 and 3 return exactly the
oracle's outputs.

Record: one JSON line per call per rank; the target rank's line carries the layer, row, slot, original and
replacement ids.
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

TARGET_RANK = 2
ROW = 0
SLOT = 0
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
    layer = contract.params['first_dense']   # first MoE layer
    experts = contract.params['experts']
    rank = _rank(case)

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        payload = {'control': 'ranklocal_routing', 'rank': rank, 'target_rank': TARGET_RANK, 'corrupted': rank == TARGET_RANK}
        if rank == TARGET_RANK:
            routes = out['expert_ids'].clone()
            present = set(routes[layer, ROW].tolist())
            original = int(routes[layer, ROW, SLOT])
            replacement = next(e for e in range(experts) if e not in present)
            routes[layer, ROW, SLOT] = replacement
            out['expert_ids'] = routes
            payload.update(layer=layer, row=ROW, slot=SLOT, original_expert=original, replacement_expert=replacement,
                           forced_rerun=False)
        _record(case, payload)
        return out

    return run
