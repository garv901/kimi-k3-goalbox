"""Negative control 1 (logits): the oracle's output with one row of the logits deliberately perturbed.

Exact perturbation, applied on every rank to its own vocabulary shard of the logits:
    row   = logits[ROW]                      ROW = 0 (present at every batch size)
    delta = SCALE * sqrt(mean(row ** 2))     SCALE = 1.0: the root mean square of the unperturbed row
    row[::STRIDE] += delta                   STRIDE = 4: shard indices 0, 4, 8, ... (one quarter of the row)
Nothing is random. delta is finite, so no NaN or inf can appear. Shapes, dtypes and devices are unchanged.

Expected effect on the oracle band (megabench/tasks/kimi.py, compare): the logits are graded per batch row by
relative L2 distance to the FP32 reference, limit = 2 x (BF16 oracle distance) + 0.002. Adding delta to one
quarter of the row adds a relative L2 of SCALE * sqrt(1/STRIDE) = 0.5 against a limit of about 0.09 on the
GB300 runs (the oracle's own ratio was 0.49), so row 0 fails by about 5x while every other row stays in band.
All other outputs (next_token, expert_ids, cache writes, KDA states, cache_length) are the oracle's and pass;
next_token is graded on the FP32 logits of the candidate's routes, so this perturbation does not touch it.

Expected evaluator verdict: AssertionError "logits: {'candidate_rel_l2_max': ..., 'oracle_rel_l2_max': ...,
'worst_ratio': ...}", status `incorrect`. Single GPU: raised by compare() inside check_trials. Distributed:
the band sums are all-reduced, so every rank raises the same failure and agree() lists it for every rank.

Record: if GOALBOX_NEGCTRL_RECORD is set to a path, one JSON line per call is appended to <path>.rank<N>.jsonl
with the row, the number of perturbed indices, delta and the relative change, because the evaluator does not
keep the candidate's stdout or stderr. Without the variable nothing is written.
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

ROW = 0
STRIDE = 4
SCALE = 1.0
RECORD_ENV = 'GOALBOX_NEGCTRL_RECORD'


def _record(case: dict, payload: dict) -> None:
    path = os.environ.get(RECORD_ENV)
    if not path:
        return
    rank = (case.get('execution') or {}).get('rank', 0)
    with open(f'{path}.rank{rank}.jsonl', 'a', encoding='utf-8') as file:
        file.write(json.dumps(payload) + '\n')


def build(case: dict):
    contract = Case(**{k: v for k, v in case.items() if k != 'execution'})

    def run(inputs: dict) -> dict:
        out = kimi.reference(contract, inputs)
        logits = out['logits'].clone()
        before = logits[ROW].clone()
        row = logits[ROW]                       # view into the clone
        delta = SCALE * row.square().mean().sqrt()
        row[::STRIDE] += delta
        out['logits'] = logits
        if os.environ.get(RECORD_ENV):
            _record(case, {'control': 'logits', 'row': ROW, 'stride': STRIDE, 'scale': SCALE,
                           'shard_vocab': int(row.numel()), 'perturbed_indices': len(range(0, row.numel(), STRIDE)),
                           'delta': float(delta),
                           'rel_l2_change_of_row_on_this_shard': float((row - before).norm() / before.norm())})
        return out

    return run
