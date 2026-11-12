"""Negative control 2 (expert routing): the oracle run on expert sets with one deliberately wrong expert.

Mechanism, identical on every rank (the oracle's expert sets are identical on all ranks by construction):
  1. oracle = kimi.reference(case, inputs)                 the normal oracle step, gives expert_ids [L, B, topk]
  2. layer = params['first_dense'] (the first MoE layer, 1), ROW = 0, SLOT = 0
     original    = expert_ids[layer, ROW, SLOT]            the lowest id of that row's top-k set (rows are sorted)
     replacement = the smallest expert id in range(experts) that is not in that row's set
     routes[layer, ROW, SLOT] = replacement                every other row, layer and slot unchanged
  3. forced = kimi.reference(case, inputs, routes=routes)  the same BF16 oracle, forced onto the corrupted routes
  4. margin = forced['route_shortfall'][layer, ROW]        how far the replacement's biased router score is below
                                                           the k-th best (the grader's own measure, in score units)
     if margin < MIN_SHORTFALL (0.05): raise RuntimeError  the replacement happened to be a near-tie, the control
                                                           is inconclusive for this seed (expected status `error`
                                                           with that message; never seen as likely: about 1 percent)
  5. return forced without 'route_shortfall'               shapes, dtypes, id range and distinctness preserved

Why step 3: the grader compares the continuous outputs against an FP32 reference run on the CANDIDATE's routes,
so a candidate that returns the oracle's logits with only expert_ids edited would fail the logits band too. Running
the oracle on the corrupted routes makes logits, next_token, cache writes and states consistent with those routes,
so the only check that fails is the routing rule: the weakest chosen expert may be at most
2 x (oracle's worst shortfall) + 0.001 below the k-th best; on the GB300 runs that allowance was 0.012 to 0.019.
A random non-top-k expert among 896 sits far below the top 16 (margin typically 0.3 to 0.5 in sigmoid units).

Expected evaluator verdict: AssertionError "expert_ids: {'candidate_max_shortfall': <margin>,
'oracle_max_shortfall': ..., 'allowed': ..., 'sets_differing_from_oracle': 1}", status `incorrect`, on every
rank (the route checksum and the forced references are collective; all ranks hold the same corrupted routes).

Record: with GOALBOX_NEGCTRL_RECORD set, one JSON line per call to <path>.rank<N>.jsonl holds layer, row, slot,
the original and replacement ids, the replacement's margin, the largest shortfall in the other rows (expected 0)
and in the later layers of the same row (may be nonzero: a different expert in layer 1 changes that row's hidden
state downstream; the grader's FP32 reference is forced onto the same routes, so the band still passes).
"""

from __future__ import annotations

import json
import os

from megabench.cases import Case
from megabench.tasks import kimi

ROW = 0
SLOT = 0
MIN_SHORTFALL = 0.05
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
    layer = contract.params['first_dense']
    experts = contract.params['experts']

    def run(inputs: dict) -> dict:
        oracle = kimi.reference(contract, inputs)
        routes = oracle['expert_ids'].clone()
        present = set(routes[layer, ROW].tolist())
        original = int(routes[layer, ROW, SLOT])
        replacement = next(e for e in range(experts) if e not in present)
        routes[layer, ROW, SLOT] = replacement
        forced = kimi.reference(contract, inputs, routes=routes)
        shortfall = forced.pop('route_shortfall')
        margin = float(shortfall[layer, ROW])
        other_rows = shortfall[:, [r for r in range(shortfall.shape[1]) if r != ROW]]
        later_layers = shortfall[layer + 1:, ROW]
        _record(case, {'control': 'routing', 'layer': layer, 'row': ROW, 'slot': SLOT,
                       'original_expert': original, 'replacement_expert': replacement,
                       'replacement_shortfall_below_kth_best': margin, 'min_shortfall_required': MIN_SHORTFALL,
                       # expected 0: the corruption is confined to row ROW
                       'max_shortfall_other_rows': float(other_rows.max()) if other_rows.numel() else 0.0,
                       # may be > 0: the replaced expert changes row ROW's hidden state, so the oracle's sets in
                       # later layers need not be top-k any more on the corrupted trajectory (candidate and FP32
                       # reference are both forced onto the same routes, so this does not add a band failure)
                       'max_shortfall_same_row_later_layers': float(later_layers.max()) if later_layers.numel() else 0.0})
        if margin < MIN_SHORTFALL:
            raise RuntimeError(f'negative control inconclusive: replacement expert {replacement} is within '
                               f'{margin:.4f} of the k-th best score (near-tie); rerun with another seed')
        return forced

    return run
