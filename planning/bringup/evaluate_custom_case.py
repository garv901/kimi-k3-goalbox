"""Milestone 2 (GPU, needs its own GO): run the unmodified evaluator (megabench.harness.runner.evaluate_case, hence
evaluate_distributed, snapshot, reference, compare, timing and launch audit) on a bring-up case that is not in the
case table: the first N real layers at TP4 with the trusted oracle as the candidate. The contract files are not
touched; this wrapper only constructs the Case object (the CLI resolves --case by id from megabench/cases.py, a
contract file, so a new case cannot be added without changing the contract digest) and writes the same record
fields as the CLI (session, contract_sha256) plus the bring-up case definition.

Usage: KIMI_K3_WEIGHTS=<root> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=<repo> python -s planning/bringup/evaluate_custom_case.py \
         --layers 56 --tp 4 --batch 1 --submission reports/gb300/negative-controls/positive_control_oracle.py \
         --session-id <id> --trials 1 --warmup 1 --reps 1 --output <dir>/result.jsonl
Exit 0 on status ok_provisional/non_megakernel/correctness_only (correctness passed), 1 otherwise."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from megabench.harness.runner import _contract_digest, evaluate_case
from tp4_real_load_only import bringup_case  # same directory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, required=True); ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--submission', required=True)
    ap.add_argument('--session-id', required=True); ap.add_argument('--method', default='bringup-oracle')
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=1); ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    case = bringup_case(args.layers, args.tp, args.batch)
    submission = Path(args.submission).resolve()
    try:
        result = evaluate_case(case, submission, device=args.device, trials=args.trials, warmup=args.warmup,
                               reps=args.reps, graph_baseline=False)
    except Exception as exc:
        result = {'case': case.to_dict(), 'submission': str(submission),
                  'status': 'incorrect' if isinstance(exc, AssertionError) else 'error',
                  'reason': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(limit=12)}
    result['session'] = {'id': args.session_id, 'method': args.method, 'case_id': case.id}
    result['contract_sha256'] = _contract_digest()
    result['bringup'] = {'wrapper': 'planning/bringup/evaluate_custom_case.py', 'case_not_in_table': True,
                         'arguments': vars(args)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(result) + '\n')
    print(f"{case.id}: {result['status']} {result.get('reason', '')}")
    return 0 if result['status'] in ('ok_provisional', 'non_megakernel', 'correctness_only') else 1


if __name__ == '__main__':
    sys.exit(main())
