#!/usr/bin/env python
"""CPU-only review of the negative controls through the evaluator's own harness functions.

Runs on a tiny Kimi geometry (the unit tests' development() shape, with 128 experts and top-8 so that a replaced
expert behaves like one of 896) and on the CPU, so no GPU is touched. Nothing in the evaluator is modified; the
controls are loaded with the evaluator's own _load_submission() and judged by its own check_trials(), snapshot(),
check_unchanged() and compare(). For each submission file it reports:
  1. sha256 and that build(case) returns a callable (the normal candidate interface), with and without the
     distributed 'execution' key, and with the real dev-case contract;
  2. the verdict of check_trials() (the single-GPU evaluator path), which must be a pass for the positive control
     and an AssertionError of exactly the expected class for each negative control, with no other failure class
     in the message (for the routing control, its documented near-tie abort, a RuntimeError, is also accepted,
     since check_trials draws random seeds; the fixed seeds in step 3 must still fail on expert_ids);
  3. the same per-trial sequence (make_inputs, snapshot, reference, candidate, check_unchanged, compare) on fixed
     seeds, to show the failure is deterministic across seeds and to print the grader's numbers;
  4. side-effect checks: the logits and routing controls leave the inputs unchanged (their failures come from
     compare, after check_unchanged passed); the immutability control's outputs are bit-identical to the oracle's.
Usage: python cpu_review.py [--seeds N] [--out DIR]; exits 0 only if every expectation holds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from megabench.cases import select_cases  # noqa: E402
from megabench.harness.correctness import check_trials, check_unchanged, compare, snapshot  # noqa: E402
from megabench.harness.runner import _load_submission  # noqa: E402
from megabench.tasks import kimi  # noqa: E402

DEV_CASE = 'kimi-k3-dev-l4-tp1-b8-s8192'
CONTROLS = {
    'positive_control_oracle.py': None,
    'negctrl_logits.py': 'logits: ',
    'negctrl_routing.py': 'expert_ids: ',
    'negctrl_immutability.py': 'candidate mutated input l3_kv_cache',
}
CLASSES = ('logits', 'kv_write', 'pe_write', 'recurrent_state', 'conv_state', 'next_token', 'expert_ids',
           'cache_length', 'candidate mutated input')


def tiny_case(batch: int = 3):
    case = select_cases('all', [DEV_CASE])[0]
    return replace(case, params=case.params | {
        'batch': batch, 'layers': 5, 'hidden': 64, 'latent_hidden': 64, 'intermediate': 64,
        'dense_intermediate': 128, 'q_heads': 4, 'head_dim': 16, 'q_lora_rank': 32,
        'kv_lora_rank': 32, 'qk_nope_dim': 8, 'qk_rope_dim': 8, 'v_head_dim': 16,
        'vocab': 128, 'experts': 128, 'topk': 8, 'context': 4, 'attn_res_block_size': 2})


def classes_in(message: str) -> list[str]:
    return [c for c in CLASSES if c in message]


def one_trial(case, candidate, seed: int) -> tuple[str | None, dict | None, dict]:
    """The evaluator's per-trial sequence on one fixed seed. Returns (failure message, details, extras)."""
    inputs = kimi.make_inputs(case, seed, 'cpu', rank=0)
    originals = snapshot(inputs)
    expected = kimi.reference(case, inputs)
    actual = candidate(inputs)
    extras = {'outputs_bit_identical_to_oracle': all(torch.equal(expected[k], actual[k]) for k in expected)
              if isinstance(actual, dict) and set(actual) == set(expected) else False}
    try:
        check_unchanged(inputs, originals)
    except AssertionError as exc:
        return str(exc), None, extras
    try:
        return None, compare(case, inputs, expected, actual, 'cpu', serial=True), extras
    except AssertionError as exc:
        return str(exc), None, extras


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(0)
    ok = True
    report = {'host': platform.node(), 'python': sys.version.split()[0], 'torch': torch.__version__,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'threads': torch.get_num_threads(),
              'cuda_initialized': torch.cuda.is_initialized(), 'controls': {}}
    case = tiny_case()
    real = select_cases('all', [DEV_CASE])[0]
    print(f'tiny geometry: layers {case.params["layers"]}, experts {case.params["experts"]}, topk {case.params["topk"]}, '
          f'batch {case.params["batch"]}; torch {torch.__version__} on {platform.node()}, CUDA_VISIBLE_DEVICES='
          f'{os.environ.get("CUDA_VISIBLE_DEVICES")!r}')
    for name, expect in CONTROLS.items():
        path = HERE / name
        entry = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'expected_prefix': expect}
        print(f'\n=== {name}  sha256 {entry["sha256"]}')
        # 1. interface
        candidate = _load_submission(path, case)
        _load_submission(path, real)                       # real dev-case contract builds too (no run)
        _load_submission(path, case, execution={'rank': 0, 'world_size': 1, 'tp_rank': 0, 'ep_rank': 0,
                                                'backend': 'none', 'device': 'cpu', 'tp_group': None, 'ep_group': None})
        entry['build_returns_callable'] = callable(candidate)
        print(f'  build(case) -> callable: {entry["build_returns_callable"]} (tiny, real dev contract, with execution key)')
        # 2. the evaluator's check_trials
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                result, _ = check_trials(case, candidate, 'cpu', trials=2)
            message = None
            entry['check_trials'] = {'status': 'pass', 'trials': len(result['trials'])}
        except AssertionError as exc:
            message = str(exc)
            entry['check_trials'] = {'status': 'incorrect (AssertionError)', 'message': message}
        except Exception as exc:  # noqa: BLE001
            message = f'{type(exc).__name__}: {exc}'
            entry['check_trials'] = {'status': 'error', 'message': message}
        entry['check_trials']['seconds'] = round(time.perf_counter() - start, 2)
        found = classes_in(message or '')
        if expect is None:
            good = message is None
        else:
            good = message is not None and message.startswith(expect) and found == classes_in(expect)
        # check_trials draws random seeds; the routing control aborts on purpose (RuntimeError, status `error`)
        # when the replacement expert happens to be a near-tie. That documented outcome is accepted here, and the
        # fixed-seed loop below must still show the expert_ids failure; the record file holds the margin.
        inconclusive = (name == 'negctrl_routing.py' and message is not None
                        and message.startswith('RuntimeError: negative control inconclusive'))
        entry['check_trials']['documented_near_tie_abort'] = inconclusive
        good = good or inconclusive
        entry['check_trials']['as_expected'] = good
        ok &= good
        print(f'  check_trials: {entry["check_trials"]["status"]}; failure classes {found}; as expected: {good}'
              + (' (documented near-tie abort on a random seed)' if inconclusive else ''))
        if message:
            print(f'    message: {message[:400]}')
        # 3 + 4. fixed seeds
        entry['seeds'] = []
        with torch.inference_mode():
            for seed in range(1, args.seeds + 1):
                message, details, extras = one_trial(case, candidate, 100 + seed)
                found = classes_in(message or '')
                row = {'seed': 100 + seed, 'failure_classes': found} | extras
                if expect is None:
                    good = message is None
                    row['worst_ratios'] = {k: round(v['worst_ratio'], 3) for k, v in details.items() if 'worst_ratio' in v}
                else:
                    good = message is not None and message.startswith(expect) and found == classes_in(expect)
                    row['message'] = message
                if name == 'negctrl_immutability.py':
                    good &= extras['outputs_bit_identical_to_oracle']
                row['as_expected'] = good
                ok &= good
                entry['seeds'].append(row)
                print(f'  seed {100 + seed}: classes {found}; oracle-identical outputs {extras["outputs_bit_identical_to_oracle"]}; '
                      f'as expected {good}' + (f'; {row["worst_ratios"]}' if 'worst_ratios' in row else ''))
                if message and name == 'negctrl_logits.py':
                    print(f'    {message[:300]}')
                if message and name == 'negctrl_routing.py':
                    print(f'    {message[:300]}')
        report['controls'][name] = entry
    report['all_expectations_met'] = bool(ok)
    print(f'\nALL EXPECTATIONS MET: {bool(ok)}')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'cpu-review.json').write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
