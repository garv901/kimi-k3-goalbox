"""Negative controls of the TRUSTED evaluator on CPU (gloo, 4 ranks): protocol dsv41-tp4-rack10-v1.

The evaluator (`megabench.harness.distributed.evaluate_distributed`) runs the development-size
DSv4.1 DSpark case at TP4 (4 processes, gloo) exactly as on the GPUs minus the launch audit and
timing: fresh inputs per trial, reference + fp32 oracle, `regrade`, per-rank gate, `agree()` across
ranks, input-mutation check. A wrapper submission around the trusted reference is correct; the
mutations below corrupt ONE rank (or one output) and must each turn the status into `incorrect`
with the offending rank named in the reason (a one-rank build failure too: agree() raises).

Needs the trusted benchmark importable (PYTHONPATH=/srv/ss/trusted/vibe-megakernel or
MEGABENCH_ROOT) and torch; skipped otherwise. ~1-3 min per control on a few CPU cores.

  PYTHONPATH=/srv/ss/trusted/vibe-megakernel python3.12 -m pytest -q scaffold_search/tests/test_evaluator_negative.py
"""
import os, sys, tempfile
from pathlib import Path

import pytest

if os.environ.get('MEGABENCH_ROOT'):
    sys.path.insert(0, os.environ['MEGABENCH_ROOT'])
try:
    import torch  # noqa: F401
    from dataclasses import replace
    from megabench.harness.distributed import evaluate_distributed
    from megabench.tests.test_dsv41 import TP4_DEV, development
except Exception as e:  # pragma: no cover
    pytest.skip(f'trusted benchmark not importable here: {e!r}', allow_module_level=True)

WORLD = 4
BASE = '''
import torch
from megabench.cases import Case
from megabench.workloads import reference

def build(contract):
    execution = contract.pop("execution")
    rank = execution["rank"]
    case = Case(**contract)
    {BUILD}
    def run(inputs):
        out = dict(reference(case, inputs))
        {MUTATION}
        return out
    return run
'''

CONTROLS = {
    # name: (build-time code, run-time mutation, expected status, substrings of the reason)
    'baseline': ('', '', 'correctness_only', ()),
    'rank3_logits_noise': ('', 'if rank == 3: out["logits"] = out["logits"] + torch.randn_like(out["logits"]) * out["logits"].abs().mean()',
                           'incorrect', ('rank 3', 'logits')),
    'rank3_wrong_pick': ('', 'if rank == 3:\n            vt = out["verify_tokens"].clone(); vt.view(-1)[0] = (vt.view(-1)[0] + 1) % case.params["vocab"]; out["verify_tokens"] = vt',
                         'incorrect', ('rank 3',)),
    'rank3_exact_output': ('', 'if rank == 3: out["accepted_count"] = out["accepted_count"] + 1',
                           'incorrect', ('rank 3', 'accepted_count')),
    'rank1_nan': ('', 'if rank == 1:\n            l = out["logits"].clone(); l.view(-1)[0] = float("nan"); out["logits"] = l',
                  'incorrect', ('rank 1',)),
    'rank2_wrong_shape': ('', 'if rank == 2: out["logits"] = out["logits"][..., :-1]', 'incorrect', ('rank 2',)),
    'rank0_missing_output': ('', 'if rank == 0: del out["committed_tokens"]', 'incorrect', ('rank 0',)),
    'rank2_mutates_input': ('', 'if rank == 2:\n            t = next(v for v in inputs.values() if isinstance(v, torch.Tensor) and not v.is_floating_point()); t.add_(1)',
                            'incorrect', ('rank 2', 'mutated input')),
    # a build failure on one rank is reported through agree() (AssertionError), so the evaluator
    # classifies it as `incorrect` with the rank and 'build failed' in the reason
    'rank2_build_fails': ('if rank == 2: raise RuntimeError("deliberate build failure")', '',
                          'incorrect', ('rank 2', 'build failed')),
    'all_ranks_wrong': ('', 'out["logits"] = out["logits"] * 0', 'incorrect', ('rank 0',)),
}


def tp4_case():
    return replace(development(**TP4_DEV), gpus=WORLD, tp=WORLD, ready=True)


def run_control(name):
    build, mutation, want, needles = CONTROLS[name]
    case = tp4_case()
    with tempfile.TemporaryDirectory(prefix=f'ss-evneg-{name}-') as d:
        sub = Path(d) / 'submission.py'
        sub.write_text(BASE.replace('{BUILD}', build).replace('{MUTATION}', mutation))
        rep = evaluate_distributed(case, sub, device='cpu', trials=1, warmup=0, reps=1)
    assert rep['status'] == want, (name, rep['status'], rep.get('reason'))
    if want == 'correctness_only':
        assert len(rep['ranks']) == WORLD and all(r['correctness']['status'] == 'pass' for r in rep['ranks'])
        assert all(all(o.get('pass') for o in t['outputs'].values()) for r in rep['ranks'] for t in r['correctness']['trials'])
    else:
        reason = rep['reason']
        for n in needles:
            assert n in reason, (name, n, reason)
        # every rank reports the failure (agree() broadcasts rank-local errors), so no rank scores
        assert all(r['status'] != 'pass' for r in rep['ranks']), [r['status'] for r in rep['ranks']]
    return rep


@pytest.mark.parametrize('name', list(CONTROLS))
def test_control(name):
    run_control(name)


if __name__ == '__main__':
    names = sys.argv[1:] or list(CONTROLS)
    fails = 0
    for n in names:
        try:
            r = run_control(n); print('PASS', n, r['status'], (r.get('reason') or '')[:160])
        except Exception as e:
            fails += 1; print('FAIL', n, repr(e)[:400])
    sys.exit(1 if fails else 0)
