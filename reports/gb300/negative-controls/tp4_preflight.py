"""CPU-only preflight for the TP4 negative controls (no GPU visible; torch.distributed on gloo).

Part A, snapshot path: builds `l3_kv_cache` tensors of the exact shape and dtype the B1, B8 and B64 cases use
([B, 8192, 512] BF16) on the CPU, runs the evaluator's own snapshot() and check_unchanged() (unmodified,
imported from megabench.harness.correctness), shows which path each size takes (host copy or the two int64
checksums), and shows that the control's single-element mutation is detected on the checksum path, including
an element in the second 256 MB chunk and a mutate-then-revert sanity check.

Part B, distributed flow: runs the evaluator's own evaluate_distributed() (4 processes, gloo, CPU) on a tiny
TP4 geometry for each submission, in a child process with a wall-clock limit so that a deadlock shows up as a
timeout instead of a hang, and prints the aggregate and per-rank status and reason plus the controls' records.

Usage: tp4_preflight.py <outdir> [--subs name,...]      (writes <outdir>/preflight.json and prints a summary)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
TP4_CASE = 'kimi-k3-syn-l8-tp4-b1-s8192'
# intermediate 128 (not the TP1 review's 64): MXFP4 down-projection shards (intermediate/tp) must be multiples of 32
TINY = {'batch': 3, 'layers': 5, 'hidden': 64, 'latent_hidden': 64, 'intermediate': 128, 'dense_intermediate': 128,
        'q_heads': 4, 'head_dim': 16, 'q_lora_rank': 32, 'kv_lora_rank': 32, 'qk_nope_dim': 8, 'qk_rope_dim': 8,
        'v_head_dim': 16, 'vocab': 128, 'experts': 128, 'topk': 8, 'context': 4, 'attn_res_block_size': 2}
SUBS = ['positive_control_oracle.py', 'negctrl_immutability.py', 'negctrl_logits.py', 'negctrl_routing.py',
        'negctrl_ranklocal_immutability.py', 'negctrl_ranklocal_routing.py', 'negctrl_ranklocal_logits.py']
CHILD_TIMEOUT_S = 900


def part_a(out: dict) -> None:
    import torch
    from megabench.harness.correctness import check_unchanged, snapshot
    rows = []
    for batch in (1, 8, 64):
        shape = (batch, 8192, 512)
        g = torch.Generator(device='cpu').manual_seed(2026)
        t = torch.randn(shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
        nbytes = t.numel() * t.element_size()
        t0 = time.perf_counter(); before = snapshot({'l3_kv_cache': t}); snap_s = time.perf_counter() - t0
        snap = before['l3_kv_cache']
        path = 'host copy' if snap.dtype == torch.bfloat16 and snap.shape == t.shape else (
            'device checksums (two int64)' if snap.dtype == torch.int64 and tuple(snap.shape) == (2,) else f'unexpected {snap.dtype} {tuple(snap.shape)}')
        again = snapshot({'l3_kv_cache': t})['l3_kv_cache']
        row = {'batch': batch, 'shape': list(shape), 'bytes': nbytes, 'over_host_limit': nbytes > (1 << 26),
               'snapshot_path': path, 'snapshot_deterministic': bool(torch.equal(snap, again)), 'snapshot_s_cpu': round(snap_s, 3)}
        if path.startswith('device checksums'):
            row['checksums_before'] = [int(snap[0]), int(snap[1])]
        # the control's mutation at (0, 0, 0)
        v0 = float(t[0, 0, 0]); t[0, 0, 0] += 1.0; v1 = float(t[0, 0, 0])
        try:
            check_unchanged({'l3_kv_cache': t}, before); msg = None
        except AssertionError as exc:
            msg = str(exc)
        row.update(value_before=v0, value_after=v1, detected_at_000=msg)
        if path.startswith('device checksums'):
            after = snapshot({'l3_kv_cache': t})['l3_kv_cache']
            row['checksums_after'] = [int(after[0]), int(after[1])]
            w = t.reshape(-1).view(torch.uint8).view(torch.int32)
            row['word0_after'] = int(w[0]); row['plain_sum_delta'] = int(after[0] - snap[0]); row['weighted_sum_delta'] = int(after[1] - snap[1])
        t[0, 0, 0] -= 1.0  # revert
        try:
            check_unchanged({'l3_kv_cache': t}, before); row['revert_passes'] = True
        except AssertionError as exc:
            row['revert_passes'] = False; row['revert_error'] = str(exc)
        if batch == 64:
            # an element in the second 1<<26-word chunk (element index > 128M) and the last element
            for name, idx in (('second_chunk', (40, 0, 0)), ('last_element', (63, 8191, 511))):
                t[idx] += 1.0
                try:
                    check_unchanged({'l3_kv_cache': t}, before); row[f'detected_{name}'] = None
                except AssertionError as exc:
                    row[f'detected_{name}'] = str(exc)
                t[idx] -= 1.0
            row['element_index_second_chunk'] = 40 * 8192 * 512
        rows.append(row)
        print(f'[A] B{batch}: {nbytes/2**20:.0f} MiB -> {path}; mutation at (0,0,0) {v0:+.4f} -> {v1:+.4f}: '
              f'{"DETECTED: " + msg if msg else "NOT DETECTED"}; revert passes {row["revert_passes"]}')
        del t, before
    out['part_a'] = rows


def tiny_case():
    from megabench.cases import select_cases
    case = select_cases('all', [TP4_CASE])[0]
    return replace(case, params=case.params | TINY)


def child(case_json: str, submission: str, result_path: str, record_path: str) -> None:
    import torch
    from megabench.cases import Case
    from megabench.harness.distributed import evaluate_distributed
    torch.set_num_threads(4)
    os.environ['GOALBOX_NEGCTRL_RECORD'] = record_path
    case = Case(**json.loads(case_json))
    t0 = time.perf_counter()
    result = evaluate_distributed(case, Path(submission), device='cpu', trials=1, warmup=0, reps=1)
    result['wall_s'] = round(time.perf_counter() - t0, 1)
    Path(result_path).write_text(json.dumps(result, default=str))


def part_b(outdir: Path, subs: list[str], out: dict) -> None:
    case = tiny_case()
    case_json = json.dumps(case.to_dict())
    out['tiny_case'] = {k: case.params[k] for k in ('batch', 'layers', 'experts', 'topk', 'vocab', 'q_heads', 'context')} | {'gpus': case.gpus, 'tp': case.tp}
    print(f'[B] tiny TP4 case on CPU/gloo: {out["tiny_case"]}')
    runs = []
    for sub in subs:
        path = HERE / sub
        res = outdir / f'tiny-{sub[:-3]}.json'; rec = outdir / f'tiny-{sub[:-3]}.record'
        log = outdir / f'tiny-{sub[:-3]}.log.txt'
        t0 = time.perf_counter()
        try:
            cp = subprocess.run([sys.executable, '-s', __file__, '--child', case_json, str(path), str(res), str(rec)],
                                stdout=open(log, 'w'), stderr=subprocess.STDOUT, timeout=CHILD_TIMEOUT_S, env=os.environ | {'OMP_NUM_THREADS': '4'})
            rc = cp.returncode
        except subprocess.TimeoutExpired:
            rc = 'TIMEOUT'
        wall = round(time.perf_counter() - t0, 1)
        entry = {'submission': sub, 'child_rc': rc, 'wall_s': wall}
        if res.exists():
            r = json.loads(res.read_text())
            entry.update(status=r.get('status'), reason=r.get('reason'),
                         ranks=[{'rank': k.get('rank'), 'status': k.get('status'), 'reason': k.get('reason'),
                                 'trials': len((k.get('correctness') or {}).get('trials', []))} for k in r.get('ranks', [])])
        records = {}
        for f in sorted(outdir.glob(f'tiny-{sub[:-3]}.record.rank*.jsonl')):
            records[f.name.split('.')[-2]] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        entry['records'] = records
        if rc == 'TIMEOUT':
            entry['note'] = 'child exceeded the wall-clock limit: possible collective mismatch or deadlock'
        runs.append(entry)
        print(f'[B] {sub}: rc={rc} wall={wall}s status={entry.get("status")} reason={str(entry.get("reason"))[:160]}')
        for k in entry.get('ranks', []):
            print(f'      rank {k["rank"]}: {k["status"]} trials={k["trials"]} reason={str(k["reason"])[:200]}')
        for rk, lines in records.items():
            print(f'      record {rk}: {json.dumps(lines)[:300]}')
    out['part_b'] = runs


def main() -> int:
    if sys.argv[1] == '--child':
        child(*sys.argv[2:6]); return 0
    outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    subs = SUBS
    if '--subs' in sys.argv:
        subs = sys.argv[sys.argv.index('--subs') + 1].split(',')
    import torch, platform
    out = {'torch': torch.__version__, 'host': platform.node(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'cuda_available': torch.cuda.is_available()}
    print(f'torch {torch.__version__} on {platform.node()}, CUDA_VISIBLE_DEVICES={out["cuda_visible_devices"]!r}, cuda_available={out["cuda_available"]}')
    part_a(out)
    part_b(outdir, subs, out)
    (outdir / 'preflight.json').write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
