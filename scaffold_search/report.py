"""Aggregate result.json files into a markdown report with the pre-registered verdicts.

  python3.12 -m scaffold_search.report RESULTS_DIR [--out report.md]

RESULTS_DIR holds <run_id>/result.json (the `results/` dir of branch `results`; `_smoke` and other
`_` dirs are skipped). Prints every run, per-arm median/IQR of the primary score and AUC, and:
- decision rule (PROTOCOL.md): arm X wins over Y if median score(X) > median score(Y) x (1 + margin)
  and median AUC(X) > median AUC(Y); margin = max(5%, IQR/median of the A0S control's scores).
"""
import argparse, glob, json, os, statistics, sys

# PROTOCOL.md comparisons: tools, knowledge, parallel engines
COMPARISONS = [('A1', 'A0S'), ('A2', 'A1'), ('A3', 'A1')]
CONTROL = 'A0S'


def iqr(xs):
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(sorted(xs), n=4, method='inclusive')
    return q[2] - q[0]


def load(d):
    rows = []
    for p in sorted(glob.glob(f'{d}/*/result.json')):
        if os.path.basename(os.path.dirname(p)).startswith('_'):
            continue
        r = json.load(open(p))
        if r.get('fake_agent'):
            continue
        rows.append(r)
    return rows


def control_iqr_rel(rows):
    c = [r['primary']['score'] for r in rows if r['arm'] == CONTROL]
    if len(c) < 2:
        return 0.0
    return iqr(c) / statistics.median(c)


def report(rows):
    out = ['# Scaffold-search report', '', '| run | arm | score | L ms | source | AUC | final score | '
           'checks (pass) | cost $ | stall min | flags |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['arm'], r['run_id'])):
        s = r['secondary']
        flags = len(r['secondary'].get('abort_flags') or []) + (r['transcript_scan']['n'] if r['transcript_scan']['flagged'] else 0)
        out.append(f'| {r["run_id"]} | {r["arm"]} | {r["primary"]["score"]} | {r["primary"]["L_ms"]:.3f} | '
                   f'{r["primary"]["source"]} | {r["auc"]} | {(r.get("final_tree") or {}).get("score")} | '
                   f'{s["checks"]["n"]} ({s["checks"]["passing"]}) | {s.get("cost_usd")} | '
                   f'{s.get("rate_limit_stall_min")} | {flags} |')
    arms = sorted({r['arm'] for r in rows})
    stats = {}
    out += ['', '| arm | n | median score | IQR score | median AUC | skill calls (runs using) |', '|---|---|---|---|---|---|']
    for a in arms:
        rs = [r for r in rows if r['arm'] == a]
        sc = [r['primary']['score'] for r in rs]
        au = [r['auc'] for r in rs if r['auc'] is not None]
        used = sum(1 for r in rs if r['secondary'].get('skill_calls'))
        stats[a] = {'n': len(rs), 'med': statistics.median(sc), 'iqr': iqr(sc), 'auc': statistics.median(au) if au else None}
        out.append(f'| {a} | {len(rs)} | {stats[a]["med"]:.4f} | {stats[a]["iqr"]:.4f} | {stats[a]["auc"]} | {used} |')
    margin = max(0.05, control_iqr_rel(rows))
    out += ['', f'Margin = max(5%, {CONTROL} IQR/median) = {margin:.4f} (relative to the comparison median).', '']
    verdicts = {}
    for x, y in COMPARISONS:
        if x in stats and y in stats:
            mx, my = stats[x]['med'], stats[y]['med']
            win = mx > my * (1 + margin) and (stats[x]['auc'] or 0) > (stats[y]['auc'] or 0)
            v = f'{x} vs {y}: {"WIN" if win else "no win"} (median {mx:.4f} vs {my:.4f}, AUC {stats[x]["auc"]} vs {stats[y]["auc"]})'
            verdicts[f'{x}-{y}'] = v
            out.append(f'- {v}')
    return '\n'.join(out) + '\n', stats, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir'); ap.add_argument('--out')
    a = ap.parse_args()
    text, _, _ = report(load(a.results_dir))
    if a.out:
        open(a.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    sys.exit(main())
