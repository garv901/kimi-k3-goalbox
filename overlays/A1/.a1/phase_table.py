"""phase_table: run the candidate's instrumented build (phase_table.cuh) on all ranks and tabulate phases.

  ./phase_table [--trial T] [--seed S] [--launches N] [--src FILE] [--group NAME=REGEX ...] [--top K]
  ./phase_table --analyze reports/phase_table-ID [same table options]   (re-analyze, no GPU)
  ./phase_table --self-test                                             (synthetic buffers, no GPU)

Contract (same as ./timeline_candidate): submission.py exports build_timeline(case) -> (run, names);
run(inputs) launches the PHASE_TABLE build and returns the filled CUDA uint64/int64 buffer of that call;
names is a list (index = id) or dict {id: name} naming barrier sites, sub-phases and marks (may be empty).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MAGIC = 0x3142545345534850
HDR = 16
K_BAR, K_BEGIN, K_END, K_MARK = 1, 2, 3, 4


# ----------------------------------------------------------------------------------------------- capture
def _capture_rank(rank: int, contract: dict, submission: str, seed: int, trial: int, launches: int, out: str,
                  rendezvous: str | None) -> None:
    import importlib.util
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from megabench.cases import Case
    from megabench.tasks.workloads import make_inputs, set_trial_scenario

    case = Case(**contract)
    device = f'cuda:{rank}'
    torch.cuda.set_device(rank)
    contract = case.to_dict()
    if rendezvous:
        from megabench.tasks.parallel import initialize
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=case.gpus,
                                timeout=timedelta(minutes=10))
        ctx = initialize(case)
        contract['execution'] = {'rank': rank, 'world_size': case.gpus, 'tp_rank': ctx.tp_rank,
                                 'ep_rank': ctx.ep_rank, 'tp_group': ctx.tp_group, 'ep_group': ctx.ep_group,
                                 'backend': 'nccl', 'device': device}
    try:
        with torch.inference_mode():
            inputs = make_inputs(case, seed, device)
            scenario = set_trial_scenario(case, inputs, trial)
            spec = importlib.util.spec_from_file_location('a1_candidate_phase_table', submission)
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(Path(submission).parent))
            spec.loader.exec_module(module)
            build = getattr(module, 'build_timeline', None)
            if not callable(build):
                raise TypeError('submission.py must export build_timeline(case) -> (run, names); see TASK.d/A1.md')
            run, names = build(contract)
            for _ in range(2):
                run(inputs)
            torch.cuda.synchronize(device)
            target = Path(out) / f'rank{rank}'
            target.mkdir(parents=True, exist_ok=True)
            for k in range(launches):
                if rendezvous:
                    dist.barrier()
                torch.cuda.synchronize(device)
                buf = run(inputs)
                torch.cuda.synchronize(device)
                if buf.dtype not in (torch.int64, torch.uint64) or buf.dim() != 1:
                    raise TypeError(f'run(inputs) must return a 1-D int64/uint64 tensor, got {buf.dtype} {tuple(buf.shape)}')
                (target / f'launch{k}.u64').write_bytes(buf.detach().view(torch.int64).cpu().numpy().tobytes())
            if rank == 0:
                if isinstance(names, dict):
                    nm = {str(int(k)): str(v) for k, v in names.items()}
                else:
                    nm = {str(i): str(v) for i, v in enumerate(names or [])}
                (Path(out) / 'names.json').write_text(json.dumps(nm))
                (Path(out) / 'meta.json').write_text(json.dumps({'case': case.id, 'seed': seed, 'trial': trial,
                                                                 'scenario': scenario, 'launches': launches,
                                                                 'ranks': case.gpus}))
            print(f'rank {rank}: captured {launches} launch(es) -> {target}', flush=True)
    finally:
        if rendezvous:
            dist.destroy_process_group()


def capture_main(a) -> int:
    import tempfile

    import torch
    from megabench.cases import select_cases
    case = select_cases('all', [a.case])[0]
    if torch.cuda.device_count() < case.gpus:
        raise SystemExit(f'{case.id} needs {case.gpus} visible GPUs, have {torch.cuda.device_count()}')
    if case.gpus == 1:
        _capture_rank(0, case.to_dict(), a.submission, a.seed, a.trial, a.launches, a.out, None)
    else:
        with tempfile.TemporaryDirectory(prefix='a1-phase-table-') as d:
            torch.multiprocessing.spawn(_capture_rank, nprocs=case.gpus, join=True,
                                        args=(case.to_dict(), a.submission, a.seed, a.trial, a.launches, a.out,
                                              (Path(d) / 'rdv').as_uri()))
    return 0


# ----------------------------------------------------------------------------------------------- decode
def decode(raw: bytes) -> dict:
    w = np.frombuffer(raw, dtype=np.uint64)
    if len(w) < HDR or int(w[0]) != MAGIC:
        raise ValueError('buffer has no PHASE_TABLE header: is the build compiled with -DPHASE_TABLE, '
                         'is phase_table_bind() called with this buffer, and does the kernel call PT_INIT()?')
    nb, R = int(w[1]), int(w[2])
    slots = (R - 8) // 4
    blocks, dropped = [], 0
    for b in range(nb):
        reg = w[HDR + b * R: HDR + (b + 1) * R]
        cnt = int(reg[0])
        n = min(cnt, slots)
        dropped += cnt - n
        rec = reg[8: 8 + 4 * n].reshape(n, 4).astype(np.int64) if n else np.zeros((0, 4), np.int64)
        blocks.append(dict(kstart=int(reg[2]), kend=int(reg[3]), ta=rec[:, 0], tb=rec[:, 1],
                           kind=(reg[8: 8 + 4 * n].reshape(n, 4)[:, 2] >> np.uint64(56)).astype(np.int64) if n else np.zeros(0, np.int64),
                           id=(reg[8: 8 + 4 * n].reshape(n, 4)[:, 2] & np.uint64(0xFFFFFFFF)).astype(np.int64) if n else np.zeros(0, np.int64),
                           spin=rec[:, 3]))
    return dict(nb=nb, slots=slots, blocks=blocks, dropped=dropped)


class Labeler:
    def __init__(self, names: dict, src: list[str]):
        self.names = names
        self.src = []
        for s in src:
            try:
                self.src.append((Path(s).name, Path(s).read_text(errors='replace').split('\n')))
            except OSError:
                pass

    def __call__(self, i: int) -> str:
        if str(i) in self.names:
            return self.names[str(i)]
        for fname, lines in self.src:
            if 0 < i <= len(lines):
                best = None
                for j in range(i - 1, max(-1, i - 400), -1):
                    m = re.search(r'//\s*[=\-]{3,}\s*(\S.*)', lines[j])
                    if m:
                        best = m.group(1)
                        break
                    m = re.match(r'(?:template\s*<.*>\s*)?(?:__device__|__global__|static|inline|void|int|float)\b[^;]*?(\w+)\s*\(', lines[j])
                    if m and not lines[j].startswith(' '):
                        best = best or m.group(1) + '()'
                        break
                    if best is None:
                        m = re.match(r'\s*//\s*(\S.*)', lines[j])
                        if m and j >= i - 6:
                            best = m.group(1)
                return f'{fname}:{i} {best.strip()[:44]}' if best else f'{fname}:{i}'
        return f'site {i}'


# ----------------------------------------------------------------------------------------------- analysis
def analyze_rank(d: dict) -> dict:
    """Barrier-interval rows of one rank, keyed by (barrier id, occurrence) in block 0's order."""
    blocks = d['blocks']
    seqs = []
    for B in blocks:
        bar = {}
        occ = collections.Counter()
        prev_rel = B['kstart']
        ctx_mark = None
        seq = []
        open_ph = collections.defaultdict(list)
        sub = []
        for k in range(len(B['kind'])):
            kind, i = int(B['kind'][k]), int(B['id'][k])
            if kind == K_BAR:
                key = (i, occ[i])
                occ[i] += 1
                bar[key] = (int(B['ta'][k]), int(B['tb'][k]), prev_rel, int(B['spin'][k]))
                seq.append((key, ctx_mark))
                prev_rel = int(B['tb'][k])
            elif kind == K_MARK:
                ctx_mark = i
            elif kind == K_BEGIN:
                open_ph[i].append(int(B['ta'][k]))
            elif kind == K_END and open_ph[i]:
                sub.append((i, open_ph[i].pop(), int(B['ta'][k])))
        seqs.append(dict(bar=bar, seq=seq, sub=sub, last_rel=prev_rel))
    order = seqs[0]['seq']
    kstart = np.array([B['kstart'] for B in blocks], np.int64)
    kend = np.array([B['kend'] for B in blocks], np.int64)
    rows = []
    prev_min_rel = int(kstart.min())
    for key, ctx in order:
        have = [s['bar'][key] for s in seqs if key in s['bar']]
        a = np.array(have, np.int64)
        arr, rel, prel, spin = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        busy = arr - prel
        mrel = int(rel.min())
        rows.append(dict(id=key[0], occ=key[1], ctx=ctx, blocks=len(have),
                         wall=(mrel - prev_min_rel) / 1e3, bmax=busy.max() / 1e3, bmean=busy.mean() / 1e3,
                         bar=max(0, mrel - int(arr.max())) / 1e3, wait=(rel - arr).mean() / 1e3,
                         spin=spin.mean() / 1e3, slow_block=int(busy.argmax())))
        prev_min_rel = mrel
    end = int(kend.max()) if (kend > 0).all() else max(s['last_rel'] for s in seqs)
    tail = (end - prev_min_rel) / 1e3
    span = (end - int(kstart.min())) / 1e3
    # sub-phases: per (id, occurrence) the max/mean over blocks
    subagg = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in seqs:
        occ = collections.Counter()
        for i, t0, t1 in s['sub']:
            subagg[i][occ[i]].append((t1 - t0) / 1e3)
            occ[i] += 1
    subs = {}
    for i, per in subagg.items():
        subs[i] = dict(n=len(per), max=sum(max(v) for v in per.values()), mean=sum(np.mean(v) for v in per.values()),
                       blocks=max(len(v) for v in per.values()))
    return dict(rows=rows, span=span, tail=tail, kskew=(int(kstart.max()) - int(kstart.min())) / 1e3, subs=subs,
                dropped=d['dropped'], nb=d['nb'])


def analyze_launch(per_rank: list[dict], label, groups) -> dict:
    nr = len(per_rank)
    keys0 = [(r['id'], r['occ']) for r in per_rank[0]['rows']]
    aligned = all([(r['id'], r['occ']) for r in R['rows']] == keys0 for R in per_rank)
    sites = collections.OrderedDict()
    for ri, R in enumerate(per_rank):
        for j, r in enumerate(R['rows']):
            S = sites.setdefault(r['id'], dict(id=r['id'], label=label(r['id']), n=0, wall=np.zeros(nr), bmax=np.zeros(nr),
                                               bmean=np.zeros(nr), bar=np.zeros(nr), wait=np.zeros(nr), spin=np.zeros(nr),
                                               rank_skew=0.0, xwait=0.0, slow=collections.Counter()))
            if ri == 0:
                S['n'] += 1
            for f in ('wall', 'bmax', 'bmean', 'bar', 'wait', 'spin'):
                S[f][ri] += r[f]
    if aligned and nr > 1:
        for j in range(len(keys0)):
            rs = [R['rows'][j] for R in per_rank]
            bm = np.array([r['bmax'] for r in rs])
            wl = np.array([r['wall'] for r in rs])
            S = sites[rs[0]['id']]
            S['rank_skew'] += bm.max() - bm.mean()
            S['xwait'] += (wl - wl.min()).mean()
            S['slow'][int(bm.argmax())] += 1
    span = np.array([R['span'] for R in per_rank])
    # contexts and groups
    ctx_wall = collections.OrderedDict()
    grp = {g: np.zeros(nr) for g, _ in groups}
    for ri, R in enumerate(per_rank):
        for r in R['rows']:
            c = label(r['ctx']) if r['ctx'] is not None else '-'
            ctx_wall.setdefault(c, np.zeros(nr))[ri] += r['wall']
            full = f'{c}|{label(r["id"])}'
            for g, rx in groups:
                if rx.search(full):
                    grp[g][ri] += r['wall']
    decomp = []
    for ri, R in enumerate(per_rank):
        w = sum(r['wall'] for r in R['rows'])
        bmx = sum(r['bmax'] for r in R['rows'])
        bmn = sum(r['bmean'] for r in R['rows'])
        decomp.append(dict(rank=ri, span=R['span'], work=bmn, intra_skew=bmx - bmn, sync=w - bmx,
                           xrank_spread_sum=0.0, tail=R['tail'], head_kstart_skew=R['kskew']))
    if aligned and nr > 1:
        for j in range(len(keys0)):
            wl = np.array([per_rank[ri]['rows'][j]['wall'] for ri in range(nr)])
            for ri in range(nr):
                decomp[ri]['xrank_spread_sum'] += wl[ri] - wl.min()
    subs = collections.OrderedDict()
    for ri, R in enumerate(per_rank):
        for i, s in R['subs'].items():
            T = subs.setdefault(i, dict(id=i, label=label(i), n=s['n'], blocks=s['blocks'], max=np.zeros(nr), mean=np.zeros(nr)))
            T['max'][ri] = s['max']
            T['mean'][ri] = s['mean']
    return dict(sites=sites, span=span, aligned=aligned, ctx=ctx_wall, groups=grp, decomp=decomp, subs=subs,
                dropped=[R['dropped'] for R in per_rank], nb=per_rank[0]['nb'],
                barriers=len(keys0))


def mean_launches(L: list[dict]) -> dict:
    if len(L) == 1:
        return L[0]
    base = L[0]
    for f in ('wall', 'bmax', 'bmean', 'bar', 'wait', 'spin'):
        for i, S in base['sites'].items():
            S[f] = np.mean([x['sites'][i][f] for x in L if i in x['sites']], axis=0)
    for i, S in base['sites'].items():
        S['rank_skew'] = float(np.mean([x['sites'][i]['rank_skew'] for x in L if i in x['sites']]))
        S['xwait'] = float(np.mean([x['sites'][i]['xwait'] for x in L if i in x['sites']]))
    base['span'] = np.mean([x['span'] for x in L], axis=0)
    for g in base['groups']:
        base['groups'][g] = np.mean([x['groups'][g] for x in L], axis=0)
    for c in base['ctx']:
        base['ctx'][c] = np.mean([x['ctx'].get(c, 0 * base['ctx'][c]) for x in L], axis=0)
    for ri, D in enumerate(base['decomp']):
        for f in ('span', 'work', 'intra_skew', 'sync', 'xrank_spread_sum', 'tail', 'head_kstart_skew'):
            D[f] = float(np.mean([x['decomp'][ri][f] for x in L]))
    return base


def short(s: str, n: int = 46) -> str:
    return s if len(s) <= n else s[: n - 17] + '~' + s[-16:]


def report(A: dict, top: int, out: Path | None, meta: dict) -> str:
    nr = len(A['span'])
    span_m = float(np.mean(A['span']))
    lines = []
    P = lines.append
    P(f"phase_table: {meta.get('case', '?')} scenario={meta.get('scenario', '?')} ranks={nr} blocks={A['nb']} "
      f"barrier crossings/launch={A['barriers']} launches averaged={meta.get('launches', 1)}")
    P('in-device span per rank (us): ' + ' '.join(f'r{i}:{s:.0f}' for i, s in enumerate(A['span'])) + f'  mean {span_m:.0f}')
    if any(A['dropped']):
        P(f"WARNING: records dropped (buffer too small): per rank {A['dropped']}; enlarge slots_per_block")
    if not A['aligned']:
        P('NOTE: barrier sequences differ across ranks; cross-rank columns (rank skew, xrank wait) are omitted')
    P('')
    P('Per phase = interval ending at a barrier site, summed over its occurrences per launch, us. wall = release-to-'
      'release (critical path), share = wall/span; busy max/mean = slowest/mean block from its previous release to '
      'its arrival; skew = max-mean; bar = last arrival->release; wait = mean block wait at the barrier; '
      'rskew = max over ranks minus mean of busy max (cross-rank imbalance of the work); xspread = mean over ranks of '
      'wall minus the fastest rank\'s wall (slow ranks show it on work phases, fast ranks on the phase that waits for peers).')
    hdr = (f'{"phase (closing barrier site)":46s} {"n":>4s} {"wall":>8s} {"wallmax":>8s} {"share":>6s} {"bmax":>8s} '
           f'{"bmean":>8s} {"skew":>7s} {"bar":>6s} {"wait":>7s} {"rskew":>7s} {"xsprd":>7s} {"slow r":>6s}')
    P(hdr)
    sites = list(A['sites'].values())
    cnt = collections.Counter(S['label'] for S in sites)
    for S in sites:
        if cnt[S['label']] > 1:
            S['label'] = f"{S['label']} #{S['id'] & 0xFFFF}"
    sites.sort(key=lambda S: -float(np.mean(S['wall'])))
    rows_json = []
    for k, S in enumerate(sites):
        wm = float(np.mean(S['wall']))
        row = dict(id=S['id'], label=S['label'], n=S['n'], wall_mean=wm, wall_max_rank=float(np.max(S['wall'])),
                   wall_per_rank=[float(x) for x in S['wall']], share=wm / span_m,
                   busy_max=float(np.mean(S['bmax'])), busy_mean=float(np.mean(S['bmean'])),
                   skew=float(np.mean(S['bmax'] - S['bmean'])), bar=float(np.mean(S['bar'])),
                   wait=float(np.mean(S['wait'])), spin=float(np.mean(S['spin'])),
                   rank_skew=float(S['rank_skew']), xrank_spread=float(S['xwait']),
                   busy_max_per_rank=[float(x) for x in S['bmax']],
                   slowest_rank=(S['slow'].most_common(1)[0][0] if S['slow'] else None))
        rows_json.append(row)
        if k < top:
            P(f'{short(row["label"]):46s} {row["n"]:4d} {wm:8.0f} {row["wall_max_rank"]:8.0f} {100 * row["share"]:5.1f}% '
              f'{row["busy_max"]:8.0f} {row["busy_mean"]:8.0f} {row["skew"]:7.0f} {row["bar"]:6.0f} {row["wait"]:7.0f} '
              f'{row["rank_skew"]:7.0f} {row["xrank_spread"]:7.0f} {str(row["slowest_rank"]) if row["slowest_rank"] is not None else "-":>6s}')
    if len(sites) > top:
        rest = sum(r['wall_mean'] for r in rows_json[top:])
        P(f'{"(%d more sites)" % (len(sites) - top):46s} {"":4s} {rest:8.0f} {"":8s} {100 * rest / span_m:5.1f}%')
    cum = 0.0
    crit = []
    for r in rows_json:
        cum += r['share']
        crit.append(dict(label=r['label'], share=r['share'], cumulative=cum))
    P('')
    P('Critical path per rank (us): span = work (mean block busy) + intra-GPU skew + sync (barrier latency/handoff) + tail. '
      'A block that waits for peer ranks counts as busy, so cross-rank waits show up as skew; see rskew/xsprd per phase. '
      'head = kernel-start skew across blocks')
    for D in A['decomp']:
        P(f"  r{D['rank']}: span {D['span']:8.0f} = work {D['work']:8.0f} + skew {D['intra_skew']:7.0f} + sync {D['sync']:7.0f} "
          f"+ tail {D['tail']:6.0f}   head {D['head_kstart_skew']:.1f}")
    k50 = next((i + 1 for i, c in enumerate(crit) if c['cumulative'] >= 0.5), len(crit))
    P(f'  {k50} of {len(crit)} phase sites make up 50% of the span; top 3: ' +
      '; '.join(f"{short(c['label'], 40)} {100 * c['share']:.1f}%" for c in crit[:3]))
    if len(A['ctx']) > 1:
        P('')
        P('Wall by context (last PT_MARK before the barrier), mean over ranks, us:')
        for c, v in sorted(A['ctx'].items(), key=lambda kv: -float(np.mean(kv[1])))[:top]:
            P(f'  {c[:50]:50s} {float(np.mean(v)):9.0f} {100 * float(np.mean(v)) / span_m:5.1f}%')
    if A['subs']:
        P('')
        P('Sub-phases (PHASE_BEGIN/END), summed over occurrences, us: max = slowest block, mean over blocks')
        for T in A['subs'].values():
            P(f"  {T['label'][:46]:46s} n {T['n']:4d} blocks {T['blocks']:4d} max {float(np.mean(T['max'])):8.0f} "
              f"mean {float(np.mean(T['mean'])):8.0f}  share(max) {100 * float(np.mean(T['max'])) / span_m:5.1f}%")
    gj = {}
    if A['groups']:
        P('')
        P('Groups (regex over "context|phase label"), share of span:')
        for g, v in A['groups'].items():
            gj[g] = dict(wall_per_rank=[float(x) for x in v], wall_mean=float(np.mean(v)), share_mean=float(np.mean(v)) / span_m,
                         share_rank0=float(v[0]) / float(A['span'][0]))
            P(f"  {g:20s} wall mean {gj[g]['wall_mean']:9.0f} us  share {100 * gj[g]['share_mean']:5.1f}% (rank 0: {100 * gj[g]['share_rank0']:5.1f}%)")
    summary = dict(meta=meta, span_per_rank=[float(x) for x in A['span']], span_mean=span_m, blocks=A['nb'],
                   barrier_crossings=A['barriers'], ranks_aligned=A['aligned'], dropped_records=A['dropped'],
                   phases=rows_json, critical_path=crit, decomposition=A['decomp'],
                   contexts={c: [float(x) for x in v] for c, v in A['ctx'].items()},
                   subphases=[dict(id=T['id'], label=T['label'], n=T['n'], blocks=T['blocks'],
                                   max_per_rank=[float(x) for x in T['max']], mean_per_rank=[float(x) for x in T['mean']])
                              for T in A['subs'].values()],
                   groups=gj)
    text = '\n'.join(lines)
    if out is not None:
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        (out / 'table.txt').write_text(text + '\n')
    return text


def analyze_dir(d: Path, a) -> str:
    names = json.loads((d / 'names.json').read_text()) if (d / 'names.json').exists() else {}
    meta = json.loads((d / 'meta.json').read_text()) if (d / 'meta.json').exists() else {}
    label = Labeler(names, a.src or [])
    groups = []
    for g in a.group or []:
        n, _, rx = g.partition('=')
        groups.append((n, re.compile(rx)))
    rdirs = sorted((p for p in d.glob('rank*') if p.is_dir()), key=lambda p: int(p.name[4:]))
    if not rdirs:
        raise SystemExit(f'no rank*/ dirs in {d}')
    nl = len(list(rdirs[0].glob('launch*.u64')))
    launches = []
    for k in range(nl):
        per_rank = [analyze_rank(decode((r / f'launch{k}.u64').read_bytes())) for r in rdirs]
        launches.append(analyze_launch(per_rank, label, groups))
    A = mean_launches(launches)
    meta['launches'] = nl
    return report(A, a.top, d, meta)


# ----------------------------------------------------------------------------------------------- self-test
def synth_buffer(nb: int, phases: list[tuple[int, float, float]], rank_extra: float, slots=64, marks=None) -> bytes:
    """Synthetic buffer: phases = (site id, base busy us, per-block spread us). Block b busy = base + spread*b/(nb-1)."""
    R = 8 + 4 * slots
    w = np.zeros(HDR + nb * R, np.uint64)
    w[0], w[1], w[2] = MAGIC, nb, R
    t0 = 1_000_000
    rel = np.full(nb, t0, np.int64)
    recs = [[] for _ in range(nb)]
    for j, (sid, base, spread) in enumerate(phases):
        if marks and j in marks:
            for b in range(nb):
                recs[b].append((int(rel[b]), int(rel[b]), (K_MARK << 56) | marks[j], 0))
        busy = np.array([base + spread * b / max(1, nb - 1) for b in range(nb)]) * 1000
        if j == len(phases) - 1:
            busy += rank_extra * 1000
        arr = rel + busy.astype(np.int64)
        r = int(arr.max()) + 1000  # 1 us barrier latency
        for b in range(nb):
            recs[b].append((int(arr[b]), r, (K_BAR << 56) | sid, 0))
        rel = np.full(nb, r, np.int64)
    for b in range(nb):
        base = HDR + b * R
        w[base + 0] = len(recs[b])
        w[base + 2] = t0
        w[base + 3] = int(rel[b]) + 2000
        for k, rc in enumerate(recs[b]):
            w[base + 8 + 4 * k: base + 12 + 4 * k] = np.array(rc, np.uint64)
    return w.tobytes()


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # 2 ranks, 4 blocks, 3 phases: A (10us, spread 2), MOE (50us, spread 0), C (5us); rank 1 slower MoE by 6us
        for r, extra in ((0, 0.0), (1, 0.0)):
            (d / f'rank{r}').mkdir()
        ph = [(10, 10.0, 2.0), (20, 50.0, 0.0), (30, 5.0, 0.0)]
        # make MOE the cross-rank slow one on rank1 by putting it last in a separate variant
        b0 = synth_buffer(4, ph, 0.0, marks={0: 900})
        ph1 = [(10, 10.0, 2.0), (20, 56.0, 0.0), (30, 5.0, 0.0)]
        b1 = synth_buffer(4, ph1, 0.0, marks={0: 900})
        (d / 'rank0' / 'launch0.u64').write_bytes(b0)
        (d / 'rank1' / 'launch0.u64').write_bytes(b1)
        (d / 'names.json').write_text(json.dumps({'10': 'A attn', '20': 'MOE experts', '30': 'C tail', '900': 'layer0'}))
        ns = argparse.Namespace(src=[], group=['moe=MOE'], top=10)
        txt = analyze_dir(d, ns)
        print(txt)
        S = json.loads((d / 'summary.json').read_text())
        ph = {p['label']: p for p in S['phases']}
        def close(a, b, tol=0.05):
            assert abs(a - b) <= tol, (a, b)
        # rank0 walls: A: 12 busy max + 1 = 13 (from kstart), MOE 51, C 6; tail 2
        close(ph['A attn']['wall_per_rank'][0], 13.0)
        close(ph['MOE experts']['wall_per_rank'][0], 51.0)
        close(ph['MOE experts']['wall_per_rank'][1], 57.0)
        close(ph['A attn']['busy_max'], 12.0)
        close(ph['A attn']['busy_mean'], 11.0)
        close(ph['A attn']['skew'], 1.0)
        close(ph['A attn']['bar'], 1.0)
        close(ph['MOE experts']['rank_skew'], 3.0)
        close(ph['MOE experts']['xrank_spread'], 3.0)
        assert ph['MOE experts']['slowest_rank'] == 1
        close(S['span_per_rank'][0], 13 + 51 + 6 + 2)
        close(S['groups']['moe']['share_rank0'], 51 / 72, 1e-3)
        assert 'layer0' in S['contexts']
        # decode error path
        try:
            decode(b'\0' * 256)
            raise AssertionError('expected ValueError')
        except ValueError:
            pass
    print('phase_table self-test: PASS')
    return 0


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trial', type=int, default=0, help='trial scenario index (0 = raw/natural rejection)')
    ap.add_argument('--seed', type=int, default=7301)
    ap.add_argument('--launches', type=int, default=1, help='captured launches to average (after 2 warmups)')
    ap.add_argument('--src', action='append', help='source file whose line numbers are the barrier ids (labels)')
    ap.add_argument('--group', action='append', help='NAME=REGEX over "context|label": print the share of matching phases')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--analyze', type=Path, help='re-analyze an existing reports/phase_table-* dir (no GPU)')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--capture', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--case', help=argparse.SUPPRESS)
    ap.add_argument('--submission', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.capture:
        return capture_main(a)
    if a.analyze:
        print(analyze_dir(a.analyze, a))
        print(f'\nwrote {a.analyze}/summary.json, table.txt')
        return 0
    info = common.case_info()
    out = (a.out or (common.WS / 'reports' / f'phase_table-{common.run_id()}')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = common.benchmark_path() + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp')
    cmd = common.gpu_run_prefix(int(info.get('gpus', 1))) + [
        common.python(), str(Path(__file__).resolve()), '--capture', '--case', info['id'],
        '--submission', str(common.submission_path()), '--seed', str(a.seed), '--trial', str(a.trial),
        '--launches', str(a.launches), '--out', str(out)]
    print('+', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env, cwd=str(common.WS))
    if rc != 0:
        print(f'phase_table: capture failed (exit {rc})', file=sys.stderr)
        return rc
    print(analyze_dir(out, a))
    print(f'\nwrote {out}/summary.json, table.txt, rank*/launch*.u64')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
