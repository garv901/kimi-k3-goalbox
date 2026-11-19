"""Rescore a finished run and write result.json (root; the checks themselves run as the run user).

  python3.12 -m scaffold_search.rescore RUN_DIR [--reps 3] [--no-final] [--no-push]

1. Candidates: passing snapshots (snapshots/index.jsonl) ordered by the agent-measured p50.
2. For the best one: extract into a fresh `make_agent_workspace` tree (trusted evaluator, no build
   cache), run the stock ./check_candidate `reps` times, each on a whole-node (4-GPU) lease, as the run user.
   L = median of the per-run p50 (gpu_noise `fast_p50_ms` of that run, not the cross-run history).
3. Automated review (review.py): every rescore run ok_provisional, 1 launch on each of the 4 ranks,
   every trial (fresh random seeds drawn by the evaluator, never seen by the agent) passing; sources
   reachable from submission.py load no tensors from files and import no reference/oracle. If it
   fails, the next-best snapshot is tried (at most 3). No valid snapshot -> the run scores the
   seed: L = this run's calibration measurement of the seed.
4. Final tree (the workspace at the end): one check, unless its source digest equals the chosen
   snapshot's.
5. result.json: primary score B3_MS/L, AUC (mean over the budget of log(B3_MS / best passing L so
   far), agent-measured p50s, seed latency before the first pass), final-tree score and secondary
   metrics from telemetry.json; written to <run_dir>/result.json and pushed to branch `results`
   as results/<run_id>/.
"""
import argparse, calendar, fcntl, glob, hashlib, json, math, os, shutil, statistics, subprocess, sys, tarfile, time

from scaffold_search import isolation, snapshot, review, telemetry, workspace

# Score normalizer. 28.95 ms is the TP8 PyTorch baseline (library kernels + torch.compile + CUDA graph +
# PDL on 8x B300) kept from protocol scaffold-search-dsv41 so that the score formula is unchanged; it is
# NOT a TP4 measurement. TP4 scores/AUCs are comparable only with each other (the decision rule uses
# ratios and differences of medians, which do not depend on this constant). Replace with the TP4
# baseline once it is measured (PROTOCOL.md, Outcomes) and regenerate the manifest.
B3_MS = 28.95
# Seed latency used for the AUC when a run has no calibration measurement. TP4 value from the Gate E
# calibration smoke (2026-11-19, rack 10 tray18, 4x GB300, stock check_candidate on a 4-GPU lease,
# --processes 1, 3 trials): seed dsv41_b8_tp4_nopf cuda-event p50 28.02 ms (per rank 28.013-28.024 ms),
# evidence docs/scaffold-search-dsv41-tp4-gateE-2026-11-19.md of the workspace repo. A run that has
# its own calibration check uses that measurement instead (seed_L); this constant only fills in when
# the calibration is missing. None would omit the AUC (result.json 'auc': null, 'auc_note').
SEED_L_FALLBACK_MS = 28.02
RESULTS_REPO = '/srv/ss/results/repo'
LOCK = '/srv/ss/results/.lock'
PY = '/usr/bin/python3.12'


def tree_digest(d):
    skip_dirs = ('.git', 'build', 'reports', '.harness', '__pycache__', '.cache', '.torch_extensions', 'progress')
    files = []
    for root, dirs, fns in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip_dirs]
        for fn in fns:
            p = os.path.join(root, fn)
            # notes/logs do not change what is built or timed; NOTES.md always changes at the end
            if fn.endswith(('.so', '.o', '.pyc', '.ncu-rep', '.nsys-rep', '.md', '.txt', '.log')) or os.path.islink(p):
                continue
            files.append(os.path.relpath(p, d))
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode()); h.update(open(os.path.join(d, rel), 'rb').read())
    return h.hexdigest()


def tar_digest(tar_path):
    """source digest of a snapshot tarball (same exclusions as tree_digest)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ss-digest-') as d:
        with tarfile.open(tar_path) as t:
            t.extractall(d, filter='data')
        return tree_digest(d)


def fresh_tree(run_id, run_dir, name, populate):
    """fresh make_agent_workspace tree at <run_dir>/rescore/<name>/ws, populated, owned by the run user."""
    info = snapshot.run_info(run_dir)
    base = f'{run_dir}/rescore'
    os.makedirs(base, exist_ok=True); os.chmod(base, 0o711)
    d = f'{base}/{name}'
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d); os.chmod(d, 0o711)
    ws = f'{d}/ws'
    subprocess.run([PY, '-m', 'megabench.integrations.make_agent_workspace', '--agent', 'plain', '--case', info['case'],
                    '--output', ws, '--benchmark-root', workspace.TRUSTED], cwd=workspace.TRUSTED, check=True,
                   capture_output=True, env=dict(os.environ, PYTHONPATH=workspace.TRUSTED))
    subprocess.run(['chmod', '-R', 'u+w', ws], check=True)
    stock = {t: open(f'{ws}/{t}').read() for t in ('check_candidate', 'profile_candidate', 'timeline_candidate')
             if os.path.exists(f'{ws}/{t}')}
    populate(ws)
    for x in ('build', '.harness'):
        shutil.rmtree(f'{ws}/{x}', ignore_errors=True)
    # the snapshot carries the harness wrappers; restore the stock tools of the fresh tree
    for t, text in stock.items():
        open(f'{ws}/{t}', 'w').write(text); os.chmod(f'{ws}/{t}', 0o755)
    import pwd
    pw = pwd.getpwnam(isolation.user_name(run_id))
    subprocess.run(['chown', '-R', f'{pw.pw_uid}:{pw.pw_gid}', ws], check=True)
    os.chmod(ws, 0o700)
    return ws


def run_check(run_id, run_dir, ws, tag):
    """one stock check_candidate on a whole-node (4-GPU) lease, as the run user."""
    env = isolation.agent_env(run_id, run_dir)
    t0 = time.time()
    p = subprocess.run(['/srv/ss/bin/gpu_lease', 'run', '-v', '--gpus', str(workspace.NGPU), '--max-minutes', '30', '--',
                        './check_candidate'], cwd=ws, env=env, preexec_fn=isolation.demote(run_id),
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    open(f'{os.path.dirname(ws)}/{tag}.log', 'w').write(out)
    info = snapshot.parse_check(ws, out)
    rec = None
    if info.get('report') and os.path.exists(info['report']):
        rec = json.loads(open(info['report']).readline())
    return {'rc': p.returncode, 'minutes': round((time.time() - t0) / 60, 2)} | info, rec


def prebuild(run_id, run_dir, ws):
    """compile outside the lease (as the run user, no GPU) when the tree has ./prebuild; the
    evaluator then loads the same digest-keyed build. Failure is harmless (it builds in the lease)."""
    if not os.access(f'{ws}/prebuild', os.X_OK):
        return None
    t0 = time.time()
    p = subprocess.run(['./prebuild'], cwd=ws, env=isolation.agent_env(run_id, run_dir),
                       preexec_fn=isolation.demote(run_id), capture_output=True, text=True, timeout=1800)
    open(f'{os.path.dirname(ws)}/prebuild.log', 'w').write(p.stdout + p.stderr)
    return {'rc': p.returncode, 'seconds': round(time.time() - t0, 1)}


def check_n(run_id, run_dir, name, populate, reps):
    ws = fresh_tree(run_id, run_dir, name, populate)
    pb = prebuild(run_id, run_dir, ws)
    runs, recs = [], []
    for i in range(reps):
        r, rec = run_check(run_id, run_dir, ws, f'check{i}')
        runs.append(r); recs.append(rec)
    p50s = [r['p50_ms'] for r in runs if r.get('p50_ms') and r.get('status') == 'ok_provisional']
    return {'ws': ws, 'runs': runs, 'p50s': p50s, 'prebuild': pb,
            'L_ms': statistics.median(p50s) if p50s else None, 'digest': tree_digest(ws)}, recs


def extract(tar_path):
    def populate(ws):
        with tarfile.open(tar_path) as t:
            t.extractall(ws, filter='data')
    return populate


def copy_seed(seed):
    def populate(ws):
        shutil.copytree(f'{workspace.REPO}/seeds/{seed}/src', ws, dirs_exist_ok=True)
        workspace.wexec(f'{ws}/prebuild', workspace.SEED_PREBUILD.replace('{ws}', ws))
    return populate


def calibrate(run_id, run_dir):
    info = snapshot.run_info(run_dir)
    res, recs = check_n(run_id, run_dir, 'calibration', copy_seed(info['seed']), 1)
    out = {'L_ms': res['L_ms'], 'status': res['runs'][0].get('status'), 'minutes': res['runs'][0]['minutes'],
           'n_processes': res['runs'][0].get('n_processes'), 'probe_errors': res['runs'][0].get('probe_errors')}
    json.dump(out, open(f'{run_dir}/logs/calibration.json', 'w'), indent=1)
    return out


def auc(snaps, seed_L, t0, budget_min):
    """time-average of log(B3 / best-so-far) over [t0, t0 + budget]."""
    T = budget_min * 60
    pts = []
    for s in snaps:
        try:
            te = calendar.timegm(time.strptime(s['end'], '%Y-%m-%dT%H:%M:%SZ'))
        except Exception:
            continue
        if s.get('p50_ms'):
            pts.append((te - t0, s['p50_ms']))
    pts.sort()
    best, last, acc = seed_L, 0.0, 0.0
    for t, L in pts:
        t = min(max(t, 0.0), T)
        acc += (t - last) * math.log(B3_MS / best)
        last = t
        best = min(best, L)
    acc += (T - last) * math.log(B3_MS / best)
    return acc / T


def push_results(run_id, files):
    os.makedirs('/srv/ss/results', exist_ok=True)
    with open(LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.isdir(f'{RESULTS_REPO}/.git'):
            subprocess.run(['git', 'clone', '-q', '-b', 'results', snapshot.REMOTE, RESULTS_REPO], check=True)
            subprocess.run(['git', '-C', RESULTS_REPO, 'config', 'user.name', 'scaffold-search harness'])
            subprocess.run(['git', '-C', RESULTS_REPO, 'config', 'user.email', 'harness@localhost'])
        g = lambda *a: subprocess.run(['git', '-C', RESULTS_REPO, *a], capture_output=True, text=True)
        g('pull', '-q', '--rebase', 'origin', 'results')
        d = f'{RESULTS_REPO}/results/' + (f'_smoke/{run_id}' if run_id.startswith(('smoke', 'test')) else run_id)
        os.makedirs(d, exist_ok=True)
        for src, name in files:
            if os.path.exists(src):
                shutil.copy2(src, f'{d}/{name}')
        g('add', '-A')
        g('commit', '-qm', f'results: {run_id}')
        for _ in range(3):
            p = g('push', '-q', 'origin', 'HEAD:results')
            if p.returncode == 0:
                return 'ok'
            g('pull', '-q', '--rebase', 'origin', 'results')
        return f'push failed: {p.stderr[-300:]}'


def rescore_run(run_dir, reps=3, final=True, push=True, max_candidates=3):
    info = snapshot.run_info(run_dir)
    run_id = info['run_id']
    run = json.load(open(f'{run_dir}/logs/run.json'))
    tel = telemetry.summarize(run_dir, run_id)
    snaps = [json.loads(l) for l in open(f'{run_dir}/snapshots/index.jsonl')] \
        if os.path.exists(f'{run_dir}/snapshots/index.jsonl') else []
    cal = run.get('calibration') or {}
    seed_L = cal.get('L_ms')
    failed = run.get('failed')   # set by run_arm containment: nothing of this run is rescored or accepted
    ranked = [] if failed else sorted((s for s in snaps if s.get('p50_ms')), key=lambda s: s['p50_ms'])
    if failed:
        final = False
        for s in snaps:
            s['invalidated'] = failed
    tried, chosen = [], None
    for s in ranked[:max_candidates]:
        try:
            res, recs = check_n(run_id, run_dir, f'snap{s["n"]:03d}', extract(f'{run_dir}/snapshots/{s["tar"]}'), reps)
        except (tarfile.TarError, EOFError) as e:   # unreadable tarball: not scorable, try the next snapshot
            tried.append({'snapshot': s['tar'], 'agent_p50_ms': s['p50_ms'], 'sha256': s.get('sha256'),
                          'error': f'snapshot tarball unreadable: {e!r}', 'L_ms': None,
                          'review_reports': {'ok': False, 'issues': ['tarball unreadable']},
                          'review_sources': {'ok': False, 'disqualifying': []}, 'runs': [], 'rescore_p50s': []})
            continue
        rv = review.review_reports(recs)
        src = review.review_sources(res['ws'])
        entry = {'snapshot': s['tar'], 'agent_p50_ms': s['p50_ms'], 'sha256': s['sha256'], 'rescore_p50s': res['p50s'],
                 'L_ms': res['L_ms'], 'review_reports': rv, 'review_sources': {k: src[k] for k in ('ok', 'disqualifying')},
                 'runs': [{k: r.get(k) for k in ('status', 'p50_ms', 'rc', 'minutes', 'verdict', 'n_processes',
                                                 'probe_errors')} for r in res['runs']], 'digest': res['digest']}
        tried.append(entry)
        if rv['ok'] and src['ok'] and res['L_ms']:
            chosen = entry
            break
    if chosen:
        L, source = chosen['L_ms'], 'snapshot'
    else:
        L, source = seed_L, 'seed-fallback'
    # final tree
    final_res = None
    if final:
        os.makedirs(f'{run_dir}/rescore', exist_ok=True)
        final_tar = f'{run_dir}/rescore/final.tgz'
        workspace.tar_source(f'{run_dir}/ws', final_tar)
        fd = tar_digest(final_tar)
        if chosen and fd == tar_digest(f'{run_dir}/snapshots/{chosen["snapshot"]}'):
            final_res = {'L_ms': chosen['L_ms'], 'status': 'ok_provisional', 'same_as': chosen['snapshot']}
        else:
            res, recs = check_n(run_id, run_dir, 'final', extract(final_tar), 1)
            r0 = res['runs'][0]
            final_res = {'L_ms': r0.get('p50_ms') if r0.get('status') == 'ok_provisional' else None,
                         'status': r0.get('status'), 'agent_view_p50_ms': r0.get('p50_ms')}
    t0 = run.get('t_agent_start')
    budget = run.get('budget_minutes')
    seed_for_auc = seed_L or SEED_L_FALLBACK_MS
    first_imp = None
    for s in sorted(snaps, key=lambda s: s.get('end') or ''):
        if seed_for_auc and s.get('p50_ms') and s['p50_ms'] < seed_for_auc * 0.99:
            te = calendar.timegm(time.strptime(s['end'], '%Y-%m-%dT%H:%M:%SZ'))
            first_imp = round((te - t0) / 60, 2); break
    tr = tel['transcript']
    cost = tr.get('cost_usd') or 0
    if failed:
        L, source = None, 'failed: ' + failed
    score = B3_MS / L if L else None
    result = {
        'schema': 'scaffold-search/result/v1',
        'run_id': run_id, 'arm': info['arm'], 'task': info['task'], 'case': info['case'],
        'budget_minutes': budget, 'fake_agent': run.get('fake_agent', False),
        'failed': failed, 'containment': run.get('containment'),
        'pins': {k: run.get(k) for k in ('claude_version', 'model', 'harness_commit', 'harness_dirty', 'trusted_commit',
                                         'trusted_patches', 'seed_sha256',
                                         'prompt_sha256', 'arm_sha256')},
        'times_utc': {k: run.get(k) for k in ('agent_start_utc', 'interrupt_utc', 'agent_end_utc', 'end_utc')},
        'calibration': cal,
        'primary': {'score': round(score, 4) if score else None, 'L_ms': L, 'source': source,
                    'snapshot': chosen['snapshot'] if chosen else None,
                    'rescore_p50s_ms': chosen['rescore_p50s'] if chosen else None, 'b3_ms': B3_MS},
        'auc': round(auc(snaps, seed_for_auc, t0, budget), 5) if t0 and budget and seed_for_auc else None,
        'auc_note': None if seed_for_auc else 'no seed calibration for this run and no SEED_L_FALLBACK_MS (TP4 uncalibrated)',
        'final_tree': final_res and {**final_res, 'score': round(B3_MS / final_res['L_ms'], 4) if final_res.get('L_ms') else None},
        'candidates_tried': tried,
        'secondary': {
            'time_to_first_improvement_min': first_imp,
            'gpu_busy_fraction': tel['gpu'].get('gpu_busy_fraction'),
            'checks': tel['checks'], 'tokens': tr.get('tokens'), 'cost_usd': cost,
            'rate_limit_stall_min': tr.get('rate_limit_stall_min'), 'tool_calls': tr.get('tool_calls'),
            'skill_calls': tr.get('skill_calls'), 'subagent_calls': tr.get('subagent_calls'),
            'lease': tel['lease'], 'score_per_usd': round(score / cost, 5) if score and cost else None,
            'abort_flags': run.get('abort_flags', []),
            'agent_exit': run.get('agent_exit'), 'gpu_clean_after': run.get('gpu_clean'),
            'bypass_pids': tel['gpu'].get('bypass_pids'),
        },
        'transcript_scan': review.scan_transcript([f'{run_dir}/logs/agent.jsonl', f'{run_dir}/logs/resume.jsonl'],
                                                  f'{run_dir}/ws', run_dir),
        'snapshots': [dict(x, invalidated=failed) for x in tel['snapshots']] if failed else tel['snapshots'],
    }
    json.dump(result, open(f'{run_dir}/result.json', 'w'), indent=1)
    if push:
        result['results_push'] = push_results(run_id, [(f'{run_dir}/result.json', 'result.json'),
                                                       (f'{run_dir}/logs/telemetry.json', 'telemetry.json'),
                                                       (f'{run_dir}/logs/run.json', 'run.json'),
                                                       (f'{run_dir}/snapshots/index.jsonl', 'snapshots_index.jsonl'),
                                                       (f'{run_dir}/snapshots/checks.jsonl', 'checks.jsonl')])
        json.dump(result, open(f'{run_dir}/result.json', 'w'), indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--no-final', action='store_true'); ap.add_argument('--no-push', action='store_true')
    a = ap.parse_args()
    r = rescore_run(a.run_dir, a.reps, not a.no_final, not a.no_push)
    print(json.dumps({k: r[k] for k in ('run_id', 'arm', 'primary', 'auc', 'final_tree')}, indent=1))


if __name__ == '__main__':
    main()
