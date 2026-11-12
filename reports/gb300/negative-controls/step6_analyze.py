#!/usr/bin/env python3
"""Step 6 analyzer: summarise one reports/gb300/step6-tp1-<control>-<ts>/ directory (CPU only, read only).
Checks the GO's expected classification per control and prints a verdict block; writes analysis.json alongside."""
import json, sys, os, re, hashlib

EXPECT = {
    'positive_control_oracle': dict(status='non_megakernel', correctness='pass', launch_audit='launch_budget_failed', reason_prefix=None),
    'negctrl_logits': dict(status='incorrect', reason_prefix='logits:'),
    'negctrl_routing': dict(status='incorrect', reason_prefix='expert_ids:'),
    'negctrl_immutability': dict(status='incorrect', reason_prefix='candidate mutated input l3_kv_cache'),
}

def read(p, default=''):
    try:
        return open(p).read()
    except FileNotFoundError:
        return default

def main(d):
    d = d.rstrip('/')
    ctrl = re.match(r'.*step6-tp1-(.+)-\d{8}T\d{6}Z$', d).group(1)
    exp = EXPECT[ctrl]
    rp = os.path.join(d, 'result.jsonl')
    if not os.path.exists(rp) and os.path.exists(rp + '.gz'):
        import gzip
        lines = gzip.open(rp + '.gz', 'rt').read().strip().splitlines()
    else:
        lines = read(rp).strip().splitlines()
    r = json.loads(lines[-1]) if lines else {}
    env = dict(l.split(': ', 1) for l in read(os.path.join(d, 'env.txt')).splitlines() if ': ' in l and not l.startswith(' '))
    rc = read(os.path.join(d, 'exit_code')).strip()
    corr = r.get('correctness'); corr_status = corr.get('status') if isinstance(corr, dict) else corr
    trials = corr.get('trials') if isinstance(corr, dict) else None
    la = (r.get('launch_audit') or {}).get('status')
    reason = r.get('reason') or ''
    # record
    rec_files = sorted(f for f in os.listdir(d) if f.startswith('negctrl-record'))
    records = []
    for f in rec_files:
        records += [json.loads(l) for l in read(os.path.join(d, f)).splitlines() if l.strip()]
    # gpu monitor: peak used memory and distinct pids
    peak_mb, pids = 0, set()
    for l in read(os.path.join(d, 'gpu-monitor.log.txt')).splitlines():
        for ent in l.split(' ', 1)[1].split(';') if ' ' in l else []:
            parts = [x.strip() for x in ent.split(',')]
            if len(parts) == 3 and parts[2].split()[0].isdigit():
                peak_mb = max(peak_mb, int(parts[2].split()[0])); pids.add(parts[1])
    # host mem
    hm = [l.split() for l in read(os.path.join(d, 'host-mem-monitor.log.txt')).splitlines()[1:] if len(l.split()) == 6]
    host = None
    if hm:
        used = [int(x[2]) for x in hm]
        host = dict(samples=len(hm), used_mb_first=used[0], used_mb_max=max(used), used_mb_last=used[-1], growth_mb_max=max(used) - used[0], swap_used_mb_max=max(int(x[5]) for x in hm))
    k8s = [l for l in read(os.path.join(d, 'k8s-monitor.log.txt')).splitlines() if l.strip()]
    k8s_bad = [l for l in k8s if 'total_gpu_requests=0' not in l]
    occ_before = read(os.path.join(d, 'occupancy-before.txt')); occ_after = read(os.path.join(d, 'occupancy-after.txt'))
    cleanup = read(os.path.join(d, 'cleanup.txt'))
    timing = dict(l.split(': ', 1) for l in read(os.path.join(d, 'timing.txt')).splitlines() if ': ' in l)
    from datetime import datetime
    wall = None
    if 'start_utc' in timing and 'end_utc' in timing:
        f = '%Y-%m-%dT%H:%M:%SZ'
        wall = (datetime.strptime(timing['end_utc'], f) - datetime.strptime(timing['start_utc'], f)).total_seconds()
    stderr = read(os.path.join(d, 'stderr.log.txt'))
    warnings = [l for l in stderr.splitlines() if re.search(r'Warning|WARN|Error|Traceback|NCCL|error', l)]
    checks = {
        'status_as_expected': r.get('status') == exp['status'],
        # the evaluator stores f"{type(e).__name__}: {e}", so the grader's own message follows "AssertionError: "
        'reason_as_expected': (re.sub(r'^AssertionError: ', '', reason).startswith(exp['reason_prefix']) if exp.get('reason_prefix') else True),
        'exit_code_1': rc == '1',
        'no_gpu_pod_during_run': not k8s_bad and len(k8s) > 0,
        'gpu_idle_after': 'compute_apps:\ngpu_uuid' in occ_after and occ_after.rstrip().endswith('total_gpu_requests=0') or False,
        'no_leftover_processes': 'no leftover megabench processes' in cleanup,
        'record_present': bool(records),
    }
    if ctrl == 'positive_control_oracle':
        checks['correctness_pass'] = corr_status == exp['correctness']
        checks['launch_audit_as_expected'] = la == exp['launch_audit']
        # a passing run lists one entry per passed trial with per-output worst_ratio (< 1 means inside the band)
        checks['three_trials_all_outputs_in_band'] = (len(trials) == 3 and all(
            all(o.get('worst_ratio', 0) < 1 for o in t.get('outputs', {}).values() if isinstance(o, dict)) for t in trials)) if isinstance(trials, list) else None
        checks.pop('record_present')  # the positive control writes no record
    if ctrl == 'negctrl_routing' and records:
        checks['margin_guard_not_triggered'] = all(x.get('replacement_shortfall_below_kth_best', 0) >= x.get('min_shortfall_required', 0.05) for x in records)
    if ctrl == 'negctrl_immutability' and records:
        checks['record_confirms_mutation'] = all(x.get('changed') is True for x in records)
    out = dict(dir=os.path.basename(d), control=ctrl, commit=env.get('commit'), submission_sha256=env.get('submission_sha256'),
               worktree_status=env.get('worktree_status', ''), host=env.get('host'), exit_code=rc, status=r.get('status'), reason=reason,
               correctness_status=corr_status, launch_audit_status=la, result_submission_sha256=r.get('submission_sha256'),
               result_keys=sorted(r.keys()), wall_s=wall, gpu_peak_used_mb=peak_mb, gpu_pids=sorted(pids), host_mem=host,
               k8s_samples=len(k8s), k8s_nonzero_gpu_samples=k8s_bad, records=records, cleanup=cleanup.strip().splitlines(),
               stderr_flagged_lines=warnings[:40], expected=exp, checks=checks, all_checks_met=all(v for v in checks.values()))
    if trials and isinstance(trials, list):
        out['trials'] = [dict(seed=t.get('seed'), scenario=t.get('scenario'), worst_ratio_max=max((o.get('worst_ratio', 0) for o in t.get('outputs', {}).values() if isinstance(o, dict)), default=None)) for t in trials]
    json.dump(out, open(os.path.join(d, 'analysis.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ('dir', 'commit', 'submission_sha256', 'exit_code', 'status', 'reason', 'correctness_status', 'launch_audit_status', 'wall_s', 'gpu_peak_used_mb', 'host_mem', 'k8s_samples', 'k8s_nonzero_gpu_samples', 'cleanup', 'stderr_flagged_lines', 'checks', 'all_checks_met')}, indent=1, default=str))
    print('RECORDS:'); [print(' ', json.dumps(x)) for x in records]
    if trials and isinstance(trials, list): print('TRIALS:', out.get('trials'))

if __name__ == '__main__':
    main(sys.argv[1])
