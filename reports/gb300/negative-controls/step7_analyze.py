"""Step 7/8 analyzer (read only): one reports/gb300/step7-tp4-<bN>-<control>-<ts>/ (or step8-tp4-ranklocal-immutability-<ts>/) directory of a distributed
negative-control run. Checks every rank's own status and reason against the control's expected failure (not only
the aggregate), the controls' per-rank records, the NCCL collective trace (sequence identical across the four
processes, no warnings), and the infrastructure monitors. Writes analysis.json beside the evidence and prints it.
NCCL parsing follows analyze_step5.py (NCCL datatype enum: 1 uint8, 2 int32, 4 int64, 7 float32, 8 float64;
ncclRedOp_t: 0 sum, 1 prod, 2 max, 3 min, 4 avg)."""
import collections, glob, gzip, json, os, re, sys
from datetime import datetime

EXPECT = {  # per-rank reason must contain the fragment; attributed_to = ranks that must appear as "rank N:" in it
    'negctrl_immutability': dict(fragment='candidate mutated input l3_kv_cache', attributed_to=[0, 1, 2, 3]),
    'negctrl_logits': dict(fragment='logits: {', attributed_to=[0, 1, 2, 3]),
    'negctrl_routing': dict(fragment='expert_ids: {', attributed_to=[0, 1, 2, 3]),
    'negctrl_ranklocal_immutability': dict(fragment='rank 2: trial 0: AssertionError: candidate mutated input l3_kv_cache', attributed_to=[2], not_attributed_to=[0, 1, 3]),
    'negctrl_ranklocal_routing': dict(fragment='different between ranks', attributed_to=[0, 1, 2, 3]),
    'negctrl_ranklocal_logits': dict(fragment='logits: {', attributed_to=[0, 1, 2, 3]),
}
PAT = re.compile(r"\[(\d+)\] NCCL INFO (\w+): opCount (\w+) sendbuff \S+ recvbuff \S+ count (\d+) datatype (\d+) op (\d+) root (\d+) comm (\S+)")


def read(p, default=''):
    try:
        return open(p, errors='replace').read()
    except FileNotFoundError:
        return default


def main(d):
    d = d.rstrip('/')
    m = re.match(r'.*step7-tp4-(b\d+)-(.+)-\d{8}T\d{6}Z$', d)
    if m: batch, ctrl = m.group(1), m.group(2)
    elif re.match(r'.*step8-tp4-ranklocal-immutability-\d{8}T\d{6}Z$', d): batch, ctrl = 'b1', 'negctrl_ranklocal_immutability'  # Step 8 directory name fixed by its GO
    else: raise SystemExit(f'unrecognized evidence directory name: {d}')
    exp = EXPECT.get(ctrl)
    rp = os.path.join(d, 'result.jsonl')
    lines = (gzip.open(rp + '.gz', 'rt').read() if not os.path.exists(rp) and os.path.exists(rp + '.gz') else read(rp)).strip().splitlines()
    r = json.loads(lines[-1]) if lines else {}
    env = dict(l.split(': ', 1) for l in read(os.path.join(d, 'env.txt')).splitlines() if ': ' in l and not l.startswith(' '))
    ranks = r.get('ranks') or []
    per_rank, checks = [], {}
    for k in ranks:
        reason = k.get('reason') or ''
        ok_status = k.get('status') == 'incorrect'
        ok_fragment = bool(exp) and exp['fragment'] in reason
        attributed = [n for n in range(4) if f'rank {n}:' in reason]
        ok_attr = bool(exp) and all(n in attributed for n in exp['attributed_to']) and all(n not in attributed for n in exp.get('not_attributed_to', []))
        per_rank.append(dict(rank=k.get('rank'), device=k.get('device'), status=k.get('status'), reason=reason, trials_recorded=len((k.get('correctness') or {}).get('trials', [])),
                             ranks_named_in_reason=attributed, status_as_expected=ok_status, reason_as_expected=ok_fragment, attribution_as_expected=ok_attr))
    checks['four_ranks_reported'] = len(ranks) == 4 and sorted(k.get('rank') for k in ranks) == [0, 1, 2, 3]
    checks['every_rank_status_incorrect'] = bool(ranks) and all(p['status_as_expected'] for p in per_rank)
    checks['every_rank_reason_as_expected'] = bool(ranks) and all(p['reason_as_expected'] for p in per_rank)
    checks['every_rank_attribution_as_expected'] = bool(ranks) and all(p['attribution_as_expected'] for p in per_rank)
    checks['aggregate_status_incorrect'] = r.get('status') == 'incorrect'
    checks['aggregate_reason_is_rank0_reason'] = bool(ranks) and r.get('reason') == ranks[0].get('reason')
    checks['distinct_devices'] = len({k.get('device') for k in ranks}) == 4
    checks['exit_code_1'] = read(os.path.join(d, 'exit_code')).strip() == '1'
    # records
    records = {}
    for f in sorted(glob.glob(os.path.join(d, 'negctrl-record.rank*.jsonl'))):
        records[os.path.basename(f).split('.')[-2]] = [json.loads(l) for l in read(f).splitlines() if l.strip()]
    checks['record_from_every_rank'] = sorted(records) == ['rank0', 'rank1', 'rank2', 'rank3']
    if ctrl == 'negctrl_immutability' and records:
        checks['record_checksum_path_on_every_rank'] = all(x.get('snapshot_path') == 'device checksums' and x.get('tensor_bytes') == 536870912 and x.get('changed') is True for v in records.values() for x in v) if batch == 'b64' else None
    if ctrl.startswith('negctrl_ranklocal') and records:
        checks['record_corrupted_only_on_rank2'] = all((x.get('corrupted') is True) == (rk == 'rank2') for rk, v in records.items() for x in v)
    # NCCL
    logs = sorted(glob.glob(os.path.join(d, 'nccl-*.log.txt'))); traces, nccl = {}, []
    for f in logs:
        txt = read(f); ops = [(mm.group(2), mm.group(4), mm.group(5), mm.group(6)) for mm in PAT.finditer(txt)]
        traces[f] = ops
        nccl.append(dict(file=os.path.basename(f), cuda_dev=sorted({mm.group(1) for mm in PAT.finditer(txt)}), collectives=len(ops), kinds=dict(collections.Counter(o[0] for o in ops)),
                         warn=txt.count('NCCL WARN'), error_lines=sum(1 for l in txt.splitlines() if 'error' in l.lower() and 'NCCL WARN' not in l), init_complete=txt.count('Init COMPLETE'),
                         ip_redacted=txt.count('<ip-redacted>')))
    base = traces[logs[0]] if logs else []
    checks['nccl_four_logs'] = len(logs) == 4
    checks['nccl_no_warnings_or_errors'] = bool(logs) and all(n['warn'] == 0 and n['error_lines'] == 0 for n in nccl)
    checks['nccl_sequence_identical_across_processes'] = bool(logs) and all(t == base for t in traces.values())
    label = {("AllReduce", "7", "0"): "F", ("AllReduce", "7", "2"): "fmax", ("AllReduce", "4", "2"): "AGREE", ("AllReduce", "4", "3"): "tokmin",
             ("AllReduce", "8", "0"): "f64sum", ("AllGather", "4", "0"): "AG64", ("AllGather", "1", "0"): "AG8", ("Broadcast", "4", "0"): "BC64", ("Broadcast", "1", "0"): "BC8"}
    rle, prev = [], None
    for k, c, dt, op in base:
        l = label.get((k, dt, op), f"{k}/{dt}/{op}")
        if prev and prev[0] == l: prev[1] += 1
        else: prev = [l, 1]; rle.append(prev)
    hist = dict(collections.Counter(f'{k}/{dt}/{op}' for k, _, dt, op in base))
    # monitors
    peak_mb, pids = 0, collections.defaultdict(set)
    for l in read(os.path.join(d, 'gpu-monitor.log.txt')).splitlines():
        for ent in (l.split(' ', 1)[1].split(';') if ' ' in l else []):
            parts = [x.strip() for x in ent.split(',')]
            if len(parts) == 3 and parts[2].split()[0].isdigit():
                peak_mb = max(peak_mb, int(parts[2].split()[0])); pids[parts[0]].add(parts[1])
    checks['one_pid_per_gpu_and_no_foreign_process'] = all(len(v) == 1 for v in pids.values()) and len(pids) == 4
    hm = [l.split() for l in read(os.path.join(d, 'host-mem-monitor.log.txt')).splitlines()[1:] if len(l.split()) == 6]
    host = dict(samples=len(hm), used_mb_first=int(hm[0][2]), used_mb_max=max(int(x[2]) for x in hm), growth_mb_max=max(int(x[2]) for x in hm) - int(hm[0][2]), available_mb_min=min(int(x[4]) for x in hm), swap_used_mb_max=max(int(x[5]) for x in hm)) if hm else None
    k8s = [l for l in read(os.path.join(d, 'k8s-monitor.log.txt')).splitlines() if l.strip()]
    checks['no_gpu_pod_during_run'] = bool(k8s) and all('total_gpu_requests=0' in l for l in k8s)
    occ_after = read(os.path.join(d, 'occupancy-after.txt'))
    checks['gpus_idle_after'] = 'compute_apps:\ngpu_uuid' in occ_after and occ_after.rstrip().endswith('total_gpu_requests=0')
    checks['no_leftover_processes'] = 'no leftover megabench processes' in read(os.path.join(d, 'cleanup.txt'))
    checks['stderr_empty'] = read(os.path.join(d, 'stderr.log.txt')).strip() == ''
    timing = dict(l.split(': ', 1) for l in read(os.path.join(d, 'timing.txt')).splitlines() if ': ' in l)
    wall = (datetime.strptime(timing['end_utc'], '%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(timing['start_utc'], '%Y-%m-%dT%H:%M:%SZ')).total_seconds() if 'end_utc' in timing else None
    out = dict(dir=os.path.basename(d), case=env.get('case'), control=ctrl, commit=env.get('commit'), worktree_status=env.get('worktree_status', ''), submission_sha256=env.get('submission_sha256'),
               result_submission_sha256=r.get('submission_sha256'), host=env.get('host'), exit_code=read(os.path.join(d, 'exit_code')).strip(), status=r.get('status'), reason=r.get('reason'),
               result_keys=sorted(r.keys()), wall_s=wall, per_rank=per_rank, records=records, nccl=nccl, nccl_histogram=hist,
               nccl_run_length=' '.join(f'{l}x{n}' if n > 1 else l for l, n in rle), gpu_peak_used_mb=peak_mb, gpu_pids={k: sorted(v) for k, v in pids.items()}, host_mem=host,
               k8s_samples=len(k8s), expected=exp, checks=checks, all_checks_met=all(v for v in checks.values() if v is not None))
    json.dump(out, open(os.path.join(d, 'analysis.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ('dir', 'case', 'commit', 'submission_sha256', 'result_submission_sha256', 'exit_code', 'status', 'wall_s', 'gpu_peak_used_mb', 'host_mem', 'k8s_samples', 'nccl_histogram', 'nccl_run_length', 'checks', 'all_checks_met')}, indent=1, default=str))
    print('REASON:', r.get('reason'))
    for p in per_rank: print(f"  rank {p['rank']} {p['device']}: {p['status']} named={p['ranks_named_in_reason']} ok={p['status_as_expected'] and p['reason_as_expected'] and p['attribution_as_expected']}")
    for n in nccl: print('  nccl', n)
    for rk, v in records.items(): print('  record', rk, json.dumps(v))


if __name__ == '__main__':
    main(sys.argv[1])
