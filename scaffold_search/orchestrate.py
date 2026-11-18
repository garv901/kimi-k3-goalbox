"""Execute a schedule: K runs at a time, the next one starts when one finishes (root, in tmux).

  tmux new -d -s ss-orch 'cd /srv/ss/harness && python3.12 -m scaffold_search.orchestrate \
      --schedule /srv/ss/schedules/main.json > /srv/ss/logs/orch-main.log 2>&1'

Each run is `python3.12 -m scaffold_search.run_arm --arm <ARM> --run-id <ID> [extra args]` in its
own window of tmux session `ss-runs-<name>` (log /srv/ss/logs/run-<ID>.log). Runs are started in
schedule order, staggered by --stagger seconds. Runs whose dir already has result.json are skipped
(restartable). The ledger (results/ledger.md + ledger.json on branch `results`) is rewritten after
each start/finish: per-run wall time and node-hours, the schedule's wall time and the cost at
$72 per node-hour (the whole node is billed while any run is active).
"""
import argparse, json, os, shlex, subprocess, sys, time

from scaffold_search import rescore

RATE = 72.0
SS = '/srv/ss'


def utc(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t if t is not None else time.time()))


def tmux(*a):
    return subprocess.run(['tmux', *a], capture_output=True, text=True)


def exit_code(run_id):
    p = f'{SS}/logs/run-{run_id}.log'
    if not os.path.exists(p):
        return None
    for line in reversed(open(p, errors='replace').read().splitlines()[-5:]):
        if line.startswith('RUN_ARM_EXIT '):
            return int(line.split()[1])
    return None


def write_ledger(sched, state, t0, push):
    rows, lines = [], [f'# Ledger: schedule `{sched["name"]}` (seed {sched["seed"]}, K={sched["k"]})', '',
                       '| run | arm | start (UTC) | end (UTC) | wall h | exit | primary score | L ms |',
                       '|---|---|---|---|---|---|---|---|']
    for r in sched['order']:
        st = state.get(r['run_id'], {})
        res = f'{SS}/runs/{r["run_id"]}/result.json'
        score = L = ''
        if os.path.exists(res):
            try:
                pr = json.load(open(res))['primary']; score, L = pr.get('score'), pr.get('L_ms')
            except Exception:
                pass
        s, e = st.get('start'), st.get('end')
        wall = round(((e or time.time()) - s) / 3600, 3) if s else None
        rows.append({'run_id': r['run_id'], 'arm': r['arm'], 'start': s, 'end': e, 'wall_h': wall,
                     'exit': st.get('exit'), 'score': score, 'L_ms': L})
        lines.append(f'| {r["run_id"]} | {r["arm"]} | {utc(s) if s else ""} | {utc(e) if e else ""} | '
                     f'{wall if wall is not None else ""} | {st.get("exit", "")} | {score} | {L} |')
    node_h = (time.time() - t0) / 3600
    lines += ['', f'Schedule wall time so far: {node_h:.2f} node-hours = ${node_h * RATE:.0f} at ${RATE:.0f}/h '
                  f'(as of {utc()}).', 'Sum of per-run wall hours: '
              f'{sum(r["wall_h"] or 0 for r in rows):.2f} (runs overlap K at a time).']
    d = f'{SS}/results/ledger'
    os.makedirs(d, exist_ok=True)
    md, js = f'{d}/ledger-{sched["name"]}.md', f'{d}/ledger-{sched["name"]}.json'
    open(md, 'w').write('\n'.join(lines) + '\n')
    json.dump({'schedule': sched['name'], 'node_hours': node_h, 'usd': node_h * RATE, 'runs': rows,
               'updated_utc': utc()}, open(js, 'w'), indent=1)
    if push:
        rescore.push_results('_ledger', [(md, f'ledger-{sched["name"]}.md'), (js, f'ledger-{sched["name"]}.json')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--k', type=int, help='override the schedule K')
    ap.add_argument('--stagger', type=int, default=90)
    ap.add_argument('--run-args', default='', help='extra run_arm args, e.g. "--budget-minutes 20"')
    ap.add_argument('--no-push', action='store_true')
    a = ap.parse_args()
    sched = json.load(open(a.schedule))
    k = a.k or sched['k']
    sess = f'ss-runs-{sched["name"]}'
    if tmux('has-session', '-t', sess).returncode != 0:
        tmux('new-session', '-d', '-s', sess, '-n', 'idle', 'sleep infinity')
    if not a.no_push:
        rescore.push_results('_schedules', [(a.schedule, f'{sched["name"]}.json')])
    pending = [r for r in sched['order'] if not os.path.exists(f'{SS}/runs/{r["run_id"]}/result.json')]
    state, active, t0, last_start = {}, {}, time.time(), 0
    print(f'{utc()} schedule {sched["name"]}: {len(pending)} runs to do, K={k}', flush=True)
    while pending or active:
        for rid in list(active):
            rc = exit_code(rid)
            if rc is not None:
                state[rid].update(end=time.time(), exit=rc)
                del active[rid]
                print(f'{utc()} finished {rid} exit {rc}', flush=True)
                write_ledger(sched, state, t0, not a.no_push)
        if pending and len(active) < k and time.time() - last_start >= a.stagger:
            r = pending.pop(0)
            rid = r['run_id']
            if os.path.exists(f'{SS}/runs/{rid}'):
                print(f'{utc()} {rid}: run dir exists without result.json; skipping (inspect it)', flush=True)
                continue
            log = f'{SS}/logs/run-{rid}.log'
            cmd = (f'cd {SS}/harness && python3.12 -m scaffold_search.run_arm --arm {shlex.quote(r["arm"])} '
                   f'--run-id {rid} {a.run_args} > {log} 2>&1; echo "RUN_ARM_EXIT $?" >> {log}')
            tmux('new-window', '-d', '-t', sess, '-n', rid, cmd)
            active[rid] = True
            state[rid] = {'start': time.time()}
            last_start = time.time()
            print(f'{utc()} started {rid} ({r["arm"]}) in tmux {sess}:{rid}', flush=True)
            write_ledger(sched, state, t0, not a.no_push)
        time.sleep(10)
    write_ledger(sched, state, t0, not a.no_push)
    print(f'{utc()} schedule {sched["name"]} done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
