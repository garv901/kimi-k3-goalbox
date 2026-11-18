#!/usr/bin/env python3
"""Freeze the TP4 seed from the builder run that ported the TP8 seed (PROTOCOL_TP4.md, step 3).

  python3.12 scripts/freeze_seed.py --run-dir /srv/ss/runs/builder-port1 [--snapshot NNN-....tgz]
         [--name dsv41_b8_tp4_nopf] [--from-seed dsv41_b8_nopf] [--add PATH ...] [--dry-run]

Takes one passing snapshot of the builder run (default: the latest passing one in
snapshots/index.jsonl) and keeps the seed's source files:
  - every path of the TP8 seed manifest (seeds/<from-seed>/MANIFEST.sha256) that is still in the
    snapshot, in its ported version;
  - every file the builder agent created (not in the workspace's initial git commit), except
    reports/, capture-*, build caches and __pycache__;
  - paths given with --add.
Workspace, harness and overlay files (they were in the initial commit) are left out: the experiment
adds them again for every run. A "Seed note" is appended to NOTES.md (as for the TP8 seed). Writes
seeds/<name>/{src/, src.tar.xz.b64, MANIFEST.sha256, SEED_SHA256, SOURCE.json} and prints the PROTOCOL_TP4.md
manifest row. Refuses to overwrite an existing seeds/<name>/. Run as root on the node (it reads the
run's root-only snapshots). Afterwards: calibrate it (scripts/run_main_tp4.sh selftest), commit the seed
dir without src/ (gitignored), and fill the seed row of PROTOCOL_TP4.md.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_PREFIX = ('reports/', 'build/', '.harness/', '.git/', 'progress/microbench/.build-cache/', '.cache/',
               '.torch_extensions/')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def skipped(rel):
    parts = rel.split('/')
    return (rel.startswith(SKIP_PREFIX) or '__pycache__' in parts or any(x.startswith('capture-') for x in parts)
            or rel.endswith(('.so', '.o', '.pyc', '.ncu-rep', '.nsys-rep', '.sqlite')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True); ap.add_argument('--snapshot')
    ap.add_argument('--name', default='dsv41_b8_tp4_nopf'); ap.add_argument('--from-seed', default='dsv41_b8_nopf')
    ap.add_argument('--add', nargs='*', default=[]); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    out = f'{ROOT}/seeds/{a.name}'
    if os.path.exists(out) and not a.dry_run:
        sys.exit(f'{out} exists; pick another --name or remove it deliberately')
    idx = [json.loads(l) for l in open(f'{run_dir}/snapshots/index.jsonl')]
    passing = [s for s in idx if s.get('status') == 'ok_provisional']
    if not passing:
        sys.exit('the builder run has no passing snapshot')
    snap = next((s for s in passing if s['tar'] == a.snapshot), None) if a.snapshot else passing[-1]
    if snap is None:
        sys.exit(f'{a.snapshot} is not a passing snapshot of this run: {[s["tar"] for s in passing]}')
    tar_path = f'{run_dir}/snapshots/{snap["tar"]}'
    ws = f'{run_dir}/ws'
    g = ['git', '-c', 'safe.directory=*', '-C', ws]
    first = subprocess.run(g + ['rev-list', '--max-parents=0', 'HEAD'], capture_output=True, text=True,
                           check=True).stdout.split()[0]
    initial = set(subprocess.run(g + ['ls-tree', '-r', '--name-only', first], capture_output=True, text=True,
                                 check=True).stdout.split('\n')) - {''}
    tp8 = [l.split(None, 1)[1].strip() for l in open(f'{ROOT}/seeds/{a.from_seed}/MANIFEST.sha256')]
    tp8 = {p[2:] if p.startswith('./') else p for p in tp8}
    with tempfile.TemporaryDirectory(prefix='freeze-') as tmp:
        with tarfile.open(tar_path) as t:
            t.extractall(tmp, filter='data')
        files = set()
        for r, _, fs in os.walk(tmp):
            for f in fs:
                p = os.path.join(r, f)
                if not os.path.islink(p):
                    files.add(os.path.relpath(p, tmp))
        keep = sorted(f for f in files if f in tp8 or f in a.add or (f not in initial and not skipped(f)))
        dropped_tp8 = sorted(tp8 - files)
        print(f'snapshot {snap["tar"]} (agent p50 {snap.get("p50_ms")} ms): {len(files)} files, keeping {len(keep)}')
        print('  from the TP8 seed manifest:', sum(f in tp8 for f in keep), '| new files:',
              [f for f in keep if f not in tp8])
        if dropped_tp8:
            print('  TP8 seed files no longer in the tree (dropped):', dropped_tp8)
        print('  left out (workspace/harness/overlay or caches):', len(files) - len(keep))
        if a.dry_run:
            return 0
        os.makedirs(f'{out}/src')
        for f in keep:
            os.makedirs(os.path.dirname(f'{out}/src/{f}'), exist_ok=True)
            shutil.copy2(f'{tmp}/{f}', f'{out}/src/{f}')
    harness = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    with open(f'{out}/src/NOTES.md', 'a') as f:
        f.write(f'\n## Seed note (added by the experiment harness)\nThis tree is the TP4 port of seed '
                f'`{a.from_seed}` made by builder run `{os.path.basename(run_dir)}` (snapshot `{snap["tar"]}`, '
                f'agent-measured p50 {snap.get("p50_ms")} ms). Measured by `./check_candidate` on this node: see '
                f'the harness calibration of each run.\n')
    rows = [f'{sha(os.path.join(out, "src", f))}  ./{f}' for f in sorted(keep)]
    open(f'{out}/MANIFEST.sha256', 'w').write('\n'.join(rows) + '\n')
    seed_sha = sha(f'{out}/MANIFEST.sha256')
    open(f'{out}/SEED_SHA256', 'w').write(seed_sha + '\n')
    json.dump({'from_seed': a.from_seed, 'builder_run': os.path.basename(run_dir), 'snapshot': snap['tar'],
               'snapshot_sha256': snap.get('sha256'), 'agent_p50_ms': snap.get('p50_ms'), 'harness_commit': harness,
               'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
              open(f'{out}/SOURCE.json', 'w'), indent=1)
    subprocess.run(f'tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 -cJf - src | base64 > src.tar.xz.b64',
                   shell=True, cwd=out, check=True)
    print(f'\nwrote {out} ({len(keep)} files)\nPROTOCOL_TP4.md row:\n| seed {a.name} (MANIFEST sha256) | {seed_sha} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
