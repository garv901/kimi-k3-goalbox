#!/usr/bin/env python3
"""Check this checkout against the sha256 manifest frozen in a protocol file.

  python3 scripts/verify_protocol.py [--protocol PROTOCOL.md] [--trusted /srv/ss/trusted/vibe-megakernel] [--quiet]

--protocol defaults to $SS_PROTOCOL, else PROTOCOL.md (TP8); the TP4 protocol is PROTOCOL_TP4.md.
Every row of the protocol's "## Manifest" table is checked, with the method used to write it:
  - `<path>`: sha256 of that file (arms, prompts, scaffold_search/gpu_lease.py)
  - `<path> (tree)`: tree hash (sha256 over `relpath \\0 sha256(file) \\n`, sorted os.walk order)
  - `seed <name> (MANIFEST sha256)`: sha256 of seeds/<name>/MANIFEST.sha256 (= SEED_SHA256), and, if
    src/ is unpacked, every file of src/ against the manifest
  - `b3 <task> (ms)`: the score constant in scaffold_search/workspace.py TASKS[<task>]['b3_ms']
  - `trusted evaluator`: the gpu_noise_multirank.patch hash (and, with --trusted, the trusted tree)
  - any row whose value is TBD fails (the protocol is not frozen yet), unless --allow-tbd
  - with --trusted: the trusted tree's .ss_commit and .ss_patches
Exit 0 only if everything matches. Any mismatch means the run conditions differ from the
ones fixed in (PROTOCOL.md: such runs are invalid).
"""
import argparse, hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fh(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def th(d):
    h = hashlib.sha256()
    for r, ds, fs in sorted(os.walk(d)):
        ds.sort()
        for f in sorted(fs):
            p = os.path.join(r, f)
            h.update(os.path.relpath(p, d).encode() + b'\0' + fh(p).encode() + b'\n')
    return h.hexdigest()


def protocol_table(path):
    rows, on = {}, False
    for line in open(path):
        if line.startswith('## '):
            on = line.strip().lower().startswith('## manifest')
            continue
        m = re.match(r'^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$', line)
        if on and m and m.group(1) != 'item' and not set(m.group(1)) <= set('-: '):
            rows[m.group(1)] = m.group(2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default=os.environ.get('SS_PROTOCOL', 'PROTOCOL.md'))
    ap.add_argument('--trusted'); ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--allow-tbd', action='store_true', help='TBD rows are reported, not counted (bootstrap '
                    'before the TP4 seed and baseline are frozen); never for starting the experiment')
    a = ap.parse_args()
    ppath = a.protocol if os.path.isabs(a.protocol) else f'{ROOT}/{a.protocol}'
    t = protocol_table(ppath)
    if not a.quiet:
        print(f'protocol: {os.path.relpath(ppath, ROOT)} ({len(t)} manifest rows)')
    want_patch = re.search(r'gpu_noise_multirank\.patch ([0-9a-f]{64})', t['trusted evaluator']).group(1)
    want_commit = re.search(r'vibe-megakernel ([0-9a-f]{40})', t['trusted evaluator']).group(1)
    checks, seeds = [], []
    for key, want in t.items():
        ms = re.match(r'^seed (\S+) \(MANIFEST sha256\)$', key)
        if key in ('trusted evaluator', 'Claude Code / model'):
            continue
        if want.strip().upper().startswith('TBD'):
            if a.allow_tbd:
                print(f'TODO {key}: TBD (allowed with --allow-tbd)')
            else:
                checks.append((key, 'set (not TBD)', lambda: 'TBD (protocol not frozen)'))
        elif ms:
            seed = f'{ROOT}/seeds/{ms.group(1)}'
            seeds.append(seed)
            checks.append((f'seed {ms.group(1)} MANIFEST.sha256', want, lambda s=seed: fh(f'{s}/MANIFEST.sha256')))
            checks.append((f'seed {ms.group(1)} SEED_SHA256 file', want, lambda s=seed: open(f'{s}/SEED_SHA256').read().strip()))
        elif re.match(r'^b3 (\S+) \(ms\)$', key):
            task = re.match(r'^b3 (\S+) \(ms\)$', key).group(1)
            def b3(task=task, want=want):
                sys.path.insert(0, ROOT)
                from scaffold_search import workspace
                got = workspace.TASKS[task]['b3_ms']
                try:
                    return want if got is not None and abs(float(want) - got) < 1e-9 else str(got)
                except ValueError:
                    return str(got)
            checks.append((f'b3 of task {task} (scaffold_search/workspace.py TASKS)', want, b3))
        elif key.endswith(' (tree)'):
            d = key[:-len(' (tree)')]
            checks.append((d, want, lambda d=f'{ROOT}/{d}': th(d)))
        elif '/' in key:
            checks.append((key, want, lambda p=f'{ROOT}/{key}': fh(p)))
    checks.append(('patches/gpu_noise_multirank.patch', want_patch,
                   lambda: fh(f'{ROOT}/scaffold_search/patches/gpu_noise_multirank.patch')))
    bad = 0
    for name, want, fn in checks:
        try:
            got = fn()
        except Exception as e:  # missing file etc.
            got = f'ERROR {e!r}'
        ok = got == want
        bad += not ok
        if not ok or not a.quiet:
            print(f'{"OK  " if ok else "FAIL"} {name}: {got[:16]}{"" if ok else " (PROTOCOL " + want[:16] + ")"}')
    for seed in seeds:
        if not os.path.isdir(f'{seed}/src'):
            print(f'SKIP {os.path.basename(seed)}: src/ not unpacked (scripts/unpack_seed.sh)')
            continue
        wrong = []
        for line in open(f'{seed}/MANIFEST.sha256'):
            h, p = line.split(None, 1)
            q = f'{seed}/src/{p.strip()}'
            if not os.path.exists(q) or fh(q) != h:
                wrong.append(p.strip())
        n = sum(len(fs) for _, _, fs in os.walk(f'{seed}/src'))
        n_manifest = len(open(f'{seed}/MANIFEST.sha256').readlines())
        if n != n_manifest:
            wrong.append(f'{n} files in src/ but {n_manifest} in the manifest')
        bad += bool(wrong)
        if wrong or not a.quiet:
            print(f'{"FAIL" if wrong else "OK  "} {os.path.basename(seed)} src/ files vs manifest' + (f': {wrong}' if wrong else ''))
    noexec = [f for f in (l.strip() for l in open(f'{ROOT}/scripts/file_modes.txt'))
              if f.startswith('overlays/') and not os.access(f'{ROOT}/{f}', os.X_OK)]
    bad += bool(noexec)
    if noexec or not a.quiet:
        print(f'{"FAIL" if noexec else "OK  "} overlay tools executable' +
              (f': {noexec} (run scripts/fix_modes.sh)' if noexec else ''))
    if a.trusted:
        try:
            c = open(f'{a.trusted}/.ss_commit').read().strip()
            pt = open(f'{a.trusted}/.ss_patches').read().split()
        except OSError as e:
            c, pt = f'ERROR {e!r}', []
        ok = c == want_commit and want_patch in pt and len(pt) == 2
        bad += not ok
        if not ok or not a.quiet:
            print(f'{"OK  " if ok else "FAIL"} trusted tree {a.trusted}: commit {c[:12]} patches {pt}')
    print(f'{os.path.basename(ppath)} manifest: ' + ('all match' if not bad else f'{bad} MISMATCH(ES)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
