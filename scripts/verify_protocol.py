#!/usr/bin/env python3
"""Check this checkout against the sha256 manifest frozen in PROTOCOL.md.

  python3 scripts/verify_protocol.py [--trusted /srv/ss/trusted/vibe-megakernel] [--quiet]

Recomputes, with the method used for the PROTOCOL.md manifest:
  - file hashes: arms/*.toml, prompts/goal_improve.md, scaffold_search/gpu_lease.py,
    scaffold_search/rescore.py (scoring constants), scaffold_search/patches/gpu_noise_multirank.patch
  - tree hashes: overlays/{S,A1,A2,A3}, scaffold_search/harness_files
    (sha256 over `relpath \\0 sha256(file) \\n`, walking directories in sorted os.walk order)
  - seed: sha256 of seeds/dsv41_b8_tp4_nopf/MANIFEST.sha256 (= SEED_SHA256), and, if src/ is unpacked,
    every file of src/ against the manifest
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


def protocol_table():
    rows = {}
    for line in open(f'{ROOT}/PROTOCOL.md'):
        m = re.match(r'^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$', line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trusted'); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    t = protocol_table()
    want_patch = re.search(r'gpu_noise_multirank\.patch ([0-9a-f]{64})', t['trusted evaluator']).group(1)
    want_commit = re.search(r'vibe-megakernel ([0-9a-f]{40})', t['trusted evaluator']).group(1)
    checks = []
    for arm in ('A0S', 'A1', 'A2', 'A3'):
        checks.append((f'arms/{arm}.toml', t[f'arms/{arm}.toml'], lambda p=f'{ROOT}/arms/{arm}.toml': fh(p)))
    for ov in ('S', 'A1', 'A2', 'A3'):
        checks.append((f'overlays/{ov}', t[f'overlays/{ov} (tree)'], lambda d=f'{ROOT}/overlays/{ov}': th(d)))
    checks.append(('scaffold_search/harness_files', t['scaffold_search/harness_files (tree)'],
                   lambda: th(f'{ROOT}/scaffold_search/harness_files')))
    checks.append(('scaffold_search/gpu_lease.py', t['scaffold_search/gpu_lease.py'],
                   lambda: fh(f'{ROOT}/scaffold_search/gpu_lease.py')))
    # scoring constants (B3_MS, SEED_L_FALLBACK_MS) and the AUC/score code are frozen too
    checks.append(('scaffold_search/rescore.py', t['scaffold_search/rescore.py'],
                   lambda: fh(f'{ROOT}/scaffold_search/rescore.py')))
    checks.append(('prompts/goal_improve.md', t['prompts/goal_improve.md'], lambda: fh(f'{ROOT}/prompts/goal_improve.md')))
    checks.append(('patches/gpu_noise_multirank.patch', want_patch,
                   lambda: fh(f'{ROOT}/scaffold_search/patches/gpu_noise_multirank.patch')))
    seed = f'{ROOT}/seeds/dsv41_b8_tp4_nopf'
    want_seed = t['seed dsv41_b8_tp4_nopf (MANIFEST sha256)']
    checks.append(('seed MANIFEST.sha256', want_seed, lambda: fh(f'{seed}/MANIFEST.sha256')))
    checks.append(('seed SEED_SHA256 file', want_seed, lambda: open(f'{seed}/SEED_SHA256').read().strip()))
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
    if os.path.isdir(f'{seed}/src'):
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
            print(f'{"FAIL" if wrong else "OK  "} seed src/ files vs manifest' + (f': {wrong}' if wrong else ''))
    else:
        print('SKIP seed src/ not unpacked (scripts/unpack_seed.sh)')
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
    print('PROTOCOL manifest: ' + ('all match' if not bad else f'{bad} MISMATCH(ES)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
