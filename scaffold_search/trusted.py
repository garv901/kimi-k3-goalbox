"""Build the read-only trusted benchmark snapshot used by every run (root).

  python3.12 -m scaffold_search.trusted [--src /root/vibe-megakernel] [--dst /srv/ss/trusted/vibe-megakernel]

`git archive HEAD` of the benchmark checkout (it must be clean), then the harness patches in
scaffold_search/patches/*.patch (applied in name order with `patch -p0` relative to
megabench/integrations/), then root-owned and read-only. Records the commit and the sha256 of each
patch in <dst>/.ss_commit and <dst>/.ss_patches.

Patches (identical for every arm; see docs/harness.md):
- gpu_noise_multirank.patch (M2): `gpu_noise probe` builds multi-GPU cases with one process per
  rank and the `execution` dict, so `check_candidate` gets its 2 extra timing processes on the
  8-rank DSv4.1 case instead of `KeyError: 'execution'`.
"""
import argparse, glob, hashlib, os, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def build(src, dst):
    st = subprocess.run(['git', '-C', src, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f'{src} has local changes:\n{st}')
    commit = subprocess.run(['git', '-C', src, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            check=True).stdout.strip()
    tmp = dst + '.new'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(f'git -C {src} archive HEAD | tar x -C {tmp}', shell=True, check=True)
    rows = []
    for p in sorted(glob.glob(f'{HERE}/patches/*.patch')):
        subprocess.run(['patch', '-p0', '--forward', '-i', p], cwd=f'{tmp}/megabench/integrations', check=True,
                       capture_output=True)
        rows.append(f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}  {os.path.basename(p)}')
    open(f'{tmp}/.ss_commit', 'w').write(commit + '\n')
    open(f'{tmp}/.ss_patches', 'w').write('\n'.join(rows) + '\n')
    subprocess.run(['chown', '-R', 'root:root', tmp], check=True)
    subprocess.run(['chmod', '-R', 'a-w,a+rX', tmp], check=True)
    if os.path.exists(dst):
        old = dst + '.old'
        shutil.rmtree(old, ignore_errors=True)
        subprocess.run(['chmod', '-R', 'u+w', dst], check=True)
        os.rename(dst, old)
        os.rename(tmp, dst)
        shutil.rmtree(old)
    else:
        os.rename(tmp, dst)
    return commit, rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/vibe-megakernel')
    ap.add_argument('--dst', default='/srv/ss/trusted/vibe-megakernel')
    a = ap.parse_args()
    print(build(a.src, a.dst))
