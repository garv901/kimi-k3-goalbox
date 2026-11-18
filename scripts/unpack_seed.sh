#!/bin/bash
# unpack_seed.sh: unpack the vendored seed (seeds/dsv41_b8_tp4_nopf/src.tar.xz.b64 -> src/) and verify
# it against MANIFEST.sha256 / SEED_SHA256 (the sha256 frozen in PROTOCOL.md). Idempotent.
# The seed is vendored as a base64 tarball (same packaging as the TP8 seed; the TP4 seed ships the
# microbenchmark sources only, the TP8 x86-64 binaries were dropped).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=$REPO/seeds/dsv41_b8_tp4_nopf
PY=${PY:-$(command -v python3.12 || command -v python3)}
verify() {
  "$PY" - "$D" <<'EOF'
import hashlib, os, sys
d = sys.argv[1]
want = open(f'{d}/SEED_SHA256').read().strip()
got = hashlib.sha256(open(f'{d}/MANIFEST.sha256', 'rb').read()).hexdigest()
assert got == want, f'MANIFEST.sha256 hash {got} != SEED_SHA256 {want}'
bad = []
for line in open(f'{d}/MANIFEST.sha256'):
    h, p = line.split(None, 1); p = p.strip()
    q = f'{d}/src/{p}'
    if not os.path.isfile(q) or hashlib.sha256(open(q, 'rb').read()).hexdigest() != h:
        bad.append(p)
n = sum(len(fs) for _, _, fs in os.walk(f'{d}/src'))
assert not bad, f'seed files differ from the manifest: {bad}'
assert n == len(open(f'{d}/MANIFEST.sha256').readlines()), f'{n} files in src/, manifest differs'
print(f'seed dsv41_b8_tp4_nopf OK: {n} files, SEED_SHA256 {want}')
EOF
}
if [ -d "$D/src" ] && verify 2>/dev/null; then exit 0; fi
[ -f "$D/src.tar.xz.b64" ] || { echo "unpack_seed: $D/src.tar.xz.b64 missing" >&2; exit 1; }
rm -rf "$D/src"
base64 -d "$D/src.tar.xz.b64" | tar -xJf - -C "$D" --no-same-owner
verify
