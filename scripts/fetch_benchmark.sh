#!/bin/bash
# fetch_benchmark.sh: clone the benchmark at the pinned commit and build the read-only trusted
# snapshot the harness reads (/srv/ss/trusted/vibe-megakernel), with
# scaffold_search/patches/gpu_noise_multirank.patch applied (scaffold_search/trusted.py). Idempotent.
#
#   SS_BENCHMARK_URL  default https://github.com/kamahori/vibe-megakernel.git (runner needs read access)
#   SS_PROTOCOL       protocol file (default PROTOCOL.md); its "trusted evaluator" row gives the pinned
#                     commit and branch:
#                       PROTOCOL.md      886eeb3e28db071f6c43b31fd0b6a9c26dbeaf27 (branch dsv41-dspark-batch8)
#                       PROTOCOL_TP4.md  c9782832d264fecd8df4619d5e96ed83aadb77db (= 886eeb3 + the TP4 cases)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/usr/bin/python3.12}
URL=${SS_BENCHMARK_URL:-https://github.com/kamahori/vibe-megakernel.git}
PROTO=$REPO/${SS_PROTOCOL:-PROTOCOL.md}
ROW=$(grep -m1 '^| trusted evaluator |' "$PROTO") || { echo "no trusted evaluator row in $PROTO" >&2; exit 1; }
COMMIT=$(echo "$ROW" | grep -o 'vibe-megakernel [0-9a-f]\{40\}' | cut -d' ' -f2)
BRANCH=$(echo "$ROW" | sed -n 's/.*(branch \([^)]*\)).*/\1/p'); BRANCH=${BRANCH:-dsv41-dspark-batch8}
[ ${#COMMIT} = 40 ] || { echo "cannot parse the pinned commit from $PROTO" >&2; exit 1; }
PATCH_SHA=bb4871ccd876220359d5b635f88380955f62c3d61974967b5c33633018cb2f75
SRC=/srv/ss/build/vibe-megakernel
DST=/srv/ss/trusted/vibe-megakernel
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
got=$(sha256sum "$REPO/scaffold_search/patches/gpu_noise_multirank.patch" | cut -d' ' -f1)
[ "$got" = "$PATCH_SHA" ] || { echo "patch sha256 $got != $PATCH_SHA" >&2; exit 1; }
[ "$(ls "$REPO"/scaffold_search/patches/*.patch | wc -l)" = 1 ] || { echo "unexpected extra patches" >&2; exit 1; }
if [ -f "$DST/.ss_commit" ] && [ "$(cat "$DST/.ss_commit")" = "$COMMIT" ] && grep -q "^$PATCH_SHA  gpu_noise_multirank.patch$" "$DST/.ss_patches" \
   && [ "$(wc -l < "$DST/.ss_patches")" = 1 ]; then
  echo "trusted tree already at $COMMIT + gpu_noise_multirank.patch: $DST"; exit 0
fi
mkdir -p /srv/ss/build; chmod 700 /srv/ss/build
if [ ! -d "$SRC/.git" ]; then
  git clone -q --branch "$BRANCH" "$URL" "$SRC"
fi
git -C "$SRC" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch -q origin "$BRANCH" || git -C "$SRC" fetch -q origin "$COMMIT"
git -C "$SRC" checkout -q --detach "$COMMIT"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "checkout failed" >&2; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has local changes" >&2; exit 1; }
mkdir -p /srv/ss/trusted
cd "$REPO" && "$PY" -m scaffold_search.trusted --src "$SRC" --dst "$DST"
echo "trusted tree: $DST ($(cat "$DST/.ss_commit"); $(cat "$DST/.ss_patches"))"
