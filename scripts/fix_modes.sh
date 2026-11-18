#!/bin/sh
# fix_modes.sh: restore the executable bits listed in scripts/file_modes.txt and make git ignore
# mode bits in this checkout (so the harness does not record it as dirty). Idempotent.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
grep -v '^#' scripts/file_modes.txt | while read -r f; do [ -n "$f" ] && chmod 755 "$f"; done
[ -d .git ] && git config core.fileMode false || true
