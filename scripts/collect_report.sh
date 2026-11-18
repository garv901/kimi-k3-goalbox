#!/bin/bash
# collect_report.sh [OUT_DIR]: report.py over every run with a result.json on this node
# (/srv/ss/runs/r*/ and the local results clone), with the decision rule of PROTOCOL.md.
#
# Writes OUT_DIR (default /srv/ss/report):
#   report.md                      every run, per-arm medians/IQR, verdicts (A1 vs A0S, A2 vs A1, A3 vs A1)
#   results/<run_id>/result.json   the inputs
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/usr/bin/python3.12}; [ -x "$PY" ] || PY=$(command -v python3.12 || command -v python3)
OUT=${1:-/srv/ss/report}
rm -rf "$OUT/results"; mkdir -p "$OUT/results"
n=0
for src in /srv/ss/runs/r[0-9]*/result.json /srv/ss/results/repo/results/r[0-9]*/result.json; do
  [ -f "$src" ] || continue
  rid=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$src")
  [ -f "$OUT/results/$rid/result.json" ] && continue
  mkdir -p "$OUT/results/$rid"; cp "$src" "$OUT/results/$rid/"; n=$((n+1))
done
echo "runs with result.json: $n (expected 16)"
cd "$H" && "$PY" -m scaffold_search.report "$OUT/results" --out "$OUT/report.md"
