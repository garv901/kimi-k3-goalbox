#!/bin/bash
# CPU-only Nsight Systems experiments for B-C2-1 (no GPU, no shared-host process controls).
#   E1  clock mapping: trace timestamps vs CLOCK_MONOTONIC_RAW / MONOTONIC / REALTIME / BOOTTIME read by the target
#   E2  injection escape: are marks from a child started with an emptied environment in the trace, and is the child's start visible
#   E3  uid separation inside one container: nsys as uid 1000 writes the report; the traced target runs as uid 1001 (setpriv with
#       ambient SETUID/SETGID) and must not be able to write the report directory
# Usage: profiler_experiments.sh IMAGE OUTDIR     (runs in Docker; the container has its own pid namespace)
set -u
IMAGE=${1:-goalbox-k3:dev}; OUT=${2:-/tmp/profexp-$$}; mkdir -p "$OUT"; chmod 777 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/goalbox/venv/bin/python
run() { docker run --rm --user "$1" --cap-drop ALL $2 -v "$HERE":/exp:ro -v "$OUT":/out -w /out --entrypoint /bin/bash "$IMAGE" -c "$3" 2>&1 | grep -v "^$\|NVIDIA\|CUDA\|license\|License\|=====\|Toolkit\|docs.nvidia\|developer.nvidia\|container image\|By pulling"; }

echo "== E1+E2: traced parent with clocks, child with emptied environment (uid 1000, no capabilities)"
run 1000:1000 "" "nsys profile --trace=nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o /out/e12 $PY /exp/profiler_child.py /out/e12.json 2>&1 | tail -3; nsys export --type sqlite --force-overwrite true -o /out/e12.sqlite /out/e12.nsys-rep >/dev/null 2>&1; echo export=\$?"
python3 - "$OUT" <<'PYEOF'
import json, sqlite3, sys
out = sys.argv[1]
rec = json.load(open(f"{out}/e12.json"))
conn = sqlite3.connect(f"{out}/e12.sqlite")
tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
strings = dict(conn.execute("select id, value from StringIds")) if "StringIds" in tables else {}
marks = {}
if "NVTX_EVENTS" in tables:
    for start, text, tid, gpid in conn.execute("select start, text, textId, globalPid from NVTX_EVENTS"):
        label = text if text is not None else strings.get(tid)
        if label and label.startswith("exp:"):
            marks[label] = (int(start), gpid)
print("   NVTX marks in trace:", sorted(marks))
info = dict(conn.execute("select name, value from TARGET_INFO_SESSION_START_TIME")) if "TARGET_INFO_SESSION_START_TIME" in tables else {}
print("   session start info:", info)
pm = [m for m in rec["marks"]]
for m in pm:
    t = marks.get(m["label"])
    if t:
        for clk in ("monotonic_raw_ns", "monotonic_ns", "realtime_ns", "boottime_ns"):
            print(f"   {m['label']}: trace_start - {clk} = {t[0] - m[clk]} ns")
child = rec.get("child_record")
print("   child ran:", bool(child), "| child marks in trace:", [k for k in marks if ":child:" in k], "| child LD_PRELOAD set:", bool(child and child["ld_preload"]))
pids = {gp for _, gp in marks.values()}
print("   distinct globalPids with exp marks:", len(pids))
if "OSRT_API" in tables:
    n = conn.execute("select count(*) from OSRT_API").fetchone()[0]
    print("   OSRT_API rows:", n)
procs = conn.execute("select count(*) from ProcessStreams").fetchone()[0] if "ProcessStreams" in tables else "n/a"
print("   ProcessStreams rows (processes known to the session):", procs)
PYEOF

echo "== E3: nsys as uid 1000, target as uid 1001 via setpriv with ambient SETUID/SETGID"
run 1000:1000 "--cap-add SETUID --cap-add SETGID" "mkdir -m 0700 /out/e3priv; nsys profile --trace=nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o /out/e3priv/e3 setpriv --reuid 1001 --regid 1001 --clear-groups --inh-caps=-all --ambient-caps=-all $PY /exp/profiler_child.py /out/e3.json child 2>&1 | tail -2; ls -ln /out/e3priv | tail -1; echo 'as 1001, write into report dir:'; setpriv --reuid 1001 --regid 1001 --clear-groups --inh-caps=-all --ambient-caps=-all sh -c 'echo forged > /out/e3priv/e3.nsys-rep 2>&1 && echo WRITABLE || echo denied'; nsys export --type sqlite --force-overwrite true -o /out/e3.sqlite /out/e3priv/e3.nsys-rep >/dev/null 2>&1; echo export=\$?"
python3 - "$OUT" <<'PYEOF'
import json, sqlite3, sys
out = sys.argv[1]
rec = json.load(open(f"{out}/e3.json")); print("   target uid:", rec["uid"], "LD_PRELOAD seen by target:", bool(rec["ld_preload"]))
conn = sqlite3.connect(f"{out}/e3.sqlite")
tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
strings = dict(conn.execute("select id, value from StringIds")) if "StringIds" in tables else {}
labels = []
if "NVTX_EVENTS" in tables:
    for text, tid in conn.execute("select text, textId from NVTX_EVENTS"):
        label = text if text is not None else strings.get(tid)
        if label and label.startswith("exp:"): labels.append(label)
print("   marks from the uid-1001 target in the uid-1000 report:", sorted(labels))
PYEOF
echo "== done; raw outputs in $OUT"
