#!/bin/bash
# E3c: as E3b, but the target keeps the spawner's group (uid 1001, gid 1000) and nsys runs with umask 007, so the
# profiler's runtime directories are group-accessible. Questions: does injection now deliver the target's events,
# and what can the target then write (report directory, profiler stream directories)?
set -u
IMAGE=${1:-goalbox-e3b:dev}; OUT=${2:-/tmp/e3c-$$}; mkdir -p "$OUT"; chmod 777 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/goalbox/venv/bin/python
docker run --rm --user 1000:1000 --cap-drop ALL --cap-add SETUID --cap-add SETGID -v "$HERE":/exp:ro -v "$OUT":/out -w /out --entrypoint /bin/bash "$IMAGE" -c '
mkdir -m 0700 /out/rep; umask 007
nsys profile --trace=nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o /out/rep/e3c \
  bash -c "umask 027; env > /out/rep/env; chmod 640 /out/rep/env; exec /usr/local/bin/dropuid 1001:1000 /out/rep/env '"$PY"' /exp/profiler_child.py /out/e3c.json child" 2>&1 | grep -v "^$" | tail -2
echo "report owner/mode: $(stat -c "%u:%g %a" /out/rep/e3c.nsys-rep)   report dir: $(stat -c "%u:%g %a" /out/rep)"
nsys export --type sqlite --force-overwrite true -o /out/e3c.sqlite /out/rep/e3c.nsys-rep >/dev/null 2>&1; echo "export=$?"
echo "--- as uid 1001 gid 1000: what is writable?"
/usr/local/bin/dropuid 1001:1000 /out/rep/env /bin/sh -c "id; echo forged > /out/rep/e3c.nsys-rep 2>/dev/null && echo REPORT-WRITABLE || echo report-denied; for d in /tmp/nsys-agent /tmp/nvidia/nsight_systems /tmp/nvidia/nsight_systems/*/; do [ -d \$d ] && { touch \$d/probe 2>/dev/null && echo \"WRITABLE \$d\" || echo \"denied \$d\"; }; done; ls -la /tmp | grep -i nsys | head -3"
' 2>&1 | grep -v "NVIDIA\|CUDA Version\|license\|License\|=====\|Toolkit\|docs.nvidia\|developer.nvidia\|container image\|By pulling\|^$"
python3 - "$OUT" <<'PYEOF'
import json, sqlite3, sys, os
out = sys.argv[1]
rec = json.load(open(f"{out}/e3c.json")) if os.path.exists(f"{out}/e3c.json") else None
print("target record:", rec and {"uid": rec["uid"], "ld_preload_set": bool(rec["ld_preload"]), "pid": rec["pid"]})
conn = sqlite3.connect(f"{out}/e3c.sqlite")
tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
strings = dict(conn.execute("select id, value from StringIds")) if "StringIds" in tables else {}
labels = sorted({(t if t is not None else strings.get(i)) for t, i in conn.execute("select text, textId from NVTX_EVENTS")} if "NVTX_EVENTS" in tables else set())
print("marks from the uid-1001 target in the uid-1000 report:", [l for l in labels if l and l.startswith("exp:")])
print("PROCESSES:", conn.execute("select pid, name from PROCESSES").fetchall() if "PROCESSES" in tables else None)
for row in conn.execute("select text from DIAGNOSTIC_EVENT").fetchall():
    if "NVTX" in row[0] or "injection" in row[0].lower(): print("  diag:", row[0][:120])
PYEOF
