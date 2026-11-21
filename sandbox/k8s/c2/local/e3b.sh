#!/bin/bash
# E3b: nsys (uid 1000) traces a target switched to uid 1001 by the file-capability helper; the report directory
# is 0700 uid 1000. Checks: target uid, injection survives (its NVTX marks are in the report), the target
# cannot write the report or call the helper, the helper refuses uid 1000 callers' misuse (target <= 1000).
set -u
IMAGE=${1:-goalbox-e3b:dev}; OUT=${2:-/tmp/e3b-$$}; mkdir -p "$OUT"; chmod 777 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/goalbox/venv/bin/python
docker run --rm --user 1000:1000 --cap-drop ALL --cap-add SETUID --cap-add SETGID -v "$HERE":/exp:ro -v "$OUT":/out -w /out --entrypoint /bin/bash "$IMAGE" -c '
mkdir -m 0700 /out/rep
nsys profile --trace=nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o /out/rep/e3b \
  bash -c "env > /out/rep/env; exec /usr/local/bin/dropuid 1001 /out/rep/env '"$PY"' /exp/profiler_child.py /out/e3b.json child" 2>&1 | grep -v "^$" | tail -2
echo "report owner: $(stat -c "%u %a" /out/rep/e3b.nsys-rep)"
nsys export --type sqlite --force-overwrite true -o /out/e3b.sqlite /out/rep/e3b.nsys-rep >/dev/null 2>&1; echo "export=$?"
echo "--- as the candidate uid (1001), via the helper itself (refuses: target must be > caller and caller must be 1000):"
/usr/local/bin/dropuid 1001 /out/rep/env /bin/sh -c "id -u; echo forged > /out/rep/e3b.nsys-rep && echo WRITABLE || echo denied-report-write; /usr/local/bin/dropuid 1000 /out/rep/env /bin/id 2>&1 | tail -1; /usr/local/bin/dropuid 1002 /out/rep/env /bin/id 2>&1 | tail -1; cat /out/rep/env >/dev/null 2>&1 && echo env-readable || echo env-unreadable; kill -0 1 2>/dev/null && echo can-signal-pid1 || echo cannot-signal-pid1"
grep Cap /proc/self/status | head -3
' 2>&1 | grep -v "NVIDIA\|CUDA\|license\|License\|=====\|Toolkit\|docs.nvidia\|developer.nvidia\|container image\|By pulling\|^$"
python3 - "$OUT" <<'PYEOF'
import json, sqlite3, sys, os
out = sys.argv[1]
rec = json.load(open(f"{out}/e3b.json")) if os.path.exists(f"{out}/e3b.json") else None
print("target record:", rec and {"uid": rec["uid"], "ld_preload_set": bool(rec["ld_preload"]), "pid": rec["pid"]})
conn = sqlite3.connect(f"{out}/e3b.sqlite")
tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
strings = dict(conn.execute("select id, value from StringIds")) if "StringIds" in tables else {}
labels = sorted({(t if t is not None else strings.get(i)) for t, i in conn.execute("select text, textId from NVTX_EVENTS")} if "NVTX_EVENTS" in tables else set())
print("marks from the uid-1001 target in the uid-1000 report:", [l for l in labels if l and l.startswith("exp:")])
PYEOF
