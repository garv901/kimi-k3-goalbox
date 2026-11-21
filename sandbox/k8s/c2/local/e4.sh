#!/bin/bash
# E4: profiler ownership in one container, process supervision in another, same uid, separate pid namespaces,
# shared /tmp (emptyDir model). The collector container runs nsys around a long-lived shell that exports the
# injection environment; the runner container starts the target with that environment. Questions: do the
# target's events reach the collector's report, and can the target reach the report file?
set -u
IMAGE=${1:-goalbox-k3:dev}; OUT=${2:-/tmp/e4-$$}; mkdir -p "$OUT/shared" "$OUT/report"; chmod 777 "$OUT/shared" "$OUT/report"
HERE=$(cd "$(dirname "$0")" && pwd); PY=/opt/goalbox/venv/bin/python
F='grep -v "NVIDIA\|CUDA Version\|license\|License\|=====\|Toolkit\|docs.nvidia\|developer.nvidia\|container image\|By pulling\|^$"'
docker rm -f e4-collector e4-runner >/dev/null 2>&1
# collector: own pid namespace, shared /tmp, private /report
docker run -d --name e4-collector --user 1000:1000 --cap-drop ALL -v "$OUT/shared":/tmp -v "$OUT/report":/report -v "$HERE":/exp:ro --entrypoint /bin/bash "$IMAGE" -c '
chmod 700 /report
nsys profile --trace=nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o /report/e4 \
  bash -c "env > /tmp/e4.env; echo ready > /tmp/e4.ready; while [ ! -e /tmp/e4.done ]; do sleep 1; done" ' >/dev/null
for i in $(seq 1 30); do [ -e "$OUT/shared/e4.ready" ] && break; sleep 1; done
echo "collector ready: $(test -e "$OUT/shared/e4.ready" && echo yes || echo no)"
# runner: same uid, own pid namespace, same shared /tmp, no /report
docker run --rm --name e4-runner --user 1000:1000 --cap-drop ALL -v "$OUT/shared":/tmp -v "$HERE":/exp:ro --entrypoint /bin/bash "$IMAGE" -c "
env -i \$(grep -v '^PWD=\|^SHLVL=\|^_=' /tmp/e4.env | tr '\n' ' ') $PY /exp/profiler_child.py /tmp/e4-target.json child 2>&1 | tail -1
echo 'runner: report reachable?'; ls /report 2>&1 | head -1; ls -la /tmp/nvidia/nsight_systems 2>/dev/null | head -4
" 2>&1 | eval $F
touch "$OUT/shared/e4.done"; sleep 6
docker logs e4-collector 2>&1 | eval $F | tail -3; docker rm -f e4-collector >/dev/null 2>&1
docker run --rm --user 1000:1000 -v "$OUT/report":/report -v "$OUT/shared":/tmp --entrypoint /bin/bash "$IMAGE" -c '
nsys export --type sqlite --force-overwrite true -o /report/e4.sqlite /report/e4.nsys-rep >/dev/null 2>&1; echo export=$?
/usr/bin/python3 - <<PYEOF
import sqlite3, json
c = sqlite3.connect("/report/e4.sqlite"); t = {r[0] for r in c.execute("select name from sqlite_master where type=\"table\"")}
s = dict(c.execute("select id, value from StringIds")) if "StringIds" in t else {}
m = sorted({(x if x is not None else s.get(i)) for x, i in c.execute("select text, textId from NVTX_EVENTS")} if "NVTX_EVENTS" in t else set())
print("=== E4: collector container + runner container, same uid, separate pid namespaces, shared /tmp")
print("  target record:", json.load(open("/tmp/e4-target.json"))["pid"] if __import__("os").path.exists("/tmp/e4-target.json") else None, "(pid in the runner namespace)")
print("  marks from the runner-container target in the collector report:", [x for x in m if x and x.startswith("exp:")])
print("  PROCESSES:", c.execute("select pid, name from PROCESSES").fetchall() if "PROCESSES" in t else None)
print("  diag:", [r[0][:100] for r in c.execute("select text from DIAGNOSTIC_EVENT") if "NVTX" in r[0] or "injection" in r[0].lower()])
PYEOF' 2>&1 | eval $F
