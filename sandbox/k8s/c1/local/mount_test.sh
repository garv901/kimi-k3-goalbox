#!/bin/bash
# C1 test 6 on a Docker host: reproduce the GPU worker pod's mount layout (request spool read-only, results
# read-write, harness read-only, weights read-only, no records) in the project image with the executor as pid 1
# (as in the pod), and let a "candidate" job try to (a) alter and inject requests, (b) change the grading code,
# (c) touch weights, (d) write its own results, (e) reach a records path. Only (d) and /work may succeed.
#   sandbox/k8s/c1/local/mount_test.sh [IMAGE=goalbox-k3:dev]
set -euo pipefail
IMAGE=${1:-goalbox-k3:dev}
T=$(mktemp -d); mkdir -p "$T"/{ctl/req/gpu/node-a/req,ctl/results,work,weights}
C=c1-mount-$$
cleanup() { docker rm -f "$C" >/dev/null 2>&1 || true; docker run --rm --user 1000:1000 -v "$T:/t" "$IMAGE" sh -c 'rm -rf /t/ctl/results/* /t/work/*' >/dev/null 2>&1 || true; rm -rf "$T"; }
trap cleanup EXIT
echo fake > "$T/weights/w.safetensors"
cat > "$T/ctl/req/gpu/node-a/req/j1.sh" <<'JOB'
set +e
echo "== candidate running as uid $(id -u) in $(pwd)"
try() { local what=$1; shift; if "$@" 2>/dev/null; then echo "WRITABLE  $what"; else echo "denied    $what"; fi; }
try "alter own request time limit"   sh -c 'echo x >> /ctl-req/gpu/node-a/req/j1.json'
try "inject a new request"           sh -c 'echo "{}" > /ctl-req/gpu/node-a/req/evil.json'
try "inject a request for node-b"    sh -c 'mkdir -p /ctl-req/gpu/node-b/req && echo "{}" > /ctl-req/gpu/node-b/req/evil.json'
try "modify grading code"            sh -c 'echo x >> /goalbox/bench/megabench/harness/correctness.py'
try "modify the oracle"              sh -c 'echo x >> /goalbox/bench/megabench/tasks/kimi.py'
try "modify the executor"            sh -c 'echo x >> /goalbox/sandbox/lib/executor.py'
try "modify weights"                 sh -c 'echo x >> /weights/w.safetensors'
try "write to /records"              sh -c 'mkdir -p /records && echo x > /records/x'
try "read the executor environment"  sh -c 'cat /proc/1/environ > /dev/null'
try "write own results (expected)"   sh -c 'echo forged > /ctl-results/gpu/node-a/out/forged.log'
try "write /work (expected)"         sh -c 'echo x > /work/x'
exit 0
JOB
echo '{"minutes": 1, "gpus": 1, "cwd": "/work"}' > "$T/ctl/req/gpu/node-a/req/j1.json"
chmod -R a+rX "$T"; chmod -R a+w "$T/ctl/results" "$T/work"
docker run -d --name "$C" --read-only --tmpfs /tmp --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
  -v "$T/ctl/req:/ctl-req:ro" -v "$T/ctl/results:/ctl-results" -v "$T/work:/work" -v "$T/weights:/weights:ro" \
  -e GOALBOX_NODE_NAME=node-a -e GOALBOX_GPUS_HERE=1 -e GOALBOX_EXECUTOR_KILL_ALL_UID=1 "$IMAGE" \
  python3 /goalbox/sandbox/lib/executor.py >/dev/null
for i in $(seq 90); do [ -e "$T/ctl/results/gpu/node-a/done/j1.json" ] && break; sleep 1; done
cat "$T/ctl/results/gpu/node-a/out/j1.log"
echo "done record: $(cat "$T/ctl/results/gpu/node-a/done/j1.json")"
echo "executor (pid 1) still running: $(docker inspect -f '{{.State.Running}}' "$C")"
echo "host view, request spool unchanged: $(ls "$T/ctl/req/gpu/node-a/req/" | tr '\n' ' ') $(ls "$T/ctl/req/gpu/" | tr '\n' ' ')"
grep -q "WRITABLE  alter\|WRITABLE  inject\|WRITABLE  modify\|WRITABLE  write to /records" "$T/ctl/results/gpu/node-a/out/j1.log" && { echo "FAIL: a protected path was writable"; exit 1; }
echo "RESULT: protected paths denied, own results and /work writable (as designed)"
