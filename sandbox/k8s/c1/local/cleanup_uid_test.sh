#!/bin/bash
# C1 test 7 inside a container (private pid namespace, executor as pid 1, like the GPU pod): the uid-wide sweep
# must kill a job's stray that escaped with setsid, and the executor must survive and run the next job.
#   sandbox/k8s/c1/local/cleanup_uid_test.sh [IMAGE=goalbox-k3:dev]
set -euo pipefail
IMAGE=${1:-goalbox-k3:dev}
T=$(mktemp -d); mkdir -p "$T"/{req/gpu/node-a/req,results}
C=c1-clean-$$
cleanup() { docker rm -f "$C" >/dev/null 2>&1 || true; docker run --rm --user 1000:1000 -v "$T:/t" "$IMAGE" sh -c 'rm -rf /t/results/*' >/dev/null 2>&1 || true; rm -rf "$T"; }
trap cleanup EXIT
printf 'setsid sleep 600 < /dev/null > /dev/null 2>&1 &\nsleep 1\necho "stray started: $(pgrep -fc "[s]leep 600") process(es) in their own session"\nexit 0\n' > "$T/req/gpu/node-a/req/j1.sh"
echo '{"minutes": 1, "gpus": 1, "cwd": "/tmp"}' > "$T/req/gpu/node-a/req/j1.json"
chmod -R a+rX "$T"; chmod -R a+w "$T/results"
docker run -d --name "$C" --read-only --tmpfs /tmp --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
  -v "$T/req:/ctl-req:ro" -v "$T/results:/ctl-results" -e GOALBOX_NODE_NAME=node-a -e GOALBOX_GPUS_HERE=1 \
  -e GOALBOX_EXECUTOR_KILL_ALL_UID=1 "$IMAGE" python3 /goalbox/sandbox/lib/executor.py >/dev/null
for i in $(seq 90); do [ -e "$T/results/gpu/node-a/done/j1.json" ] && break; sleep 1; done
echo "== job 1"; cat "$T/results/gpu/node-a/out/j1.log"; echo "done: $(cat "$T/results/gpu/node-a/done/j1.json")"
left=$(docker exec "$C" bash -c 'pgrep -fc "[s]leep 600" || true')
echo "stray processes left after the sweep: ${left:-0}"
echo "executor (pid 1) still running: $(docker inspect -f '{{.State.Running}}' "$C")"
printf 'echo "second job ran after the sweep"\n' > "$T/req/gpu/node-a/req/j2.sh"
echo '{"minutes": 1, "gpus": 1, "cwd": "/tmp"}' > "$T/req/gpu/node-a/req/j2.json"
for i in $(seq 90); do [ -e "$T/results/gpu/node-a/done/j2.json" ] && break; sleep 1; done
echo "== job 2"; cat "$T/results/gpu/node-a/out/j2.log"
[ "${left:-0}" = 0 ] && grep -q "second job ran" "$T/results/gpu/node-a/out/j2.log" && echo "RESULT: stray killed, executor survived, next job served" || { echo "FAIL"; exit 1; }
