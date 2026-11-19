#!/bin/bash
# B-C1-1 regression on a Docker host: the GPU worker pod's two-container layout (grader uid 2000, runner
# uid 1000, separate pid namespaces, one shared "emptyDir") with the real grader, spawner, broker spool
# files and the hostile / killer control candidates, on the CPU.
#   sandbox/k8s/c1/local/verdict_test.sh [IMAGE=goalbox-k3:dev]
# Prints every ATTACK line the hostile candidate reported (all must be "denied") and the grader's verdicts.
set -euo pipefail
IMAGE=${1:-goalbox-k3:dev}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
T=$(mktemp -d); mkdir -p "$T"/{req/gpu/node-a/req,results,work,job}
chmod 777 "$T/job" "$T/results" "$T/work"
A=vt-grader-$$; B=vt-runner-$$
cleanup() { docker rm -f $A $B >/dev/null 2>&1 || true
            docker run --rm --user 0:0 -v "$T:/t" "$IMAGE" sh -c 'rm -rf /t/results/* /t/work/* /t/job/* /t/job/.[!.]* 2>/dev/null' >/dev/null 2>&1 || true
            rm -rf "$T"; }
trap cleanup EXIT
cp "$REPO"/megabench/tests/candidates/{oracle_candidate,hostile_candidate,killer_candidate}.py "$T/work/"
chmod 644 "$T/work"/*.py
GRADE_ARGS='["--trials","1","--warmup","0","--reps","1","--timeout","300","--no-scan","--tiny","--reference-reps","0"]'
req() {  # rid kind submission-or-script
  local rid=$1 kind=$2
  if [ "$kind" = grade ]; then
    printf '{"kind":"grade","minutes":3,"gpus":1,"cwd":"/tmp","topology":{},"case":"kimi-k3-dev-l4-tp1-b8-s8192","submission":"/work/%s","harness_args":%s,"device":"cpu"}\n' "$3" "$GRADE_ARGS" > "$T/req/gpu/node-a/req/.$rid.tmp"
  else
    cp "$3" "$T/req/gpu/node-a/req/$rid.sh"
    printf '{"kind":"exec","minutes":2,"gpus":1,"cwd":"/work","topology":{}}\n' > "$T/req/gpu/node-a/req/.$rid.tmp"
  fi
  mv "$T/req/gpu/node-a/req/.$rid.tmp" "$T/req/gpu/node-a/req/$rid.json"
}
wait_done() { for i in $(seq 240); do [ -e "$T/results/gpu/node-a/done/$1.json" ] && return 0; sleep 1; done; echo "TIMEOUT waiting for $1"; return 1; }
common=(--read-only --tmpfs /tmp:size=2g --cap-drop ALL --security-opt no-new-privileges -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/goalbox/bench -e OMP_NUM_THREADS=2)
docker run -d --name $A "${common[@]}" --user 2000:2000 -e HOME=/tmp -e GOALBOX_NODE_NAME=node-a -e GOALBOX_GPUS_HERE=1 -e GOALBOX_JOB_DIR=/job \
  -e GOALBOX_GRADER_TMP=/tmp -e GOALBOX_BENCH=/goalbox/bench -e GOALBOX_RUNNER_STALE_S=8 \
  -v "$T/req:/ctl-req:ro" -v "$T/results:/ctl-results" -v "$T/work:/work:ro" -v "$T/job:/job" "$IMAGE" python3 /goalbox/sandbox/lib/executor.py >/dev/null
start_runner() { docker run -d --name $B "${common[@]}" --user 1000:1000 -e HOME=/tmp -e GOALBOX_JOB_DIR=/job -e GOALBOX_SPAWNER_KILL_ALL_UID=1 \
  -v "$T/work:/work" -v "$T/job:/job" "$IMAGE" python3 /goalbox/sandbox/lib/spawner.py >/dev/null; }
start_runner
sleep 3
echo "== layout: grader uid $(docker exec $A id -u), runner uid $(docker exec $B id -u); runner sees ctl-results? $(docker exec $B sh -c 'test -e /ctl-results && echo yes || echo no'); runner sees grader pids? $(docker exec $B sh -c 'grep -l "[e]xecutor.py" /proc/[0-9]*/cmdline 2>/dev/null | wc -l')"

echo "== 1. exec job: candidate shell probes the boundary"
cat > "$T/attack.sh" <<'SH'
try() { local what=$1; shift; if "$@" 2>/dev/null; then echo "ATTACK $what: WRITABLE"; else echo "ATTACK $what: denied"; fi; }
try forge_done_record        sh -c 'echo "{\"exit\":0}" > /ctl-results/gpu/node-a/done/x.json'
try alter_pending_request    sh -c 'echo x > /ctl-req/gpu/node-a/req/x.json'
try modify_records           sh -c 'mkdir -p /records && echo x > /records/x'
try modify_evaluator_code    sh -c 'echo x >> /goalbox/bench/megabench/harness/correctness.py'
try write_job_root           sh -c 'mkdir /job/evil && echo x > /job/evil/x'
try alter_grader_pointer     sh -c 'echo evil > /job/GRADER'
try enter_grader_root        sh -c 'cd /job/$(cat /job/GRADER 2>/dev/null) && echo x > ./x'
try modify_own_job_spec      sh -c 'd=$(dirname "$0"); echo x > /job/*/*/job.json'
echo "visible processes: $(ls /proc | grep -c '^[0-9]')"
exit 0
SH
req exec-1 exec "$T/attack.sh"; wait_done exec-1
grep ATTACK "$T/results/gpu/node-a/out/exec-1.log" | sed 's/^/   /'; echo "   done: $(cat "$T/results/gpu/node-a/done/exec-1.json")"

echo "== 2. grade job: oracle (trusted pass)"
req grade-1 grade oracle_candidate.py; wait_done grade-1
echo "   done: $(cat "$T/results/gpu/node-a/done/grade-1.json")"
python3 -c "import json,sys; v=json.load(open(sys.argv[1])); r=v['record'] or {}; print('   verdict:', v['state'], 'tiny' if r.get('tiny_self_test') else '', 'snapshot', v['graded_snapshot'])" "$T/results/gpu/node-a/verdict/grade-1.json"
grep -q '"state": "correctness_only"' "$T/results/gpu/node-a/done/grade-1.json" || { echo "   grader log:"; sed 's/^/   | /' "$T/results/gpu/node-a/out/grade-1.log" | tail -40; }

echo "== 3. grade job: hostile candidate (attacks from inside the candidate process)"
req grade-2 grade hostile_candidate.py; wait_done grade-2
echo "   done: $(cat "$T/results/gpu/node-a/done/grade-2.json")"
cat "$T/results/gpu/node-a/candidate/grade-2/"*.stderr | grep ATTACK | sed 's/^/   /'
[ -e "$T/results/gpu/node-a/done/other-job.json" ] && echo "   FAIL: forged other-job record exists" || echo "   no forged records in the results spool"

echo "== 4. grade job: killer candidate (uid-wide SIGKILL from inside the runner)"
req grade-3 grade killer_candidate.py; wait_done grade-3
echo "   done: $(cat "$T/results/gpu/node-a/done/grade-3.json")"
grep KILLER "$T/results/gpu/node-a/candidate/grade-3/"*.stderr | head -3 | sed 's/^.*KILLER/   KILLER/'
echo "   runner container still running: $(docker inspect -f '{{.State.Running}}' $B); spawner (pid 1, signal-immune from inside) alive: $(docker exec $B sh -c 'grep -q spawner.py /proc/1/cmdline && echo yes || echo no')"
echo "   grader container still running: $(docker inspect -f '{{.State.Running}}' $A)"
req grade-4 grade oracle_candidate.py; wait_done grade-4
echo "   next job: $(cat "$T/results/gpu/node-a/done/grade-4.json")"

echo "== 5. runner unresponsive (paused): infrastructure outcome, restart request, recovery"
docker pause $B >/dev/null; sleep 10
req grade-5 grade oracle_candidate.py; wait_done grade-5
echo "   done: $(cat "$T/results/gpu/node-a/done/grade-5.json")"
echo "   restart requested: $(cat "$T/job/RESTART-RUNNER" 2>/dev/null || echo none)"
docker unpause $B >/dev/null
echo "   liveness probe command exit in runner: $(docker exec $B test ! -e /job/RESTART-RUNNER; echo $?)  (1 = kubelet would restart the container)"
docker restart $B >/dev/null; sleep 5
echo "   after restart, flag cleared: $(test -e "$T/job/RESTART-RUNNER" && echo no || echo yes)"
req grade-6 grade oracle_candidate.py; wait_done grade-6
echo "   done: $(cat "$T/results/gpu/node-a/done/grade-6.json")"
echo "== grader log tail"; tail -5 "$T/results/gpu/node-a/executor.log" | sed 's/^/   /'
