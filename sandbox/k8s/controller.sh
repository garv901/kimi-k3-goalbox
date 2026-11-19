#!/bin/bash
# Controller process (60-controller.yaml): GPU broker in k8s mode, GPU-spool mirroring, run-time budget.
#
# - Broker: sandbox/lib/broker.py with GOALBOX_GPU_MODE=k8s on the agent's spool (/run-home/.gpu), copying accepted
#   requests into /ctl/req/gpu/<node>/req (read-only in the GPU pods) and reading /ctl/results/gpu/<node>.
# - Mirror (D-T4): /ctl (requests and results) is copied into /records/gpu every minute, add-only.
# - Budget (section 16; C1 test 9): sandbox/lib/budget.py keeps a ledger of *run time* in /records/controller/
#   budget.json, persisted across controller restarts. A tick counts only while the agent pod is Running and
#   both GPU workers are Ready; otherwise the interval is recorded as an interruption and not counted. At
#   48 h - 30 min the controller writes /work/STOP-REQUESTED (TASK.md tells the agent to stop and commit
#   within 15 min); at 48 h it deletes the agent pod (its Role allows deleting exactly that pod). The agent
#   pod's activeDeadlineSeconds (72 h) is only a backstop.
# Everything the controller writes is in /ctl or /records, which the agent cannot mount.
set -euo pipefail
NS=${GOALBOX_NAMESPACE:-goalbox-k3}
RUN_HOME=/run-home
HOURS=${GOALBOX_RUN_HOURS:-48}
TICK=${GOALBOX_BUDGET_TICK_S:-30}
mkdir -p /records/gpu /records/controller "$RUN_HOME/.gpu"/{req,status,out} /ctl/req /ctl/results
log() { echo "$(date -u +%FT%TZ) $*" | tee -a /records/controller/controller.log; }
log "controller start nodes=$GOALBOX_GPU_NODES hours=$HOURS tick=${TICK}s"
RUN_DIR=/tmp/run; mkdir -p "$RUN_DIR/gpu"; ln -sfn "$RUN_HOME" "$RUN_DIR/home"
python3 /goalbox/sandbox/lib/broker.py "$RUN_DIR" &
BROKER=$!
( while sleep 60; do cp -rn /ctl/. /records/gpu/ 2>/dev/null || true; done ) &
# Budget ledger: one observation per tick.
( while :; do
    phase=$(kubectl -n "$NS" get pod goalbox-agent -o jsonpath='{.status.phase}' 2>/dev/null || echo unknown)
    ready=$(kubectl -n "$NS" get pod -l app.kubernetes.io/component=gpu-worker -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status}{" "}{end}' 2>/dev/null || echo "")
    workers_ready=$(echo "$ready" | tr ' ' '\n' | grep -c True || true)
    broker_ok=0; kill -0 $BROKER 2>/dev/null && broker_ok=1
    running=no; reason=""
    if [ "$phase" = Running ] && [ "$workers_ready" -ge 2 ] && [ $broker_ok = 1 ]; then running=yes
    else reason="agent=$phase workers_ready=$workers_ready broker=$broker_ok"; fi
    out=$(python3 /goalbox/sandbox/lib/budget.py /records/controller/budget.json --running $running --reason "$reason" --budget-hours "$HOURS")
    event=$(echo "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["event"] or "")')
    case $event in
      warn) echo "The run-time budget ends in 30 minutes: stop experiments, commit, and write the final PROGRESS.md entry." > /work/STOP-REQUESTED
            log "STOP-REQUESTED written ($out)" ;;
      stop) log "budget used up: deleting goalbox-agent ($out)"; echo "$(date -u +%FT%TZ)" > /records/controller/DEADLINE
            kubectl -n "$NS" delete pod goalbox-agent --grace-period=120 2>&1 | tee -a /records/controller/controller.log || true ;;
    esac
    case $phase in Succeeded|Failed) [ -e /records/controller/AGENT-ENDED ] || { log "agent pod $phase ($out)"; echo "$phase $(date -u +%FT%TZ)" > /records/controller/AGENT-ENDED; } ;; esac
    sleep "$TICK"
  done ) &
wait $BROKER
