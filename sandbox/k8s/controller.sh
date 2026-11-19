#!/bin/bash
# Controller process (60-controller.yaml): GPU broker in k8s mode, trajectory stream capture, 48 h budget.
#
# - Broker: sandbox/lib/broker.py with GOALBOX_GPU_MODE=k8s on the agent's spool (/run-home/.gpu), copying
#   accepted requests into /ctl/gpu/<node>/req for the executors.
# - Stream capture (D-T1): `kubectl logs -f goalbox-agent -c agent` into /records/stream/agent-<n>.jsonl, restarted
#   if the stream drops (the file is append-only; the records verifier checks each is valid JSONL).
# - Budget (section 16): 30 min before GOALBOX_RUN_HOURS the controller writes /work/STOP-REQUESTED (the
#   agent's TASK.md tells it to stop experiments and commit within 15 min); the hard stop is the agent pod's
#   activeDeadlineSeconds (70-agent.yaml), enforced by Kubernetes, so no delete right is needed here.
# Everything the controller writes is in /ctl or /records, which the agent cannot mount.
set -euo pipefail
NS=${GOALBOX_NAMESPACE:-goalbox-k3}
RUN_HOME=/run-home
mkdir -p /records/stream /records/gpu /records/controller "$RUN_HOME/.gpu"/{req,status,out} /ctl/gpu
log() { echo "$(date -u +%FT%TZ) $*" | tee -a /records/controller/controller.log; }
log "controller start nodes=$GOALBOX_GPU_NODES hours=${GOALBOX_RUN_HOURS:-48}"
# The broker expects RUN_DIR/home/.gpu: give it a run dir whose home is the agent's home.
RUN_DIR=/tmp/run; mkdir -p "$RUN_DIR"; ln -sfn "$RUN_HOME" "$RUN_DIR/home"; mkdir -p "$RUN_DIR/gpu"
python3 /goalbox/sandbox/lib/broker.py "$RUN_DIR" &
BROKER=$!
# Mirror the GPU spool into the records volume every minute (D-T4: every GPU job and result).
( while sleep 60; do rsync -a --ignore-existing /ctl/gpu/ /records/gpu/ 2>/dev/null || cp -rn /ctl/gpu/. /records/gpu/ 2>/dev/null || true; done ) &
# Stream the agent's stdout (stream-json) into records; each reconnect starts a new numbered file.
( n=0; while :; do
    if kubectl -n "$NS" get pod goalbox-agent >/dev/null 2>&1; then
      n=$((n+1)); f=/records/stream/agent-$(printf %03d $n).jsonl
      log "streaming agent stdout to $f"
      kubectl -n "$NS" logs -f goalbox-agent -c agent >> "$f" 2>>/records/controller/stream-errors.log || true
      phase=$(kubectl -n "$NS" get pod goalbox-agent -o jsonpath='{.status.phase}' 2>/dev/null || echo unknown)
      log "stream ended (pod phase $phase)"
      case $phase in Succeeded|Failed) break ;; esac
    fi
    sleep 15
  done ) &
# Budget timer.
( start=$(date +%s); hours=${GOALBOX_RUN_HOURS:-48}
  warn=$((start + hours*3600 - 1800)); end=$((start + hours*3600))
  while [ "$(date +%s)" -lt "$warn" ]; do sleep 60; done
  echo "The 48-hour budget ends in 30 minutes: stop experiments, commit, and write the final PROGRESS.md entry." > /work/STOP-REQUESTED
  log "STOP-REQUESTED written"
  while [ "$(date +%s)" -lt "$end" ]; do sleep 60; done
  log "DEADLINE reached (Kubernetes activeDeadlineSeconds stops the agent pod)"; echo "$(date -u +%FT%TZ)" > /records/controller/DEADLINE ) &
wait $BROKER
