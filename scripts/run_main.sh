#!/bin/bash
# run_main.sh: the experiment schedule fixed in PROTOCOL.md.
#
#   bash scripts/run_main.sh start            # create the schedule (if absent) and start the orchestrator
#   bash scripts/run_main.sh status           # orchestrator log, per-run state, lease status
#   bash scripts/run_main.sh rescore RUN_ID   # finish a run whose agent phase completed but whose
#                                             # rescore was interrupted (no result.json)
#   bash scripts/run_main.sh stop             # stop the orchestrator and all runs (scripts/node_ops/killrun.sh)
#
# Schedule: arms A0S=4, A1=4, A2=4, A3=4; K=3 concurrent runs; seed 20261118; run ids r01..r16
# (waves: r01 A2, r02 A2, r03 A0S | r04 A1, r05 A3, r06 A2 | r07 A2, r08 A1, r09 A3 |
#  r10 A3, r11 A0S, r12 A3 | r13 A0S, r14 A0S, r15 A1 | r16 A1). Each run: ~2 h agent + ~15-20 min of
# calibration/rescore; ~14 h for all 16 (the orchestrator starts the next run as soon as one ends).
# Pushes (run branches, results, ledger) go to the snapshot repo if one is configured
# (/srv/ss/snapshot_repo, written by bootstrap_node.sh from SS_SNAPSHOT_REPO); otherwise the
# orchestrator and the runs get --no-push and everything stays under /srv/ss.
set -euo pipefail
SS=/srv/ss; H=$SS/harness; PY=/usr/bin/python3.12
SCHED=$SS/schedules/main.json; LOG=$SS/logs/orch-main.log
EXPECTED='r01:A2 r02:A2 r03:A0S r04:A1 r05:A3 r06:A2 r07:A2 r08:A1 r09:A3 r10:A3 r11:A0S r12:A3 r13:A0S r14:A0S r15:A1 r16:A1'
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
cd "$H"
push_args() { if [ -s $SS/snapshot_repo ]; then echo ""; else echo "--no-push"; fi; }

case "${1:-}" in
start)
  "$PY" scripts/verify_protocol.py --quiet --trusted $SS/trusted/vibe-megakernel || {
    echo "PROTOCOL manifest mismatch: refusing to start (runs would not be valid)" >&2; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "$H has uncommitted changes" >&2; exit 1; }
  $SS/bin/gpu_lease status >/dev/null || { echo "gpu_lease daemon down: bash scripts/bootstrap_node.sh" >&2; exit 1; }
  tmux has-session -t ss-proxy 2>/dev/null || { echo "proxy down: bash scripts/bootstrap_node.sh" >&2; exit 1; }
  if tmux has-session -t ss-orch-main 2>/dev/null; then echo "ss-orch-main already running"; exit 1; fi
  if [ ! -f $SCHED ]; then
    "$PY" -m scaffold_search.schedule --name main --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 \
      --prefix r --out $SCHED > /dev/null
  fi
  got=$("$PY" -c "import json; print(' '.join(r['run_id']+':'+r['arm'] for r in json.load(open('$SCHED'))['order']))")
  [ "$got" = "$EXPECTED" ] || { echo "schedule differs from the one in PROTOCOL.md: $got" >&2; exit 1; }
  P=$(push_args)
  RA=""; [ -n "$P" ] && RA="--run-args=--no-push"
  echo "starting orchestrator (K=3, 16 runs) $P; log $LOG"
  # stagger 90 s between starts; restartable: runs with a result.json are skipped,
  # run dirs without one are skipped too and reported in the log (inspect, then rerun under a new
  # id per PROTOCOL.md and log it in the ledger)
  tmux new -d -s ss-orch-main "cd $H && $PY -m scaffold_search.orchestrate --schedule $SCHED $P $RA >> $LOG 2>&1"
  sleep 5; tail -5 $LOG
  ;;
status)
  tail -20 $LOG 2>/dev/null || echo "no $LOG yet"
  echo; for d in $SS/runs/r[0-9]*; do
    [ -d "$d" ] || continue; r=$(basename "$d")
    if [ -f "$d/result.json" ]; then s=$("$PY" -c "import json; r=json.load(open('$d/result.json')); print('score', r['primary']['score'], 'L', r['primary']['L_ms'], r['primary']['source'], 'auc', r['auc'])")
    else s="running/incomplete: $(tail -1 "$d/logs/run_arm.log" 2>/dev/null | cut -c1-120)"; fi
    echo "$r $s"
  done
  echo; $SS/bin/gpu_lease status | "$PY" -c "
import json,sys; d=json.load(sys.stdin)
print('active:', [(a['run'], a['n'], int(a['held_s'])) for a in d['active']])
print('queue :', [(q['run'], q['n'], int(q['waited_s'])) for q in d['queue']])"
  ;;
rescore)
  RID=${2:?run id}; D=$SS/runs/$RID
  [ -f "$D/result.json" ] && { echo "$RID already has result.json"; exit 1; }
  "$PY" -c "import json,sys; r=json.load(open('$D/logs/run.json')); sys.exit(0 if r.get('t_end') else 1)" || {
    echo "$RID: the agent phase did not finish (no t_end in logs/run.json); PROTOCOL.md: rerun under a new id instead" >&2; exit 1; }
  "$PY" -m scaffold_search.rescore "$D" $(push_args)
  "$PY" -c "from scaffold_search import isolation; isolation.retire_user('$RID', '$D')"
  ;;
stop)
  tmux kill-session -t ss-orch-main 2>/dev/null || true
  for d in $SS/runs/r[0-9]*; do [ -f "$d/result.json" ] || bash scripts/node_ops/killrun.sh "$(basename "$d")"; done
  tmux kill-session -t ss-runs-main 2>/dev/null || true
  ;;
*) sed -n 2,19p "$0"; exit 2;;
esac
