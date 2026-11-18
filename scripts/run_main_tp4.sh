#!/bin/bash
# run_main_tp4.sh: the TP4 protocol (PROTOCOL_TP4.md) on 4-GPU slots (one container per slot, bootstrapped with
# SS_PROTOCOL=PROTOCOL_TP4.md; SS_NGPU=4 is the default).
#
#   bash scripts/run_main_tp4.sh port [MINUTES]          # step 2: builder run builder-port1 (TP8 seed -> TP4)
#   bash scripts/run_main_tp4.sh freeze [ARGS...]        # step 3: scripts/freeze_seed.py on builder-port1
#   bash scripts/run_main_tp4.sh selftest                # step 4: run_selftest.sh with the TP4 arms and seed
#   bash scripts/run_main_tp4.sh start e4a --slot I      # E4a: this slot runs wave position I (0-3), K=1
#   bash scripts/run_main_tp4.sh start e4b               # E4b: A1 x4 + A3 x4, K=3 on this one slot
#   bash scripts/run_main_tp4.sh status [e4a|e4b]
#   bash scripts/run_main_tp4.sh rescore RUN_ID
#   bash scripts/run_main_tp4.sh stop
#
# E4a: 24 runs (tp4/A0S, A1, A2, A3 x 6), Latin-square schedule over 4 slots (seed 20261119, run ids t01..t24):
#   slot 0: t01 t05 t09 t13 t17 t21 | slot 1: t02 t06 ... t22 | slot 2: t03 ... t23 | slot 3: t04 ... t24
#   each slot runs its 6 runs one at a time (~2.3 h each, ~14 h). Waves (position = slot):
#   t01 A3, t02 A2, t03 A0S, t04 A1 | t05 A2, t06 A0S, t07 A1, t08 A3 | t09 A1, t10 A3, t11 A2, t12 A0S |
#   t13 A0S, t14 A1, t15 A3, t16 A2 | t17 A3, t18 A1, t19 A0S, t20 A2 | t21 A1, t22 A3, t23 A2, t24 A0S
# E4b: 8 runs on one slot, K=3 (seed 20261120, run ids c01..c08):
#   c01 A1, c02 A3, c03 A3 | c04 A1, c05 A3, c06 A1 | c07 A3, c08 A1   (~7 h)
set -euo pipefail
SS=/srv/ss; H=$SS/harness; PY=/usr/bin/python3.12
E4A='t01:tp4/A3 t02:tp4/A2 t03:tp4/A0S t04:tp4/A1 t05:tp4/A2 t06:tp4/A0S t07:tp4/A1 t08:tp4/A3 t09:tp4/A1 t10:tp4/A3 t11:tp4/A2 t12:tp4/A0S t13:tp4/A0S t14:tp4/A1 t15:tp4/A3 t16:tp4/A2 t17:tp4/A3 t18:tp4/A1 t19:tp4/A0S t20:tp4/A2 t21:tp4/A1 t22:tp4/A3 t23:tp4/A2 t24:tp4/A0S'
E4B='c01:tp4/A1 c02:tp4/A3 c03:tp4/A3 c04:tp4/A1 c05:tp4/A3 c06:tp4/A1 c07:tp4/A3 c08:tp4/A1'
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
cd "$H"
grep -qx 'SS_PROTOCOL=PROTOCOL_TP4.md' $SS/node.env 2>/dev/null && grep -qx 'SS_NGPU=4' $SS/node.env || {
  echo "this node is not bootstrapped for the TP4 protocol: SS_PROTOCOL=PROTOCOL_TP4.md bash scripts/bootstrap_node.sh" >&2; exit 1; }
export SS_PROTOCOL=PROTOCOL_TP4.md
push_args() { if [ -s $SS/snapshot_repo ]; then echo ""; else echo "--no-push"; fi; }
daemons_up() {
  $SS/bin/gpu_lease status >/dev/null || { echo "gpu_lease daemon down: bash scripts/bootstrap_node.sh" >&2; exit 1; }
  tmux has-session -t ss-proxy 2>/dev/null || { echo "proxy down: bash scripts/bootstrap_node.sh" >&2; exit 1; }
}

case "${1:-}" in
port)
  MIN=${2:-}
  daemons_up
  [ -d $SS/runs/builder-port1 ] && { echo "builder-port1 exists; for another attempt use: $PY -m scaffold_search.run_arm --arm tp4/PORT --run-id builder-port2 --no-calibrate --no-rescore" >&2; exit 1; }
  P=$(push_args)
  tmux new -d -s ss-port "cd $H && $PY -m scaffold_search.run_arm --arm tp4/PORT --run-id builder-port1 --no-calibrate --no-rescore ${MIN:+--budget-minutes $MIN} $P > $SS/logs/run-builder-port1.log 2>&1; echo \"RUN_ARM_EXIT \$?\" >> $SS/logs/run-builder-port1.log"
  echo "builder-port1 started in tmux ss-port; log $SS/logs/run-builder-port1.log"
  ;;
freeze)
  shift; "$PY" scripts/freeze_seed.py --run-dir $SS/runs/builder-port1 "$@"
  ;;
selftest)
  SS_ARMS_DIR=arms/tp4 SS_SEED=dsv41_b8_tp4_nopf bash scripts/run_selftest.sh "${@:2}"
  ;;
start)
  EXP=${2:?e4a or e4b}
  "$PY" scripts/verify_protocol.py --quiet --trusted $SS/trusted/vibe-megakernel || {
    echo "PROTOCOL_TP4.md manifest mismatch or TBD rows: refusing to start (runs would not be valid)" >&2; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "$H has uncommitted changes" >&2; exit 1; }
  daemons_up
  SCHED=$SS/schedules/$EXP.json
  case "$EXP" in
    e4a) [ "${3:-}" = --slot ] && [ -n "${4:-}" ] || { echo "usage: start e4a --slot I (I = 0..3)" >&2; exit 2; }
         SLOT=$4; WANT=$E4A; ORCH="--slots $SLOT"; NAME=e4a-slot${SLOT//,/_}
         [ -f $SCHED ] || "$PY" -m scaffold_search.schedule --name e4a --arms tp4/A0S=6,tp4/A1=6,tp4/A2=6,tp4/A3=6 \
           --k 4 --seed 20261119 --prefix t --latin --out $SCHED > /dev/null ;;
    e4b) WANT=$E4B; ORCH=""; NAME=e4b
         [ -f $SCHED ] || "$PY" -m scaffold_search.schedule --name e4b --arms tp4/A1=4,tp4/A3=4 --k 3 --seed 20261120 \
           --prefix c --out $SCHED > /dev/null ;;
    *) echo "unknown experiment $EXP" >&2; exit 2;;
  esac
  got=$("$PY" -c "import json; print(' '.join(r['run_id']+':'+r['arm'] for r in json.load(open('$SCHED'))['order']))")
  [ "$got" = "$WANT" ] || { echo "schedule differs from the one in PROTOCOL_TP4.md: $got" >&2; exit 1; }
  tmux has-session -t ss-orch-$NAME 2>/dev/null && { echo "ss-orch-$NAME already running"; exit 1; }
  P=$(push_args); RA=""; [ -n "$P" ] && RA="--run-args=--no-push"
  LOG=$SS/logs/orch-$NAME.log
  echo "starting orchestrator $NAME $ORCH $P; log $LOG"
  tmux new -d -s ss-orch-$NAME "cd $H && $PY -m scaffold_search.orchestrate --schedule $SCHED $ORCH $P $RA >> $LOG 2>&1"
  sleep 5; tail -5 $LOG
  ;;
status)
  for L in $SS/logs/orch-${2:-e4}*.log; do [ -f "$L" ] && { echo "== $L"; tail -8 "$L"; }; done
  echo; for d in $SS/runs/[tc][0-9]* $SS/runs/builder-port*; do
    [ -d "$d" ] || continue; r=$(basename "$d")
    if [ -f "$d/result.json" ]; then s=$("$PY" -c "import json; r=json.load(open('$d/result.json')); p=r['primary']; print(r['arm'], 'score', p['score'], 'L', p['L_ms'], p['source'], 'auc', r['auc'])")
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
    echo "$RID: the agent phase did not finish (no t_end in logs/run.json); rerun under a new id instead" >&2; exit 1; }
  "$PY" -m scaffold_search.rescore "$D" $(push_args)
  "$PY" -c "from scaffold_search import isolation; isolation.retire_user('$RID', '$D')"
  ;;
stop)
  for s in $(tmux ls -F '#S' 2>/dev/null | grep -E '^ss-(orch|runs)-e4|^ss-port$' || true); do tmux kill-session -t "$s"; done
  for d in $SS/runs/[tc][0-9]* $SS/runs/builder-port*; do [ -d "$d" ] && [ ! -f "$d/result.json" ] && bash scripts/node_ops/killrun.sh "$(basename "$d")"; done
  true
  ;;
*) sed -n 2,23p "$0"; exit 2;;
esac
