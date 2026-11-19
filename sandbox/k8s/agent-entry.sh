#!/bin/bash
# Entrypoint of the agent container (70-agent.yaml). Prepares the agent's home on first start, exports the
# credential from the mounted Secret file, writes the pinned Claude Code settings and runs the goal in a
# fresh headless session (DECISIONS section 16, DECISIONS-LOG L7). Stdout is the stream-json trajectory
# (D-T1); the controller streams it into the records volume. Nothing here is reachable by the agent
# before it starts; after that the agent runs as the same uid, which is the documented exposure boundary
# (DECISIONS-LOG L9).
set -euo pipefail
GOAL_FILE=${GOALBOX_GOAL_FILE:-/goalbox/goal/kimi-k3-tp8.txt}
HOME=${HOME:-/home/agent}
cd "$HOME"
if [ ! -e "$HOME/.goalbox-initialized" ]; then
  mkdir -p "$HOME/.claude" "$HOME/.gpu"/{req,status,out} /work
  # Pinned configuration (section 16): model, effort, no web tools, no subagent model override, auto mode.
  cat > "$HOME/.claude/settings.json" <<'JSON'
{
  "model": "claude-opus-5-5",
  "permissions": {"deny": ["WebSearch", "WebFetch"]},
  "env": {"CLAUDE_CODE_EFFORT_LEVEL": "high", "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
  "includeCoAuthoredBy": false
}
JSON
  if [ ! -e /work/TASK.md ]; then
    PYTHONPATH=/goalbox/bench /opt/goalbox/venv/bin/python -s -m megabench.integrations.make_kimi_workspace --out /work --bench /goalbox/bench
  fi
  date -u +%FT%TZ > "$HOME/.goalbox-initialized"
fi
unset CLAUDE_CODE_SUBAGENT_MODEL
if [ -r /goalbox/token ]; then
  CLAUDE_CODE_OAUTH_TOKEN=$(cat /goalbox/token); export CLAUDE_CODE_OAUTH_TOKEN
fi
export PATH=/work/.venv/bin:/goalbox/bin:/goalbox/sandbox/agent-tools:$PATH
[ -s "$GOAL_FILE" ] || { echo "goal file $GOAL_FILE missing" >&2; exit 2; }
SESSION_ID=${GOALBOX_SESSION_ID:-$(cat /proc/sys/kernel/random/uuid)}
# The stream-json trajectory goes to a pod-local emptyDir that the capture sidecar copies into the records,
# not to stdout: pod logs are readable by the shared cluster account (DECISIONS-LOG L14). Stdout keeps only
# these two marker lines.
STREAM_DIR=${GOALBOX_STREAM_DIR:-/stream}
STREAM=$STREAM_DIR/agent-$SESSION_ID-$(date -u +%Y%m%dT%H%M%SZ).jsonl
echo "{\"type\":\"goalbox\",\"event\":\"start\",\"session_id\":\"$SESSION_ID\",\"utc\":\"$(date -u +%FT%TZ)\",\"claude_code\":\"$(claude --version 2>/dev/null | head -1)\",\"stream\":\"$(basename "$STREAM")\"}"
set +e
cd /work
claude -p "/goal $(cat "$GOAL_FILE")" \
  --output-format stream-json --verbose --include-partial-messages --forward-subagent-text \
  --permission-mode auto --model claude-opus-5-5 --effort high --session-id "$SESSION_ID" \
  --disallowedTools "WebSearch,WebFetch" \
  ${GOALBOX_MAX_BUDGET_USD:+--max-budget-usd "$GOALBOX_MAX_BUDGET_USD"} \
  >> "$STREAM" 2>> "$STREAM_DIR/agent-stderr.txt"
code=$?
set -e
echo "{\"type\":\"goalbox\",\"event\":\"exit\",\"code\":$code,\"utc\":\"$(date -u +%FT%TZ)\"}"
touch "$HOME/.goalbox-finished"
sleep 90   # let the capture sidecar take its final pass before the pod completes
exit $code
