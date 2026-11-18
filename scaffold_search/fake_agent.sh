#!/bin/bash
# Stand-in for `claude -p` in harness smoke tests (run as the run user, cwd = workspace).
#   fake_agent.sh work     submit the seed once through ./check_candidate, note it, then idle
#   fake_agent.sh summary  append a final summary to NOTES.md
set -u
case "${1:-work}" in
  work)
    echo '{"type":"system","subtype":"init","session_id":"fake-session","model":"fake"}'
    ./check_candidate > reports/fake_check.log 2>&1
    rc=$?
    p50=$(grep -o 'p50 [0-9.]* ms' reports/fake_check.log | head -1)
    printf '| fake | resubmitted the seed unchanged | rc=%s | %s | 1 |\n' "$rc" "$p50" >> NOTES.md
    git add NOTES.md && git commit -qm "fake agent: seed check" || true
    echo '{"type":"result","subtype":"success","total_cost_usd":0,"num_turns":1}'
    exec sleep 100000
    ;;
  summary)
    printf '\n## Final summary (fake agent)\nThe fake agent only re-checked the seed.\n' >> NOTES.md
    git add NOTES.md && git commit -qm "fake agent: summary" || true
    echo '{"type":"result","subtype":"success","total_cost_usd":0,"num_turns":1}'
    ;;
esac
