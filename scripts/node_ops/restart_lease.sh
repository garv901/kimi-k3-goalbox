# restart the gpu_lease daemon at a moment with no active lease and no queue (max 20 min; queued clients of the new version resubmit)
install -m 755 /srv/ss/harness/scaffold_search/gpu_lease.py /srv/ss/bin/gpu_lease.py
end=$(( $(date +%s) + 1200 ))
while [ $(date +%s) -lt $end ]; do
  if /srv/ss/bin/gpu_lease status | python3.12 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if not d['active'] else 1)"; then
    tmux kill-session -t ss-lease; sleep 0.3
    tmux new -d -s ss-lease "/srv/ss/bin/gpu_lease daemon 2>&1 | tee -a /srv/ss/logs/ss-lease.log"
    sleep 1; /srv/ss/bin/gpu_lease status >/dev/null && echo RESTARTED $(date -u +%FT%TZ); exit 0
  fi
  sleep 1
done
echo TIMEOUT; exit 1
