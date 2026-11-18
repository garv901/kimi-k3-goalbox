# killrun.sh RUN_ID... : stop harness runs (tmux windows/sessions, user processes, leases)
for r in "$@"; do
  pkill -f "run_arm .*--run-id $r( |$)"; pkill -KILL -u ss-$r
  /srv/ss/bin/gpu_lease kill-run $r >/dev/null
done
