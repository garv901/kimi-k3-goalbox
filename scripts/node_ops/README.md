# node_ops: small operational scripts

- `killrun.sh RUN_ID...` - stop harness runs: their run_arm, every process of the run user, their leases.
- `restart_lease.sh` - reinstall `scaffold_search/gpu_lease.py` to /srv/ss/bin and restart the lease
  daemon (tmux `ss-lease`) at a moment with no active lease (waits up to 20 min). Queued clients
  resubmit; running leases are adopted.
