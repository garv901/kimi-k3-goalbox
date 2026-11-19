#!/bin/bash
# Operator-side (off-node) snapshot puller: no credential on the node. Every INTERVAL seconds copy a run's
# logs, snapshots (tarballs + index), reports and NOTES.md out of the container into a local git repo and
# commit. Run on the operator's machine (laptop) in tmux for the whole pilot; stop with Ctrl-C after result.json.
#   tools/pull_run.sh TRAY_SSH_HOST CONTAINER RUN_ID LOCAL_REPO [INTERVAL=300]
# e.g. tools/pull_run.sh pod4-gb300-10-tray18 ss p01 ~/Documents/workspace/scaffold-search-tp4/pilot-snapshots 300
set -u
TRAY=$1; CT=$2; RUN=$3; REPO=$4; IV=${5:-300}
mkdir -p "$REPO" && cd "$REPO" && { [ -d .git ] || git init -q -b main .; }
while true; do
  T=$(date -u +%FT%TZ)
  mkdir -p "runs/$RUN"
  ssh "$TRAY" "export HOME=/mnt/shared/home/dc; docker exec $CT tar -C /srv/ss/runs/$RUN -cf - --ignore-failed-read logs snapshots rescore/final.tgz result.json ws/NOTES.md ws/reports 2>/dev/null; docker exec $CT sh -c 'cat /srv/ss/lease.jsonl' > /dev/null" \
    | tar -x -C "runs/$RUN" 2>/dev/null
  ssh "$TRAY" "export HOME=/mnt/shared/home/dc; docker exec $CT cat /srv/ss/lease.jsonl /srv/ss/proxy.jsonl 2>/dev/null" > "runs/$RUN/node-lease-proxy.jsonl" 2>/dev/null
  git add -A >/dev/null 2>&1
  if git commit -qm "pull $RUN $T" 2>/dev/null; then echo "$T committed $(git rev-parse --short HEAD) ($(du -sh runs/$RUN | cut -f1))"; else echo "$T no change"; fi
  [ -f "runs/$RUN/result.json" ] && { echo "$T result.json present; final pull done"; break; }
  sleep "$IV"
done
