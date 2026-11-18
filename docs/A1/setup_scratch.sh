#!/bin/sh
# Build scratch workspaces under /srv/ss/build/a1/scratch from the read-only run dir (never edits it).
set -e
S=/srv/ss/build/a1/scratch; O=/root/megabench-runs/20261117-dsv41-dspark-b8-claude-goal
OV=/srv/ss/build/a1/stage/overlay; D=/srv/ss/build/a1/stage/docs_A1
mkdir -p $S
for n in orig seed seed_inst; do
  rm -rf $S/$n; mkdir -p $S/$n
  (cd $O && tar cf - --exclude=.git --exclude=.experiment --exclude=reports --exclude=build --exclude=__pycache__ .) | (cd $S/$n && tar xf -)
  cp -a $OV/. $S/$n/
  cp $D/gpu_run_stub $S/$n/gpu_run
done
# seed = original sources with the build defaulting to -DNO_PF
for n in seed seed_inst; do
  sed -i "s/os.environ.get('MK_FLAGS', '')/os.environ.get('MK_FLAGS', '-DNO_PF')/g" $S/$n/mk_host.py
  grep -c "MK_FLAGS', '-DNO_PF'" $S/$n/mk_host.py
done
cd $S/seed_inst && python3 $D/instrument_seed.py
