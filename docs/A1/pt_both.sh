#!/bin/sh
# one 8-GPU lease: phase_table captures of the NO_PF and the PF (prefetch) instrumented builds (prebuilt)
cd /srv/ss/build/a1/scratch/seed_inst
export A1_NO_GPU_RUN=1
MK_FLAGS=-DNO_PF ./phase_table --launches 3 --group "moe=^verify L[0-9]+\|(M|N|O|P):" --out reports/pt3_nopf > /srv/ss/logs/a1pt3_nopf.log 2>&1
MK_FLAGS= ./phase_table --launches 3 --group "moe=^verify L[0-9]+\|(M|N|O|P):" --out reports/pt3_pf > /srv/ss/logs/a1pt3_pf.log 2>&1
