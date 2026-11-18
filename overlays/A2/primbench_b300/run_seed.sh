#!/bin/bash
# Build every primbench family for one arch and run it, one GPU per job.
# Copy of megabench/primbench/run_seed.sh (vibe-megakernel @ dsv41-dspark-batch8), adapted for B300
# by the A2 overlay of scaffold-search:
#   - ARCH (default sm_103a = B300; sm_100a = B200) selects the -gencode target;
#   - on sm_103a tc_mma is built with -DPB_NO_I8 (ptxas: ".kind::i8 not supported on sm_103a");
#   - `build` and `run GROUP` are separate, so the four run groups can go to four 1-GPU jobs.
# Usage:
#   bash run_seed.sh build OUT_DIR                            # CPU only: compile into OUT_DIR/bin
#   GPU="./gpu_run -g 1 --" bash run_seed.sh run OUT_DIR g1   # groups g1..g4, or all
#   bash run_seed.sh rows OUT_DIR > rows.jsonl                # bench lines -> primdb rows
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cmd=${1:?usage: run_seed.sh build|run|rows OUT_DIR [GROUP]}
out=${2:?usage: run_seed.sh build|run|rows OUT_DIR [GROUP]}
ARCH=${ARCH:-sm_103a}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
GPU=${GPU:-}
ALL="mma_sync simt tc_mma tc_mma2 dequant tmem sync ingress mem gemm_small gemm_small_tc"

case "$cmd" in
build)
  mkdir -p "$out/bin"
  "$NVCC" --version | tail -1 > "$out/nvcc.txt"
  echo "$ARCH" > "$out/arch.txt"
  python3 "$here/gen_tmem.py" > "$out/tmem.cu"
  extra_tc=""; [ "$ARCH" = sm_103a ] && extra_tc="-DPB_NO_I8"
  pids=()
  for b in $ALL; do
    src="$here/$b.cu"; [ "$b" = tmem ] && src="$out/tmem.cu"
    x=""; [ "$b" = tc_mma ] && x="$extra_tc"
    "$NVCC" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" $x -I"$here" \
      -o "$out/bin/$b" "$src" -lcuda > "$out/build_$b.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  for b in $ALL; do [ -x "$out/bin/$b" ] || { echo "build failed: $b"; tail -5 "$out/build_$b.log"; }; done
  exit $rc ;;
run)
  group=${3:-all}
  case "$group" in
    g1) fams="tc_mma tc_overhead simt mma_sync" ;;
    g2) fams="tmem tc_mma2 dequant sync" ;;
    g3) fams="ingress mem" ;;
    g4) fams="gemm_small gemm_small_tc" ;;
    all) fams="simt mma_sync tc_mma tc_overhead tmem dequant tc_mma2 sync ingress mem gemm_small gemm_small_tc" ;;
    *) echo "unknown group $group"; exit 2 ;;
  esac
  for f in $fams; do
    t0=$(date +%s)
    if [ "$f" = tc_overhead ]; then
      $GPU "$out/bin/tc_mma" overhead > "$out/$f.jsonl"
    else
      $GPU "$out/bin/$f" > "$out/$f.jsonl"
    fi
    echo "$f: $(wc -l < "$out/$f.jsonl") lines in $(( $(date +%s) - t0 )) s"
  done
  $GPU nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader \
    > "$out/gpu.txt" 2>/dev/null || true ;;
rows)
  python3 "$here/to_rows.py" "$out" ;;
*) echo "unknown command $cmd"; exit 2 ;;
esac
