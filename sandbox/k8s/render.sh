#!/bin/bash
# Render the manifests with the deployment-specific values that are not committed (node names, image
# references, host paths). Output: sandbox/k8s/rendered/ (gitignored). Nothing here talks to the cluster.
#
#   Phase C1 (namespace, RBAC, network, storage, proxy overlay, test pods; no GPU, no agent):
#     C1_IMAGE=<public arm64 python image> sandbox/k8s/render.sh c1
#   Full deployment:
#     IMAGE=... TRAY_A=... TRAY_B=... WEIGHTS_DIR=... HF_DIR=... CACHE_DIR=... sandbox/k8s/render.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-full}
OUT=$HERE/rendered/$MODE; rm -rf "$OUT"; mkdir -p "$OUT"
sub() { envsubst "$1" < "$2" > "$OUT/$(basename "$3")"; }
if [ "$MODE" = c1 ]; then
  : "${C1_IMAGE:?public arm64 image for the C1 proxy and test pods, e.g. docker.io/library/python:3.12-slim}"
  export C1_IMAGE
  for f in 00-namespace 10-rbac 20-network 30-storage; do sub '' "$HERE/manifests/$f.yaml" "$f.yaml"; done
  sub '$C1_IMAGE' "$HERE/c1/40-proxy-c1.yaml" 40-proxy-c1.yaml
  sub '$C1_IMAGE' "$HERE/c1/75-c1-test-pods.yaml" 75-c1-test-pods.yaml
else
  : "${IMAGE:?image reference, e.g. registry/goalbox-k3@sha256:...}"
  : "${TRAY_A:?node name of the first GPU tray}" "${TRAY_B:?node name of the second GPU tray}"
  : "${WEIGHTS_DIR:?host path of the converted weights}" "${HF_DIR:?host path of the HF snapshot}" "${CACHE_DIR:?host path for conversion output}"
  WORKERS=${WORKERS:-4}
  export IMAGE TRAY_A TRAY_B WEIGHTS_DIR HF_DIR CACHE_DIR WORKERS
  for f in "$HERE"/manifests/*.yaml; do
    name=$(basename "$f")
    case $name in
      90-conversion-job.yaml)
        for tray in "$TRAY_A" "$TRAY_B"; do
          short=$(echo "$tray" | tr -c 'a-z0-9\n' '-' | sed 's/-*$//' | tail -c 20)
          TRAY=$tray TRAY_SHORT=$short envsubst '$IMAGE $TRAY $TRAY_SHORT $HF_DIR $CACHE_DIR $WORKERS' < "$f" > "$OUT/90-conversion-job-$short.yaml"
        done ;;
      80-secret.yaml) ;;   # never rendered or applied: the operator creates the Secret directly
      *) sub '$IMAGE $TRAY_A $TRAY_B $WEIGHTS_DIR $HF_DIR $CACHE_DIR' "$f" "$name" ;;
    esac
  done
fi
( cd "$OUT" && sha256sum *.yaml ) > "$OUT/SHA256SUMS"
echo "rendered $MODE into $OUT"; cat "$OUT/SHA256SUMS"
