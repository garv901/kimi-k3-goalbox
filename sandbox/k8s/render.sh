#!/bin/bash
# Render the manifests with the deployment-specific values that are not committed (node names, the
# API server CIDR, the image reference, host paths). Output: sandbox/k8s/rendered/ (gitignored).
#
#   IMAGE=... TRAY_A=... TRAY_B=... API_SERVER_CIDR=... WEIGHTS_DIR=... HF_DIR=... CACHE_DIR=... \
#   sandbox/k8s/render.sh
#
# Nothing here talks to the cluster. Applying is a separate, authorized step (see k8s/README.md).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/rendered; mkdir -p "$OUT"
: "${IMAGE:?image reference, e.g. registry/goalbox-k3@sha256:...}"
: "${TRAY_A:?node name of the first GPU tray}" "${TRAY_B:?node name of the second GPU tray}"
: "${API_SERVER_CIDR:?API server address as a CIDR, e.g. x.x.x.x/32}"
: "${WEIGHTS_DIR:?host path of the converted weights}" "${HF_DIR:?host path of the HF snapshot}" "${CACHE_DIR:?host path for conversion output}"
WORKERS=${WORKERS:-4}
export IMAGE TRAY_A TRAY_B API_SERVER_CIDR WEIGHTS_DIR HF_DIR CACHE_DIR WORKERS
for f in "$HERE"/manifests/*.yaml; do
  name=$(basename "$f")
  case $name in
    90-conversion-job.yaml)
      for tray in "$TRAY_A" "$TRAY_B"; do
        TRAY=$tray TRAY_SHORT=$(echo "$tray" | tr -c 'a-z0-9\n' '-' | sed 's/-*$//' | tail -c 20) envsubst '$IMAGE $TRAY $TRAY_SHORT $HF_DIR $CACHE_DIR $WORKERS' < "$f" > "$OUT/90-conversion-job-$(echo "$tray" | tail -c 8).yaml"
      done ;;
    80-secret.yaml) ;;   # never rendered or applied: the operator creates the Secret directly
    *) envsubst '$IMAGE $TRAY_A $TRAY_B $API_SERVER_CIDR $WEIGHTS_DIR $HF_DIR $CACHE_DIR' < "$f" > "$OUT/$name" ;;
  esac
done
echo "rendered into $OUT"
