#!/bin/bash
# Build the goalbox image natively on an aarch64 Docker host (rack 3 trays have docker + buildx;
# no cluster privileges are needed). Prints the local image ID (sha256 of the image config) and
# writes it with the build arguments to sandbox/k8s/IMAGE-BUILD.txt. The registry digest exists
# only after a push, which needs a registry login that this script does not have.
#
#   sandbox/k8s/build-image.sh [TAG]        TAG default goalbox-k3:dev
#   CLAUDE_CODE_VERSION=2.1.322 TORCH_VERSION=2.14.1 (defaults in the Dockerfile)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TAG=${1:-goalbox-k3:dev}
[ "$(uname -m)" = aarch64 ] || { echo "build on an aarch64 host (got $(uname -m))"; exit 1; }
ARGS=()
for v in CLAUDE_CODE_VERSION TORCH_VERSION TORCH_INDEX NSYS_PKG NCU_PKG; do
  [ -n "${!v:-}" ] && ARGS+=(--build-arg "$v=${!v}")
done
docker buildx build --platform linux/arm64 --load -t "$TAG" "${ARGS[@]}" -f "$HERE/Dockerfile" "$REPO"
ID=$(docker inspect --format '{{.Id}}' "$TAG")
SIZE=$(docker inspect --format '{{.Size}}' "$TAG")
{
  echo "built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tag: $TAG"
  echo "image_id: $ID"
  echo "size_bytes: $SIZE"
  echo "labels: $(docker inspect --format '{{json .Config.Labels}}' "$TAG")"
  echo "base: $(grep -m1 '^FROM' "$HERE/Dockerfile" | cut -d' ' -f2)"
  echo "repo_digests: $(docker inspect --format '{{json .RepoDigests}}' "$TAG")   # local daemon value; the registry digest is known only after a push"
} | tee "$HERE/IMAGE-BUILD.txt"
