#!/bin/bash
# C1 test 4 on a Docker host (no Kubernetes): run netproxy.py in the project image on a private bridge network
# and probe it from a second container with c1_net_tests.py's proxy cases. Shows destination, port, method and
# wildcard restrictions and that the allowed tunnel carries TLS to the allowed host only.
#   sandbox/k8s/c1/local/proxy_test.sh [IMAGE=goalbox-k3:dev]
set -euo pipefail
IMAGE=${1:-goalbox-k3:dev}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
NET=goalbox-c1-$$
docker network create "$NET" >/dev/null
trap 'docker rm -f c1-proxy-$$ >/dev/null 2>&1; docker network rm "$NET" >/dev/null 2>&1' EXIT
docker run -d --name "c1-proxy-$$" --network "$NET" --network-alias goalbox-proxy --read-only --tmpfs /var/run/goalbox \
  --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges "$IMAGE" \
  python3 /goalbox/sandbox/lib/netproxy.py /var/run/goalbox --tcp 3128 >/dev/null
sleep 2
docker run --rm --network "$NET" --user 1000:1000 --cap-drop ALL --read-only -v "$HERE/../c1_net_tests.py:/c1/c1_net_tests.py:ro" \
  -e C1_PROXY=goalbox-proxy:3128 "$IMAGE" python3 /c1/c1_net_tests.py --role agent --json | sed 's/^{.*/(json summary omitted)/' \
  | grep -E "proxy_|summary" || true
echo "== proxy decisions (net.log)"
docker exec "c1-proxy-$$" cat /var/run/goalbox/net.log
