#!/bin/bash
# Phase C1: apply the namespace scaffolding and run the security validation. RUN BY THE OPERATOR with an
# identity that may create a namespace (the build agent's credentials cannot: see c1 README). Every kubectl
# call is printed before it runs. Nothing GPU-related, no Secret, no agent pod.
#
#   C1_IMAGE=docker.io/library/python:3.12-slim sandbox/k8s/c1/c1-validate.sh [apply|test|cleanup|all]
#
# Optional: KUBECTL="kubectl --context=..." ; HTTPS_PROXY for a Rancher socks path is honoured by kubectl.
# Evidence goes to $EVIDENCE (default reports/gb300/c1-security-<utc>/), raw (the build agent redacts it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(cd "$HERE/.." && pwd); REPO=$(cd "$K8S/../.." && pwd)
NS=goalbox-k3
KUBECTL=${KUBECTL:-kubectl}
STEP=${1:-all}
EVIDENCE=${EVIDENCE:-$REPO/reports/gb300/c1-security-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$EVIDENCE"
run() { echo "+ $*" | tee -a "$EVIDENCE/commands.log"; "$@" 2>&1 | tee -a "$EVIDENCE/commands.log"; return "${PIPESTATUS[0]}"; }
k() { run $KUBECTL "$@"; }

apply() {
  : "${C1_IMAGE:?set C1_IMAGE}"
  C1_IMAGE=$C1_IMAGE "$K8S/render.sh" c1 | tee "$EVIDENCE/render.log"
  cp "$K8S/rendered/c1/"*.yaml "$K8S/rendered/c1/SHA256SUMS" "$EVIDENCE/"
  echo "== pre-apply inspection (expect: no privileged, no hostPath, no Secret data, no cluster-scoped RBAC)"
  grep -nE "privileged: *true|hostPath|hostNetwork|hostPID|hostIPC|ClusterRole|kind: Secret|stringData|^data:" "$K8S/rendered/c1/"*.yaml | tee "$EVIDENCE/pre-apply-grep.txt" || echo "(none)"
  python3 "$K8S/check_manifests.py" "$K8S/manifests" | tee "$EVIDENCE/check_manifests.json"
  echo "== whoami"; k auth whoami | tee "$EVIDENCE/whoami.txt"
  k apply -f "$K8S/rendered/c1/00-namespace.yaml"
  k apply -f "$K8S/rendered/c1/10-rbac.yaml"
  k apply -f "$K8S/rendered/c1/20-network.yaml"
  k apply -f "$K8S/rendered/c1/30-storage.yaml"
  k -n $NS create configmap goalbox-proxy-src --from-file=netproxy.py="$REPO/sandbox/lib/netproxy.py" \
      --from-file=blocked-packages.txt="$REPO/sandbox/blocked-packages.txt" --dry-run=client -o yaml | $KUBECTL apply -f -
  k -n $NS create configmap goalbox-c1-tests --from-file=c1_net_tests.py="$HERE/c1_net_tests.py" --dry-run=client -o yaml | $KUBECTL apply -f -
  k apply -f "$K8S/rendered/c1/40-proxy-c1.yaml"
  k apply -f "$K8S/rendered/c1/75-c1-test-pods.yaml"
  echo "== admission outcomes (any rejection appears above as an apply error; events below)"
  k -n $NS get events --sort-by=.lastTimestamp | tail -40 | tee "$EVIDENCE/events-after-apply.txt"
  k -n $NS wait --for=condition=Ready pod/c1-agentlike pod/c1-unlabeled --timeout=300s || true
  k -n $NS wait --for=condition=Available deployment/goalbox-proxy --timeout=300s || true
  k -n $NS get all,netpol,cnp,pvc,sa,role,rolebinding -o wide | tee "$EVIDENCE/resources.txt"
  k -n $NS get pvc | tee "$EVIDENCE/pvc.txt"
}

test_rbac() {
  echo "== RBAC can-i matrix (test 1, 2, 5)" | tee "$EVIDENCE/rbac-matrix.txt"
  local subjects=("system:serviceaccount:inference-wideep:wideep-session" "system:serviceaccount:$NS:goalbox-agent" \
                  "system:serviceaccount:$NS:goalbox-gpu" "system:serviceaccount:$NS:goalbox-proxy" "system:serviceaccount:$NS:goalbox-controller")
  local verbs=("get pods" "list pods" "watch pods" "create pods" "delete pods" "delete pods --subresource= " "patch pods" "create pods/exec" "get pods/log" \
               "get secrets" "list secrets" "create secrets" "get persistentvolumeclaims" "create persistentvolumeclaims" "delete persistentvolumeclaims" \
               "create deployments.apps" "patch deployments.apps" "create jobs.batch" "create networkpolicies.networking.k8s.io" "delete networkpolicies.networking.k8s.io" \
               "create ciliumnetworkpolicies.cilium.io" "create roles.rbac.authorization.k8s.io" "create rolebindings.rbac.authorization.k8s.io" "create serviceaccounts" \
               "create configmaps" "patch configmaps" "get events")
  printf '%-48s' "verb (in $NS)"; for s in "${subjects[@]}"; do printf '%-14s' "${s##*:}"; done; echo
  for v in "${verbs[@]}"; do
    [ "$v" = "delete pods --subresource= " ] && continue
    printf '%-48s' "$v"
    for s in "${subjects[@]}"; do printf '%-14s' "$($KUBECTL auth can-i $v -n $NS --as="$s" 2>/dev/null || echo err)"; done; echo
  done | tee -a "$EVIDENCE/rbac-matrix.txt"
  echo "controller may delete only goalbox-agent:" | tee -a "$EVIDENCE/rbac-matrix.txt"
  for name in goalbox-agent goalbox-gpu-a goalbox-controller c1-agentlike; do
    printf '  delete pod %-20s %s\n' "$name" "$($KUBECTL auth can-i delete pods/$name -n $NS --as=system:serviceaccount:$NS:goalbox-controller 2>/dev/null)"
  done | tee -a "$EVIDENCE/rbac-matrix.txt"
  echo "shared account outside its namespace (cluster-wide read scope, for the record):" | tee -a "$EVIDENCE/rbac-matrix.txt"
  for v in "list pods --all-namespaces" "get pods/log -n $NS" "get nodes" "create namespaces"; do
    printf '  %-36s %s\n' "$v" "$($KUBECTL auth can-i $v --as=system:serviceaccount:inference-wideep:wideep-session 2>/dev/null)"
  done | tee -a "$EVIDENCE/rbac-matrix.txt"
}

test_network() {
  echo "== in-pod network tests (tests 3, 4)"
  local node_ip; node_ip=$($KUBECTL get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo 10.13.84.1)
  k -n $NS exec c1-agentlike -- env C1_NODE_IP="$node_ip" python3 /c1/c1_net_tests.py --role agent --json | tee "$EVIDENCE/net-agentlike.txt" || true
  k -n $NS exec c1-unlabeled -- env C1_NODE_IP="$node_ip" python3 /c1/c1_net_tests.py --role unlabeled --json | tee "$EVIDENCE/net-unlabeled.txt" || true
  echo "== proxy log (allow/deny decisions)"
  local proxy_pod; proxy_pod=$($KUBECTL -n $NS get pod -l app.kubernetes.io/component=proxy -o jsonpath='{.items[0].metadata.name}')
  k -n $NS exec "$proxy_pod" -- cat /var/run/goalbox/net.log | tee "$EVIDENCE/proxy-net.log.txt" || true
  echo "== PVC access from an agent-like pod is impossible by construction (no API identity): shown by the can-i matrix;"
  echo "   additionally try to mount the records PVC from a pod created AS the agent ServiceAccount:"
  $KUBECTL -n $NS --as=system:serviceaccount:$NS:goalbox-agent apply -f - <<YAML 2>&1 | tee "$EVIDENCE/pvc-mount-as-agent.txt" || true
apiVersion: v1
kind: Pod
metadata: {name: c1-records-grab, namespace: $NS}
spec:
  restartPolicy: Never
  containers: [{name: x, image: "$C1_IMAGE", command: [sleep, "10"], volumeMounts: [{name: r, mountPath: /records}]}]
  volumes: [{name: r, persistentVolumeClaim: {claimName: goalbox-records}}]
YAML
  echo "== admission: a privileged pod and a hostPath pod attempted by the operator (expect rejection or a recorded acceptance)"
  $KUBECTL -n $NS apply --dry-run=server -f - <<YAML 2>&1 | tee "$EVIDENCE/admission-privileged.txt" || true
apiVersion: v1
kind: Pod
metadata: {name: c1-priv-probe, namespace: $NS}
spec:
  restartPolicy: Never
  containers: [{name: x, image: "$C1_IMAGE", command: [sleep, "1"], securityContext: {privileged: true}}]
YAML
  $KUBECTL -n $NS apply --dry-run=server -f - <<YAML 2>&1 | tee "$EVIDENCE/admission-hostpath.txt" || true
apiVersion: v1
kind: Pod
metadata: {name: c1-hostpath-probe, namespace: $NS}
spec:
  restartPolicy: Never
  containers: [{name: x, image: "$C1_IMAGE", command: [sleep, "1"], volumeMounts: [{name: h, mountPath: /host}]}]
  volumes: [{name: h, hostPath: {path: /, type: Directory}}]
YAML
  k -n $NS get events --sort-by=.lastTimestamp | tail -60 | tee "$EVIDENCE/events-after-tests.txt"
}

cleanup() {
  k -n $NS delete pod c1-agentlike c1-unlabeled c1-records-grab --ignore-not-found --wait=false
  k -n $NS delete configmap goalbox-c1-tests --ignore-not-found
  echo "proxy Deployment, PVCs, policies, RBAC and the namespace stay for the next phase."
}

case $STEP in
  apply) apply ;;
  test) test_rbac; test_network ;;
  cleanup) cleanup ;;
  all) apply; test_rbac; test_network; cleanup ;;
  *) echo "usage: $0 [apply|test|cleanup|all]"; exit 2 ;;
esac
echo "evidence in $EVIDENCE"
