# C1b: exact operator commands (read-only checks, then the validation run)

Run from a shell with your own Rancher kubeconfig. Every command before the final one is read-only.
Checklist: `C1-OPERATOR-CHECKLIST.md`; rationale: `C1-OPERATOR-SAFETY-REVIEW.md`. Verified on
2026-11-19: the operator script `sandbox/k8s/c1/c1-validate.sh`, the C1 net tests and manifests 00, 10, 20,
30, 40-proxy-c1, 75 are byte-identical to the reviewed state at `7f8e025` (rendered hashes below).

## 1. Identity and impersonation
```
kubectl auth whoami
kubectl auth can-i create namespaces
kubectl auth can-i impersonate serviceaccounts
kubectl auth can-i impersonate groups
kubectl config view --minify -o jsonpath='{.users[0].user}' | grep -q token && echo "WARNING: kubeconfig carries an inline token (fine, but never pass it on the command line)"
```
Expected: your user (not a service account); `yes` to the three can-i lines.

## 2. Namespace absence
```
kubectl get namespace goalbox-k3 2>&1
kubectl get pv -o json | python3 -c 'import json,sys; print([p["metadata"]["name"] for p in json.load(sys.stdin)["items"] if p["spec"].get("claimRef",{}).get("namespace")=="goalbox-k3"] or "no PVs claim goalbox-k3")'
```
Expected: `Error from server (NotFound)` and `no PVs claim goalbox-k3`. Anything else: stop.

## 3. StorageClass reclaim policy and binding mode
```
kubectl get storageclass csi-cephfs-sc-k8s -o custom-columns='NAME:.metadata.name,PROVISIONER:.provisioner,RECLAIM:.reclaimPolicy,BINDING:.volumeBindingMode,EXPAND:.allowVolumeExpansion'
kubectl get storageclass -o custom-columns='NAME:.metadata.name,DEFAULT:.metadata.annotations.storageclass\.kubernetes\.io/is-default-class'
```
Record the output in the evidence. `Retain` means PVs outlive their PVCs (rollback must delete them by
name); `Delete` means PVC deletion frees the CephFS subvolume. `WaitForFirstConsumer` means the PVCs stay
Pending through C1 (expected, no pod mounts them in C1 except the test pods, which mount `work` only).

## 4. Rendered manifest hashes
```
cd <repo>; git rev-parse --short HEAD
C1_IMAGE=docker.io/library/python:3.12-slim sandbox/k8s/render.sh c1 >/dev/null
sha256sum -c reports/gb300/c1-security-prep-20261119/rendered-c1/SHA256SUMS --ignore-missing
```
Expected: `OK` for 00, 10, 20, 30 (40 and 75 depend on the image string you pin next).
Reviewed hashes (unpinned image): 00 `f8b9d084…`, 10 `0e4ba1ff…`, 20 `59a38f06…`, 30 `ab6a4aab…`.

## 5. The pinned C1 image
```
export C1_IMAGE=docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
kubectl get pods -A -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | grep -c 'python:3.12-slim' 
```
Expected: the digest above is the multi-arch index (arm64 included) read on 2026-11-19; at least 2 cluster
pods already run `python:3.12-slim`. The two test pods and the proxy use this image.

## 6. Resource creation and rollback boundaries (dry run, no mutation)
```
C1_IMAGE=$C1_IMAGE sandbox/k8s/render.sh c1 >/dev/null
grep -nE 'privileged: *true|hostPath|hostNetwork|hostPID|hostIPC|ClusterRole|kind: Secret|stringData|^data:' sandbox/k8s/rendered/c1/*.yaml || echo "no forbidden fields"
grep -h '^kind:' sandbox/k8s/rendered/c1/*.yaml | sort | uniq -c
kubectl apply --dry-run=client -f sandbox/k8s/rendered/c1/
kubectl apply --dry-run=server -f sandbox/k8s/rendered/c1/00-namespace.yaml
```
Expected kinds: Namespace 1, ServiceAccount 4, Role 1, RoleBinding 1, NetworkPolicy 4,
CiliumNetworkPolicy 1, PersistentVolumeClaim 3, ConfigMap 2, Service 1, Deployment 1, Pod 2. All
namespaced in `goalbox-k3` except the Namespace itself. Rollback (safety review §10) deletes only these,
plus PVs whose `claimRef.namespace` is `goalbox-k3` if the reclaim policy is `Retain`.

## 7. The validation run (the only mutating step)
```
C1_IMAGE=$C1_IMAGE sandbox/k8s/c1/c1-validate.sh all 2>&1 | tee /tmp/c1-run.log
ls reports/gb300/c1-security-*/
```
Hand back the evidence directory and the log. They contain your username and node names; redaction before
commit is the build agent's job. Leave the namespace and PVCs in place unless the run fails (then §10).
