# C1 live validation: operator checklist

Companion to `C1-OPERATOR-SAFETY-REVIEW.md` (details and rationale there). Tick every line before
`sandbox/k8s/c1/c1-validate.sh all`.

**Kubernetes identity**
- [ ] `kubectl auth whoami` shows your own identity (Rancher); the kubeconfig comes from a file or context,
      never `KUBECTL="kubectl --token=..."` (the script logs `$KUBECTL`).
- [ ] `kubectl auth can-i impersonate serviceaccounts` → yes (the can-i matrix and the PVC probe need it).
- [ ] `kubectl auth can-i create namespaces` → yes.

**Namespace absence**
- [ ] `kubectl get ns goalbox-k3` → `NotFound`. If it exists, stop: the script would adopt it and the
      rollback would delete something this run did not create.

**StorageClass reclaim policy**
- [ ] `kubectl get sc csi-cephfs-sc-k8s -o jsonpath='{.reclaimPolicy} {.volumeBindingMode}'` exists.
      Note the values: `Retain` means the three PVs (900 GiB of requests) survive PVC deletion and must be
      deleted by name in a rollback; `WaitForFirstConsumer` means the PVCs stay Pending during C1 (fine).

**PVC ownership and cleanup**
- [ ] The three PVCs (`goalbox-work` 500Gi, `goalbox-ctl` 200Gi, `goalbox-records` 200Gi) are intended to
      stay after C1 for the GPU phase. Their PVs are cluster-scoped objects created by the provisioner.
- [ ] Rollback order if needed: pods → ConfigMaps → Deployment/Service → CiliumNetworkPolicy →
      NetworkPolicies → RoleBinding → Role → ServiceAccounts → PVCs (note PV names first) → Namespace →
      released PVs by name (review §10).

**Exact resources affected**
- [ ] Rendered set hashes match `reports/gb300/c1-security-prep-20261119/rendered-c1/SHA256SUMS` for
      `00`, `10`, `20`, `30` (the pinned `C1_IMAGE` changes `40` and `75` only).
- [ ] `grep -nE "privileged: *true|hostPath|hostNetwork|hostPID|hostIPC|ClusterRole|kind: Secret|stringData|^data:" sandbox/k8s/rendered/c1/*.yaml` prints nothing.
- [ ] Created in `goalbox-k3` only: Namespace, 4 SAs, 1 Role, 1 RoleBinding, 4 NetworkPolicies,
      1 CiliumNetworkPolicy, 3 PVCs, 2 ConfigMaps, proxy Service + Deployment, 2 test pods (deleted at the
      end). Read-only outside: `auth whoami`, `auth can-i`, `get nodes`.
- [ ] `C1_IMAGE=docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b`
      exported (reviewed digest; two cluster pods already run this image).

**Safe rollback**
- [ ] You have the §10 command list at hand; every command is `--ignore-not-found`; it touches nothing
      outside `goalbox-k3` except the Namespace itself and the PVs whose `claimRef.namespace` is
      `goalbox-k3`.

**Dry run first** (no mutation):
```
sandbox/k8s/render.sh c1 && kubectl apply --dry-run=client -f sandbox/k8s/rendered/c1/ && kubectl apply --dry-run=server -f sandbox/k8s/rendered/c1/00-namespace.yaml
```
**Then**: `C1_IMAGE=$C1_IMAGE sandbox/k8s/c1/c1-validate.sh all`, and hand back `reports/gb300/c1-security-<utc>/`
(raw: it contains your username and node names; the build agent redacts before commit).
