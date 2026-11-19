# C1 operator safety review: `sandbox/k8s/c1/c1-validate.sh`

Read-only review at commit `be8905a` (2026-11-19) of the script the operator runs with their own cluster
identity. Nothing in this review changed the cluster; the only cluster calls made for it were `kubectl auth
can-i`, `get ns`, `api-resources`, `version` and a pod-image listing with the shared account's token.

## 1. Resources the script creates, updates or deletes

All through `kubectl apply` (create-or-update) unless noted. Rendered sources and sha256 sums:
`reports/gb300/c1-security-prep-20261119/rendered-c1/`.

| step | kind | name | notes |
|---|---|---|---|
| apply 00 | Namespace | `goalbox-k3` | the only cluster-scoped object applied |
| apply 10 | ServiceAccount x4 | `goalbox-controller`, `goalbox-agent`, `goalbox-gpu`, `goalbox-proxy` | all `automountServiceAccountToken: false` |
| apply 10 | Role | `goalbox-controller` | pods get/list/watch; delete on `resourceNames: [goalbox-agent]` only |
| apply 10 | RoleBinding | `goalbox-controller` | Role above to SA `goalbox-controller`; no ClusterRole, no ClusterRoleBinding |
| apply 20 | NetworkPolicy x4 | `default-deny`, `agent`, `proxy`, `gpu-workers` | select pods in `goalbox-k3` only |
| apply 20 | CiliumNetworkPolicy | `controller` | egress to entity `kube-apiserver` (6443/443) and kube-dns:53 |
| apply 30 | PersistentVolumeClaim x3 | `goalbox-work` 500Gi, `goalbox-ctl` 200Gi, `goalbox-records` 200Gi | RWX, class `csi-cephfs-sc-k8s` |
| create+apply | ConfigMap | `goalbox-proxy-src` | `netproxy.py`, `blocked-packages.txt` (code, not secret) |
| create+apply | ConfigMap | `goalbox-c1-tests` | `c1_net_tests.py` |
| apply 40 | Service | `goalbox-proxy` | ClusterIP 3128 |
| apply 40 | Deployment (1 replica) | `goalbox-proxy` | public python image + ConfigMap (L17); uid 1000, caps dropped, read-only root |
| apply 75 | Pod x2 | `c1-agentlike`, `c1-unlabeled` | `sleep 7200`, `activeDeadlineSeconds: 7200`, no GPU, SA `goalbox-agent` |
| test | Pod (attempted, expected **Forbidden**) | `c1-records-grab` | created with `--as=system:serviceaccount:goalbox-k3:goalbox-agent`; mounts `goalbox-records` |
| test | Pod x2 `--dry-run=server` (never persisted) | `c1-priv-probe`, `c1-hostpath-probe` | admission probes only |
| cleanup | **delete** Pod x3 | `c1-agentlike`, `c1-unlabeled`, `c1-records-grab` | `--ignore-not-found` |
| cleanup | **delete** ConfigMap | `goalbox-c1-tests` | |

Indirect, cluster-scoped: the CSI provisioner creates **three PersistentVolumes** for the PVCs (names
`pvc-<uuid>`). They are not applied by the script but exist because of it; see §10.

No `patch`, `replace`, `edit`, `scale`, `label` or `annotate` commands are present.

## 2. Namespaces affected

Every applied object carries `namespace: goalbox-k3`; the three ConfigMap commands, the exec, wait, get,
delete and dry-run commands all use `-n goalbox-k3`. Outside the namespace the script only **reads**:
`kubectl auth whoami`, `auth can-i` (several impersonated subjects, including
`system:serviceaccount:inference-wideep:wideep-session`; `can-i` with `--as` authorizes nothing and reads
nothing), `get nodes` (one InternalIP, used as a forbidden-destination test target). The Namespace object
and the dynamically provisioned PVs are the only cluster-scoped state it leaves behind. Nothing in
`inference-wideep`, `inference-bench`, `kube-system` or any other namespace is touched.

## 3. Commands whose effect depends on pre-existing cluster state

- **`apply 00-namespace.yaml`**: if `goalbox-k3` already existed, `apply` would adopt it and the later
  rollback (§10) would delete a namespace this run did not create. Verified 2026-11-19 (read-only): the
  namespace does **not** exist. Pre-check before running: `kubectl get ns goalbox-k3` must print NotFound.
- **`apply 10/20/30/40`**: `apply` overwrites same-named objects. With the namespace absent none can exist.
- **`apply 20-network.yaml`** needs the CRD `ciliumnetworkpolicies.cilium.io/v2`: verified present
  (`kubectl api-resources`). Without it the four NetworkPolicies would apply and the CNP would fail,
  stopping the script (`set -e`) with a partial namespace (§4).
- **`apply 30-storage.yaml`** needs StorageClass `csi-cephfs-sc-k8s`. **Unverified**: the shared account
  may not read StorageClasses. The manifest comment claims reclaim policy Retain; also unverified. If the
  class binds immediately, 900 GiB of CephFS quota is allocated at apply time; if it waits for the first
  consumer, the PVCs stay Pending during C1 (no pod mounts them), which the script tolerates. Pre-check:
  `kubectl get sc csi-cephfs-sc-k8s -o yaml` (operator identity).
- **Image pulls** (`docker.io/library/python:3.12-slim`): nodes must reach Docker Hub anonymously.
  Verified: two running pods in the cluster already use exactly this image (§8).
- **`auth can-i --as=...`** requires the operator's identity to hold `impersonate` on serviceaccounts
  cluster-wide; otherwise the matrix prints `err`/Forbidden and must be rerun with a permitted identity.
  The `--as=...goalbox-agent apply` PVC probe has the same dependency (§6).
- **`--dry-run=server`** admission probes pass through the admission chain (Kyverno, KEDA webhooks). They
  persist nothing; Kyverno honours dryRun for validate/mutate and does not run generate rules on it.
- **`kubectl exec`** through the Rancher proxy can drop long streams. The agent-role test takes up to about
  2 min (29 probes, 6 s timeouts). If the exec is cut, the `|| true` keeps the script going and the
  evidence file is truncated; rerun `c1-validate.sh test` after `apply` finished (idempotent, read-only).
- **`get nodes` InternalIP**: read-only; used only as a destination that must be blocked.
- **Error handling**: `set -euo pipefail`; an `apply` failure aborts before later mutations; `wait`,
  `exec`, dry-runs and the PVC probe are `|| true` so tests continue and record their failure.

## 4. Cleanup scope and what remains

`cleanup` deletes only `c1-agentlike`, `c1-unlabeled`, `c1-records-grab` (should never exist) and ConfigMap
`goalbox-c1-tests`, all in `goalbox-k3`, with `--ignore-not-found`. It deletes nothing it did not create.

After **successful completion**, the namespace keeps the scaffolding intended for the next phase:
Namespace, 4 SAs, Role, RoleBinding, 4 NetworkPolicies, 1 CiliumNetworkPolicy, 3 PVCs (+3 PVs), ConfigMap
`goalbox-proxy-src`, Service and Deployment `goalbox-proxy` (one running pod). No GPU, no Secret, no agent.

After a **partial failure** (script aborted by `set -e`): whatever `apply` steps completed remain, in
manifest order, and test pods may be left running (they self-terminate after 2 h via
`activeDeadlineSeconds`; the pod objects stay until deleted). `c1-validate.sh cleanup` is safe to run at any
point; the full rollback is §10.

## 5. Secrets and operator credentials

- No command reads, lists, creates, copies or prints a Secret. The only Secret-related calls are
  `auth can-i get/list/create secrets --as=<subject>`, which return yes/no and no data.
- Neither ConfigMap contains credentials (proxy source, blocked-package list, test script).
- The operator's kubeconfig is read by `kubectl` only; the script never echoes `$KUBECONFIG`, tokens or
  `HTTPS_PROXY`. **Unsafe assumption to avoid**: do not set `KUBECTL="kubectl --token=..."`; the `run`
  wrapper echoes `$KUBECTL` into `commands.log`. Use a kubeconfig/context instead.
- `kubectl auth whoami` writes the operator's **username and groups** (not a credential) to `whoami.txt` and
  `commands.log`; `events-*.txt`, `resources.txt` and `net-*.txt` contain node names and addresses. The
  evidence directory is **raw**: the build agent redacts it before commit (host addresses, user names).

## 6. Records-PVC access test: what it actually proves

The probe `kubectl --as=system:serviceaccount:goalbox-k3:goalbox-agent apply` of a pod that mounts
`goalbox-records` is rejected at **authorization** (`create pods` denied), before admission. It therefore
demonstrates the RBAC layer only. **It does not exercise a mount restriction, and none exists**: Kubernetes
has no built-in rule tying PVC mounts to ServiceAccounts, and this project adds no admission policy for it.
Any identity that can create pods in `goalbox-k3` (the operator; not the controller, whose Role has no
`create`) can mount the records PVC.

The agent-controlled workloads cannot mount it because they cannot create pods at all, enforced by three
independent layers, each tested live by the script:

1. no ServiceAccount token is mounted in the agent, GPU and proxy pods (`api_token_absent`);
2. the API server is unreachable from them (`kubernetes_api`, `kubernetes_api_6443` blocked by the
   default-deny policy; the CiliumNetworkPolicy opens it for the controller only);
3. RBAC: the SAs have no bindings (`can-i` matrix, all `no`).

Recommended addition before relying on this (not yet in the script): a positive control showing that a pod
created **by the operator** with `serviceAccountName: goalbox-agent` and the records PVC *does* get
admitted, so nobody later assumes an admission layer exists. Decision for the operator: accept the
three-layer model (recommended, consistent with DECISIONS section 3) or add a Kyverno policy restricting
`goalbox-records` to pods with the controller's SA, which would be the first project-specific admission
rule and needs cluster-admin rights.

## 7. Independent verification of GPU-job outcomes: BLOCKER for autonomous execution

The controller's broker reads the executor's `done/<id>.json` and `out/<id>.log` from the results half of
the control volume, which is **writable by candidate code** (same uid in the GPU pod; the request half is
read-only, DECISIONS-LOG L13). The harness (`runner.py`) also runs in that pod as that uid; its verdict is
written by the job itself. The broker treats results as untrusted *format* (malformed records become
`MALFORMED_RESULT`, strings are truncated) but it cannot verify their *content*: a candidate can forge a
`correct`/timing result for its own job, or delete and rewrite the log. The add-only mirror of the spool
into records (D-T4) time-stamps the forgery but does not detect it; the nsys audit runs inside the same job.

What is independently recorded today: the request (read-only), the controller's dispatch log, the mirror,
the agent's transcript and stream. What is **not**: a correctness/timing verdict that the candidate could
not have written. Recorded as **blocker B-C1-1**: results reported to the agent during the run are
advisory; the run's conclusion cannot rest on them. Closure options, for the operator's decision before
the GPU phase:

- **(a) Post-run independent grading** (minimum): after the agent stops, the operator submits the agent's
  final committed candidate (by git digest, from the records copy, not from `/work`) to a fresh GPU job in
  a pod where no agent-authored code has run, and that verdict is the result of record. Candidate code still
  executes in that job, so the harness must write its verdict through a channel the candidate cannot reach.
- **(b) Verdict channel the candidate cannot write**: run the harness driver and the candidate child under
  different uids in the GPU container (needs `CAP_SETUID`/`CAP_SETGID` in that container, a capability
  increase to approve), with the verdict file owned by the driver uid; or run the executor in a second
  container (uid 2000) that alone mounts the results volume and receives the job's exit status and stdout
  through a shared emptyDir, so candidate code never has a path to `/ctl-results`. Neither is implemented.
- **(c) Record-side cross-checks** (cheap, partial): the controller's mirror hashes every result at mirror
  time; the executor writes `done` records only after the job process exits, and the broker rejects a
  `done` record whose mtime precedes the request dispatch; nsys SQLite traces are copied by the executor
  with their sha256. These narrow forgery windows but do not close them.

Until (a) and at least one of (b)/(c) are in place and demonstrated, the goal run's result is not
independently verifiable; the 48 h run may still *start* under the trajectory-is-primary-data rule, but
its outcome must be labelled accordingly. This blocker does not affect C1.

## 8. Image provenance and pull requirements

- `C1_IMAGE=docker.io/library/python:3.12-slim`, Docker Official Image. Resolved on rack 3 on 2026-11-19:
  index digest `sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b` (multi-arch,
  includes `linux/arm64/v8`; created 2026-11-05). Use the pinned reference below so the cluster pulls the
  reviewed image, not a later retag.
- Verified on that exact image on rack 3 Docker: `netproxy.py` from bind-mounted files (ConfigMap layout,
  uid 1000, read-only root, caps dropped) serves all 18 proxy cases with the expected allow/deny decisions
  (`c1_net_tests.py --role agent`; the 5 non-proxy probes fail there only because a Docker bridge has
  external access, which the Kubernetes policies remove).
- Pull: anonymous from Docker Hub; two pods on the cluster already run `python:3.12-slim`, so the nodes'
  pull path works. `imagePullPolicy: IfNotPresent`. No registry credential is needed or used. Not
  substituted by any other image.

## 9. Commands

Dry run / render only (no cluster mutation):

```
cd <clone> && git pull origin gb300-tp8-e2e-decisions && git log --oneline -1   # expect be8905a or later
export C1_IMAGE=docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
sandbox/k8s/render.sh c1                              # prints sha256 of each rendered file
sha256sum -c <(sed 's|  |  sandbox/k8s/rendered/c1/|' sandbox/k8s/rendered/c1/SHA256SUMS)   # optional: compare with the review's hashes below
grep -nE "privileged: *true|hostPath|hostNetwork|hostPID|hostIPC|ClusterRole|kind: Secret|stringData|^data:" sandbox/k8s/rendered/c1/*.yaml ; echo "grep exit $? (1 = nothing found)"
kubectl auth whoami
kubectl get ns goalbox-k3                              # must be NotFound
kubectl get sc csi-cephfs-sc-k8s -o yaml | grep -E "reclaimPolicy|volumeBindingMode"
kubectl apply --dry-run=client -f sandbox/k8s/rendered/c1/
kubectl apply --dry-run=server -f sandbox/k8s/rendered/c1/00-namespace.yaml
```
(Server-side dry run of the namespaced files is only possible once the namespace exists.)

Rendered-file hashes with the unpinned tag (the pinned `C1_IMAGE` changes `40-proxy-c1.yaml` and
`75-c1-test-pods.yaml` only; the other four must match exactly):
`00` f8b9d084…38df, `10` 0e4ba1ff…96b3, `20` 59a38f06…c92d, `30` ab6a4aab…74c0
(full values in `reports/gb300/c1-security-prep-20261119/rendered-c1/SHA256SUMS`).

State-changing command, after the operator's approval:

```
C1_IMAGE=docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b \
  sandbox/k8s/c1/c1-validate.sh all
```
Equivalent in stages: `... c1-validate.sh apply`, then `test`, then `cleanup` (set `EVIDENCE=<dir>` to keep
one evidence directory across stages).

## 10. Rollback (removes C1-created resources only)

Precondition: `goalbox-k3` was created by this run (NotFound before the run, §3). Then, in order:

```
NS=goalbox-k3
kubectl -n $NS delete pod c1-agentlike c1-unlabeled c1-records-grab --ignore-not-found
kubectl -n $NS delete configmap goalbox-c1-tests goalbox-proxy-src --ignore-not-found
kubectl -n $NS delete deployment goalbox-proxy --ignore-not-found
kubectl -n $NS delete service goalbox-proxy --ignore-not-found
kubectl -n $NS delete cnp controller --ignore-not-found
kubectl -n $NS delete networkpolicy default-deny agent proxy gpu-workers --ignore-not-found
kubectl -n $NS delete rolebinding goalbox-controller --ignore-not-found
kubectl -n $NS delete role goalbox-controller --ignore-not-found
kubectl -n $NS delete serviceaccount goalbox-controller goalbox-agent goalbox-gpu goalbox-proxy --ignore-not-found
kubectl get pv -o json | python3 -c 'import json,sys; [print(p["metadata"]["name"], p["spec"]["persistentVolumeReclaimPolicy"]) for p in json.load(sys.stdin)["items"] if (p["spec"].get("claimRef") or {}).get("namespace")=="goalbox-k3"]'   # note the PV names first
kubectl -n $NS delete pvc goalbox-work goalbox-ctl goalbox-records --ignore-not-found
kubectl get ns $NS -o jsonpath='{.metadata.creationTimestamp}'; echo   # confirm it is this run's namespace
kubectl delete ns $NS
```
PVs: with reclaim policy Delete they disappear with the PVCs; with Retain they stay `Released` and hold
CephFS space until the operator deletes them explicitly (`kubectl delete pv <name>` for the names noted
above, and only those). Nothing else in the cluster references C1 objects, so no other workload is affected.
Partial-failure rollback is the same list; every command is `--ignore-not-found`.

## Verdict

Ready to execute with the pinned `C1_IMAGE`, after the pre-checks in §9 (namespace NotFound, StorageClass
present, identity may impersonate). Unsafe assumptions found and recorded: StorageClass existence and
reclaim policy unverified (§3); PVC probe proves RBAC only, no mount-level control exists (§6); in-run
GPU-job results are candidate-writable, blocker B-C1-1 for the GPU phase, not for C1 (§7); evidence
directory is raw and needs redaction before commit (§5).
