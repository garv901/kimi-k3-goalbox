# Phase C1: namespace and security validation

Scope (operator's conditional GO, 2026-11-19): namespace, namespaced RBAC, default-deny and scoped network
policies, PVCs, proxy, disposable non-GPU test pods. No GPU, no conversion, no Secret, no agent.

## Who runs what

The build agent's only non-Rancher Kubernetes credential (the shared account's token) cannot create anything in
`goalbox-k3` (DECISIONS-LOG L18), so the cluster part is run by the operator:

```
# on the laptop, with the operator's kubeconfig (and the Rancher socks path if needed)
cd <repo>
C1_IMAGE=docker.io/library/python:3.12-slim sandbox/k8s/c1/c1-validate.sh all
# -> prints every kubectl command, applies 00/10/20/30 + the C1 proxy overlay + two test pods,
#    runs the RBAC can-i matrix and the in-pod network tests, probes admission with --dry-run=server,
#    deletes the test pods, and leaves the evidence in reports/gb300/c1-security-<utc>/ (raw; redact before commit)
```

Steps can be run separately: `apply`, `test`, `cleanup`. The rendered manifests and their sha256 are copied
into the evidence directory before anything is applied, and `pre-apply-grep.txt` records the search for
privileged settings, hostPath, host namespaces, Secret data and cluster-scoped RBAC in the rendered set.

## What the tests show

| C1 test | where |
|---|---|
| 1 shared account cannot create pods, read Secrets, reach records, change workloads | `rbac-matrix.txt` (can-i as `inference-wideep:wideep-session`) |
| 2 agent, GPU and proxy ServiceAccounts have no API access | `rbac-matrix.txt` (all `no`), `net-agentlike.txt` (`api_token_absent`, `kubernetes_api` blocked) |
| 3 permitted and forbidden destinations from a sandbox pod | `net-agentlike.txt`, `net-unlabeled.txt` |
| 4 proxy restricts destinations and protocols; no unrestricted tunnel | `net-agentlike.txt` proxy cases, `proxy-net.log.txt`; also `local/proxy_test.sh` on Docker |
| 5 records PVC cannot be mounted by agent-controlled workloads | `rbac-matrix.txt` (agent SA: no create pods), `pvc-mount-as-agent.txt` (Forbidden) |
| 6 candidate code cannot modify control state, grading, protected artifacts, records | `local/verdict_test.sh` (Docker, grader + runner containers, hostile and killer controls), `local/mount_test.sh` (single-container layout, superseded), `check_manifests.json`, `sandbox/k8s/TRUSTED-VERDICT.md` |
| 7 cleanup kills only the job's session | `test_sandbox.SpoolProtocolTests` (unrelated same-uid bystander survives), `local/cleanup_uid_test.sh` (uid-wide sweep inside a private pid namespace) |
| 8 admission behaviour | `events-after-apply.txt`, `admission-privileged.txt`, `admission-hostpath.txt` |
| 9 budget timer and interruptions | `test_sandbox.BudgetLedgerTests`, `sandbox/lib/budget.py`, `controller.sh` |
