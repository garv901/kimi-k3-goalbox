# Phase C1 security validation: preparation evidence (2026-11-19)

Phase C1 (operator's conditional GO, 2026-11-19) covers the `goalbox-k3` namespace, its RBAC, network
policies, PVCs, proxy and disposable test pods, and nine security tests. This directory holds everything that
could be produced **without a credential that may create resources in the cluster**: the only non-Rancher
credential available to the build agent cannot create a namespace or anything inside `goalbox-k3`
(`rbac-readonly-recon-dc.txt`; DECISIONS-LOG L18), and the operator's Rancher identity is not to be used by
the build agent. The cluster part is scripted for the operator: `sandbox/k8s/c1/c1-validate.sh` (see
`sandbox/k8s/c1/README.md`); its evidence lands in a sibling directory `c1-security-<utc>/`.

## Contents

| file | what |
|---|---|
| `rendered-c1/*.yaml`, `rendered-c1/SHA256SUMS` | the exact manifests `c1-validate.sh apply` applies, rendered with `C1_IMAGE=docker.io/library/python:3.12-slim` |
| `check_manifests.json` | automated inspection of all manifests (production set + C1 overlay): no privileged, no host namespaces, hostPath only for weights/HF/cache in the GPU and conversion pods (not in C1), no Secret data, no cluster-scoped resources, request spool read-only in GPU pods, controller Role limited to pods get/list/watch + delete `goalbox-agent` |
| `rbac-readonly-recon-dc.txt` | can-i answers for the shared account (test 1, RBAC part) |
| `unittest-full.log.txt` | 32 CPU tests on rack 3 incl. `BudgetLedgerTests` (test 9) and the same-uid bystander regression in `SpoolProtocolTests` (test 7) |
| `local-proxy-test.log.txt` | test 4 on Docker: 18 proxy cases, allow/deny decisions from the proxy's own log |
| `local-mount-and-cleanup-test.log.txt` | test 6 (GPU-pod mount layout, executor as pid 1, candidate job tries 11 writes) and test 7 (uid-wide sweep in a private pid namespace) on Docker |

## Pre-apply inspection of the rendered C1 set

```
grep -nE "privileged: *true|hostPath|hostNetwork|hostPID|hostIPC|ClusterRole|kind: Secret|stringData|^data:" rendered-c1/*.yaml
```
returns nothing. Cluster-scoped objects in the set: the `Namespace` only. Images: `docker.io/library/python:3.12-slim`
(C1 proxy and test pods; L17). Rendering leaves no unexpanded `${...}`.

## Results of the tests that ran here

- **Test 4, proxy** (`local-proxy-test.log.txt`): CONNECT to the 13 allowlisted hosts on 443 is the only tunnel;
  denied: non-allowlisted hosts, any other port on an allowed host (22, 80), raw IPs, the metadata address, an
  arbitrary `*.huggingface.co` subdomain (no wildcard, so no DNS side channel through the proxy), plain-HTTP GET
  to any host, POST to the index; `GET http://pypi.goalbox/simple/<pkg>/` serves allowed packages and refuses
  the blocked list (vllm, flashinfer, …). A TLS handshake inside the allowed tunnel reaches api.anthropic.com
  (certificate verified; HTTP 404 on `/` as expected). The allowed hosts are download-only for an anonymous
  client; the Anthropic API is the agent's own channel. No unrestricted relay exists through the proxy.
- **Test 6, candidate writes** (`local-mount-and-cleanup-test.log.txt`, first half): with the GPU pod's mounts
  (request spool ro, results rw, image ro, weights ro, no records) a job running as the executor's uid could
  not alter its request, inject requests (own or other node), modify grading code, oracle, executor or
  weights, or write a records path. It could write its own results and `/work` (by design), and read the
  executor's environment (same uid; no secret is in that environment). The request spool was unchanged on
  the host afterwards.
- **Test 7, cleanup** (second half, plus `SpoolProtocolTests`): session-scoped mode leaves an unrelated same-uid
  process alive; in the pod's uid-wide mode (private pid namespace) a `setsid` stray is killed, the executor
  (pid 1) survives and serves the next job.
- **Test 9, timer** (`BudgetLedgerTests`): run time counts only across ticks where everything was running;
  interruptions and controller gaps are recorded and excluded; warn then stop fire once each, 30 min apart;
  state survives a controller restart. Pod `activeDeadlineSeconds` is a 72 h backstop only.
- **Test 1/2/5, RBAC**: designed and checked statically (`check_manifests.json`, Role definitions); the live
  `can-i` matrix needs the namespace to exist (operator step).

## Pending the operator's run (`c1-validate.sh`)

Live RBAC matrix for the five subjects; in-pod network tests from the agent-labelled and the unlabelled pod
(DNS, direct HTTPS, API server, metadata, node address, proxy reachability); the records-PVC mount attempt
as the agent ServiceAccount; admission outcomes (apply events, `--dry-run=server` of a privileged and a
hostPath pod); the proxy's live decision log.

## Unresolved weaknesses (to carry into the readiness report)

1. The shared cluster account reads pods and pod logs in every namespace. Pod specs of `goalbox-k3` are
   therefore visible to all its users (no secrets are in specs; the token is a Secret reference). Stdout
   carries only marker lines (L14). Fixing the account's ClusterRole is outside this project's scope.
2. The results half of the control spool is candidate-writable: a candidate can forge or delay its own job's
   result (it misleads the agent, not the record) and can kill the executor (restart). Request side is
   kernel-enforced read-only.
3. The agent's token is readable by the agent process by design (`/goalbox/token`); the only egress is the
   proxy allowlist, and the GPU pods never see it.
4. NetworkPolicy is one layer; the proxy allowlist and the absence of ServiceAccount tokens are the others.
   Egress through allowed hosts is download-only, but a Hugging Face or PyPI *download* of attacker-prepared
   content remains possible (it would still run inside the sandbox).
5. The production proxy image cannot be pushed yet (registry 401); C1 runs the proxy from a public image with
   a ConfigMap (L17). Not a security gap for C1, a blocker for the GPU phase (image for the workers).
6. Admission (Kyverno) behaviour is unknown until the apply: a rejection of the agent-like test pods will be
   recorded and adapted to within the fixed requirements (no privilege increase).
