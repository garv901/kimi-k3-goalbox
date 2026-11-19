# goalbox-k3 on Kubernetes

Everything the 48-hour run needs on the cluster, as manifests that are rendered and then applied by the
operator's own identity. Nothing in this directory contacts the cluster by itself. Design rationale and the
deviations from `planning/DECISIONS.md` are in `planning/DECISIONS-LOG.md` (L1, L2, L3, L7, L9, L10).

## Pieces

| file | what |
|---|---|
| `Dockerfile`, `build-image.sh` | aarch64 image: CUDA 13.4 toolkit, nsys, ncu, torch cu130 venv, Claude Code pinned, and this repository's harness, sandbox scripts and goal text baked in read-only (the image digest pins them). `IMAGE-BUILD.txt` records the last local build. |
| `manifests/00-namespace.yaml` | namespace `goalbox-k3` |
| `manifests/10-rbac.yaml` | four ServiceAccounts, one Role (controller: pods get/list/watch, pods/log get), one RoleBinding. No cluster-scoped objects, no secrets access. |
| `manifests/20-network.yaml` | default deny; agent to proxy + DNS only; proxy to public 443 (private ranges excluded) + DNS; GPU workers to each other + DNS; controller to the API server (CIDR rendered) + DNS |
| `manifests/30-storage.yaml` | PVCs `goalbox-work` (agent rw), `goalbox-ctl` (controller + GPU workers), `goalbox-records` (controller + capture sidecar) |
| `manifests/40-proxy.yaml` | allowlist proxy (`sandbox/lib/netproxy.py --tcp 3128`) |
| `manifests/50-gpu-workers.yaml` | two long-lived GPU pods, 4 GPUs each, pinned to the two trays, running `sandbox/lib/executor.py` |
| `manifests/60-controller.yaml` | broker (k8s mode), stream capture, budget warning (`controller.sh`) |
| `manifests/70-agent.yaml` | the goal agent (CPU only) with the `capture` sidecar; `activeDeadlineSeconds` = 48 h + 5 min |
| `manifests/80-secret.yaml` | the credential Secret, by name only; created by the operator from a file |
| `manifests/90-conversion-job.yaml` | checkpoint conversion on one tray (rendered once per tray) |
| `render.sh` | substitutes node names, API server CIDR, image, host paths into `rendered/` (gitignored) |
| `check_manifests.py` | static security checks; part of the CPU test suite |
| `agent-entry.sh` | agent container entrypoint: settings, token from file, `claude -p "/goal ..."` headless |
| `controller.sh` | controller entrypoint |

## Order of operations (each step is a separate authorization)

1. Image: `sandbox/k8s/build-image.sh` on a rack 3 tray (done locally; push waits for registry access).
2. Render: `IMAGE=... TRAY_A=... TRAY_B=... API_SERVER_CIDR=... WEIGHTS_DIR=... HF_DIR=... CACHE_DIR=... sandbox/k8s/render.sh`
3. Namespace, RBAC, network, storage, proxy: `kubectl apply -f rendered/00-*.yaml -f rendered/10-*.yaml -f rendered/20-*.yaml -f rendered/30-*.yaml -f rendered/40-*.yaml`, then copy the pull secret into the namespace (`kubectl get secret vccr-creds -n inference-wideep -o yaml | <strip namespace/uid/resourceVersion> | kubectl apply -n goalbox-k3 -f -`).
4. RBAC verification (before anything sensitive exists), as the shared account and as the controller SA:
   ```
   for v in "get secrets" "list secrets" "create pods" "delete pods" "get pods" "create pods/exec" "get persistentvolumeclaims"; do
     printf '%-24s dc: %s  controller: %s\n' "$v" \
       "$(kubectl auth can-i $v -n goalbox-k3 --as=system:serviceaccount:inference-wideep:wideep-session)" \
       "$(kubectl auth can-i $v -n goalbox-k3 --as=system:serviceaccount:goalbox-k3:goalbox-controller)"
   done
   ```
   Expected: every `dc` answer `no`; controller `yes` only for `get pods`.
5. Conversion Jobs (one per tray), then `tools/conversion_sizing.py --out-dir` beforehand for the free-space check and `planning/bringup/verify_conversion.py` afterwards (run by the Job itself; output in `VERIFY.txt`); compare the two trays' `MANIFEST.json`.
6. GPU workers (this is the reservation): `kubectl apply -f rendered/50-*.yaml`; `kubectl -n goalbox-k3 get pod -o wide`.
7. Pre-run validation jobs through the broker spool (sandbox self-test, synthetic TP8 oracle, real TP8 B1, vLLM baselines) — DECISIONS section 10.
8. Secret: `kubectl -n goalbox-k3 create secret generic goalbox-claude-token --from-file=token=<file>` (operator only).
9. Controller, then agent: `kubectl apply -f rendered/60-*.yaml`, `kubectl apply -f rendered/70-*.yaml`. The run starts when the agent pod starts.

## Trust boundaries (what each process can reach)

| process | uid | network | volumes | API identity |
|---|---|---|---|---|
| agent | 1000 | proxy:3128 only (allowlist: api.anthropic.com, PyPI index/wheels minus blocked packages, Hugging Face) | `/work` rw, `/home/agent` rw, `/tmp`; token file ro | none |
| capture sidecar | 2000 | none needed (pod policy = agent's) | `/agent-home` ro, `/records` rw | none |
| proxy | 1000 | from agent; to public 443 | none | none |
| GPU executor + candidate code | 1000 | other GPU pod only | `/work` rw, `/home/agent` rw, `/ctl` rw, `/weights` ro, `/dev/shm` | none |
| controller | 2000 | API server, DNS | `/work`, `/run-home`, `/ctl`, `/records` | pods get/list/watch, pods/log get |

Known limits (recorded in the decisions log): candidate code runs as the same uid as the executor and can write
the work and ctl volumes (L2); the agent process holds the OAuth token in its environment (L9); NetworkPolicy
is one layer and depends on Cilium enforcing it (test 1 of the readiness report exercises it from inside the pods).
