# C2 readiness evidence (2026-11-19, CPU only)

| file | what |
|---|---|
| `unittest-full.log.txt` | 36 CPU tests on rack 3 (venv Python, no GPU), including the new transport-validation and contamination-rule tests and the export-audit record in a grade verdict |
| `check_manifests.json` | static manifest checks including the new toleration rule for the `nvidia.com/gpu` taint |
| `c1-unchanged-since-review.txt` | `git diff --stat` of the operator script, C1 net tests, C1 manifests and renderer against the reviewed commit `7f8e025`: empty |
| `rendered-c1-SHA256SUMS-pinned.txt` | hashes of the rendered C1 set with the pinned image digest (00/10/20/30 equal the prep evidence; 40/75 differ only by the image string) |

Read-only cluster facts gathered with the dc service account (no mutation): all 18 rack-10 trays advertise
4 `nvidia.com/gpu` and carry the `nvidia.com/gpu` taint; the device plugin is `k8s-device-plugin:v0.19.2`
with a config manager (its ConfigMap is not readable by dc); tray14 reports driver 595.71.05 and product
NVIDIA-GB300; no pod requesting GPUs was scheduled on tray14 or tray15; RuntimeClasses are not listable by
dc (the operator must confirm `runtimeClassName: nvidia` exists, C2 step 0).

Decisions: L21 (export audit), L22 (contamination policy), L23 (toleration, blocker B-C2-1). Plan:
`sandbox/k8s/C2-GPU-SMOKE-PLAN.md`. Operator commands for C1b: `reports/gb300/C1B-OPERATOR-COMMANDS.md`.
