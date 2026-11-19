#!/usr/bin/env python3
"""Static checks of the goalbox-k3 manifests (conditional-GO items 2, 3, 4; DECISIONS section 3).
Runs on the unrendered templates (placeholders are legal YAML strings). Exit 0 only if every rule holds.

    python3 sandbox/k8s/check_manifests.py [sandbox/k8s/manifests]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

NS = "goalbox-k3"
PRIVATE = {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8", "100.64.0.0/10"}


def load(directory: Path) -> list[dict]:
    docs = []
    files = sorted(directory.glob("*.yaml")) + sorted((directory.parent / "c1").glob("*.yaml"))
    for path in files:
        for doc in yaml.safe_load_all(path.read_text()):
            if doc:
                doc["_file"] = path.name
                doc["_c1"] = path.parent.name == "c1"
                docs.append(doc)
    return docs


def pods(docs):
    """(name, podSpec, kind) for Pods, Deployments and Jobs."""
    out = []
    for doc in docs:
        kind = doc["kind"]
        if kind == "Pod":
            out.append((doc["metadata"]["name"], doc["spec"], kind))
        elif kind in ("Deployment", "Job"):
            out.append((doc["metadata"]["name"], doc["spec"]["template"]["spec"], kind))
    return out


def volume_kinds(spec) -> dict:
    return {v["name"]: [k for k in v if k != "name"][0] for v in spec.get("volumes", [])}


def check(directory: Path) -> list[str]:
    docs = load(directory)
    problems = []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    kinds = {d["kind"] for d in docs}
    from_c1 = {(d["kind"], d["metadata"].get("name")): d["_c1"] for d in docs}
    need(not kinds & {"ClusterRole", "ClusterRoleBinding"}, "cluster-scoped RBAC objects are not allowed")
    for doc in docs:
        if doc["kind"] != "Namespace":
            need(doc["metadata"].get("namespace") == NS, f"{doc['_file']}: {doc['kind']} {doc['metadata'].get('name')} not in {NS}")
    for doc in docs:
        if doc["kind"] == "RoleBinding":
            for subject in doc["subjects"]:
                need(subject.get("kind") == "ServiceAccount" and subject.get("namespace") == NS,
                     f"RoleBinding {doc['metadata']['name']}: subject outside the namespace: {subject}")
        if doc["kind"] == "Role":
            for rule in doc["rules"]:
                need("secrets" not in rule["resources"], f"Role {doc['metadata']['name']} grants secrets")
                mutating = {"create", "delete", "patch", "update", "*"} & set(rule["verbs"])
                if mutating and {"pods", "*"} & set(rule["resources"]):
                    need(set(rule["verbs"]) == {"delete"} and rule.get("resourceNames") == ["goalbox-agent"],
                         f"Role {doc['metadata']['name']} can create or change pods beyond deleting goalbox-agent")
                need("*" not in rule["verbs"] and "*" not in rule["resources"], f"Role {doc['metadata']['name']} has a wildcard")
    sa_automount = {d["metadata"]["name"]: d.get("automountServiceAccountToken") for d in docs if d["kind"] == "ServiceAccount"}
    for name, spec, kind in pods(docs):
        label = f"{kind} {name}"
        vols = volume_kinds(spec)
        sa = spec.get("serviceAccountName", "default")
        automount = spec.get("automountServiceAccountToken", sa_automount.get(sa, True))
        if name != "goalbox-controller":
            need(automount is False, f"{label}: API token would be mounted (sa {sa})")
            need(sa != "default", f"{label}: uses the default ServiceAccount")
        containers = spec.get("containers", []) + spec.get("initContainers", [])
        for c in containers:
            sc = c.get("securityContext", {})
            clabel = f"{label}/{c['name']}"
            need(sc.get("allowPrivilegeEscalation") is False, f"{clabel}: allowPrivilegeEscalation not false")
            need(sc.get("readOnlyRootFilesystem") is True, f"{clabel}: root filesystem writable")
            need(not sc.get("privileged"), f"{clabel}: privileged")
            caps = sc.get("capabilities", {})
            need(caps.get("drop") == ["ALL"], f"{clabel}: capabilities not dropped")
            added = set(caps.get("add", []))
            if c in spec.get("initContainers", []):
                need(added <= {"CHOWN", "FOWNER", "DAC_OVERRIDE"}, f"{clabel}: init container adds {added}")
            else:
                need(added <= {"IPC_LOCK"}, f"{clabel}: adds capabilities {added}")
                run_as = sc.get("runAsUser", spec.get("securityContext", {}).get("runAsUser"))
                need(run_as not in (None, 0), f"{clabel}: not pinned to a non-root uid")
            need("resources" in c and "limits" in c["resources"], f"{clabel}: no resource limits")
            for mount in c.get("volumeMounts", []):
                kind_of = vols.get(mount["name"])
                if kind_of == "hostPath":
                    is_cache = mount["mountPath"] == "/cache" and kind == "Job"
                    need(mount.get("readOnly") is True or is_cache, f"{clabel}: hostPath {mount['mountPath']} mounted read-write")
        # Per-role rules
        mounts = {c["name"]: [m["name"] for m in c.get("volumeMounts", [])] for c in containers}
        if name.startswith("c1-"):
            need(set(vols.values()) <= {"configMap", "emptyDir"}, f"{label}: test pod mounts {set(vols.values())}")
        if name == "goalbox-agent":
            need("hostPath" not in vols.values(), f"{label}: hostPath volume present")
            need("records" not in mounts.get("agent", []), f"{label}: agent container mounts records")
            need("ctl" not in vols, f"{label}: ctl volume present")
            secret_mounts = [(cn, m) for cn, ms in mounts.items() for m in ms if vols.get(m) == "secret"]
            need(secret_mounts == [("agent", "token")], f"{label}: Secret mounted in {secret_mounts}")
            need(vols.get("token") == "secret" and spec["volumes"][[v["name"] for v in spec["volumes"]].index("token")]["secret"].get("defaultMode") == 0o400,
                 f"{label}: token volume mode is not 0400")
            cap = next(c for c in containers if c["name"] == "capture")
            agent = next(c for c in containers if c["name"] == "agent")
            need(cap["securityContext"]["runAsUser"] != agent["securityContext"]["runAsUser"], f"{label}: capture shares the agent's uid")
            need(all(m.get("readOnly") for m in cap["volumeMounts"] if m["name"] == "work"), f"{label}: capture has the agent home read-write")
            need(spec.get("enableServiceLinks") is False, f"{label}: service links would leak cluster addresses")
        if "gpu" in name and kind == "Pod":
            need("records" not in vols, f"{label}: records volume present")
            by_name = {c["name"]: c for c in spec.get("containers", [])}
            need(set(by_name) == {"grader", "runner"}, f"{label}: containers {sorted(by_name)} (expected grader and runner)")
            need(not spec.get("shareProcessNamespace"), f"{label}: shares the pid namespace between grader and runner")
            need(not spec.get("hostPID") and not spec.get("hostIPC") and not spec.get("hostNetwork"), f"{label}: host namespace")
            grader, runner = by_name.get("grader", {}), by_name.get("runner", {})
            g_uid = grader.get("securityContext", {}).get("runAsUser"); r_uid = runner.get("securityContext", {}).get("runAsUser")
            need(g_uid not in (None, 0) and r_uid not in (None, 0) and g_uid != r_uid, f"{label}: grader and runner must run as different non-root uids")
            g_mounts = {m["mountPath"]: m for m in grader.get("volumeMounts", [])}
            r_mounts = {m["mountPath"]: m for m in runner.get("volumeMounts", [])}
            need(not any(m["name"] == "ctl" for m in r_mounts.values()), f"{label}: runner mounts the control volume")
            need(not any(m["name"] == "records" for m in list(r_mounts.values()) + list(g_mounts.values())), f"{label}: records mounted")
            ctl = [m for m in g_mounts.values() if m["name"] == "ctl"]
            need(any(m.get("subPath") == "req" and m.get("readOnly") for m in ctl), f"{label}: request spool not mounted read-only in the grader")
            need(all(m.get("subPath") in ("req", "results") for m in ctl), f"{label}: ctl mounted without a subPath")
            need(g_mounts.get("/work", {}).get("readOnly") is True, f"{label}: /work writable in the grader")
            need(g_mounts.get("/dev/shm", {}).get("name") != r_mounts.get("/dev/shm", {}).get("name"), f"{label}: grader and runner share /dev/shm")
            need(g_mounts.get("/tmp", {}).get("name") != r_mounts.get("/tmp", {}).get("name"), f"{label}: grader and runner share /tmp")
            shared = {m["name"] for m in g_mounts.values() if not m.get("readOnly")} & {m["name"] for m in r_mounts.values() if not m.get("readOnly")}
            need(shared <= {"job"}, f"{label}: volumes writable from both containers: {shared}")
            need("livenessProbe" in runner, f"{label}: runner has no liveness probe (restart channel)")
            need(any(t.get("key") == "nvidia.com/gpu" for t in spec.get("tolerations", [])),
                 f"{label}: no toleration for the nvidia.com/gpu taint carried by every GPU node")
            need(spec.get("runtimeClassName") == "nvidia", f"{label}: runtimeClassName")
            need("secret" not in vols.values(), f"{label}: a Secret is mounted in a GPU pod")
            c = containers[0]
            need(c["resources"]["limits"].get("nvidia.com/gpu") == 4, f"{label}: must hold exactly 4 GPUs")
        if name == "goalbox-controller":
            need("secret" not in vols.values() and "hostPath" not in vols.values(), f"{label}: Secret or hostPath volume")
        if name == "goalbox-proxy":
            allowed = {"emptyDir", "configMap"} if from_c1.get((kind, name)) else {"emptyDir"}   # C1 overlay: code from a ConfigMap
            need(set(vols.values()) <= allowed, f"{label}: proxy mounts {set(vols.values())}")
    # Network policies
    pols = {d["metadata"]["name"]: d["spec"] for d in docs if d["kind"] == "NetworkPolicy"}
    need("default-deny" in pols and pols["default-deny"]["podSelector"] == {} and set(pols["default-deny"]["policyTypes"]) == {"Ingress", "Egress"},
         "no default-deny NetworkPolicy for both directions")
    agent = pols.get("agent", {})
    need("ingress" not in agent, "agent policy allows ingress")
    for rule in agent.get("egress", []):
        for peer in rule["to"]:
            need("ipBlock" not in peer, "agent egress to an ipBlock")
            if "podSelector" in peer and "namespaceSelector" not in peer:
                need(peer["podSelector"]["matchLabels"].get("app.kubernetes.io/component") == "proxy", f"agent egress to {peer}")
                need([p["port"] for p in rule["ports"]] == [3128], "agent egress to the proxy on a port other than 3128")
            if "namespaceSelector" in peer:
                need(peer["namespaceSelector"]["matchLabels"].get("kubernetes.io/metadata.name") == "kube-system" and
                     peer.get("podSelector", {}).get("matchLabels", {}).get("k8s-app") == "kube-dns", f"agent egress to {peer}")
                need({p["port"] for p in rule["ports"]} == {53}, "agent DNS egress on a port other than 53")
    proxy = pols.get("proxy", {})
    for rule in proxy.get("egress", []):
        for peer in rule["to"]:
            if "ipBlock" in peer:
                need(set(peer["ipBlock"].get("except", [])) >= PRIVATE - {"100.64.0.0/10"}, "proxy egress does not exclude private ranges")
                need([p["port"] for p in rule["ports"]] == [443], "proxy public egress on a port other than 443")
    for rule in proxy.get("ingress", []):
        for peer in rule["from"]:
            need(peer.get("podSelector", {}).get("matchLabels", {}).get("app.kubernetes.io/component") == "agent", f"proxy ingress from {peer}")
    gpu = pols.get("gpu-workers", {})
    for rule in gpu.get("egress", []):
        for peer in rule["to"]:
            need("ipBlock" not in peer, "GPU workers may reach an ipBlock")
            if "namespaceSelector" not in peer:
                need(peer["podSelector"]["matchLabels"].get("app.kubernetes.io/component") == "gpu-worker", f"GPU egress to {peer}")
    for rule in gpu.get("ingress", []):
        for peer in rule["from"]:
            need(peer.get("podSelector", {}).get("matchLabels", {}).get("app.kubernetes.io/component") == "gpu-worker", f"GPU ingress from {peer}")
    cnps = {d["metadata"]["name"]: d["spec"] for d in docs if d["kind"] == "CiliumNetworkPolicy"}
    ctrl = cnps.get("controller")
    need(ctrl is not None and ctrl["endpointSelector"]["matchLabels"].get("app.kubernetes.io/component") == "controller",
         "no CiliumNetworkPolicy for the controller")
    for rule in (ctrl or {}).get("egress", []):
        need("toCIDR" not in rule and "toCIDRSet" not in rule, "controller egress to a CIDR")
        for entity in rule.get("toEntities", []):
            need(entity == "kube-apiserver", f"controller egress to entity {entity}")
        for ep in rule.get("toEndpoints", []):
            need(ep["matchLabels"].get("k8s-app") == "kube-dns", f"controller egress to endpoints {ep}")
    for name_, spec_ in cnps.items():
        need("ingress" not in spec_ or not spec_["ingress"], f"CiliumNetworkPolicy {name_} allows ingress")
    # Every pod role has a policy selecting it
    for name, spec, kind in pods(docs):
        comp = None
        for doc in docs:
            if doc["metadata"]["name"] == name and doc["kind"] == kind:
                labels = (doc["metadata"].get("labels") if kind == "Pod" else doc["spec"]["template"]["metadata"].get("labels")) or {}
                comp = labels.get("app.kubernetes.io/component")
        selected = [p for p, s in pols.items() if s["podSelector"].get("matchLabels", {}).get("app.kubernetes.io/component") == comp]
        selected += [p for p, s in cnps.items() if s["endpointSelector"].get("matchLabels", {}).get("app.kubernetes.io/component") == comp]
        need(selected or comp in ("conversion", None), f"{kind} {name} (component {comp}) has no NetworkPolicy of its own")
    # Secret manifest has no data
    for doc in docs:
        if doc["kind"] == "Secret":
            need(not doc.get("data") and not doc.get("stringData"), "Secret manifest carries data")
    return problems


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "manifests"
    problems = check(directory)
    print(json.dumps({"manifests": str(directory), "ok": not problems, "problems": problems}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
