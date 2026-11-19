#!/usr/bin/env python3
"""GPU job executor: runs inside each long-lived GPU worker pod (Kubernetes mode, DECISIONS-LOG L2).

The agent never reaches this pod. The controller's broker (broker.py, GOALBOX_GPU_MODE=k8s) copies an
accepted request into the controller-owned spool at CTL_REQ/gpu/<node>/req/<id>.{json,sh}; this
process picks it up, runs the command with the job's environment (topology for two-tray jobs,
CUDA_VISIBLE_DEVICES for the requested GPU count), writes CTL_RESULTS/gpu/<node>/out/<id>.log and
CTL_RESULTS/gpu/<node>/status/<id>, and finally CTL_RESULTS/gpu/<node>/done/<id>.json {"exit": ..}.

Two mounts, because candidate code runs in this pod as the same uid: the request side is mounted
read-only here (only the controller writes it), so candidate code cannot inject or alter requests
(time limits, GPU counts, commands); the results side is read-write and therefore untrusted content
to the broker, which only ever reports it back to the agent that submitted the job.

Before writing the done record it kills every process left in the job's session (a stray candidate
process could otherwise keep running into the next job) and, when GOALBOX_WEIGHTS_MANIFEST is set,
checks the weight files it names are unchanged (size + mtime) as a cheap tamper tripwire; the real
check is the read-only mount.

Settings: GOALBOX_CTL_REQ (request root, default /ctl-req), GOALBOX_CTL_RESULTS (results root,
default /ctl-results), GOALBOX_NODE_NAME (this worker's name, default hostname), GOALBOX_GPUS_HERE
(default 4), GOALBOX_EXECUTOR_KILL_ALL_UID (1 inside the pod: sweep every process of the uid).
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

NODE = os.environ.get("GOALBOX_NODE_NAME") or socket.gethostname()
GPUS_HERE = int(os.environ.get("GOALBOX_GPUS_HERE", "4"))
REQ = Path(os.environ.get("GOALBOX_CTL_REQ", "/ctl-req")) / "gpu" / NODE / "req"       # read-only mount in the pod
SPOOL = Path(os.environ.get("GOALBOX_CTL_RESULTS", "/ctl-results")) / "gpu" / NODE     # read-write mount
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
PASS_ENV = ("PATH", "HOME", "LANG", "TERM", "CUDA_HOME", "PYTHONPATH", "KIMI_K3_WEIGHTS", "NVIDIA_IMEX_CHANNELS",
            "NCCL_MNNVL_ENABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "LD_LIBRARY_PATH", "TMPDIR",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")


def log(message: str) -> None:
    with (SPOOL / "executor.log").open("a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def status(rid: str, line: str) -> None:
    with (SPOOL / "status" / rid).open("a") as file:
        file.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def weights_fingerprint(manifest: Path) -> dict:
    files = json.loads(manifest.read_text())["files"]
    return {name: [os.stat(name).st_size, int(os.stat(name).st_mtime)] for name in files if os.path.exists(name)}


def run_job(rid: str) -> None:
    req = json.loads((REQ / f"{rid}.json").read_text())
    minutes, gpus = int(req["minutes"]), int(req["gpus"])
    cwd = req["cwd"]
    topo = req.get("topology") or {}
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(min(gpus, GPUS_HERE)))
    env["GOALBOX_JOB_ID"] = rid
    for key in ("GOALBOX_NNODES", "GOALBOX_NODE_RANK", "GOALBOX_MASTER_ADDR", "GOALBOX_MASTER_PORT"):
        if key in topo:
            env[key] = str(topo[key])
    manifest = os.environ.get("GOALBOX_WEIGHTS_MANIFEST")
    before = weights_fingerprint(Path(manifest)) if manifest else None
    out = (SPOOL / "out" / f"{rid}.log").open("w")
    status(rid, f"running on {NODE}, GPUs {env['CUDA_VISIBLE_DEVICES']}, {minutes} min" +
           (f", node_rank {topo.get('GOALBOX_NODE_RANK')} of {topo.get('GOALBOX_NNODES')}" if topo else ""))
    log(f"{rid} start cwd={cwd} gpus={gpus} topo={topo}")
    try:
        proc = subprocess.Popen(["bash", str(REQ / f"{rid}.sh")], cwd=cwd if os.path.isdir(cwd) else str(SPOOL),
                                stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    except OSError as exc:
        out.write(f"executor: could not start: {exc}\n"); out.close()
        finish(rid, 2, "START_FAILED"); return
    deadline = time.time() + 60 * minutes + 60
    state = "COMPLETED"
    while proc.poll() is None:
        if time.time() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
            state = "TIMEOUT"
        time.sleep(2)
    code = proc.returncode if state != "TIMEOUT" else 124
    # Nothing from this job may survive into the next one.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    leftovers = kill_strays(proc.pid)
    if leftovers:
        out.write(f"executor: killed {leftovers} leftover process(es)\n")
        state += "+STRAYS_KILLED"
    if before is not None and weights_fingerprint(Path(manifest)) != before:
        out.write("executor: WEIGHT FILES CHANGED during the job\n")
        state += "+WEIGHTS_CHANGED"
        code = code or 3
    out.close()
    finish(rid, code, state if code == 0 or state != "COMPLETED" else "FAILED")


def _session_of(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as file:
            fields = file.read().rsplit(")", 1)[1].split()
        return int(fields[3])          # session id (field 6 of /proc/<pid>/stat)
    except (OSError, IndexError, ValueError):
        return None


KILL_ALL_UID = os.environ.get("GOALBOX_EXECUTOR_KILL_ALL_UID") == "1"


def kill_strays(job_pid: int) -> int:
    """Kill every process left by the job. In the GPU pod (GOALBOX_EXECUTOR_KILL_ALL_UID=1, a private pid
    namespace where the executor is the only process that should exist between jobs) that is every other
    process of this uid, which also catches a candidate that called setsid. Elsewhere (tests on shared
    hosts) only the job's session is swept: start_new_session=True made the job its own session leader,
    so detached children keep its session id, and nothing outside the job is touched."""
    killed = 0
    for _ in range(3):
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            if KILL_ALL_UID:
                try:
                    if entry.stat().st_uid != os.getuid():
                        continue
                except OSError:
                    continue
            elif _session_of(pid) != job_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
                found = True
            except (ProcessLookupError, PermissionError):
                continue
        if not found:
            break
        time.sleep(0.2)
    return killed


def finish(rid: str, code: int, state: str) -> None:
    status(rid, f"done {code} {state}")
    tmp = SPOOL / "done" / f".{rid}.tmp"
    tmp.write_text(json.dumps({"exit": code, "state": state, "node": NODE, "finished_utc": time.time()}))
    os.replace(tmp, SPOOL / "done" / f"{rid}.json")
    log(f"{rid} done {code} {state}")


def main() -> None:
    for d in ("status", "out", "done"):
        (SPOOL / d).mkdir(parents=True, exist_ok=True)
    try:
        REQ.mkdir(parents=True, exist_ok=True)      # tests on a host; in the pod the mount is read-only and pre-created
    except OSError:
        pass
    (SPOOL / "READY").write_text(json.dumps({"node": NODE, "gpus": GPUS_HERE, "pid": os.getpid(), "started_utc": time.time()}))
    log(f"executor start node={NODE} gpus={GPUS_HERE}")
    seen = {p.stem for p in (SPOOL / "done").glob("*.json")}
    while True:
        reqs = sorted((p for p in REQ.glob("*.json") if p.stem not in seen), key=lambda p: p.stat().st_mtime)
        for req in reqs[:1]:
            rid = req.stem
            seen.add(rid)
            if not ID_RE.fullmatch(rid) or not (REQ / f"{rid}.sh").is_file():
                log(f"bad request {rid!r}"); finish(rid, 2, "REJECTED"); continue
            run_job(rid)
        (SPOOL / "HEARTBEAT").write_text(str(time.time()))
        time.sleep(2)


if __name__ == "__main__":
    main()
