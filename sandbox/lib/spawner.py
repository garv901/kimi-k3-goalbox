#!/usr/bin/env python3
"""Candidate-side process launcher: runs in the `runner` container of each GPU worker pod (uid 1000,
the candidate's uid; DECISIONS-LOG L19). It is *untrusted plumbing*: candidate code runs as the same uid
in the same container and can stop or alter it. Nothing it writes is a verdict; the grader (executor.py,
other container, other uid) decides every job's outcome from what it observes itself.

Shared with the grader: the pod-local emptyDir GOALBOX_JOB_DIR (default /job). Per job the grader creates
  /job/<rid>/            owned by the grader, 0755: the runner can read, not write
  /job/<rid>/job.json    {"kind": "exec"|"grade", "cwd", "env": {...}}            (0644)
  /job/<rid>/run.sh      the command (exec jobs)                                  (0644)
  /job/<rid>/out/        0777: this process writes stdout.log, exit, cleanup.json
  /job/<rid>/spawn/      0755 (grade jobs): the harness ranks create spawn requests in it, each a 0777
                         directory with cmd.json and two FIFOs; this process starts
                         `python -m megabench.harness.isolation --remote <dir>` for each and writes pid
  /job/<rid>/KILL, /job/<rid>/spawn/<id>/KILL   written by the grader: kill that process group
  /job/<rid>/END         written by the grader: kill everything of this job, write out/cleanup.json

Cleanup is session-scoped (every process in a job's sessions) and, with GOALBOX_SPAWNER_KILL_ALL_UID=1
(the pod: this container runs nothing else), uid-wide. A candidate can defeat this cleanup; the grader
then detects leftover GPU processes with NVML and has the container restarted (liveness probe on
/job/RESTART-RUNNER), which kills everything regardless of what ran here.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

VOLUME = Path(os.environ.get("GOALBOX_JOB_DIR", "/job"))
JOB = VOLUME              # the grader's root inside the volume, from VOLUME/GRADER (refreshed every tick)
KILL_ALL_UID = os.environ.get("GOALBOX_SPAWNER_KILL_ALL_UID") == "1"
PASS_ENV = ("PATH", "HOME", "LANG", "TERM", "CUDA_HOME", "PYTHONPATH", "KIMI_K3_WEIGHTS", "NVIDIA_IMEX_CHANNELS",
            "NCCL_MNNVL_ENABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "LD_LIBRARY_PATH", "TMPDIR",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
PYTHON = os.environ.get("GOALBOX_PYTHON") or ("/opt/goalbox/venv/bin/python" if os.path.exists("/opt/goalbox/venv/bin/python") else sys.executable)
RUNNER = VOLUME / "runner"
procs: dict[str, dict] = {}        # key -> {"proc", "dir", "job", "kind"}
sessions: dict[str, set[int]] = {}  # rid -> session ids (pids of the session leaders) this job started
handled_exec: set[str] = set()
ended: set[str] = set()


def log(message: str) -> None:
    try:
        with (RUNNER / "spawner.log").open("a") as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def job_env(spec: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV}
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items() if isinstance(k, str)})
    return env


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def start_exec(rid: str, spec: dict) -> None:
    job = JOB / rid
    cwd = spec.get("cwd") or "/tmp"
    out = (job / "out" / "stdout.log").open("ab")
    try:
        proc = subprocess.Popen(["bash", str(job / "run.sh")], cwd=cwd if os.path.isdir(cwd) else "/tmp",
                                stdout=out, stderr=subprocess.STDOUT, env=job_env(spec), start_new_session=True)
    except OSError as exc:
        out.write(f"spawner: could not start: {exc}\n".encode()); out.close()
        write_atomic(job / "out" / "exit", "127")
        return
    out.close()
    procs[f"exec:{rid}"] = {"proc": proc, "dir": job, "job": rid, "kind": "exec"}
    sessions.setdefault(rid, set()).add(proc.pid)
    log(f"{rid} exec pid {proc.pid}")


def start_child(rid: str, spawn: Path) -> None:
    try:
        request = json.loads((spawn / "cmd.json").read_text())
    except (OSError, ValueError) as exc:
        log(f"{rid} bad spawn request {spawn.name}: {exc}")
        return
    env = job_env({"env": request.get("env") or {}})
    env.setdefault("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    stderr = (spawn / "stderr").open("ab")
    cwd = request.get("cwd") or "/tmp"
    try:
        proc = subprocess.Popen([PYTHON, "-m", "megabench.harness.isolation", "--remote", str(spawn)],
                                cwd=cwd if os.path.isdir(cwd) else "/tmp", stdout=stderr, stderr=stderr, env=env,
                                start_new_session=True)
    except OSError as exc:
        stderr.write(f"spawner: could not start the candidate process: {exc}\n".encode()); stderr.close()
        write_atomic(spawn / "FAILED", f"spawn: {exc}")
        return
    stderr.close()
    write_atomic(spawn / "pid", str(proc.pid))
    procs[f"child:{rid}:{spawn.name}"] = {"proc": proc, "dir": spawn, "job": rid, "kind": "child"}
    sessions.setdefault(rid, set()).add(proc.pid)
    log(f"{rid} candidate process {spawn.name} pid {proc.pid}")


def killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _session_of(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as file:
            return int(file.read().rsplit(")", 1)[1].split()[3])
    except (OSError, IndexError, ValueError):
        return None


def sweep(sessions: set[int]) -> int:
    killed = 0
    for _ in range(3):
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            pid = int(entry.name)
            try:
                if KILL_ALL_UID:
                    if entry.stat().st_uid != os.getuid():
                        continue
                elif _session_of(pid) not in sessions:
                    continue
                os.kill(pid, signal.SIGKILL)
                killed += 1; found = True
            except (OSError, ProcessLookupError, PermissionError):
                continue
        if not found:
            break
        time.sleep(0.2)
    return killed


def end_job(rid: str) -> None:
    mine = [k for k, v in procs.items() if v["job"] == rid]
    for k in mine:
        killpg(procs[k]["proc"])
        procs[k]["proc"].wait()
        del procs[k]
    killed = sweep(sessions.pop(rid, set()))
    try:
        write_atomic(JOB / rid / "out" / "cleanup.json", json.dumps({"killed": killed, "uid_wide": KILL_ALL_UID, "utc": time.time()}))
    except OSError:
        pass
    log(f"{rid} ended, swept {killed}")
    ended.add(rid)


def grader_root() -> Path:
    try:
        name = (VOLUME / "GRADER").read_text().strip()
        if name.startswith("grader-") and "/" not in name and (VOLUME / name).is_dir():
            return VOLUME / name
    except OSError:
        pass
    return VOLUME


def tick() -> None:
    global JOB
    JOB = grader_root()
    for job in sorted(JOB.iterdir()) if JOB.is_dir() else []:
        rid = job.name
        if rid in ended or rid == "runner" or rid.startswith(".") or not (job / "job.json").is_file():
            continue
        if (job / "END").exists():
            end_job(rid); continue
        try:
            spec = json.loads((job / "job.json").read_text())
        except (OSError, ValueError):
            continue
        if spec.get("kind") == "exec" and rid not in handled_exec and (job / "out").is_dir():
            handled_exec.add(rid)
            start_exec(rid, spec)
        if spec.get("kind") == "grade" and (job / "spawn").is_dir():
            for spawn in sorted((job / "spawn").iterdir()):
                if (spawn / "cmd.json").is_file() and not (spawn / "pid").exists() and not (spawn / "FAILED").exists():
                    start_child(rid, spawn)
    for key, entry in list(procs.items()):
        proc, d = entry["proc"], entry["dir"]
        if (d / "KILL").exists() and proc.poll() is None:
            killpg(proc); log(f"{entry['job']} killed {key} on request")
        code = proc.poll()
        if code is not None:
            if entry["kind"] == "exec":
                write_atomic(d / "out" / "exit", str(code))
            else:
                try:
                    write_atomic(d / "exited", str(code))
                except OSError:
                    pass
            log(f"{entry['job']} {key} exited {code}")
            del procs[key]


def main() -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    write_atomic(RUNNER / "READY", json.dumps({"pid": os.getpid(), "uid": os.getuid(), "started_utc": time.time()}))
    log(f"spawner start uid={os.getuid()} job_dir={JOB} kill_all_uid={KILL_ALL_UID}")
    while True:
        try:
            tick()
        except Exception as exc:  # noqa: BLE001 - keep serving; the grader classifies what it does not see
            log(f"tick error: {type(exc).__name__}: {exc}")
        try:
            (RUNNER / "HEARTBEAT").write_text(str(time.time()))
        except OSError:
            pass
        time.sleep(0.5)


if __name__ == "__main__":
    main()
