#!/usr/bin/env python3
"""Trusted GPU-job grader: runs in the `grader` container of each GPU worker pod (uid 2000; DECISIONS-LOG
L13, L19). Candidate code never runs in this container: it runs in the `runner` container (uid 1000,
spawner.py) of the same pod. The two share only the pod-local emptyDir GOALBOX_JOB_DIR (default /job).

Requests arrive on the controller-written spool GOALBOX_CTL_REQ/gpu/<node>/req/<rid>.json (read-only
mount; the runner has no mount of it at all). Results go to GOALBOX_CTL_RESULTS/gpu/<node>/ (mounted in
this container only; the runner cannot reach it):
  out/<rid>.log        what this process saw: its own lines, the harness output (grade) or the candidate's
                       stdout copied from the runner with a "candidate-authored" header (exec)
  verdict/<rid>.json   grade jobs: the harness record this process read from its private result file
  candidate/<rid>/     the candidate processes' stderr files, copied from the runner (untrusted content)
  done/<rid>.json      {"exit", "state", "kind", "trusted", ...}: written last, by this process only

Two kinds of job. `exec` runs the agent's command in the runner; its exit code and output are
candidate-reported and the done record says so (trusted: false). `grade` runs the evaluator
(`python -m megabench evaluate`) *here*, with the candidate processes spawned remotely in the runner
(GOALBOX_SPAWN_DIR, megabench.harness.isolation remote mode): the verdict is what the evaluator computed
from the outputs it received and hashed itself (trusted: true). The graded sources are a snapshot this
process copies into /job/<rid>/candidate (read-only to the runner) before the scan, so what was scanned is
what was imported. Harness options come from the request as the controller's broker wrote them, never
from the agent.

Outcome classes decided here, independently of anything the candidate wrote: COMPLETED/FAILED (exec,
candidate-reported exit), TIMEOUT (this process's deadline), the evaluator status (grade: ok_provisional,
incorrect, candidate_error, ...), GRADER_FAILED (evaluator exited without a record), INFRA_RUNNER_DOWN
(no spawner heartbeat), REJECTED. Leftover GPU processes after a job (NVML) make this process request a
restart of the runner container (/job/RESTART-RUNNER, liveness probe) and tag the state.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

NODE = os.environ.get("GOALBOX_NODE_NAME") or socket.gethostname()
GPUS_HERE = int(os.environ.get("GOALBOX_GPUS_HERE", "4"))
REQ = Path(os.environ.get("GOALBOX_CTL_REQ", "/ctl-req")) / "gpu" / NODE / "req"
SPOOL = Path(os.environ.get("GOALBOX_CTL_RESULTS", "/ctl-results")) / "gpu" / NODE
JOB_VOLUME = Path(os.environ.get("GOALBOX_JOB_DIR", "/job"))
JOB = JOB_VOLUME          # replaced in main() by a grader-owned, unpredictably named root (the runner cannot pre-create entries)
PRIVATE = Path(os.environ.get("GOALBOX_GRADER_TMP", "/tmp")) / "grader"
SNAPSHOT_LIMIT = int(os.environ.get("GOALBOX_SNAPSHOT_LIMIT_BYTES", str(4 << 30)))
PYTHON = os.environ.get("GOALBOX_PYTHON") or ("/opt/goalbox/venv/bin/python" if os.path.exists("/opt/goalbox/venv/bin/python") else sys.executable)
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
CASE_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")
PASS_ENV = ("PATH", "HOME", "LANG", "TERM", "CUDA_HOME", "PYTHONPATH", "KIMI_K3_WEIGHTS", "NVIDIA_IMEX_CHANNELS",
            "NCCL_MNNVL_ENABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "LD_LIBRARY_PATH", "TMPDIR",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
TOPO_KEYS = ("GOALBOX_NNODES", "GOALBOX_NODE_RANK", "GOALBOX_MASTER_ADDR", "GOALBOX_MASTER_PORT")
RUNNER_STALE_S = float(os.environ.get("GOALBOX_RUNNER_STALE_S", "30"))


def log(message: str) -> None:
    with (SPOOL / "executor.log").open("a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def status(rid: str, line: str) -> None:
    with (SPOOL / "status" / rid).open("a") as file:
        file.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def weights_fingerprint(manifest: Path) -> dict:
    files = json.loads(manifest.read_text())["files"]
    return {name: [os.stat(name).st_size, int(os.stat(name).st_mtime)] for name in files if os.path.exists(name)}


# ------------------------------------------------------------------------------------------ runner state
def runner_state() -> dict:
    """What the untrusted spawner says about itself; used only to classify infrastructure failures."""
    try:
        ready = json.loads((JOB_VOLUME / "runner" / "READY").read_text())
        beat = float((JOB_VOLUME / "runner" / "HEARTBEAT").read_text())
    except (OSError, ValueError):
        return {"up": False}
    return {"up": time.time() - beat < RUNNER_STALE_S, "started_utc": ready.get("started_utc", 0), "uid": ready.get("uid")}


def gpu_processes() -> list[str]:
    """Compute processes on the visible GPUs (NVML view; pids are the host's, only presence matters).
    GOALBOX_LEFTOVER_CHECK=0 disables the check (CPU tests on shared hosts)."""
    if os.environ.get("GOALBOX_LEFTOVER_CHECK", "1") == "0" or not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def request_runner_restart(reason: str) -> None:
    write_atomic(JOB_VOLUME / "RESTART-RUNNER", json.dumps({"utc": time.time(), "reason": reason}))
    log(f"runner restart requested: {reason}")


def clear_restart_if_restarted() -> None:
    flag = JOB_VOLUME / "RESTART-RUNNER"
    if not flag.exists():
        return
    try:
        asked = json.loads(flag.read_text())["utc"]
    except (OSError, ValueError, KeyError):
        asked = 0
    state = runner_state()
    if state.get("started_utc", 0) > asked:
        flag.unlink(missing_ok=True)
        log("runner restarted; restart flag cleared")


# ---------------------------------------------------------------------------------------------- job dirs
def make_job_dir(rid: str, spec: dict) -> Path:
    job = JOB / rid
    if job.exists():
        shutil.rmtree(job, ignore_errors=True)
        if job.exists():
            job.rename(JOB / f".stale-{rid}-{int(time.time())}")
    job.mkdir(mode=0o755)
    os.chmod(job, 0o755)
    (job / "out").mkdir(mode=0o777); os.chmod(job / "out", 0o777)
    if spec["kind"] == "grade":
        (job / "spawn").mkdir(mode=0o755); os.chmod(job / "spawn", 0o755)
    write_atomic(job / "job.json", json.dumps(spec), 0o644)
    return job


def discard_job_dir(job: Path) -> None:
    shutil.rmtree(job, ignore_errors=True)
    if job.exists():                     # entries the runner created in its own subdirectories cannot be removed
        try:
            job.rename(JOB / f".trash-{job.name}-{int(time.time())}")
        except OSError:
            pass


def snapshot_candidate(submission: Path, job: Path) -> tuple[Path, dict]:
    """Copy the submission's directory into the job (read-only to the runner); return path and digest."""
    src = submission.parent
    dst = job / "candidate"
    total = 0
    files = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] in (".git", "__pycache__", ".venv", "reports"):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
        if total > SNAPSHOT_LIMIT:
            raise ValueError(f"candidate directory exceeds {SNAPSHOT_LIMIT} bytes")
        files.append(rel)
    digest = hashlib.sha256()
    for rel in files:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, target)
        os.chmod(target, 0o644)
        digest.update(str(rel).encode() + b"\0" + hashlib.sha256(target.read_bytes()).digest())
    for d in [dst] + [p for p in dst.rglob("*") if p.is_dir()]:
        os.chmod(d, 0o755)
    return dst / submission.name, {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


# -------------------------------------------------------------------------------------------------- jobs
def job_env(req: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV}
    gpus = int(req.get("gpus", 1))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(min(max(gpus, 1), GPUS_HERE)))
    for key in TOPO_KEYS:
        if key in (req.get("topology") or {}):
            env[key] = str(req["topology"][key])
    return env


def run_exec(rid: str, req: dict, out, deadline: float) -> tuple[int, str, dict]:
    """The agent's command, run by the untrusted spawner. Everything about its result is candidate-reported."""
    env = job_env(req)
    env["GOALBOX_JOB_ID"] = rid
    spec = {"kind": "exec", "cwd": req.get("cwd", "/work"), "env": env}
    job = make_job_dir(rid, spec)
    shutil.copyfile(REQ / f"{rid}.sh", job / "run.sh"); os.chmod(job / "run.sh", 0o644)
    exit_file = job / "out" / "exit"
    state = "COMPLETED"
    while not exit_file.exists():
        if time.time() > deadline:
            write_atomic(job / "KILL", str(time.time()))
            state = "TIMEOUT"
            break
        if not runner_state()["up"]:
            state = "INFRA_RUNNER_DOWN"
            break
        time.sleep(1)
    reported = None
    if state == "COMPLETED":
        try:
            reported = int(exit_file.read_text().strip())
        except (OSError, ValueError):
            state = "MALFORMED_EXIT"
    out.write("## candidate-authored output (runner container; not authoritative) ##\n"); out.flush()
    try:
        with (job / "out" / "stdout.log").open("rb") as src:
            shutil.copyfileobj(src, out.buffer)
        out.buffer.flush()
    except OSError:
        out.write("(no output file)\n")
    out.write("## end of candidate-authored output ##\n")
    code = reported if reported is not None else {"TIMEOUT": 124, "INFRA_RUNNER_DOWN": 8}.get(state, 3)
    if state == "COMPLETED" and code != 0:
        state = "FAILED"
    return code, state, {"job": job, "reported_exit": reported}


def run_grade(rid: str, req: dict, out, deadline: float) -> tuple[int, str, dict]:
    """The evaluator runs here; only the candidate processes run in the runner."""
    from_status = {}
    case, submission = req.get("case", ""), Path(req.get("submission", ""))
    if not CASE_RE.fullmatch(case) or not submission.is_absolute() or not submission.is_file():
        out.write(f"grader: rejected grade request (case {case!r}, submission {submission})\n")
        return 2, "REJECTED", {}
    env = job_env(req)
    env["GOALBOX_JOB_ID"] = rid
    spec = {"kind": "grade", "cwd": "/tmp", "env": env}
    job = make_job_dir(rid, spec)
    try:
        graded_path, snapshot = snapshot_candidate(submission, job)
    except (OSError, ValueError) as exc:
        out.write(f"grader: cannot snapshot the candidate: {exc}\n")
        discard_job_dir(job)
        return 2, "REJECTED", {}
    out.write(f"grader: snapshot of {submission.parent} -> {snapshot['files']} files, {snapshot['bytes']} bytes, "
              f"sha256 {snapshot['sha256']}\n")
    private = PRIVATE / rid
    private.mkdir(parents=True, exist_ok=True)
    result_path = private / "result.jsonl"
    harness_args = [str(a) for a in (req.get("harness_args") or [])]
    cmd = [PYTHON, "-m", "megabench", "evaluate", "--case", case, "--submission", str(graded_path),
           "--method", "goalbox-grade", "--session-id", rid, "--device", str(req.get("device", "cuda:0")),
           *harness_args, "--output", str(result_path)]
    henv = dict(env, GOALBOX_SPAWN_DIR=str(job / "spawn"), TMPDIR=str(private), HOME=str(private))
    out.write(f"grader: {' '.join(cmd[2:])}\n"); out.flush()
    proc = subprocess.Popen(cmd, cwd=str(private), stdout=out, stderr=subprocess.STDOUT, env=henv, start_new_session=True)
    state = "GRADED"
    while proc.poll() is None:
        if time.time() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
            for spawn in (job / "spawn").iterdir():
                write_atomic(spawn / "KILL", str(time.time()))
            state = "TIMEOUT"
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    record = None
    if result_path.exists():
        try:
            record = json.loads(result_path.read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            record = None
    (SPOOL / "candidate" / rid).mkdir(parents=True, exist_ok=True)
    for spawn in sorted((job / "spawn").iterdir()):
        for name in ("stderr", "exited", "FAILED"):
            if (spawn / name).is_file():
                try:
                    shutil.copyfile(spawn / name, SPOOL / "candidate" / rid / f"{spawn.name}.{name}")
                except OSError:
                    pass
    (SPOOL / "candidate" / rid / "README").write_text("Files written by candidate processes in the runner container; not authoritative.\n")
    if state == "TIMEOUT":
        code, state = 124, "TIMEOUT"
    elif record is None:
        code, state = 7, "GRADER_FAILED"
        from_status = {"grader_exit": proc.returncode}
    else:
        code = int(record.get("exit_code", record.get("worker_exit_code", 1)) or 0) if "exit_code" in record else None
        state = str(record.get("status", "GRADER_FAILED"))[:40]
    verdict = {"rid": rid, "node": NODE, "case": case, "submission": str(submission), "graded_snapshot": snapshot,
               "harness_args": harness_args, "grader_exit": proc.returncode, "state": state, "utc": time.time(),
               "record": record}
    write_atomic(SPOOL / "verdict" / f"{rid}.json", json.dumps(verdict))
    if code is None:
        from_status = {"status": state}
        code = _exit_code_for(state)
    out.write(f"grader: verdict {state} (exit {code}); record at verdict/{rid}.json\n")
    return code, state, {"job": job, "verdict_sha256": hashlib.sha256(json.dumps(verdict).encode()).hexdigest(), **from_status}


def _exit_code_for(status_name: str) -> int:
    try:
        sys.path.insert(0, os.environ.get("GOALBOX_BENCH", "/goalbox/bench"))
        from megabench.harness.status import EXIT_CODES
        return EXIT_CODES.get(status_name, 1)
    except Exception:  # noqa: BLE001
        return 0 if status_name in ("ok_provisional", "correctness_only") else 1


def run_job(rid: str) -> None:
    try:
        req = json.loads((REQ / f"{rid}.json").read_text())
        minutes = int(req["minutes"])
        kind = req.get("kind", "exec")
        assert kind in ("exec", "grade")
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        log(f"{rid} bad request: {exc}"); finish(rid, 2, "REJECTED", "exec", False); return
    if kind == "exec" and not (REQ / f"{rid}.sh").is_file():
        finish(rid, 2, "REJECTED", kind, False); return
    manifest = os.environ.get("GOALBOX_WEIGHTS_MANIFEST")
    before = weights_fingerprint(Path(manifest)) if manifest else None
    out = (SPOOL / "out" / f"{rid}.log").open("w", encoding="utf-8", errors="replace")
    status(rid, f"{kind} on {NODE}, {minutes} min")
    log(f"{rid} start kind={kind} gpus={req.get('gpus')} topo={req.get('topology')}")
    if not runner_state()["up"]:
        out.write("grader: the runner container is not ready (no spawner heartbeat)\n"); out.close()
        request_runner_restart("no heartbeat before job")
        finish(rid, 8, "INFRA_RUNNER_DOWN", kind, kind == "grade"); return
    deadline = time.time() + 60 * minutes + 60
    gpu_before = gpu_processes()
    try:
        code, state, extra = (run_grade if kind == "grade" else run_exec)(rid, req, out, deadline)
    except Exception as exc:  # noqa: BLE001 - never leave a job without a done record
        log(f"{rid} grader exception: {type(exc).__name__}: {exc}")
        out.write(f"grader: internal error {type(exc).__name__}: {exc}\n")
        code, state, extra = 7, "GRADER_FAILED", {}
    job = extra.pop("job", None)
    cleanup = None
    if job is not None:
        write_atomic(job / "END", str(time.time()))
        for _ in range(40):
            if (job / "out" / "cleanup.json").exists():
                try:
                    cleanup = json.loads((job / "out" / "cleanup.json").read_text())
                except (OSError, ValueError):
                    cleanup = {"malformed": True}
                break
            time.sleep(0.5)
        if cleanup and cleanup.get("killed"):
            out.write(f"runner cleanup (candidate-side report): killed {cleanup['killed']} leftover process(es)\n")
            state += "+STRAYS_KILLED"
    leftovers = sorted(set(gpu_processes()) - set(gpu_before))      # only what appeared during the job
    if leftovers:
        out.write(f"grader: {len(leftovers)} GPU process(es) still present after the job; runner restart requested\n")
        request_runner_restart(f"{rid}: {len(leftovers)} leftover GPU process(es)")
        state += "+RUNNER_RESTART"
    if before is not None and weights_fingerprint(Path(manifest)) != before:
        out.write("grader: WEIGHT FILES CHANGED during the job\n")
        state += "+WEIGHTS_CHANGED"
        code = code or 3
    out.close()
    if job is not None:
        discard_job_dir(job)
    finish(rid, code, state, kind, kind == "grade", **extra)


def finish(rid: str, code: int, state: str, kind: str, trusted: bool, **extra) -> None:
    status(rid, f"done {code} {state}")
    record = {"exit": code, "state": state, "kind": kind, "trusted": trusted, "node": NODE, "finished_utc": time.time(), **extra}
    write_atomic(SPOOL / "done" / f"{rid}.json", json.dumps(record))
    log(f"{rid} done {code} {state} trusted={trusted}")


def main() -> None:
    for d in ("status", "out", "done", "verdict", "candidate"):
        (SPOOL / d).mkdir(parents=True, exist_ok=True)
    try:
        REQ.mkdir(parents=True, exist_ok=True)      # tests on a host; in the pod the mount is read-only and pre-created
    except OSError:
        pass
    global JOB
    JOB_VOLUME.mkdir(parents=True, exist_ok=True)
    JOB = JOB_VOLUME / f"grader-{os.urandom(6).hex()}"
    JOB.mkdir(mode=0o755); os.chmod(JOB, 0o755)
    write_atomic(JOB_VOLUME / "GRADER", JOB.name, 0o644)      # tells the spawner where this grader's jobs are
    for stale in JOB_VOLUME.glob("grader-*"):                 # a previous grader incarnation (container restart)
        if stale != JOB:
            shutil.rmtree(stale, ignore_errors=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_atomic(SPOOL / "READY", json.dumps({"node": NODE, "gpus": GPUS_HERE, "pid": os.getpid(), "uid": os.getuid(),
                                              "started_utc": time.time()}))
    log(f"grader start node={NODE} gpus={GPUS_HERE} uid={os.getuid()} job_dir={JOB}")
    seen = {p.stem for p in (SPOOL / "done").glob("*.json")}
    while True:
        clear_restart_if_restarted()
        reqs = sorted((p for p in REQ.glob("*.json") if p.stem not in seen), key=lambda p: p.stat().st_mtime)
        for req in reqs[:1]:
            rid = req.stem
            seen.add(rid)
            if not ID_RE.fullmatch(rid):
                log(f"bad request id {rid!r}"); finish(rid, 2, "REJECTED", "exec", False); continue
            run_job(rid)
        (SPOOL / "HEARTBEAT").write_text(str(time.time()))
        time.sleep(2)


if __name__ == "__main__":
    main()
