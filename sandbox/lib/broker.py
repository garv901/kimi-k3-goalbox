#!/usr/bin/env python3
"""GPU broker for one goalbox run (runs on the host, outside the sandbox).

GOALBOX_GPU_MODE=slurm (default when sbatch exists): each request becomes an sbatch job.
GOALBOX_GPU_MODE=local: the broker runs each request itself on this machine's GPUs
(GOALBOX_LOCAL_GPUS, default all), one request at a time, killed at its time limit.

The sandboxed agent cannot reach Slurm. `gpu-run` inside the sandbox drops a request in
/home/agent/.gpu/req/ (= RUN_DIR/home/.gpu/req). The broker turns each request into an
sbatch job that runs the command in the same sandbox view, in `gpu` mode: no network,
only the job's GPUs. One job per run at a time; others wait in the spool.

Request:  req/<id>.json  {"minutes": int, "gpus": int, "cwd": "/work/..."}
          req/<id>.sh    the command (runs inside the sandbox; never parsed by the host)
Status:   status/<id>    last line is the current state; "done <exit> <slurm state>" at the end
Output:   out/<id>.log
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
GB = Path(__file__).resolve().parents[1]  # the sandbox/ directory
NAME = RUN.name
SPOOL = RUN / "home/.gpu"
import shlex, shutil, signal
PARTITION = os.environ.get("GOALBOX_PARTITION", "main")
MAX_MIN, MAX_GPUS = int(os.environ.get("GOALBOX_MAX_MIN", 120)), int(os.environ.get("GOALBOX_MAX_GPUS", 8))
MODE = os.environ.get("GOALBOX_GPU_MODE", "slurm" if shutil.which("sbatch") else "local")
LOCAL_GPUS = [g for g in os.environ.get("GOALBOX_LOCAL_GPUS", "").split(",") if g]
LOCAL = {}  # rid -> (Popen, deadline)
JOBS = RUN / "gpu"
LOG = (RUN / "broker.log").open("a", buffering=1)
for d in ("req", "status", "out"):
    (SPOOL / d).mkdir(parents=True, exist_ok=True)
JOBS.mkdir(exist_ok=True)


def log(msg):
    LOG.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def status(rid, line):
    with (SPOOL / "status" / rid).open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def job_state(jobid):
    q = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%T|%r|%S"], capture_output=True, text=True)
    if q.stdout.strip():
        return "active", q.stdout.strip()
    s = subprocess.run(["sacct", "-n", "-X", "-P", "-j", jobid, "-o", "State,ExitCode"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    if s:
        st, code = (s[0].split("|") + ["", ""])[:2]
        if st and not st.startswith(("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")):
            return "done", f"{code.split(':')[0] or '?'} {st}"
    return "active", "UNKNOWN"


def submit(rid):
    try:
        req = json.loads((SPOOL / "req" / f"{rid}.json").read_text())
        minutes, gpus, cwd = int(req["minutes"]), int(req["gpus"]), str(req["cwd"])
        assert 1 <= minutes <= MAX_MIN, f"minutes must be 1..{MAX_MIN}"
        assert 1 <= gpus <= MAX_GPUS, f"gpus must be 1..{MAX_GPUS}"
        assert re.fullmatch(r"/(work|home/agent|tmp)(/[A-Za-z0-9_.+-]+)*/?", cwd), "cwd must be under /work, /home/agent or /tmp"
        assert (SPOOL / "req" / f"{rid}.sh").is_file(), "missing command file"
    except Exception as e:
        status(rid, f"rejected: {e}"); status(rid, "done 2 REJECTED"); log(f"{rid} rejected: {e}")
        return None
    if MODE == "local":
        return submit_local(rid, minutes, gpus, cwd)
    script = JOBS / f"{rid}.sbatch"
    script.write_text(
        "#!/bin/bash\n"
        f"export GOALBOX_CWD='{cwd}'\n"
        f"exec '{GB}/lib/sandbox.sh' gpu '{RUN}' -- bash '/home/agent/.gpu/req/{rid}.sh'\n")
    cmd = ["sbatch", "--parsable", "-p", PARTITION, f"--gres=gpu:{gpus}", f"--time={minutes}",
           "-J", f"gb-{NAME}", "-o", str(SPOOL / "out" / f"{rid}.log"),
           *shlex.split(os.environ.get("GOALBOX_SBATCH_ARGS", "")), str(script)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"^(\d+)", r.stdout.strip(), re.M)
    if r.returncode or not m:
        msg = (r.stderr or r.stdout).strip().replace("\n", " | ")[:500]
        status(rid, f"sbatch failed: {msg}"); status(rid, "done 2 SUBMIT_FAILED"); log(f"{rid} sbatch failed: {msg}")
        return None
    jobid = m.group(1)
    status(rid, f"submitted job {jobid} on {PARTITION}, {gpus} GPU, {minutes} min")
    log(f"{rid} -> job {jobid} ({PARTITION}, {gpus} GPU, {minutes} min, cwd {cwd})")
    return jobid


def submit_local(rid, minutes, gpus, cwd):
    visible = LOCAL_GPUS or [str(i) for i in range(int(subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")))]
    if gpus > len(visible):
        status(rid, f"rejected: {gpus} GPUs requested, {len(visible)} available"); status(rid, "done 2 REJECTED")
        return None
    env = dict(os.environ, GOALBOX_CWD=cwd, CUDA_VISIBLE_DEVICES=",".join(visible[:gpus]))
    out = (SPOOL / "out" / f"{rid}.log").open("w")
    proc = subprocess.Popen([str(GB / "lib/sandbox.sh"), "gpu", str(RUN), "--", "bash", f"/home/agent/.gpu/req/{rid}.sh"],
                            stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    LOCAL[rid] = (proc, time.time() + 60 * minutes + 60)
    status(rid, f"running locally on GPUs {env['CUDA_VISIBLE_DEVICES']}, {minutes} min")
    log(f"{rid} -> local pid {proc.pid} (GPUs {env['CUDA_VISIBLE_DEVICES']}, {minutes} min, cwd {cwd})")
    return f"local:{rid}"


def local_state(rid):
    proc, deadline = LOCAL[rid]
    code = proc.poll()
    if code is None and time.time() > deadline:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(); return "done", "1 TIMEOUT"
    if code is None:
        return "active", "RUNNING"
    return "done", f"{code} {'COMPLETED' if code == 0 else 'FAILED'}"


def main():
    log(f"broker start mode={MODE} partition={PARTITION} max_min={MAX_MIN} max_gpus={MAX_GPUS}")
    active = None  # (rid, jobid, last)
    seen = {p.name for p in (SPOOL / "status").iterdir()}
    while True:
        if active:
            rid, jobid, last = active
            kind, info = local_state(rid) if jobid.startswith("local:") else job_state(jobid)
            if kind == "done":
                status(rid, f"done {info}"); log(f"{rid} job {jobid} done {info}"); active = None
            elif info != last:
                status(rid, info.replace("|", " ")); active = (rid, jobid, info)
        if not active:
            reqs = sorted((p for p in (SPOOL / "req").glob("*.json") if p.stem not in seen),
                          key=lambda p: p.stat().st_mtime)
            if reqs:
                rid = reqs[0].stem
                seen.add(rid)
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid):
                    log(f"bad request id {rid!r}"); continue
                jobid = submit(rid)
                if jobid:
                    active = (rid, jobid, "")
        time.sleep(3)


main()
