#!/usr/bin/env python3
"""GPU broker for one goalbox run (runs on the host, outside the sandbox).

GOALBOX_GPU_MODE=slurm (default when sbatch exists): each request becomes an sbatch job.
GOALBOX_GPU_MODE=local: the broker runs each request itself on this machine's GPUs
(GOALBOX_LOCAL_GPUS, default all), one request at a time, killed at its time limit.
GOALBOX_GPU_MODE=k8s (controller pod, DECISIONS-LOG L2): the broker copies each request into the
controller-owned spool GOALBOX_CTL/req/gpu/<node>/req (mounted read-only in the GPU pods) for the
executor (executor.py) in the GPU worker pods named in GOALBOX_GPU_NODES (comma list, 4 GPUs each),
and reads their results under GOALBOX_CTL/results/gpu/<node> (read-write in the GPU pods, so
untrusted: it is only ever reported back to the agent). 1..4 GPUs run on the first worker; 8 GPUs
run on both with GOALBOX_NNODES=2, node ranks 0/1 and the first worker as rendezvous master
(GOALBOX_MASTER_PORT, default 29500). The two outputs are concatenated into one log.

The sandboxed agent cannot reach Slurm. `gpu-run` inside the sandbox drops a request in
/home/agent/.gpu/req/ (= RUN_DIR/home/.gpu/req). The broker turns each request into an
sbatch job that runs the command in the same sandbox view, in `gpu` mode: no network,
only the job's GPUs. One job per run at a time; others wait in the spool.

Request:  req/<id>.json  {"minutes": int, "gpus": int, "cwd": "/work/..."}            (kind "exec", default)
                         {"kind": "grade", "case": ID, "submission": "/work/...", "minutes": int}
          req/<id>.sh    the command (exec; runs in the runner container; never parsed by the host)
Grade requests (DECISIONS-LOG L19): the broker fixes the GPU count from the case, the harness options
(GOALBOX_GRADE_ARGS) and the device; the agent chooses only the case and the submission path under /work.
In k8s mode the trusted grader evaluates it; its done record carries "trusted": true and a verdict file.
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
PARTITION = os.environ.get("GOALBOX_PARTITION") or None  # None: the cluster's default partition
MAX_MIN, MAX_GPUS = int(os.environ.get("GOALBOX_MAX_MIN", 120)), int(os.environ.get("GOALBOX_MAX_GPUS", 8))
MODE = os.environ.get("GOALBOX_GPU_MODE", "slurm" if shutil.which("sbatch") else "local")
CTL = Path(os.environ.get("GOALBOX_CTL", "/ctl"))
GPU_NODES = [n for n in os.environ.get("GOALBOX_GPU_NODES", "").split(",") if n]
MASTER_PORT = int(os.environ.get("GOALBOX_MASTER_PORT", "29500"))
GRADE_ARGS = os.environ.get("GOALBOX_GRADE_ARGS", "--trials 2 --warmup 2 --reps 10 --timeout 3600").split()
GRADE_DEVICE = os.environ.get("GOALBOX_GRADE_DEVICE", "cuda:0")
WORK_ROOT = os.environ.get("GOALBOX_WORK_ROOT", "/work")
DISPATCHED = {}  # rid -> dispatch time (a done record older than its request is refused)
K8S = {}  # rid -> list of nodes the job was dispatched to
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


def case_gpus(case_id):
    """GPU count of a graded case from the benchmark's own case table (never from the agent)."""
    sys.path.insert(0, os.environ.get("GOALBOX_BENCH", str(GB.parent)))
    from megabench.cases import select_cases
    return int(select_cases("all", [case_id])[0].gpus)


def submit(rid):
    grade = None
    try:
        req = json.loads((SPOOL / "req" / f"{rid}.json").read_text())
        kind = req.get("kind", "exec")
        assert kind in ("exec", "grade"), "kind must be exec or grade"
        minutes, cwd = int(req["minutes"]), str(req["cwd"])
        assert 1 <= minutes <= MAX_MIN, f"minutes must be 1..{MAX_MIN}"
        assert re.fullmatch(r"/(work|home/agent|tmp)(/[A-Za-z0-9_.+-]+)*/?", cwd), "cwd must be under /work, /home/agent or /tmp"
        if kind == "grade":
            case = str(req["case"])
            assert re.fullmatch(r"[A-Za-z0-9._-]{1,80}", case), "bad case id"
            submission = str(req["submission"])
            assert re.fullmatch(re.escape(WORK_ROOT) + r"(/[A-Za-z0-9_.+-]+)+", submission), f"submission must be a file under {WORK_ROOT}"
            gpus = case_gpus(case)
            grade = {"kind": "grade", "case": case, "submission": submission, "harness_args": GRADE_ARGS, "device": GRADE_DEVICE}
        else:
            gpus = int(req["gpus"])
            assert 1 <= gpus <= MAX_GPUS, f"gpus must be 1..{MAX_GPUS}"
            assert (SPOOL / "req" / f"{rid}.sh").is_file(), "missing command file"
    except Exception as e:
        status(rid, f"rejected: {e}"); status(rid, "done 2 REJECTED"); log(f"{rid} rejected: {e}")
        return None
    if MODE == "k8s":
        return submit_k8s(rid, minutes, gpus, cwd, grade)
    if grade:   # local and slurm modes have no separate grader: run the evaluator as the job's command
        (SPOOL / "req" / f"{rid}.sh").write_text(
            f"exec python -m megabench evaluate --case {grade['case']} --submission {grade['submission']} --method goalbox-grade "
            f"--session-id {rid} --device {grade['device']} {' '.join(grade['harness_args'])} --output /tmp/grade-{rid}.jsonl\n")
    if MODE == "local":
        return submit_local(rid, minutes, gpus, cwd)
    script = JOBS / f"{rid}.sbatch"
    script.write_text(
        "#!/bin/bash\n"
        f"export GOALBOX_CWD='{cwd}'\n"
        f"exec '{GB}/lib/sandbox.sh' gpu '{RUN}' -- bash '/home/agent/.gpu/req/{rid}.sh'\n")
    cmd = ["sbatch", "--parsable", *(["-p", PARTITION] if PARTITION else []), f"--gres=gpu:{gpus}", f"--time={minutes}",
           "-J", f"gb-{NAME}", "-o", str(SPOOL / "out" / f"{rid}.log"),
           *shlex.split(os.environ.get("GOALBOX_SBATCH_ARGS", "")), str(script)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"^(\d+)", r.stdout.strip(), re.M)
    if r.returncode or not m:
        msg = (r.stderr or r.stdout).strip().replace("\n", " | ")[:500]
        status(rid, f"sbatch failed: {msg}"); status(rid, "done 2 SUBMIT_FAILED"); log(f"{rid} sbatch failed: {msg}")
        return None
    jobid = m.group(1)
    status(rid, f"submitted job {jobid} on {PARTITION or 'the default partition'}, {gpus} GPU, {minutes} min")
    log(f"{rid} -> job {jobid} ({PARTITION or 'default partition'}, {gpus} GPU, {minutes} min, cwd {cwd})")
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


def submit_k8s(rid, minutes, gpus, cwd, grade=None):
    per_node = 4
    if gpus > per_node and gpus != 2 * per_node:
        status(rid, f"rejected: {gpus} GPUs: use 1..{per_node} (one tray) or {2*per_node} (two trays)")
        status(rid, "done 2 REJECTED"); return None
    nodes = GPU_NODES[:1] if gpus <= per_node else GPU_NODES[:2]
    if len(nodes) < (1 if gpus <= per_node else 2):
        status(rid, "rejected: not enough GPU workers configured"); status(rid, "done 2 REJECTED"); return None
    script = (SPOOL / "req" / f"{rid}.sh").read_bytes() if grade is None else None
    for index, node in enumerate(nodes):
        spool = CTL / "req" / "gpu" / node
        (spool / "req").mkdir(parents=True, exist_ok=True)
        for d in ("status", "out", "done", "verdict", "candidate"):
            (CTL / "results" / "gpu" / node / d).mkdir(parents=True, exist_ok=True)
        topology = ({"GOALBOX_NNODES": 2, "GOALBOX_NODE_RANK": index, "GOALBOX_MASTER_ADDR": nodes[0],
                     "GOALBOX_MASTER_PORT": MASTER_PORT} if len(nodes) == 2 else {})
        if script is not None:
            (spool / "req" / f"{rid}.sh").write_bytes(script)
        request = {"kind": "exec", "minutes": minutes, "gpus": min(gpus, per_node), "cwd": cwd, "topology": topology}
        if grade:
            request |= grade
        tmp = spool / "req" / f".{rid}.tmp"
        tmp.write_text(json.dumps(request))
        os.replace(tmp, spool / "req" / f"{rid}.json")
    K8S[rid] = nodes
    DISPATCHED[rid] = time.time()
    status(rid, f"dispatched to {', '.join(nodes)} ({gpus} GPU, {minutes} min)")
    log(f"{rid} -> k8s {nodes} ({gpus} GPU, {minutes} min, cwd {cwd})")
    return f"k8s:{rid}"


def k8s_state(rid):
    """The grader's done records decide the state. Grade jobs: trusted verdict (the evaluator ran in the
    grader container). Exec jobs: the exit code is candidate-reported and labelled so. A done record older
    than the dispatch, or malformed, is an infrastructure-class failure, never a pass."""
    nodes = K8S[rid]
    done = [CTL / "results" / "gpu" / n / "done" / f"{rid}.json" for n in nodes]
    if not all(p.exists() for p in done):
        running = [n for n, p in zip(nodes, done) if not p.exists()]
        return "active", "RUNNING on " + " ".join(running)
    records = []
    for p in done:
        try:
            r = json.loads(p.read_text())
            codes_ok = isinstance(r.get("exit"), int) and isinstance(r.get("state"), str)
            assert codes_ok and float(r.get("finished_utc", 0)) >= DISPATCHED.get(rid, 0) - 5, "stale or malformed done record"
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            log(f"{rid} refused done record {p}: {e}")
            r = {"exit": 8, "state": "MALFORMED_RESULT", "kind": "exec", "trusted": False}
        records.append(r)
    trusted = all(r.get("trusted") is True for r in records)
    with (SPOOL / "out" / f"{rid}.log").open("w") as out:
        for node, record in zip(nodes, records):
            tag = "trusted grader verdict" if record.get("trusted") else "candidate-reported exit"
            out.write(f"===== {node} (exit {record.get('exit')} {str(record.get('state'))[:40]}; {tag})\n")
            src = CTL / "results" / "gpu" / node / "out" / f"{rid}.log"
            out.write(src.read_text(errors="replace") if src.exists() else "(no output)\n")
            verdict = CTL / "results" / "gpu" / node / "verdict" / f"{rid}.json"
            if record.get("trusted") and verdict.exists():
                try:
                    v = json.loads(verdict.read_text())
                    rec = v.get("record") or {}
                    out.write(f"----- verdict: {v.get('state')} {('- ' + str(rec.get('reason'))) if rec.get('reason') else ''}"
                              f" (graded snapshot sha256 {str(v.get('graded_snapshot', {}).get('sha256', '?'))[:16]})\n")
                except (OSError, ValueError):
                    out.write("----- verdict file unreadable\n")
    code = max(int(r["exit"]) for r in records)
    states = sorted({str(r.get("state"))[:40] for r in records})
    label = "COMPLETED" if code == 0 and not trusted else "+".join(states)
    return "done", f"{code} {label}" + ("" if trusted else " (candidate-reported)")


def local_state(rid):
    proc, deadline = LOCAL[rid]
    code = proc.poll()
    if code is None and time.time() > deadline:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(); return "done", "1 TIMEOUT"
    if code is None:
        return "active", "RUNNING"
    return "done", f"{code} {'COMPLETED' if code == 0 else 'FAILED'}"


def main():
    log(f"broker start mode={MODE} partition={PARTITION or 'default'} max_min={MAX_MIN} max_gpus={MAX_GPUS}"
        + (f" nodes={GPU_NODES}" if MODE == "k8s" else ""))
    active = None  # (rid, jobid, last)
    seen = {p.name for p in (SPOOL / "status").iterdir()}
    while True:
        if active:
            rid, jobid, last = active
            kind, info = (local_state(rid) if jobid.startswith("local:") else
                          k8s_state(rid) if jobid.startswith("k8s:") else job_state(jobid))
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


if __name__ == "__main__":
    main()
