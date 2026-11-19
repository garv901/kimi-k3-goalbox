"""Evaluate an agent's submitted megakernels against independent workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..cases import Case, select_cases
from ..sol import (DEFAULT_HBM_BANDWIDTH_SOURCE,
                   DEFAULT_HBM_BANDWIDTH_TB_S, estimate_case)
from ..workloads import make_inputs, reference
from .benchmark import _audit_launches, _measure  # noqa: F401 - re-exported for tooling
from .correctness import check_trials  # noqa: F401 - re-exported for tooling
from .scan import scan_submission
from .status import EXIT_CODES, STATUSES


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_FILES = (
    "megabench/cases.py", "megabench/runner.py", "megabench/workloads.py",
    "megabench/sol.py",
    "megabench/harness/runner.py", "megabench/harness/correctness.py",
    "megabench/harness/benchmark.py", "megabench/harness/distributed.py",
    "megabench/tasks/workloads.py", "megabench/tasks/common.py",
    "megabench/tasks/quantization.py", "megabench/tasks/distributed.py",
    "megabench/tasks/parallel.py", "megabench/tasks/kimi.py",
    "megabench/tasks/kimi_checkpoint.py",
    "megabench/harness/status.py", "megabench/harness/isolation.py", "megabench/harness/launch.py",
    "megabench/harness/deadlines.py", "megabench/harness/scan.py", "megabench/harness/nsys_audit.py",
)


def _contract_digest() -> str:
    digest = hashlib.sha256()
    for name in CONTRACT_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _load_submission(path: Path, case: Case, *, execution: dict | None = None) -> Callable:
    spec = importlib.util.spec_from_file_location("megabench_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import submission {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError("submission must export build(case: dict) -> callable")
    contract = case.to_dict()
    if execution is not None:
        contract["execution"] = execution
    runner = build(contract)
    if not callable(runner):
        raise TypeError("build(case) must return a callable run(inputs)")
    return runner


def provenance(case: Case, submission: Path, *, device: str) -> dict:
    """Fields every record keeps, pass or fail (ISSUES item 11)."""
    import torch

    source = submission.read_bytes()
    return {"case": case.to_dict(), "submission": str(submission),
            "submission_sha256": hashlib.sha256(source).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": {"hostname": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": device, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}}


def evaluate_case(case: Case, submission: Path, *, device: str, trials: int, warmup: int, reps: int,
                  graph_baseline: bool = True, scan: bool = True, reference_reps: int | None = None) -> dict:
    """Evaluate one case. Every case runs through the per-rank path with the candidate in its own
    process (D-I2); TP1 is a world of one. `graph_baseline` is accepted for compatibility and ignored
    (the oracle cannot be graph-captured; ISSUES item 3)."""
    import torch

    record = provenance(case, submission, device=device)
    if not case.ready:
        return record | {"status": "not_implemented", "reason": case.note}
    if scan:
        record["scan"] = scan_submission(submission)
        if record["scan"]["status"] != "pass":
            rules = ", ".join(record["scan"]["rules_hit"])
            return record | {"status": "scan_failed", "reason": f"automated source/artifact scan: {rules}"}
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            return record | {"status": "unavailable", "reason": "CUDA not visible"}
        props = torch.cuda.get_device_properties(device)
        record["hardware"].update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
    from .distributed import evaluate_distributed
    result = evaluate_distributed(case, submission, device=device, trials=trials, warmup=warmup, reps=reps,
                                  reference_reps=reference_reps)
    record |= result
    record.setdefault("launch_audit", {"status": "see nsys_audit", "max_gpu_launches_per_rank": case.max_gpu_launches,
                                       "note": "the graded audit is the nsys trace of the timed repetitions "
                                               "(megabench.harness.nsys_audit); in-process counts are advisory"})
    return record


def _worker(args: argparse.Namespace) -> int:
    case = select_cases("all", [args.case])[0]
    submission = Path(args.submission).resolve()
    try:
        result = evaluate_case(case, submission, device=args.device, trials=args.trials,
                               warmup=args.warmup, reps=args.reps, scan=not args.no_scan,
                               reference_reps=args.reference_reps)
    except Exception as exc:
        from .status import classify
        failure = classify(exc, phase="setup")
        try:
            result = provenance(case, submission, device=args.device)
        except Exception:  # noqa: BLE001
            result = {"case": case.to_dict(), "submission": str(submission)}
        result |= {"status": failure.status, "reason": failure.message, "failure": failure.record(),
                   "traceback": traceback.format_exc(limit=12)}
    if result.get("status") not in STATUSES and result.get("status") not in ("node_done",):
        result["status_note"] = f"non-standard status {result.get('status')!r}"
    with Path(args.result).open("x", encoding="utf-8") as file:
        json.dump(result, file)
    return 0


def _docker_worker_command(args: argparse.Namespace, case: Case,
                           tempdir: Path, result_path: Path) -> tuple[list[str], str, str]:
    """Run only this case in a disposable, unprivileged container."""
    submission = Path(args.submission).resolve()
    run_id = uuid.uuid4().hex
    name = f"megabench-{run_id}"
    label = f"org.megabench.run={run_id}"
    if submission.is_relative_to(ROOT):
        container_submission = Path("/workspace") / submission.relative_to(ROOT)
        extra_mount: list[str] = []
    else:
        container_submission = Path("/submission") / submission.name
        extra_mount = ["--volume", f"{submission.parent}:/submission:ro"]
    cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
           "--label", label, "--network", "none", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--pids-limit", "1024", "--shm-size", "1g",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--tmpfs", "/tmp:rw,exec,nosuid,size=8g",
           "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
           "--env", "TRITON_CACHE_DIR=/tmp/triton",
           "--env", "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions",
           "--workdir", "/workspace",
           "--volume", f"{ROOT}:/workspace:ro",
           "--volume", f"{tempdir}:/results:rw", *extra_mount]
    for group in os.getgroups():
        cmd.extend(["--group-add", str(group)])
    if args.docker_gpus:
        cmd.extend(["--gpus", args.docker_gpus])
    cmd.extend([args.docker_image, "python", "-m", "megabench", "_worker",
                "--case", case.id, "--submission", str(container_submission),
                "--device", args.device, "--trials", str(args.trials),
                "--warmup", str(args.warmup), "--reps", str(args.reps),
                "--result", f"/results/{result_path.name}"])
    if args.no_scan:
        cmd.append("--no-scan")
    if args.reference_reps is not None:
        cmd.extend(["--reference-reps", str(args.reference_reps)])
    return cmd, name, run_id


def _stop_owned_container(name: str, run_id: str) -> str | None:
    """On timeout, stop only the container carrying our unique run label."""
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "--format",
             '{{ index .Config.Labels "org.megabench.run" }}', name],
            capture_output=True, text=True, timeout=10)
        if inspected.returncode or inspected.stdout.strip() != run_id:
            return "timed-out container could not be verified for cleanup"
        stopped = subprocess.run(["docker", "stop", "--time", "1", name],
                                 capture_output=True, text=True, timeout=15)
        if stopped.returncode:
            return f"could not stop timed-out container: {stopped.stderr[-500:]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not stop timed-out container: {type(exc).__name__}: {exc}"
    return None


def _run_local_worker(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Own the worker's process group so a timeout also stops spawned ranks."""
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as worker:
        try:
            stdout, stderr = worker.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.communicate()
            raise
        return subprocess.CompletedProcess(cmd, worker.returncode, stdout, stderr)


def _run_one(args: argparse.Namespace, case: Case, tempdir: Path) -> dict:
    if not case.ready:
        return {"case": case.to_dict(), "status": "not_implemented", "reason": case.note}
    submission = str(Path(args.submission).resolve())
    result_path = tempdir / f"{case.id}.json"
    container = None
    if args.docker_image:
        cmd, name, run_id = _docker_worker_command(args, case, tempdir, result_path)
        container = (name, run_id)
    else:
        cmd = [sys.executable, "-m", "megabench", "_worker", "--case", case.id,
               "--submission", str(Path(args.submission).resolve()), "--device", args.device,
               "--trials", str(args.trials), "--warmup", str(args.warmup),
               "--reps", str(args.reps), "--result", str(result_path)]
        if args.no_scan:
            cmd.append("--no-scan")
        if args.reference_reps is not None:
            cmd.extend(["--reference-reps", str(args.reference_reps)])
    try:
        proc = (subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
                if container else _run_local_worker(cmd, args.timeout))
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if proc.returncode:
                result["worker_exit_code"] = proc.returncode
        else:
            result = {"case": case.to_dict(), "status": "evaluator_error",
                      "reason": f"worker exited {proc.returncode} without a record",
                      "stderr_tail": proc.stderr[-4000:]}
        result["submission"] = submission
        if container:
            result["evaluation_runtime"] = {"kind": "docker", "image": args.docker_image,
                                            "gpus": args.docker_gpus}
        return result
    except subprocess.TimeoutExpired:
        result = {"case": case.to_dict(), "submission": submission,
                  "status": "timeout",
                  "reason": f"worker exceeded {args.timeout}s"}
        if container:
            warning = _stop_owned_container(*container)
            if warning:
                result["cleanup_warning"] = warning
        return result
    except OSError as exc:
        return {"case": case.to_dict(), "submission": submission,
                "status": "infrastructure",
                "reason": f"could not start worker: {type(exc).__name__}: {exc}"}


def _output_path(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "runs" / f"{stamp}-{os.getpid()}.jsonl"


def aggregate_sessions(paths: list[Path], suite: str, method: str) -> dict:
    """Combine one independently produced result per case for one method."""
    expected = {case.id: case for case in select_cases(suite) if case.ready}
    if not expected:
        raise ValueError(f"suite {suite} has no ready cases")
    results: dict[str, dict] = {}
    session_ids: set[str] = set()
    source_dirs: set[Path] = set()
    contract_sha256 = _contract_digest()
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected exactly one case result")
        row = rows[0]
        case_id = row.get("case", {}).get("id")
        session = row.get("session")
        if case_id not in expected:
            raise ValueError(f"{path}: unexpected case {case_id}")
        if row["case"] != expected[case_id].to_dict():
            raise ValueError(f"{path}: case contract differs from current {case_id}")
        if row.get("contract_sha256") != contract_sha256:
            raise ValueError(f"{path}: benchmark contract digest differs")
        if case_id in results:
            raise ValueError(f"duplicate case {case_id}")
        if not isinstance(session, dict) or session.get("case_id") != case_id:
            raise ValueError(f"{path}: missing matching session metadata")
        if session.get("method") != method or not session.get("id"):
            raise ValueError(f"{path}: method or session ID mismatch")
        if session["id"] in session_ids:
            raise ValueError(f"duplicate session ID {session['id']}")
        source_dir = Path(row["submission"]).resolve().parent
        if source_dir in source_dirs:
            raise ValueError(f"shared candidate directory {source_dir}")
        session_ids.add(session["id"])
        source_dirs.add(source_dir)
        results[case_id] = row
    missing = sorted(expected.keys() - results.keys())
    passed = [row for row in results.values() if row["status"] == "ok_provisional"]
    latencies = {case_id: row.get("candidate_timing", {}).get("cuda_event_p50_ms") for case_id, row in results.items()}
    return {"suite": suite, "method": method,
            "contract_sha256": contract_sha256, "cases": results,
            "missing_cases": missing, "passed": len(passed),
            "total": len(expected), "all_passed": not missing and len(passed) == len(expected),
            "cuda_event_p50_ms": latencies}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megabench")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list agent-writing challenge cells")
    listing.add_argument("--suite", choices=("core", "p0", "p1", "p2", "p3", "planned", "all"),
                         default="all")
    listing.add_argument("--json", action="store_true")
    sol = sub.add_parser("sol", help="estimate optimistic HBM speed-of-light floors")
    sol.add_argument("--suite", choices=("core", "p0", "p1", "p2", "p3", "planned", "all"),
                     default="core")
    sol.add_argument("--case", help="show one case, regardless of suite")
    sol.add_argument("--bandwidth-tb-s", type=float,
                     default=DEFAULT_HBM_BANDWIDTH_TB_S,
                     help="assumed peak HBM TB/s per GPU (default: B300 8.0)")
    sol.add_argument("--measured-ms", type=float,
                     help="candidate CUDA-event median in ms; requires --case")
    sol.add_argument("--result", type=Path,
                     help="one MegaBench evaluate JSONL result; requires --case")
    sol.add_argument("--json", action="store_true")
    evaluation = sub.add_parser("evaluate", help="evaluate one case from one agent session")
    evaluation.add_argument("--submission", required=True)
    evaluation.add_argument("--case", required=True)
    evaluation.add_argument("--method", required=True)
    evaluation.add_argument("--session-id", required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--trials", type=int, default=3)
    evaluation.add_argument("--warmup", type=int, default=2)
    evaluation.add_argument("--reps", type=int, default=5)
    evaluation.add_argument("--timeout", type=int, default=180)
    evaluation.add_argument("--docker-image", help="run each case in this existing local image")
    evaluation.add_argument("--docker-gpus", help="Docker GPU selection, e.g. device=6")
    evaluation.add_argument("--no-graph-baseline", dest="graph_baseline", action="store_false",
                            help="accepted for compatibility; no graph baseline exists for these cases")
    evaluation.add_argument("--no-scan", action="store_true", help="skip the automated source/artifact scan (controls only)")
    evaluation.add_argument("--reference-reps", type=int, default=None,
                            help="timed repetitions of the eager oracle (default: 3 synthetic, 0 real weights)")
    evaluation.add_argument("--output")
    aggregate = sub.add_parser("aggregate", help="combine independent case sessions")
    aggregate.add_argument("--suite", choices=("core", "p0", "p1", "p2", "p3"),
                           default="p0")
    aggregate.add_argument("--method", required=True)
    aggregate.add_argument("--input", action="append", required=True)
    aggregate.add_argument("--output")
    worker = sub.add_parser("_worker")
    worker.add_argument("--case", required=True)
    worker.add_argument("--submission", required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--trials", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--reps", type=int, required=True)
    worker.add_argument("--result", required=True)
    worker.add_argument("--no-graph-baseline", dest="graph_baseline", action="store_false")
    worker.add_argument("--no-scan", action="store_true")
    worker.add_argument("--reference-reps", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return _worker(args)
    if args.command == "list":
        cases = select_cases(args.suite)
        if args.json:
            print(json.dumps([case.to_dict() for case in cases], indent=2))
        else:
            for case in cases:
                state = "ready" if case.ready else "planned"
                print(f"{case.id:44} {case.suite.upper():3} {state:7} "
                      f"{case.model} / {case.phase}")
        return 0
    if args.command == "sol":
        if (args.measured_ms is not None or args.result is not None) and not args.case:
            parser.error("--measured-ms and --result require --case")
        if args.measured_ms is not None and args.result is not None:
            parser.error("use either --measured-ms or --result")
        try:
            cases = select_cases("all", [args.case]) if args.case else select_cases(args.suite)
            measured_ms = args.measured_ms
            if args.result is not None:
                rows = [json.loads(line) for line in args.result.read_text().splitlines()
                        if line.strip()]
                if len(rows) != 1:
                    raise ValueError("--result must contain exactly one case result")
                row = rows[0]
                if row.get("case") != cases[0].to_dict():
                    raise ValueError("--result case contract does not match --case")
                measured_ms = row.get("candidate_timing", {}).get("cuda_event_p50_ms")
                if measured_ms is None:
                    raise ValueError("--result has no candidate CUDA-event median")
            estimates = [estimate_case(case, args.bandwidth_tb_s, measured_ms)
                         for case in cases]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        report = {
            "method": "optimistic_one_read_hbm_floor",
            "assumed_bandwidth_tb_s_per_gpu": args.bandwidth_tb_s,
            "default_bandwidth_source": DEFAULT_HBM_BANDWIDTH_SOURCE,
            "assumptions": [
                "Per slowest rank: each weight byte the rank holds and needs, read once.",
                "Routed experts: expected distinct experts under uniform routing; MLA cache replicated per rank.",
                "Peak bandwidth; no scratch, NCCL, synchronization, or compute costs.",
                "This is an HBM lower bound, not a prediction of achievable runtime.",
            ],
            "cases": estimates,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Optimistic one-read HBM floor at {args.bandwidth_tb_s:g} TB/s per GPU")
            print("Peak-bandwidth lower bound; excludes compute, scratch, and synchronization.")
            for item in estimates:
                if item["status"] == "estimated":
                    suffix = (f"  gap {item['gap_to_hbm_floor_x']:.1f}x"
                              if "gap_to_hbm_floor_x" in item else "")
                    print(f"{item['case_id']:44} {item['minimum_gb']:7.3f} GB  "
                          f"{item['hbm_floor_ms']:7.3f} ms{suffix}")
                    if args.case:
                        for component in item["components"]:
                            print(f"  {component['name']:40} "
                                  f"{component['bytes'] / 1e9:8.4f} GB")
                else:
                    print(f"{item['case_id']:44} unavailable: {item['reason']}")
        return 0
    if args.command == "aggregate":
        try:
            summary = aggregate_sessions([Path(name) for name in args.input],
                                         args.suite, args.method)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        if args.output:
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as output:
                json.dump(summary, output, indent=2)
        print(json.dumps(summary))
        return 0 if summary["all_passed"] else 1
    if min(args.trials, args.reps, args.timeout) < 1 or args.warmup < 0:
        parser.error("trials, reps, timeout must be positive; warmup >= 0")
    if args.docker_gpus and not args.docker_image:
        parser.error("--docker-gpus requires --docker-image")
    if args.docker_image and args.docker_image.startswith("-"):
        parser.error("--docker-image must be an image name or digest")
    if args.docker_image and args.device.startswith("cuda") and not args.docker_gpus:
        parser.error("CUDA Docker evaluation requires explicit --docker-gpus")
    submission = Path(args.submission).resolve()
    if not submission.is_file():
        parser.error(f"submission does not exist: {submission}")
    try:
        cases = select_cases("all", [args.case])
    except ValueError as exc:
        parser.error(str(exc))
    path = _output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    case = cases[0]
    results = []
    with tempfile.TemporaryDirectory(prefix="megabench-worker-") as directory:
        with path.open("x", encoding="utf-8") as output:
            result = _run_one(args, case, Path(directory))
            result["session"] = {"id": args.session_id, "method": args.method,
                                 "case_id": case.id}
            result["contract_sha256"] = _contract_digest()
            results.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            timing = result.get("candidate_timing", {}).get("cuda_event_p50_ms")
        print(f"{case.id}: {result['status']}" + (f" ({timing:.3f} ms CUDA-event p50, slowest rank)" if timing else "") +
              (f" - {result['reason']}" if result.get("reason") else ""))
    status = results[0]["status"]
    print(json.dumps({"results": str(path), "status": status, "exit_code": EXIT_CODES.get(status, 1),
                      "passed": int(status in ("ok_provisional", "correctness_only")), "total": 1}))
    return EXIT_CODES.get(status, 1)
