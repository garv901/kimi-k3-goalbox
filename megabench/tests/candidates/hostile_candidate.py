"""Control candidate for the trusted-verdict channel (C1 blocker B-C1-1 regressions).

Behaves like the oracle candidate but, during `build`, attempts every write the boundary must refuse and
reports each attempt on stderr as `ATTACK <name>: WRITABLE|denied(<errno>)`. In a unit test on one host
(same uid) several attempts succeed, which is exactly the finding that motivated the two-container layout;
in the pod layout (sandbox/k8s/c1/local/verdict_test.sh) every attempt must be denied. The verdict must be
the same either way, because nothing the attempts touch is read by the grader.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # the snapshot holds the sibling oracle_candidate.py
from oracle_candidate import build as _oracle_build  # noqa: E402


def _attempt(name: str, fn) -> None:
    try:
        fn()
        print(f"ATTACK {name}: WRITABLE", file=sys.stderr, flush=True)
    except OSError as exc:
        print(f"ATTACK {name}: denied({exc.errno}:{type(exc).__name__})", file=sys.stderr, flush=True)


def _write(path: str, text: str = "forged\n"):
    def go():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return go


def _attacks() -> None:
    here = Path(__file__).resolve()
    job = here.parents[1] if here.parent.name == "candidate" else None      # /job/<root>/<rid>/candidate/submission.py
    rid = job.name if job else "unknown"
    results = os.environ.get("HOSTILE_RESULTS", "/ctl-results/gpu/node-a")
    forged = '{"exit": 0, "state": "ok_provisional", "kind": "grade", "trusted": true, "finished_utc": 9999999999}'
    _attempt("forge_done_record", _write(f"{results}/done/{rid}.json", forged))
    _attempt("forge_verdict", _write(f"{results}/verdict/{rid}.json", '{"state": "ok_provisional"}'))
    _attempt("replace_grader_output_log", _write(f"{results}/out/{rid}.log", "all good\n"))
    _attempt("modify_other_job_result", _write(f"{results}/done/other-job.json", forged))
    _attempt("alter_pending_request", _write(f"/ctl-req/gpu/node-a/req/{rid}.json", '{"minutes": 999}'))
    _attempt("inject_request", _write("/ctl-req/gpu/node-a/req/evil.json", "{}"))
    _attempt("modify_records", _write("/records/transcripts/forged.jsonl"))
    _attempt("modify_controller_state", _write("/ctl/req/gpu/node-a/req/evil.json", "{}"))
    _attempt("modify_evaluator_code", _write("/goalbox/bench/megabench/harness/correctness.py", "# forged\n"))
    if job is not None:
        _attempt("alter_own_job_spec", _write(str(job / "job.json"), '{"kind": "exec"}'))
        _attempt("replace_graded_source", _write(str(job / "candidate" / "submission.py"), "# forged\n"))
        _attempt("add_file_to_job_dir", _write(str(job / "evil"), "x"))
        _attempt("fake_exit_file", _write(str(job / "out" / "exit"), "0"))          # allowed: out/ is the runner's
    grader_pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if b"executor.py" in (entry / "cmdline").read_bytes():
                    grader_pids.append(int(entry.name))
            except OSError:
                pass
    print(f"ATTACK see_grader_process: {'VISIBLE ' + str(grader_pids) if grader_pids else 'denied(not visible)'}",
          file=sys.stderr, flush=True)
    for pid in grader_pids:
        try:
            os.kill(pid, 0)
            print("ATTACK signal_grader: WRITABLE", file=sys.stderr, flush=True)
        except OSError as exc:
            print(f"ATTACK signal_grader: denied({exc.errno})", file=sys.stderr, flush=True)
    try:
        Path("/proc/1/root/ctl-results").exists() and Path("/proc/1/root/ctl-results/forged").write_text("x")
        print("ATTACK magic_link_into_grader_root: " + ("WRITABLE" if Path("/proc/1/root/ctl-results").exists() else "denied(no such root)"),
              file=sys.stderr, flush=True)
    except OSError as exc:
        print(f"ATTACK magic_link_into_grader_root: denied({exc.errno})", file=sys.stderr, flush=True)


def build(case: dict):
    _attacks()
    return _oracle_build(case)
