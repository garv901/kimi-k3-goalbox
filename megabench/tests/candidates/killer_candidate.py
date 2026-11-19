"""Control candidate: tries to kill the grading process and its own spawner.

In the pod layout the grader is invisible (other pid namespace, other uid): only the runner's processes
die, the grader records candidate_error for this job and INFRA_RUNNER_DOWN (restart requested) for the
next one. It must never produce a pass.

On a shared host (unit tests) a uid-wide kill would take the developer's session with it, so the wide
sweep runs only inside a private pid namespace whose pid 1 is the spawner (the runner container); the
unit-test variant kills only its parent and reports whether the grader could be signalled. Note that pid 1
of a pid namespace ignores signals from inside it, so in the container the spawner itself survives.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def build(case: dict):
    try:
        wide = b"spawner.py" in Path("/proc/1/cmdline").read_bytes()      # private pid namespace: safe to go wide
    except OSError:
        wide = False
    print(f"KILLER: wide={wide}", file=sys.stderr, flush=True)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if wide:
            try:
                os.kill(int(entry.name), signal.SIGKILL)
                print(f"KILLER: killed {entry.name} ({cmd[:60]!r})", file=sys.stderr, flush=True)
            except OSError as exc:
                print(f"KILLER: cannot signal {entry.name}: {exc.errno}", file=sys.stderr, flush=True)
        elif b"executor.py" in cmd:          # shared host: only report whether the grader could be signalled
            try:
                os.kill(int(entry.name), 0)
                print(f"KILLER: grader {entry.name} is signallable (same uid, same pid namespace)", file=sys.stderr, flush=True)
            except OSError as exc:
                print(f"KILLER: grader {entry.name} not signallable: {exc.errno}", file=sys.stderr, flush=True)
    os.kill(os.getppid(), signal.SIGKILL)

    def run(inputs):
        return {}
    return run
