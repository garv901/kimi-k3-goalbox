"""Shared helpers for the A1 tools (workspace discovery, benchmark path, case info)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent  # workspace root (the overlay copies .a1/ there)


def benchmark_path() -> str:
    """PYTHONPATH entry of the trusted benchmark: env, then the workspace check_candidate, then defaults."""
    env = os.environ.get('MEGABENCH_ROOT')
    if env:
        return env
    for name in ('check_candidate', 'timeline_candidate', 'profile_candidate'):
        p = WS / name
        if p.exists():
            m = re.search(r'PYTHONPATH=("?)([^"\s:]+)', p.read_text(errors='replace'))
            if m and Path(m.group(2), 'megabench').is_dir():
                return m.group(2)
    for c in os.environ.get('PYTHONPATH', '').split(':') + ['/srv/ss/trusted/vibe-megakernel', '/root/vibe-megakernel']:
        if c and Path(c, 'megabench').is_dir():
            return c
    raise SystemExit('cannot find the benchmark (set MEGABENCH_ROOT)')


def case_info() -> dict:
    p = WS / 'case.json'
    if p.exists():
        return json.loads(p.read_text())
    for name in ('check_candidate', 'profile_candidate'):
        q = WS / name
        if q.exists():
            m = re.search(r'--case\s+(\S+)', q.read_text())
            if m:   # no case.json: assume the protocol's whole node (dsv41-tp4-rack10-v1: 4 ranks)
                return {'id': m.group(1), 'gpus': 4}
    raise SystemExit('no case.json in the workspace')


def submission_path() -> Path:
    return WS / 'submission.py'


def python() -> str:
    return '/usr/bin/python3.12' if os.path.exists('/usr/bin/python3.12') else sys.executable


def gpu_run_prefix(gpus: int) -> list[str]:
    """Every GPU command goes through ./gpu_run (the run's lease). A1_NO_GPU_RUN=1 runs directly."""
    if os.environ.get('A1_NO_GPU_RUN') == '1':
        return []
    g = WS / 'gpu_run'
    if not g.exists():
        raise SystemExit('./gpu_run not found in the workspace (set A1_NO_GPU_RUN=1 to run without a lease)')
    return [str(g), '-g', str(gpus), '--']


def run_id() -> str:
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + f'-{os.getpid()}'
