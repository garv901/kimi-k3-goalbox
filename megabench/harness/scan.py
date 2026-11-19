"""Automated source and artifact scan of a submission (D-L7, D-L6, section 15).

No person reviews the agent's code, so the rules that used to be `review_required` become
checks here. `scan_submission(directory)` returns a record with `status` "pass" or "fail"
and one finding per rule hit. The rules are deliberately syntactic and conservative: a hit
fails the submission, and the message tells the agent what to remove.

Sources (every .py, .cu, .cuh, .cpp, .h, .hpp, .sh, .txt, .toml under the submission directory
except `reports/`, `.git/`, `reference/` and `__pycache__/`):
  oracle_import      imports or executes the trusted oracle (megabench.*, reference/megabench)
  file_tensors       loads tensors from files (torch.load, safetensors, numpy load, pickle)
  cuda_graph         CUDA graph capture or launch APIs (Python or CUDA)
  device_launch      device-side launch APIs (dynamic parallelism)
  timer_macro        the graded build defines the in-kernel timer macro
  outside_reads      opens absolute paths outside the sandbox mounts (/work, /home/agent, /tmp, /weights)
Artifacts (every .so, .cubin, .fatbin, .ptx, .o under the directory): cuobjdump symbols and SASS
  device_launch      calls into cudaLaunchDevice / cudaGetParameterBuffer
  timer_macro        the megatimer record buffer symbol is present
The scan also records every artifact's sha256, so the graded run is tied to exact binaries.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".sh", ".txt", ".toml", ".cfg", ".mk"}
ARTIFACT_SUFFIXES = {".so", ".cubin", ".fatbin", ".ptx", ".o", ".a"}
SKIP_DIRS = {"reports", ".git", "reference", "__pycache__", ".venv", "node_modules"}
TIMER_MACRO = "MEGATIMER"
TIMER_SYMBOL = "megatimer_records"
ALLOWED_ABSOLUTE = ("/work", "/home/agent", "/tmp", "/weights", "/usr", "/dev", "/proc", "/goalbox/bench")

RULES = {
    "oracle_import": [
        (r"^\s*(from|import)\s+megabench\b", "import of the trusted oracle package megabench"),
        (r"reference/megabench|megabench[/.]tasks|megabench[/.]harness", "reference to the oracle modules"),
        (r"importlib\.[a-z_.]*\(.*megabench", "dynamic import of megabench"),
    ],
    "file_tensors": [
        (r"\btorch\.load\s*\(", "torch.load"),
        (r"\bsafetensors\b", "safetensors file loading"),
        (r"\bload_file\s*\(|\bsafe_open\s*\(", "safetensors file loading"),
        (r"\bnp\.load\s*\(|\bnumpy\.load\s*\(|\bnp\.fromfile\s*\(|\bnumpy\.fromfile\s*\(", "numpy file loading"),
        (r"\bpickle\.load\b|\bjoblib\.load\b|\btorch\.from_file\b", "pickled or raw tensor loading"),
        (r"\bnp\.memmap\s*\(|\bnumpy\.memmap\s*\(|\bmmap\.mmap\s*\(", "memory-mapped file reads"),
    ],
    "cuda_graph": [
        (r"\bCUDAGraph\b|torch\.cuda\.graph\s*\(|make_graphed_callables|torch\.cuda\.graphs", "torch CUDA graph API"),
        (r"\bcuda(Graph|StreamBeginCapture|StreamEndCapture)\w*\s*\(|\bcuGraph\w*\s*\(", "CUDA graph driver/runtime API"),
        (r"\bcudaLaunchKernelExC?\b.*[Gg]raph", "graph launch"),
    ],
    "device_launch": [
        (r"\bcudaLaunchDevice\w*\b|\bcudaGetParameterBuffer\w*\b", "device-side launch API (dynamic parallelism)"),
        (r"-rdc=true.*-lcudadevrt|-lcudadevrt", "links the device runtime (dynamic parallelism)"),
    ],
    "timer_macro": [
        (rf"-D\s*{TIMER_MACRO}\b", f"-D{TIMER_MACRO} in a build command"),
        (rf"^\s*#\s*define\s+{TIMER_MACRO}\b", f"#define {TIMER_MACRO} in a source"),
    ],
}
CUDA_ONLY = {"device_launch", "timer_macro"}
# A __global__ function that launches with <<< >>> is dynamic parallelism in CUDA sources.
DEVICE_LAUNCH_CUDA = re.compile(r"<<<")
OPEN_RE = re.compile(r"""(?:open|Path|os\.open|np\.load|torch\.load|safe_open|load_file)\s*\(\s*['"](/[^'"]*)['"]""")


def iter_files(directory: Path, suffixes: set[str]):
    for path in sorted(directory.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(directory).parts[:-1]):
            continue
        if path.is_file() and path.suffix in suffixes:
            yield path


def _strip_comments(text: str, suffix: str) -> str:
    if suffix == ".py":
        return "\n".join(line.split("#", 1)[0] if not line.lstrip().startswith("#") else "" for line in text.splitlines())
    if suffix in (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return re.sub(r"//[^\n]*", "", text)
    return text


def scan_sources(directory: Path) -> list[dict]:
    findings = []
    for path in iter_files(directory, SOURCE_SUFFIXES):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _strip_comments(raw, path.suffix)
        relative = str(path.relative_to(directory))
        for rule, patterns in RULES.items():
            for pattern, why in patterns:
                for match in re.finditer(pattern, text, flags=re.M):
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append({"rule": rule, "file": relative, "line": line, "detail": why,
                                     "text": match.group(0)[:120]})
        if path.suffix in (".cu", ".cuh"):
            findings.extend(_device_launch_in_kernels(text, relative))
        for match in OPEN_RE.finditer(text):
            target = match.group(1)
            if not target.startswith(ALLOWED_ABSOLUTE):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"rule": "outside_reads", "file": relative, "line": line,
                                 "detail": "absolute path outside the sandbox mounts", "text": target[:120]})
    return findings


def _device_launch_in_kernels(text: str, relative: str) -> list[dict]:
    """A <<< >>> launch inside a __global__ or __device__ function body is dynamic parallelism."""
    findings = []
    for match in re.finditer(r"__(global|device)__[^{;]*\{", text):
        depth, i = 1, match.end()
        start = i
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[start:i]
        if DEVICE_LAUNCH_CUDA.search(body):
            line = text.count("\n", 0, match.start()) + 1
            findings.append({"rule": "device_launch", "file": relative, "line": line,
                             "detail": "kernel launch inside a device function (dynamic parallelism)",
                             "text": "<<< >>> in __global__/__device__ body"})
    return findings


def scan_artifacts(directory: Path) -> tuple[list[dict], list[dict]]:
    """cuobjdump symbol and SASS scan of every compiled artifact; also returns their hashes."""
    findings, artifacts = [], []
    cuobjdump = shutil.which("cuobjdump") or ("/usr/local/cuda/bin/cuobjdump" if Path("/usr/local/cuda/bin/cuobjdump").exists() else None)
    for path in iter_files(directory, ARTIFACT_SUFFIXES):
        relative = str(path.relative_to(directory))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {"file": relative, "sha256": digest, "bytes": path.stat().st_size, "scanned": False}
        if path.suffix == ".ptx":
            text = path.read_text(encoding="utf-8", errors="replace")
            entry["scanned"] = True
            findings.extend(_scan_text_artifact(text, relative))
        elif cuobjdump is not None:
            out = ""
            for args in (["-symbols"], ["-sass"], ["-ptx"]):
                try:
                    result = subprocess.run([cuobjdump, *args, str(path)], capture_output=True, text=True, timeout=600)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    continue
                if result.returncode == 0:
                    out += result.stdout
                    entry["scanned"] = True
            findings.extend(_scan_text_artifact(out, relative))
        else:
            entry["error"] = "cuobjdump not available"
        artifacts.append(entry)
    return findings, artifacts


def _scan_text_artifact(text: str, relative: str) -> list[dict]:
    findings = []
    for pattern, rule, why in ((r"cudaLaunchDevice|cudaGetParameterBuffer|cudaCDP2", "device_launch",
                                "device-side launch symbol in compiled code"),
                               (TIMER_SYMBOL, "timer_macro", "in-kernel timer buffer symbol in the graded build"),
                               (r"\bcudaGraphLaunch\b|\bcuGraphLaunch\b", "cuda_graph", "graph launch symbol in compiled code")):
        match = re.search(pattern, text)
        if match:
            findings.append({"rule": rule, "file": relative, "line": None, "detail": why, "text": match.group(0)[:120]})
    return findings


def scan_submission(submission: Path) -> dict:
    """Scan the directory that holds the submission. Returns the scan record."""
    directory = Path(submission).resolve()
    if directory.is_file():
        directory = directory.parent
    findings = scan_sources(directory)
    artifact_findings, artifacts = scan_artifacts(directory)
    findings.extend(artifact_findings)
    rules_hit = sorted({f["rule"] for f in findings})
    return {"status": "fail" if findings else "pass", "directory": str(directory), "rules_hit": rules_hit,
            "findings": findings[:200], "finding_count": len(findings), "artifacts": artifacts,
            "rules": sorted(RULES) + ["outside_reads"]}
