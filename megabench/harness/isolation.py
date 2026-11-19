"""Run the candidate in its own process; share inputs and outputs through CUDA IPC (D-I2).

The evaluator rank process (the parent) owns the inputs, the oracle outputs, the snapshot key
and the record. The candidate process (the child) imports the submission, calls `build`, and
runs `run(inputs)` on request. Tensors cross the boundary through torch.multiprocessing
reductions: CUDA tensors as IPC memory handles (no copy), CPU tensors through shared memory.
The child never receives the key, the expected outputs or a handle to the parent's state.

Timing is owned by the parent. The child records two interprocess CUDA events around each
run (`started` before, `finished` after, on its stream) and tells the parent when `started`
is recorded; the parent makes its stream wait for `started`, records its own timing event,
waits for `finished`, records the second timing event and synchronizes. The GPU time between
the parent's two events is the candidate's step time plus a few microseconds of event
latency. Before every timed repetition the parent overwrites the candidate's previous output
buffers with a poison value, so an early `finished` event (recorded before the kernel ends)
leaves poison or partial results in the outputs that the parent grades after the timing phase.
The parent also wraps every timed repetition in an NVTX range `megabench:rank<R>:rep<N>` so
that an Nsight Systems trace of the process tree (nsys_audit.py) can count the kernels of
exactly that repetition (D-L5).

Two transports carry the same protocol. *Local*: a multiprocessing Pipe to a child this process spawned
(tests, single-container runs; tensors through torch's reductions). *Remote* (GOALBOX_SPAWN_DIR set, the
GPU pod): the parent writes a spawn request into a directory of the pod-local `/job` emptyDir and the
untrusted spawner in the runner container (different uid, mount and pid namespace) starts
`python -m megabench.harness.isolation --remote <dir>`; the two talk over FIFOs with tensors encoded by
transport.py (raw CUDA IPC). The parent never learns a pid: liveness is EOF on the channel, a kill is a
`KILL` file the spawner acts on, and a child that never connects is a Timeout in `build`. DECISIONS-LOG L19.

Protocol over the channel (child side in `_child_main`):
  parent -> child  ("build",)                      child -> ("ready", events) | ("error", text, tb)
  parent -> child  ("run", inputs)                 child -> ("started",) then ("done", outputs) | ("error", text, tb)
  parent -> child  ("profile", inputs)             child -> ("done", outputs, advisory_counts) | ("error", ...)
  parent -> child  ("stop",)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

from .status import CandidateError, Infrastructure, Timeout

POISON = -12345.0


def _import_submission(path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("megabench_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import submission {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError("submission must export build(case: dict) -> callable")
    return build


def _child_main(conn, submission: str, contract: dict, device: str, group: dict | None, stderr_path: str,
                remote: bool = False) -> None:
    """Candidate process entry point. Everything after `build` runs the candidate's code."""
    import torch

    if remote:
        from . import transport
        keep = transport.KeepAlive()
        raw_send, raw_recv = conn.send, conn.recv
        conn.send = lambda msg: raw_send(transport.encode(msg, keep))        # type: ignore[assignment]
        conn.recv = lambda: transport.decode(raw_recv())                     # type: ignore[assignment]
    sys.stderr = open(stderr_path, "a", buffering=1)
    sys.stdout = sys.stderr
    os.environ.setdefault("MEGABENCH_CANDIDATE_PROCESS", "1")
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.set_device(torch.device(device))
    runner = None
    try:
        message = conn.recv()
        if message[0] != "build":
            raise RuntimeError(f"unexpected first message {message[0]!r}")
        if group is not None:
            import torch.distributed as dist
            from ..tasks.parallel import initialize
            from ..cases import Case

            dist.init_process_group(group["backend"], init_method=group["init_method"], world_size=group["world_size"],
                                    rank=group["rank"], timeout=timedelta(seconds=group["timeout_s"]),
                                    device_id=torch.device(device) if cuda else None)
            case = Case(**{k: v for k, v in contract.items() if k != "execution"})
            ctx = initialize(case)
            contract["execution"].update(tp_group=ctx.tp_group, ep_group=ctx.ep_group, backend=dist.get_backend())
        build = _import_submission(submission)
        runner = build(contract)
        if not callable(runner):
            raise TypeError("build(case) must return a callable run(inputs)")
        events = None
        if cuda:
            started = torch.cuda.Event(interprocess=True)
            finished = torch.cuda.Event(interprocess=True)
            # Interprocess events must be recorded once before their handle can be shared.
            started.record()
            finished.record()
            torch.cuda.synchronize()
            events = (started.ipc_handle(), finished.ipc_handle())
        conn.send(("ready", events))
    except BaseException as exc:  # noqa: BLE001 - everything goes back to the parent
        tb = traceback.format_exc(limit=12)
        print(tb, file=sys.stderr, flush=True)   # D-S5: the candidate's stderr file keeps the traceback
        conn.send(("error", f"{type(exc).__name__}: {exc}", tb))
        return
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        kind = message[0]
        if kind == "stop":
            break
        try:
            inputs = message[1]
            if kind == "run":
                if cuda:
                    started.record()
                conn.send(("started",))
                with torch.inference_mode():
                    outputs = runner(inputs)
                if cuda:
                    finished.record()
                    finished.synchronize()
                # `outputs` stays referenced here until the next message: the parent receives the tensors as
                # shared storage (CUDA IPC on the GPU, file descriptors on the CPU) that this process owns.
                conn.send(("done", outputs))
            elif kind == "profile":
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
                with torch.inference_mode(), profile(activities=activities) as prof:
                    outputs = runner(inputs)
                    if cuda:
                        torch.cuda.synchronize()
                gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                cpu_names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU]
                conn.send(("done", outputs, {"advisory_gpu_events": len(gpu),
                                             "advisory_names": sorted({e.name for e in gpu})[:64],
                                             "advisory_graph_launch_hint": any("cudagraphlaunch" in n.lower() for n in cpu_names)}))
            else:
                conn.send(("error", f"unknown message {kind!r}", ""))
            del inputs
        except BaseException as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=12)
            print(tb, file=sys.stderr, flush=True)   # D-S5: the candidate's stderr file keeps the traceback
            conn.send(("error", f"{type(exc).__name__}: {exc}", tb))
    if group is not None:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


class CandidateProcess:
    """Parent-side handle. All exceptions raised to the caller are typed (status module)."""

    def __init__(self, submission: Path, contract: dict, device: str, *, group: dict | None,
                 stderr_path: Path, build_timeout_s: float):
        import torch
        import torch.multiprocessing as mp

        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.stderr_path = Path(stderr_path)
        self.spawn_dir: Path | None = None
        self.process = None
        self.remote = bool(os.environ.get("GOALBOX_SPAWN_DIR"))
        if self.remote:
            self._start_remote(submission, contract, device, group, build_timeout_s)
        else:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_path.touch()
            ctx = mp.get_context("spawn")
            self.conn, child_conn = ctx.Pipe()
            self.process = ctx.Process(target=_child_main, name="megabench-candidate",
                                       args=(child_conn, str(submission), contract, device, group, str(self.stderr_path)),
                                       daemon=False)
            self.process.start()
            child_conn.close()
        self.last_outputs: dict | None = None
        self.events = None
        self.build_ms = None
        start = time.perf_counter()
        self.conn.send(("build",))
        reply = self._recv(build_timeout_s, phase="build")
        self.build_ms = (time.perf_counter() - start) * 1000
        if reply[0] == "error":
            self.close()
            raise CandidateError(reply[1], phase="build")
        if self.cuda and reply[1] is not None:
            self.events = tuple(torch.cuda.Event.from_ipc_handle(self.device, handle) for handle in reply[1])

    def _start_remote(self, submission, contract, device, group, build_timeout_s) -> None:
        """Remote transport: write the spawn request where the runner container's spawner reads it, then
        wait for the candidate process to connect. The stderr file is written by the candidate (untrusted
        content, kept as evidence) inside the spawn directory."""
        import json

        from . import transport
        from .status import Infrastructure

        self.spawn_dir = transport.make_spawn_dir(Path(os.environ["GOALBOX_SPAWN_DIR"]))
        self.stderr_path = self.spawn_dir / "stderr"
        request = {"submission": str(submission), "contract": contract, "device": device, "group": group,
                   "stderr": str(self.stderr_path), "cwd": str(Path(submission).parent),
                   "env": {k: v for k, v in os.environ.items()
                           if k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "KIMI_K3_WEIGHTS", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
                                    "NVIDIA_IMEX_CHANNELS", "NCCL_MNNVL_ENABLE", "GOALBOX_JOB_ID", "MASTER_ADDR", "MASTER_PORT")}}
        tmp = self.spawn_dir / ".cmd.tmp"
        tmp.write_text(json.dumps(request))
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.spawn_dir / "cmd.json")
        try:
            self.conn = transport.connect_parent(self.spawn_dir, build_timeout_s)
        except TimeoutError as exc:
            spawned = (self.spawn_dir / "pid").exists()
            if not spawned:
                raise Infrastructure(f"the runner did not start a candidate process: {exc}", phase="build") from exc
            raise Timeout(f"candidate process connected nothing within {build_timeout_s:.0f} s", phase="build") from exc
        except transport.TransportError as exc:
            raise CandidateError(f"candidate process failed to start: {exc}", phase="build") from exc
        self._keep = transport.KeepAlive()
        raw_send, raw_recv = self.conn.send, self.conn.recv
        self._mappings: dict = {}
        self.conn.send = lambda msg: raw_send(transport.encode(msg, self._keep))       # type: ignore[assignment]
        self.conn.recv = lambda: transport.decode(raw_recv(), self._mappings)          # type: ignore[assignment]

    def _alive(self) -> bool:
        return self.process.is_alive() if self.process is not None else True   # remote: liveness is channel EOF

    def _exitcode(self):
        return self.process.exitcode if self.process is not None else "unknown (remote)"

    def _recv(self, timeout_s: float, *, phase: str):
        if not self.conn.poll(timeout_s):
            if not self._alive():
                raise CandidateError(f"candidate process exited with code {self._exitcode()} during {phase}", phase=phase)
            self.kill()
            raise Timeout(f"candidate exceeded {timeout_s:.0f} s in {phase}", phase=phase)
        try:
            return self.conn.recv()
        except EOFError:
            raise CandidateError(f"candidate process died during {phase} (exit code {self._exitcode()})", phase=phase)

    def stderr_tail(self, limit: int = 4000) -> str:
        try:
            data = self.stderr_path.read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", errors="replace")

    def poison_outputs(self) -> None:
        """Overwrite the candidate's last output buffers so a reused buffer cannot pass as fresh."""
        import torch

        if not self.last_outputs:
            return
        for value in self.last_outputs.values():
            if isinstance(value, torch.Tensor) and value.numel():
                if value.is_floating_point():
                    value.fill_(POISON)
                else:
                    value.fill_(-1)
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def run(self, inputs: dict, timeout_s: float, *, timed: bool = False, nvtx: str | None = None) -> tuple[dict, dict]:
        """One call. Returns (outputs, timing) with timing = {host_ms, cuda_event_ms or None}."""
        import torch

        if self.cuda and timed and self.events is not None:
            self.poison_outputs()
        stream = torch.cuda.current_stream(self.device) if self.cuda else None
        start_event = end_event = None
        if nvtx and self.cuda:
            torch.cuda.nvtx.range_push(nvtx)
        host_start = time.perf_counter()
        self.conn.send(("run", inputs))
        reply = self._recv(timeout_s, phase="candidate")
        if reply[0] == "started" and self.cuda and self.events is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream.wait_event(self.events[0])
            start_event.record(stream)
            reply = self._recv(timeout_s, phase="candidate")
        elif reply[0] == "started":
            reply = self._recv(timeout_s, phase="candidate")
        if reply[0] == "error":
            if nvtx and self.cuda:
                torch.cuda.nvtx.range_pop()
            raise CandidateError(reply[1], phase="candidate")
        outputs = reply[1]
        if start_event is not None:
            stream.wait_event(self.events[1])
            end_event.record(stream)
            end_event.synchronize()
        elif self.cuda:
            torch.cuda.synchronize(self.device)
        host_ms = (time.perf_counter() - host_start) * 1000
        if nvtx and self.cuda:
            torch.cuda.nvtx.range_pop()
        self.last_outputs = outputs if isinstance(outputs, dict) else None
        return outputs, {"host_ms": host_ms,
                         "cuda_event_ms": start_event.elapsed_time(end_event) if start_event is not None else None}

    def profile(self, inputs: dict, timeout_s: float) -> tuple[dict, dict]:
        self.conn.send(("profile", inputs))
        reply = self._recv(timeout_s, phase="candidate")
        if reply[0] == "error":
            raise CandidateError(reply[1], phase="candidate")
        self.last_outputs = reply[1] if isinstance(reply[1], dict) else None
        return reply[1], reply[2]

    def kill(self) -> None:
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
                self.process.join(10)
        elif self.spawn_dir is not None:
            try:
                (self.spawn_dir / "KILL").write_text(str(time.time()))     # the spawner kills the process group
            except OSError:
                pass

    def close(self) -> None:
        try:
            if self._alive():
                self.conn.send(("stop",))
                if self.process is not None:
                    self.process.join(30)
        except (OSError, ValueError, BrokenPipeError):
            pass
        self.kill()
        try:
            self.conn.close()
        except OSError:
            pass


def _remote_main(spawn_dir: str) -> int:
    """Entry point in the runner container: `python -m megabench.harness.isolation --remote DIR`."""
    import json

    from . import transport

    d = Path(spawn_dir)
    request = json.loads((d / "cmd.json").read_text())
    try:
        conn = transport.connect_child(d)
    except OSError as exc:
        (d / "FAILED").write_text(f"connect: {exc}")
        return 1
    _child_main(conn, request["submission"], request["contract"], request["device"], request["group"], request["stderr"],
                remote=True)
    conn.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--remote":
        raise SystemExit(_remote_main(sys.argv[2]))
    raise SystemExit("usage: python -m megabench.harness.isolation --remote SPAWN_DIR")
