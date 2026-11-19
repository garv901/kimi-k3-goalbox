"""Cross-container transport for the candidate protocol (DECISIONS-LOG L19).

In the GPU pod the evaluator rank (the *grader*, uid 2000, container `grader`) and the candidate process
(uid 1000, container `runner`) live in different containers with different uids, mount namespaces and pid
namespaces. They share one pod-local emptyDir (`/job`) and the GPUs. This module gives the harness

  * `Channel`: a `multiprocessing.connection`-style send/recv/poll over two FIFOs in a spawn directory
    (`to_child`, written by the grader, mode 0644; `from_child`, written by the candidate, mode 0622);
  * `encode()` / `decode()`: tensors are never pickled by torch's reductions on this path (they need a
    shared `/dev/shm` ref-counter file and the same uid). CPU tensors travel by value; CUDA tensors travel
    as raw CUDA IPC memory handles (cudaIpcGetMemHandle on the sender, cudaIpcOpenMemHandle on the
    receiver) with the offset inside the allocation, rebuilt through `__cuda_array_interface__`.

The sender keeps every exported tensor alive until the next message (`KeepAlive`), exactly as the local
pipe mode does. The receiver closes an IPC mapping when the last tensor built on it is garbage-collected.
Nothing here is trusted: the grader treats what it decodes as candidate output and grades it.
"""
from __future__ import annotations

import ctypes
import io
import os
import pickle
import select
import struct
import time
from pathlib import Path
from typing import Any


class TransportError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- channel
class Channel:
    """Length-prefixed pickle frames over a (read fd, write fd) pair. Not thread-safe."""

    def __init__(self, read_fd: int, write_fd: int):
        self.read_fd, self.write_fd = read_fd, write_fd
        self._buffer = b""

    def send(self, obj: Any) -> None:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        data = struct.pack(">Q", len(payload)) + payload
        view = memoryview(data)
        while view:
            n = os.write(self.write_fd, view)
            view = view[n:]

    def poll(self, timeout_s: float | None = None) -> bool:
        if len(self._buffer) >= 8:
            return True
        ready, _, _ = select.select([self.read_fd], [], [], timeout_s)
        return bool(ready)

    def _read_exact(self, n: int, deadline: float | None) -> bytes:
        while len(self._buffer) < n:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self.read_fd], [], [], remaining)
            if not ready:
                raise TimeoutError("channel read timed out")
            chunk = os.read(self.read_fd, 1 << 20)
            if not chunk:
                raise EOFError("peer closed the channel")
            self._buffer += chunk
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def recv(self, timeout_s: float | None = None) -> Any:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        (length,) = struct.unpack(">Q", self._read_exact(8, deadline))
        if length > (64 << 30):
            raise TransportError(f"frame of {length} bytes refused")
        return pickle.loads(self._read_exact(length, deadline))

    def close(self) -> None:
        for fd in (self.read_fd, self.write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def make_spawn_dir(root: Path) -> Path:
    """Grader side: a fresh spawn directory the runner can read, with the two FIFOs."""
    import uuid

    d = Path(root) / uuid.uuid4().hex[:12]
    d.mkdir(mode=0o777)
    os.chmod(d, 0o777)       # the runner (other uid) writes pid, stderr, exited, FAILED here; the parent dir is 0755
    os.mkfifo(d / "to_child", 0o644)
    os.mkfifo(d / "from_child", 0o622)
    os.chmod(d / "to_child", 0o644)      # mkfifo honours the umask; make the modes explicit
    os.chmod(d / "from_child", 0o622)
    return d


def connect_parent(spawn_dir: Path, timeout_s: float) -> Channel:
    """Grader side: wait (up to timeout_s) for the candidate process to open its ends."""
    d = Path(spawn_dir)
    read_fd = os.open(d / "from_child", os.O_RDONLY | os.O_NONBLOCK)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            write_fd = os.open(d / "to_child", os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as exc:                       # ENXIO: no reader yet
            if time.monotonic() > deadline:
                os.close(read_fd)
                raise TimeoutError(f"no candidate process connected to {d} within {timeout_s:.0f} s") from exc
            if (d / "FAILED").exists():
                os.close(read_fd)
                raise TransportError((d / "FAILED").read_text(errors="replace")[:500])
            time.sleep(0.05)
    for fd in (read_fd, write_fd):
        os.set_blocking(fd, True)
    return Channel(read_fd, write_fd)


def connect_child(spawn_dir: Path) -> Channel:
    """Candidate side (runner container): open the FIFOs; blocks until the grader has its ends open."""
    d = Path(spawn_dir)
    read_fd = os.open(d / "to_child", os.O_RDONLY)
    write_fd = os.open(d / "from_child", os.O_WRONLY)
    return Channel(read_fd, write_fd)


# ----------------------------------------------------------------------------------------- tensor codec
class KeepAlive:
    """Holds the tensors the sender exported since its previous message."""

    def __init__(self):
        self.items: list = []

    def reset(self, items: list) -> None:
        self.items = items


_cudart = None
_cuda = None


def _rt():
    global _cudart, _cuda
    if _cudart is None:
        import torch

        _cudart = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}" if torch.version.cuda else "libcudart.so")
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _cudart, _cuda


class _IpcMapping:
    """One cudaIpcOpenMemHandle mapping; closed when the last tensor built on it dies."""

    def __init__(self, handle: bytes, device: int):
        import torch

        rt, _ = _rt()
        torch.cuda.set_device(device)
        ptr = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(handle, 64)
        err = rt.cudaIpcOpenMemHandle(ctypes.byref(ptr), buf, ctypes.c_uint(1))   # cudaIpcMemLazyEnablePeerAccess
        if err != 0:
            raise TransportError(f"cudaIpcOpenMemHandle failed with {err}")
        self.base = ptr.value
        self.device = device

    def __del__(self):
        try:
            rt, _ = _rt()
            rt.cudaIpcCloseMemHandle(ctypes.c_void_p(self.base))
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


class _View:
    """A `__cuda_array_interface__` object over `nbytes` at `ptr`; keeps its mapping alive."""

    def __init__(self, mapping: _IpcMapping, offset: int, nbytes: int):
        self.mapping = mapping
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (mapping.base + offset, False),
                                         "version": 2, "strides": None}


def _export_cuda(tensor) -> dict:
    rt, cu = _rt()
    t = tensor.contiguous()
    ptr = t.data_ptr()
    base = ctypes.c_void_p(); size = ctypes.c_size_t()
    if cu.cuMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)) != 0:
        raise TransportError("cuMemGetAddressRange failed")
    handle = ctypes.create_string_buffer(64)
    err = rt.cudaIpcGetMemHandle(handle, ctypes.c_void_p(base.value))
    if err != 0:
        raise TransportError(f"cudaIpcGetMemHandle failed with {err}")
    return {"handle": handle.raw, "offset": ptr - base.value, "nbytes": t.numel() * t.element_size(),
            "dtype": str(t.dtype).removeprefix("torch."), "shape": tuple(t.shape), "device": t.device.index, "_keep": t}


def encode(obj: Any, keep: KeepAlive) -> Any:
    """Replace tensors in a (nested) dict/list/tuple by portable descriptions. Returns the new object."""
    import torch

    kept: list = []

    def enc(x):
        if isinstance(x, torch.Tensor):
            if x.device.type == "cuda":
                d = _export_cuda(x)
                kept.append(d.pop("_keep"))
                return ("__tensor_cuda__", d)
            buf = io.BytesIO()
            torch.save(x.detach().contiguous(), buf)
            return ("__tensor_cpu__", buf.getvalue())
        if isinstance(x, dict):
            return {k: enc(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(enc(v) for v in x)
        return x

    out = enc(obj)
    keep.reset(kept)
    return out


def decode(obj: Any, mappings: dict | None = None) -> Any:
    """Inverse of encode. `mappings` caches IPC mappings by handle for the lifetime of the message set."""
    import torch

    cache = mappings if mappings is not None else {}

    def dec(x):
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "__tensor_cpu__":
            return torch.load(io.BytesIO(x[1]), weights_only=True)
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "__tensor_cuda__":
            d = x[1]
            key = (d["handle"], d["device"])
            if key not in cache:
                cache[key] = _IpcMapping(d["handle"], d["device"])
            view = _View(cache[key], d["offset"], d["nbytes"])
            raw = torch.as_tensor(view, device=f"cuda:{d['device']}")
            return raw.view(getattr(torch, d["dtype"])).view(*d["shape"]) if d["nbytes"] else \
                torch.empty(d["shape"], dtype=getattr(torch, d["dtype"]), device=f"cuda:{d['device']}")
        if isinstance(x, dict):
            return {k: dec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [dec(v) for v in x]
        if isinstance(x, tuple):
            return tuple(dec(v) for v in x)
        return x

    return dec(obj)
