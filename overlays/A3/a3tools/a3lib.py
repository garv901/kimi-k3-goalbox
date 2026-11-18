"""Shared code of ./capture and ./task_ab (A3 overlay).

Capture format (one directory per capture, reports/capture-<stamp>/):
  manifest.json   capture settings, case tolerances, record index per rank
  rank<r>.pt      torch.save dict:
    params: bytes of the captured launch's parameter block (SS_CAP_PARAMS)
    records: [{name, layer, seq, regions: [{name, addr, bytes, dtype, data(uint8 cpu)}]}]
    inputs: {name: {ptr, nbytes, shape, dtype, fp}}      (regenerated from seed/trial/rank)
    outputs: {name: {ptr, nbytes, shape, dtype}}          (fresh buffers on restore)
    allocs: [{base, nbytes, kind, data(uint8 cpu) | None}] other device memory the params or
            regions point into, snapshotted AFTER the launch
    final_outputs: {name: cpu tensor}                     end-to-end outputs of the launch
"""
from __future__ import annotations

import ctypes
import glob
import importlib
import json
import os
import struct
import sys

import torch

WS = os.environ.get('A3_WS', os.getcwd())

DTYPES = {0: torch.uint8, 1: torch.int8, 2: torch.uint16, 3: torch.int16, 4: torch.uint32, 5: torch.int32,
          6: torch.uint64, 7: torch.int64, 8: torch.float16, 9: torch.bfloat16, 10: torch.float32, 11: torch.float64}
DTYPE_NAMES = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'u64', 7: 'i64', 8: 'f16', 9: 'bf16',
               10: 'f32', 11: 'f64'}
FLOAT_CODES = (8, 9, 10, 11)
MAGIC = 0x53534350


# ----------------------------------------------------------------------------- candidate hooks
def import_hooks(cand_dir: str):
    """Import <cand_dir>/ss_task.py fresh (each candidate dir has its own)."""
    path = os.path.join(cand_dir, 'ss_task.py')
    if not os.path.exists(path):
        raise SystemExit(f'{path} missing: the candidate must provide ss_task.py with build_ext(flags) '
                         '(see TASK.d/A3.md)')
    if cand_dir not in sys.path:
        sys.path.insert(0, cand_dir)
    ref = os.path.join(cand_dir, 'reference')
    if os.path.isdir(ref) and ref not in sys.path:
        sys.path.insert(1, ref)
    spec = importlib.util.spec_from_file_location('ss_task', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task_module():
    """The benchmark task module of the workspace's reference snapshot (has make_inputs)."""
    import megabench.tasks as tasks
    found = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(tasks.__file__), '*.py'))):
        name = os.path.basename(f)[:-3]
        if name.startswith('_') or name in ('common', 'distributed', 'parallel', 'quantization'):
            continue
        try:
            m = importlib.import_module(f'megabench.tasks.{name}')
        except Exception:
            continue
        if hasattr(m, 'make_inputs') and hasattr(m, 'reference'):
            found.append(m)
    # the case's own task module has the trial scenarios; helpers (shared make_inputs) come last
    found.sort(key=lambda m: (not hasattr(m, 'set_trial_scenario'),))
    if not found:
        raise SystemExit('no task module with make_inputs() under reference/megabench/tasks; '
                         'define make_inputs(case, seed, trial, device, rank) in ss_task.py')
    return found[0]


def default_case():
    from megabench.cases import CASE
    return CASE


def make_inputs(hooks, case, seed: int, trial: int, device: str, rank: int) -> dict:
    if hasattr(hooks, 'make_inputs'):
        return hooks.make_inputs(case, seed, trial, device, rank)
    tm = task_module()
    v = tm.make_inputs(case, seed, device, rank=rank)
    if hasattr(tm, 'set_trial_scenario'):
        tm.set_trial_scenario(case, v, trial)
    return v


def tolerances(case) -> dict:
    keys = ('atol', 'rtol', 'bf16_rtol', 'noise_factor')
    return {k: float(getattr(case, k)) for k in keys if hasattr(case, k)}


# ----------------------------------------------------------------------------- records
def parse_records(buf: bytes) -> list[dict]:
    out, off = [], 0
    while off + 128 <= len(buf):
        magic, nreg, layer, seq, total = struct.unpack_from('<IIiIQ', buf, off)
        if magic != MAGIC or total == 0:
            break
        name = buf[off + 24:off + 72].split(b'\0')[0].decode()
        regs = []
        for i in range(nreg):
            h = off + 128 + 128 * i
            rname = buf[h:h + 48].split(b'\0')[0].decode()
            addr, nbytes = struct.unpack_from('<QQ', buf, h + 48)
            dtype, = struct.unpack_from('<i', buf, h + 64)
            doff, = struct.unpack_from('<Q', buf, h + 72)
            regs.append(dict(name=rname, addr=addr, bytes=nbytes, dtype=dtype,
                             data=torch.frombuffer(bytearray(buf[off + doff:off + doff + nbytes]), dtype=torch.uint8)
                             if nbytes else torch.empty(0, dtype=torch.uint8)))
        out.append(dict(name=name, layer=layer, seq=seq, regions=regs))
        off += total
    return out


def find_record(records: list[dict], name: str, layer: int) -> dict:
    hits = [r for r in records if r['name'] == name and r['layer'] == layer]
    if not hits:
        have = sorted({(r['name'], r['layer']) for r in records})
        raise SystemExit(f'no capture record {name!r} at layer {layer}; have {have}')
    return hits[0]


# ----------------------------------------------------------------------------- device extents
_cuda = None


def _libcuda():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL('libcuda.so.1')
    return _cuda


def driver_extent(ptr: int):
    """(base, size) of the driver allocation containing ptr, or None (not device memory)."""
    if ptr < (1 << 16):
        return None
    base, size = ctypes.c_uint64(0), ctypes.c_size_t(0)
    r = _libcuda().cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr))
    return (base.value, size.value) if r == 0 else None


def torch_blocks(device) -> list[tuple[int, int]]:
    out = []
    for seg in torch.cuda.memory_snapshot():
        if seg.get('device', None) not in (None, device):
            continue
        for b in seg['blocks']:
            if b['state'] == 'active_allocated':
                out.append((b['address'], b['size']))
    return out


def word_values(params: bytes) -> list[int]:
    n = len(params) // 8
    return list(struct.unpack_from(f'<{n}Q', params, 0))


# ----------------------------------------------------------------------------- comparison
def compare(got: torch.Tensor, want: torch.Tensor, dtype: int, tol: dict) -> dict:
    """Compare one region. Floats: relative L2 and max-abs band of the case's tolerances.
    Integers: exact."""
    t = {2: torch.int16, 4: torch.int32, 6: torch.int64}.get(dtype, DTYPES[dtype])
    g, w = got.view(t), want.view(t)
    d = dict(dtype=DTYPE_NAMES[dtype], elems=int(w.numel()), bitwise_equal=bool(torch.equal(got, want)))
    if dtype in FLOAT_CODES:
        gd, wd = g.double(), w.double()
        finite = bool(torch.isfinite(gd).all())
        diff = (gd - wd)
        rel = float(diff.norm() / wd.norm().clamp_min(1e-30))
        mx = float(diff.abs().max()) if diff.numel() else 0.0
        peak = float(wd.abs().max()) if wd.numel() else 0.0
        rtol = tol.get('bf16_rtol', 0.008) if dtype in (8, 9) else tol.get('rtol', 0.003)
        lim_abs = tol.get('atol', 0.003) + rtol * peak
        d.update(rel_l2=rel, max_abs=mx, rel_limit=rtol, abs_limit=lim_abs, finite=finite,
                 ok=bool(finite and (rel <= rtol) and (mx <= lim_abs)))
    else:
        neq = int((g != w).sum())
        d.update(mismatches=neq, ok=neq == 0)
    return d


def fingerprint(t: torch.Tensor) -> int:
    """Byte sum of a tensor (chunked, no large temporaries)."""
    b = t.contiguous().view(-1).view(torch.uint8)
    return int(sum(int(c.sum(dtype=torch.int64)) for c in torch.split(b, 1 << 26)))


def watchdog(seconds: int, what: str):
    """Hard limit for a GPU job: exit 124 (killing NCCL waits, hung kernels' host side) after `seconds`."""
    import threading

    def fire():
        sys.stderr.write(f'[a3] {what}: timeout after {seconds}s, exiting\n')
        sys.stderr.flush()
        os._exit(124)
    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
