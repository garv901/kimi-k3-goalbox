"""Validate submission outputs against a task reference, and detect input mutation.

Input integrity (D-I1, D-I4): `snapshot()` records every input tensor before the candidate
runs, `check_unchanged()` compares after. Tensors up to 64 MB (and any tensor whose byte
size is not a multiple of 4) are copied to the host and compared byte for byte, so the
check is exact. Larger tensors (the real weight shards) get a keyed hash computed on the
device: two independent sums of the tensor's 32-bit words weighted by secret, odd, pseudo-
random 64-bit coefficients derived from a key that is drawn fresh for every evaluation and
lives only in the evaluator process (the candidate runs in another process, D-I2). A
mutation whose word deltas are not all zero passes both sums with probability about 2^-128
unless the candidate knows the key. The previous fixed-weight checksums were linear with
public coefficients and could be preserved by a sum-preserving edit (ISSUES item 17).
"""

from __future__ import annotations

import secrets
import time
from typing import Callable

from ..cases import Case
from ..tasks.workloads import compare_outputs, make_inputs, reference

HOST_COPY_LIMIT = 1 << 26        # bytes: at or below this a tensor is copied to the host (exact)
CHUNK_WORDS = 1 << 26            # 32-bit words per hashed chunk (256 MiB)
MASK64 = (1 << 64) - 1


def new_key() -> int:
    """A fresh 64-bit key per evaluation. Never passed to the candidate."""
    return secrets.randbits(64)


_WEIGHTS: dict = {}


def _coefficients(key: int, device, which: int):
    """Odd pseudo-random int64 coefficients for one chunk position, cached per (key, device, which)."""
    import torch

    cache_key = (key, str(device), which)
    if cache_key not in _WEIGHTS:
        generator = torch.Generator(device=device).manual_seed((key ^ (0x9E3779B97F4A7C15 * (which + 1))) & ((1 << 63) - 1))
        draw = torch.randint(-(1 << 62), (1 << 62) - 1, (CHUNK_WORDS,), generator=generator, device=device, dtype=torch.int64)
        _WEIGHTS.clear()
        _WEIGHTS[cache_key] = draw * 2 + 1
    return _WEIGHTS[cache_key]


def _chunk_multiplier(key: int, which: int, chunk: int) -> int:
    """A secret odd multiplier per chunk index so that reusing one coefficient vector across chunks
    does not make two chunks interchangeable."""
    x = (key ^ (which * 0xD1B54A32D192ED03) ^ ((chunk + 1) * 0x2545F4914F6CDD1D)) & MASK64
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x ^= x >> 33
    return (x | 1) & ((1 << 62) - 1)


def keyed_hash(value, key: int):
    """Two keyed 64-bit sums over the int32 words of a contiguous tensor, as a CPU int64[2] tensor."""
    import torch

    words = value.reshape(-1).view(torch.uint8).view(torch.int32)
    total = torch.zeros(2, dtype=torch.int64, device=value.device)
    for index, start in enumerate(range(0, words.numel(), CHUNK_WORDS)):
        chunk = words[start:start + CHUNK_WORDS].to(torch.int64)
        for which in range(2):
            coefficients = _coefficients(key, value.device, which)[:chunk.numel()]
            # int64 arithmetic wraps modulo 2^64 on the device, which is the intended ring
            total[which] += (chunk * coefficients).sum() * _chunk_multiplier(key, which, index)
    return total.cpu()


def _needs_hash(value) -> bool:
    nbytes = value.numel() * value.element_size()
    return nbytes > HOST_COPY_LIMIT and value.is_contiguous() and nbytes % 4 == 0


def snapshot(inputs: dict, key: int | None = None) -> dict:
    """Record every input: exact host copy (as bytes) for small tensors, keyed hash for large ones.

    `key` must be the evaluation's secret key when the inputs contain large tensors."""
    import torch

    result = {"_key_used": key is not None}
    for name, value in inputs.items():
        value = value.detach()
        if _needs_hash(value):
            if key is None:
                raise ValueError(f"snapshot of {name} ({value.numel() * value.element_size()} bytes) needs a key")
            result[name] = ("hash", keyed_hash(value, key))
        else:
            # Bytes, not torch.equal: exact for every dtype (fp8, bool, bf16 NaN payloads included).
            result[name] = ("bytes", value.contiguous().view(torch.uint8).to("cpu", copy=True))
    return result


def changed_inputs(inputs: dict, before: dict, key: int | None = None) -> list[str]:
    """Names of inputs whose snapshot differs now."""
    import torch

    after = snapshot(inputs, key)
    return [name for name in inputs
            if before[name][0] != after[name][0] or not torch.equal(after[name][1], before[name][1])]


def check_unchanged(inputs: dict, before: dict, key: int | None = None) -> None:
    changed = changed_inputs(inputs, before, key)
    if changed:
        from .status import InputMutation
        raise InputMutation(f"candidate mutated input(s) {', '.join(changed)}", phase="immutability")


def structural_check(case: Case, expected: dict, actual, device: str) -> None:
    """Rank-local checks that need no collective: keys, tensor type, shape, dtype, device (D-S2).
    Raises Incorrect. Runs before any agreement so that one bad rank cannot desynchronize the group."""
    import torch
    from .status import CandidateError, Incorrect

    if not isinstance(actual, dict):
        raise CandidateError(f"run() must return a dict of tensors, got {type(actual).__name__}")
    if set(actual) != set(expected):
        raise Incorrect(f"output keys: expected {sorted(expected)}, got {sorted(actual)}")
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise CandidateError(f"{name}: output must be a Tensor, got {type(got).__name__}")
        if got.shape != want.shape or got.dtype != want.dtype:
            raise Incorrect(f"{name}: expected {tuple(want.shape)} {want.dtype}, got {tuple(got.shape)} {got.dtype}")
        if got.device != want.device:
            raise Incorrect(f"{name}: output is on {got.device}, expected {want.device}")


def compare(case: Case, inputs: dict, expected: dict, actual, device: str, *, serial: bool = False) -> dict:
    details = compare_outputs(case, inputs, expected, actual, serial=serial)
    return _compare(expected, actual, case, device) if details is None else details


def _compare(expected: dict, actual: dict, case: Case, device: str) -> dict:
    import torch

    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AssertionError(f"output keys: expected {sorted(expected)}, got "
                             f"{sorted(actual) if isinstance(actual, dict) else type(actual)}")
    details = {}
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{name}: output must be a Tensor")
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: expected {tuple(want.shape)} {want.dtype}, "
                                 f"got {tuple(got.shape)} {got.dtype}")
        if got.device != want.device:
            raise AssertionError(f"{name}: output is on {got.device}, expected {device}")
        if want.is_floating_point():
            delta = (got.float() - want.float()).abs()
            rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
            close = torch.isclose(got.float(), want.float(),
                                  atol=case.atol, rtol=rtol, equal_nan=False)
            details[name] = {"max_abs_error": float(delta.max().item()),
                             "mismatched_elements": int((~close).sum().item())}
            if not bool(close.all()):
                raise AssertionError(f"{name}: {details[name]}")
        else:
            mismatch = int((got != want).sum().item())
            details[name] = {"mismatched_elements": mismatch}
            if mismatch:
                raise AssertionError(f"{name}: {mismatch} integer mismatches")
    return details


def check_trials(case: Case, candidate: Callable, device: str, trials: int, *, key: int | None = None,
                 synchronize: Callable | None = None) -> tuple[dict, float | None]:
    """Single-process correctness trials (TP1 path): fresh inputs, immutability, grading.

    `candidate` is the callable from the isolation layer (its exceptions are already typed) or a
    plain callable. Exceptions propagate typed: Incorrect / InputMutation / CandidateError."""
    import torch
    from .status import classify

    key = new_key() if key is None else key
    result = {"status": "pass", "trials": []}
    first_call_ms: float | None = None
    trial_seeds = [secrets.randbits(32) for _ in range(trials)]
    for trial_index, seed in enumerate(trial_seeds):
        inputs = make_inputs(case, seed, device)
        originals = snapshot(inputs, key)
        expected = reference(case, inputs)
        first_start = time.perf_counter() if trial_index == 0 else None
        try:
            actual = candidate(inputs)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        except Exception as exc:
            raise classify(exc, phase="candidate") from exc
        if first_start is not None:
            first_call_ms = (time.perf_counter() - first_start) * 1000
        check_unchanged(inputs, originals, key)
        structural_check(case, expected, actual, device)
        try:
            details = compare(case, inputs, expected, actual, device, serial=case.gpus == 1)
        except AssertionError as exc:
            raise classify(exc, phase="grade") from exc
        result["trials"].append({"seed": seed, "scenario": "seeded", "outputs": details})
        del inputs, originals, expected, actual
    return result, first_call_ms
