"""CPU-only, read-only check of the evaluator's device-checksum snapshot (megabench.harness.correctness.snapshot):
for tensors above 64 MB it keeps two exact int64 sums of the int32 words, plain and weighted by (i mod 65521) + 1 per
2^26-word chunk. Both sums are linear, so a pair of word changes +d at word i and -d at word i + 65521 (same weight,
same chunk) leaves both unchanged. This script shows (1) a single-word change is detected, (2) the compensating pair is
not, (3) the same pair on a 64 MB tensor (host-copy path, torch.equal) is detected. Run with PYTHONPATH=<repo>,
CUDA_VISIBLE_DEVICES="". Nothing in the evaluator is modified."""
import torch
from megabench.harness.correctness import check_unchanged, snapshot

torch.manual_seed(0)
t = torch.randn(1 << 26, dtype=torch.bfloat16)  # 2^26 BF16 = 128 MB = 2^25 int32 words, one checksum chunk
inputs = {'x': t}
before = snapshot(inputs)
path = 'device checksums' if before['x'].dtype == torch.int64 and before['x'].numel() == 2 else 'host copy'
print(f'tensor bytes {t.numel() * t.element_size()} path {path} sums_before {before["x"].tolist()}')
words = t.view(torch.uint8).view(torch.int32)  # the same word view snapshot() uses; shares storage with t


def detected() -> str:
    try:
        check_unchanged(inputs, before)
        return 'NOT detected'
    except AssertionError as e:
        return f'detected ({e})'


words[12345] += 1
print(f'single word +1 at 12345: {detected()}')
words[12345] -= 1
print(f'revert: {detected()}')
for i, d in [(100, 1), (7_000_000, 12345), (0, -3)]:
    j = i + 65521
    bf_before = (float(t[2 * i]), float(t[2 * i + 1]), float(t[2 * j]), float(t[2 * j + 1]))
    words[i] += d
    words[j] -= d
    bf_after = (float(t[2 * i]), float(t[2 * i + 1]), float(t[2 * j]), float(t[2 * j + 1]))
    after = snapshot(inputs)
    print(f'pair +{d} at word {i}, -{d} at word {j}: bf16 values {bf_before} -> {bf_after}, '
          f'sums_after {after["x"].tolist()}, {detected()}')
    words[i] -= d
    words[j] += d
print(f'final revert: {detected()}')
s = torch.randn(1 << 25, dtype=torch.bfloat16)  # 64 MB exactly: host-copy path
inputs2 = {'y': s}
before2 = snapshot(inputs2)
w2 = s.view(torch.uint8).view(torch.int32)
w2[100] += 1
w2[100 + 65521] -= 1
try:
    check_unchanged(inputs2, before2)
    print(f'64 MB tensor (path {"host copy" if before2["y"].dtype == torch.bfloat16 else "?"}), same pair: NOT detected')
except AssertionError as e:
    print(f'64 MB tensor (path host copy, {before2["y"].dtype}), same pair: detected ({e})')
