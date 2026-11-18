"""Verification helper (not part of the overlay): insert phase_table.cuh stamps into the DSv4.1 b8 seed.

Run in a scratch copy of the seed: python3 instrument_seed.py. Edits mk_head.cu / mk_tail.cu (stamps only,
compiled to nothing without -DPHASE_TABLE) and appends build_timeline() to submission.py. This is what
an agent would do by hand following TASK.d/A1.md; the profiler's MKPROF stamps are ported 1:1.
"""


def sub(text, old, new, count=1):
    n = text.count(old)
    assert n == count, (old[:70], n)
    return text.replace(old, new)


h = open('mk_head.cu').read()
t = open('mk_tail.cu').read()

h = sub(h, '#include "mk_params.h"\n', '#include "mk_params.h"\n#include "phase_table.cuh"\n')
# grid barrier: thread 0 stamps arrival after the block-level __syncthreads, release after the wait
h = sub(h, '__device__ __forceinline__ void grid_sync(unsigned* bar) {\n  __syncthreads();\n  if (threadIdx.x == 0) {\n',
        '__device__ __forceinline__ void grid_sync(unsigned* bar, int tag = 0) {\n  __syncthreads();\n  if (threadIdx.x == 0) {\n    PT_ARRIVE(tag);\n')
h = sub(h, '      while (*vgen == gen) {}\n    }\n    __threadfence();\n  }\n  __syncthreads();\n}',
        '      while (*vgen == gen) {}\n    }\n    __threadfence();\n    PT_RELEASE(tag);\n  }\n  __syncthreads();\n}')
# cross-GPU barrier = grid sync, flag exchange, grid sync: tag both halves with the call site
h = sub(h, '__device__ void xbar(Ctx& c) {\n  grid_sync(c.bar);', '__device__ void xbar(Ctx& c, int tag = 0) {\n  grid_sync(c.bar, tag | 0x10000);')
h = sub(h, '  c.epoch = e;\n  grid_sync(c.bar);\n}', '  c.epoch = e;\n  grid_sync(c.bar, tag | 0x20000);\n}')
h = sub(h, '#define XBAR(c) do { xbar(c); tmark', '#define XBAR(c) do { xbar(c, __LINE__); tmark')
h = sub(h, '#define GSYNC(c) do { grid_sync((c).bar); tmark', '#define GSYNC(c) do { grid_sync((c).bar, __LINE__); tmark')

# in-phase counter waits
t = sub(t, '        while (*cp < (unsigned)bpr) {}\n', '        PT_SPIN(while (*cp < (unsigned)bpr) {});\n')
t = sub(t, '          while (*cp < (unsigned)(INTER / 32)) {}\n', '          PT_SPIN(while (*cp < (unsigned)(INTER / 32)) {});\n')
# context marks: layer start; back to "outside" after each layer
t = sub(t, '  unsigned long long pht = gtime();\n  int phn = 0;\n', '  unsigned long long pht = gtime();\n  int phn = 0;\n  PT_MARK(0x30000 | (draft ? 0x100 : 0) | l);\n')
t = sub(t, '    run_block(c, sh, dsm, b, l, false);\n', '    run_block(c, sh, dsm, b, l, false);\n    PT_MARK(0x3FFFF);\n')
t = sub(t, '    run_block(c, sh, dsm, b, s, true);\n', '    run_block(c, sh, dsm, b, s, true);\n    PT_MARK(0x3FFFF);\n')
# kernel start / end
t = sub(t, '  c.nmark = 0;\n', '  PT_INIT();\n  c.nmark = 0;\n')
t = sub(t, '  GSYNC(c);\n  if (blockIdx.x == 0 && threadIdx.x == 0) *(volatile unsigned long long*)(c.scr + 64) = c.epoch;\n}',
        '  GSYNC(c);\n  if (blockIdx.x == 0 && threadIdx.x == 0) *(volatile unsigned long long*)(c.scr + 64) = c.epoch;\n  PT_FINISH();\n}')
t = sub(t, '  m.def("scratch_bytes", &mk_scratch_bytes);\n', '  m.def("scratch_bytes", &mk_scratch_bytes);\n  PHASE_TABLE_PYBIND(m);\n')
open('mk_head.cu', 'w').write(h)
open('mk_tail.cu', 'w').write(t)

SUB = r'''

# ---- phase_table instrumented build (used only by ./phase_table; see TASK.d/A1.md)
def build_timeline(case: dict):
    import re
    import torch
    base = os.environ.get('MK_FLAGS', mk_host.DEFAULT_FLAGS if hasattr(mk_host, 'DEFAULT_FLAGS') else '-DNO_PF')
    os.environ['MK_FLAGS'] = base + ' -DPHASE_TABLE'
    runner = mk_host.Runner(case)
    blocks, slots = runner.nblocks, 2048
    buf = torch.zeros(16 + blocks * (8 + 4 * slots), dtype=torch.int64, device=runner.device)
    rc = runner.ext.phase_table_bind(buf.data_ptr(), buf.numel())
    assert rc == 0, f'phase_table_bind returned {rc}: build lacks -DPHASE_TABLE'
    # names: barrier ids are mk.cu line numbers (+0x10000 / 0x20000 for the two halves of an XBAR)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mk.cu')).read().split('\n')
    names, label, func = {}, '?', '?'
    for i, line in enumerate(src, 1):
        if re.match(r'(__device__|__global__)\s', line):
            func, label = line.split('(')[0].split()[-1], None
        m = re.search(r'// =+ (.*)', line)
        if m:
            label = m.group(1)[:28]
        if 'GSYNC' in line or 'XBAR' in line:
            lab = label or func
            names[i] = lab
            names[i | 0x10000] = lab + ' [xbar part 1]'
            names[i | 0x20000] = lab + ' [xbar flag exchange]'
    for l in range(64):
        names[0x30000 | l] = f'verify L{l}'
        names[0x30100 | l] = f'draft D{l}'
    names[0x3FFFF] = 'outside layers'

    def run(inputs):
        runner(inputs)
        return buf
    return run, names
'''
s = open('submission.py').read()
if 'def build_timeline' not in s:
    open('submission.py', 'w').write(s + SUB)
print('instrumented ok')
