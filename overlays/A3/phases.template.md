# Phase contract table (template)

Copy to `phases.md` and fill it from measurements of the CURRENT candidate (`./phase_table` for
windows, `-Xptxas -v` / `cuobjdump -res-usage` for registers and spills, the source for layouts).
One row per phase. The row is the contract an `engine` subagent must keep: the integrator changes a
row only between engine rounds, and then re-captures (`./capture`). A filled example for a toy
kernel is in `a3tools/toy/phases.md`.

Contract version: v1 (candidate commit `<sha>`, capture `reports/capture-<stamp>`)
Kernel-wide budget: `<threads>` threads/block, `<blocks>` blocks, `<regs>` regs/thread (spills `<B>`),
`<dyn smem>` dynamic + `<static smem>` static shared memory, `<cols>` TMEM columns.

| phase | task (`SS_TASK_REGISTER`) | capture in → out (`SS_CAP` names) | inputs (buffer: dtype, layout) | outputs (buffer: dtype, layout) | smem / regs / TMEM / blocks | sync in → sync out | in-kernel window (µs, max over ranks) | task_ab time (µs) | owner / status |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

Columns:
- **task**: name of the standalone entry point under `#ifdef SS_TASK` (empty: not yet extracted).
- **capture in → out**: `SS_CAP` record names at the phase's start and end; the layers captured.
- **inputs / outputs**: every buffer the phase reads / writes, with dtype, shape and layout (row
  stride, padding, swizzle, which rows are live), plus the rounding points other phases rely on.
  Anything held in registers or shared memory across the phase boundary must be listed too (it is
  NOT captured; the task must recompute it or get it from a captured buffer).
- **smem / regs / TMEM / blocks**: the phase's share of the kernel-wide budget; an engine change that
  raises the kernel's registers or spills hurts every other phase.
- **sync in → sync out**: what starts the phase (grid barrier, counter, flag) and what it signals.
- **in-kernel window**: barrier-to-barrier time in the full kernel; **task_ab time**: the standalone
  time of the same phase (`./task_ab` median). Their ratio tells how faithful the standalone is.
- **owner / status**: engine agent working on it, or merged/rejected with the measured gain.

## Merge log
| date | phase | variant | task_ab A→B (µs) | check_candidate before → after (ms) | kept? |
|---|---|---|---|---|---|
