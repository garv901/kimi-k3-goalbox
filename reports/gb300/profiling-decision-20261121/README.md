# B-C2-1 profiling experiments: CPU-only evidence (2026-11-21)

Experiments ran on rack 3 tray11 in Docker containers (own pid namespaces, no GPU, no uid-wide process
controls) with Nsight Systems 2026.3.2 from the project image. Scripts: `sandbox/k8s/c2/local/`
(`profiler_experiments.sh`, `profiler_child.py`, `e3b.sh`, `e3c.sh`, `dropuid.c`, `Dockerfile.e3b`; the
E3b image is an experiment image, not the deployed one).

| file | what |
|---|---|
| `nsys-experiments-analysis.txt` | E1 clock mapping (trace vs CLOCK_MONOTONIC/BOOTTIME/REALTIME/MONOTONIC_RAW), E2 injection escape (child with emptied environment), E3b and E3c uid separation (marks, process lists, profiler diagnostics) |
| `unittest-full.log.txt` | 37 CPU tests on rack 3 including `audit_windows` (invalid unless complete) and the evaluator's repetition windows |

Key readings: trace time = CLOCK_MONOTONIC minus the session's `systemClockNs` within 10 us (0.2 ms on
the first mark, library warm-up); CLOCK_MONOTONIC_RAW is 23.7 s off; an untraced child is absent from the
trace and its process list; a uid-1001 target under a uid-1000 profiler keeps the report unwritable but
delivers no events (E3b, E3c). Analysis and recommendation: `sandbox/k8s/PROFILING-DECISION.md`; decision L24.
