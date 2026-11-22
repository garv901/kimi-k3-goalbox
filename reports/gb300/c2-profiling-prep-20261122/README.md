# C2 profiling experiment preparation: CPU evidence (2026-11-22)

Nothing ran on a GPU. `controls-cpu-check.txt`: the evaluator rejects `--trials 0` (so the controls need the
proposed controls-only `--mechanism-test` flag, awaiting the owners' GO); the four control candidates import
and run twice on tiny CPU inputs with structurally valid outputs; the scanner flags the graph control
(`cuda_graph`) and, as expected for every control, the oracle import of the shared helper (controls run with
`--no-scan`). Matrix: `sandbox/k8s/C2-PROFILING-TEST-MATRIX.md`; decision request:
`reports/gb300/DECISION-REQUEST-B-C2-1.md`.
