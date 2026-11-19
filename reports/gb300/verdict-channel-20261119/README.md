# Trusted verdict channel: CPU evidence (2026-11-19)

Blocker B-C1-1 (C1-OPERATOR-SAFETY-REVIEW §7): GPU-job verdicts were candidate-writable. Design and
boundary analysis: `sandbox/k8s/TRUSTED-VERDICT.md`; decisions L19, L20.

| file | what |
|---|---|
| `unittest-full.log.txt` | 34 CPU tests on rack 3 (venv Python, no GPU): harness, kimi oracle, sandbox incl. `GradeProtocolTests`, `TransportTests`, `SpoolProtocolTests` |
| `docker-two-container-regression.log.txt` | `sandbox/k8s/c1/local/verdict_test.sh` on the project image: grader (uid 2000) and runner (uid 1000) containers, separate pid namespaces, one shared job directory, real broker spool files |
| `check_manifests.json` | static checks of all manifests with the new two-container GPU pod rules |

## Regression controls (requirement 5) and outcomes

| attempt | where | outcome |
|---|---|---|
| forge a successful verdict (`done/<rid>.json`, `verdict/<rid>.json`) | exec shell, candidate `build()` | denied: no results mount in the runner (ENOENT); unit test: the grader's record overwrites and the broker refuses records older than the dispatch |
| modify another job's result | candidate `build()` | denied (ENOENT); same-uid unit test: inert, the broker never dispatched it |
| replace evaluator outputs (grader log, private result file) | candidate `build()` | denied (ENOENT); the grader's `/tmp` is its own emptyDir |
| alter a pending broker request / inject one | exec shell, candidate | denied (ENOENT: no request mount) |
| alter own `job.json`, graded snapshot, add files to the job dir | candidate | denied (EACCES: 0755 dirs owned by the grader) |
| kill or interfere with the grading process | killer candidate, uid-wide SIGKILL | grader not visible (0 pids), not signallable; the spawner (pid 1) survives signals from inside; job graded `incorrect`, next job passes |
| see the grader through `/proc/1/root` | candidate | denied: pid 1 is the spawner, its root is the runner's |
| modify protected trajectory records | exec shell, candidate | denied (ENOENT: no records mount in either container) |
| write the job root, write own `out/exit` | exec shell, candidate | allowed by design (runner-owned scratch; `exit` is candidate-reported) |
| agent-supplied harness options, submission outside /work, unknown case | unit test | rejected by the broker |
| runner unresponsive | Docker (paused) | `INFRA_RUNNER_DOWN` (exit 8), `/job/RESTART-RUNNER` written, liveness probe fails, restart clears the flag, next job passes |

Verdicts observed: oracle `correctness_only` (trusted, tiny self-test geometry), hostile `correctness_only`
(its attempts preserved as candidate-authored stderr), killer `incorrect`. No forged record appeared in the
results spool. Numerical tolerances and the launch budget were not touched (requirement 8): the evaluator
code paths (`correctness.py`, `benchmark.py`, `nsys_audit.py`, deadlines) are unchanged; only the process
transport changed.

## Not covered here (live tests, TRUSTED-VERDICT.md)

GPU visibility in the runner container, CUDA IPC across containers, Kyverno admission of the two-container
pod, the kubelet restart path, interprocess CUDA events and nsys tracing of remote candidate processes.
