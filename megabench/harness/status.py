"""Typed evaluation statuses, their exit codes and the exception classes that carry them.

Every evaluation record has one `status` from STATUSES (D-S1). A status is set by the code
that detects the condition, not inferred from an exception type after the fact: the phase
guards in `phases` wrap each stage of an evaluation and translate what went wrong there.

  ok_provisional     correct, within the launch budget (graded cases: this is the only pass)
  correctness_only   correct on the CPU (no launch audit possible)
  non_megakernel     correct, but the launch audit found more than the budget (or a graph launch)
  incorrect          the grader rejected the outputs
  input_mutation     an input changed during the candidate's run (D-I3)
  candidate_error    the candidate raised, crashed, or returned unusable objects
  evaluator_error    the trusted side raised (oracle, grader, snapshot): not the candidate's fault
  infrastructure     the environment failed (NCCL, device, process group, missing weights)
  timeout            a phase deadline or the whole-case deadline passed
  scan_failed        the automated source/artifact scan (D-L7) rejected the submission
  unavailable        the hardware for the case is not visible
  not_implemented    the case has no oracle
Exit codes are stable so that automation can act on them without parsing JSON (ISSUES item 8).
"""

from __future__ import annotations

STATUSES = ("ok_provisional", "correctness_only", "non_megakernel", "incorrect", "input_mutation",
            "candidate_error", "evaluator_error", "infrastructure", "timeout", "scan_failed",
            "unavailable", "not_implemented")
EXIT_CODES = {"ok_provisional": 0, "correctness_only": 0, "non_megakernel": 3, "incorrect": 4,
              "input_mutation": 5, "candidate_error": 6, "evaluator_error": 7, "infrastructure": 8,
              "timeout": 9, "scan_failed": 10, "unavailable": 11, "not_implemented": 12}
# A case counts as failed for the candidate (D-S3) when its status is one of these.
CANDIDATE_FAILURES = frozenset({"incorrect", "input_mutation", "candidate_error", "timeout",
                                "non_megakernel", "scan_failed"})
# These are not the candidate's fault; the evaluation must be repeated, not scored.
RETRY_STATUSES = frozenset({"evaluator_error", "infrastructure", "unavailable"})

assert set(EXIT_CODES) == set(STATUSES)


class EvaluationFailure(Exception):
    """A typed failure. `status` is one of STATUSES; `message` is the grader/phase message
    without an exception-type prefix (ISSUES item 12); `phase` names where it happened."""

    status = "evaluator_error"

    def __init__(self, message: str, *, phase: str | None = None, rank: int | None = None,
                 cause: BaseException | None = None):
        super().__init__(message)
        self.message, self.phase, self.rank, self.cause = message, phase, rank, cause

    def record(self) -> dict:
        out = {"status": self.status, "message": self.message}
        if self.phase is not None:
            out["phase"] = self.phase
        if self.rank is not None:
            out["rank"] = self.rank
        if self.cause is not None:
            out["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return out


class Incorrect(EvaluationFailure):
    status = "incorrect"


class InputMutation(EvaluationFailure):
    status = "input_mutation"


class CandidateError(EvaluationFailure):
    status = "candidate_error"


class EvaluatorError(EvaluationFailure):
    status = "evaluator_error"


class Infrastructure(EvaluationFailure):
    status = "infrastructure"


class Timeout(EvaluationFailure):
    status = "timeout"


class ScanFailed(EvaluationFailure):
    status = "scan_failed"


_INFRA_MARKERS = ("NCCL", "ncclInternalError", "ncclSystemError", "ncclRemoteError", "CUDA error: unspecified launch failure",
                  "CUDA driver error", "device-side assert", "Connection reset", "Socket Timeout",
                  "process group", "ProcessGroup", "No space left on device", "no converted weights",
                  "CUDA error: out of memory")


def classify(exc: BaseException, *, phase: str, rank: int | None = None) -> EvaluationFailure:
    """Turn an arbitrary exception raised in `phase` into a typed failure.

    The phase decides who is to blame: exceptions raised while the candidate's code runs
    (`build`, `candidate`) are candidate_error; grader assertions are incorrect; exceptions in
    the trusted phases are evaluator_error; messages that name the fabric or the device are
    infrastructure in every phase; timeouts stay timeouts."""
    if isinstance(exc, EvaluationFailure):
        if exc.phase is None:
            exc.phase = phase
        if exc.rank is None:
            exc.rank = rank
        return exc
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutExpired" or (
            type(exc).__name__ == "DistBackendError" and "Timeout" in text):
        return Timeout(text, phase=phase, rank=rank, cause=exc)
    if phase in ("candidate", "build"):
        # Everything the candidate's own code raises is its error, including out-of-memory and a
        # fabric error during its run; the group-level merge in distributed.py reports the ranks.
        return CandidateError(text, phase=phase, rank=rank, cause=exc)
    if any(marker in text for marker in _INFRA_MARKERS):
        return Infrastructure(text, phase=phase, rank=rank, cause=exc)
    if phase == "grade" and isinstance(exc, AssertionError):
        return Incorrect(str(exc), phase=phase, rank=rank, cause=exc)
    if phase == "immutability" and isinstance(exc, AssertionError):
        return InputMutation(str(exc), phase=phase, rank=rank, cause=exc)
    return EvaluatorError(text, phase=phase, rank=rank, cause=exc)


def merge_group(per_rank: list[dict]) -> dict:
    """Group status from per-rank statuses (D-S1). Precedence: the candidate's own faults first,
    then timeouts, then infrastructure, then evaluator errors; a group passes only if every rank passed."""
    statuses = [rank.get("status") for rank in per_rank]
    order = ("input_mutation", "incorrect", "scan_failed", "candidate_error", "non_megakernel", "timeout",
             "infrastructure", "evaluator_error", "unavailable")
    for status in order:
        if status in statuses:
            culprits = [rank["rank"] for rank in per_rank if rank.get("status") == status]
            first = next(rank for rank in per_rank if rank.get("status") == status)
            return {"status": status, "ranks_with_status": culprits,
                    "message": first.get("failure", {}).get("message") or first.get("reason", ""),
                    "phase": first.get("failure", {}).get("phase")}
    if all(status in ("pass", "ok_provisional", "correctness_only") for status in statuses):
        return {"status": "pass", "ranks_with_status": list(range(len(per_rank)))}
    return {"status": "evaluator_error", "ranks_with_status": [rank["rank"] for rank in per_rank],
            "message": f"unexpected per-rank statuses {statuses}"}
