"""Evaluate a submission with one evaluator process per local GPU and the candidate in its own
process per rank (D-I2). Every case goes through this path, including TP1 (world size 1).

Per rank: init the evaluator process group -> start the candidate process (build) -> trials
(fresh inputs, keyed snapshot, oracle, candidate run, immutability, structural checks,
agreement, band grading) -> timing (warmup, timed reps under NVTX ranges, immutability,
grading of the last timed outputs) -> reference timing (synthetic cases) -> advisory
in-process profile of the candidate -> immutability -> record. Each phase is guarded (D-S2):
a rank that fails reports its typed failure at the next agreement point, so no rank is left
alone in a collective. Statuses per rank and for the group follow status.py (D-S1).

Two trays (DECISIONS section 2): launch.py reads GOALBOX_NNODES / GOALBOX_NODE_RANK /
GOALBOX_MASTER_ADDR / GOALBOX_MASTER_PORT; this module spawns only the local ranks and
gathers every rank's record on global rank 0 at the end.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..cases import Case
from ..tasks.parallel import initialize
from ..workloads import make_inputs, reference
from .benchmark import _measure, classify_events, measure_candidate
from .correctness import changed_inputs, compare, new_key, snapshot, structural_check
from .deadlines import Deadlines, for_case
from . import transport
from .isolation import CandidateProcess
from .launch import Topology, topology_from_env
from .status import EvaluationFailure, Incorrect, InputMutation, classify, merge_group


class GroupAbort(Exception):
    def __init__(self, failures: list[dict]):
        super().__init__(f"{len(failures)} rank(s) failed")
        self.failures = failures


def agree(local_failure: dict | None) -> None:
    """Exchange typed failures. Raises GroupAbort on every rank if any rank failed (D-S2)."""
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_failure)
    failures = [f for f in gathered if f]
    if failures:
        raise GroupAbort(failures)


def shared_seed() -> int:
    values = [secrets.randbits(32) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


class Phase:
    """Run one guarded step; return (value, failure_record_or_None) and time it."""

    def __init__(self, rank: int, record: dict):
        self.rank, self.record = rank, record

    def __call__(self, name: str, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            value = fn(*args, **kwargs)
        except GroupAbort:
            raise
        except BaseException as exc:  # noqa: BLE001 - typed below
            failure = classify(exc, phase=name, rank=self.rank)
            entry = failure.record() | {"traceback": traceback.format_exc(limit=8)}
            self.record.setdefault("phases", {})[name] = {"ms": (time.perf_counter() - start) * 1000, "ok": False}
            return None, entry
        self.record.setdefault("phases", {})[name] = {"ms": (time.perf_counter() - start) * 1000, "ok": True}
        return value, None


def _immutability(inputs: dict, originals: dict, key: int, when: str) -> None:
    changed = changed_inputs(inputs, originals, key)
    if changed:
        raise InputMutation(f"candidate mutated input(s) {', '.join(changed[:8])} ({len(changed)} total) {when}",
                            phase="immutability")


def _grade(case: Case, inputs: dict, expected: dict, actual: dict, device: str) -> dict:
    try:
        return compare(case, inputs, expected, actual, device, serial=case.gpus == 1)
    except AssertionError as exc:
        raise Incorrect(str(exc), phase="grade") from exc


def _rank_worker(local_rank: int, contract: dict, submission: str, device_kind: str, first_device: int,
                 trials: int, warmup: int, reps: int, topo_dict: dict, output_dir: str, deadlines_dict: dict,
                 options: dict) -> None:
    case = Case(**contract)
    topo = Topology(**topo_dict)
    deadlines = Deadlines(**deadlines_dict)
    rank = topo.global_rank(local_rank)
    device = f"cuda:{first_device + local_rank}" if device_kind == "cuda" else "cpu"
    if device_kind == "cuda":
        torch.cuda.set_device(first_device + local_rank)
    result = {"rank": rank, "local_rank": local_rank, "node_rank": topo.node_rank, "device": device,
              "host": platform.node(), "status": "pass", "phases": {}}
    phase = Phase(rank, result)
    candidate = None
    initialized = False
    own_failure: dict | None = None
    try:
        dist.init_process_group("nccl" if device_kind == "cuda" else "gloo", init_method=topo.rendezvous(output_dir, offset=0),
                                world_size=case.gpus, rank=rank, timeout=timedelta(seconds=deadlines.collective_s),
                                device_id=torch.device(device) if device_kind == "cuda" else None)
        initialized = True
        ctx = initialize(case)
        key = new_key()
        execution = None
        group = None
        if case.gpus > 1:
            execution = {"rank": rank, "world_size": case.gpus, "tp_rank": ctx.tp_rank, "ep_rank": ctx.ep_rank,
                         "backend": dist.get_backend(), "device": device, "node_rank": topo.node_rank,
                         "local_rank": local_rank, "nnodes": topo.nnodes}
            group = {"backend": dist.get_backend(), "init_method": topo.rendezvous(output_dir, offset=1),
                     "world_size": case.gpus, "rank": rank, "timeout_s": deadlines.collective_s}
        contract_for_candidate = dict(contract)
        if execution is not None:
            contract_for_candidate["execution"] = execution
        stderr_path = Path(output_dir) / f"candidate-rank{rank}.stderr"

        def start_candidate():
            return CandidateProcess(Path(submission), contract_for_candidate, device, group=group,
                                    stderr_path=stderr_path, build_timeout_s=deadlines.build_s)

        candidate, own_failure = phase("build", start_candidate)
        agree(own_failure)
        result["build_ms"] = candidate.build_ms
        result["correctness"] = {"status": "pass", "trials": []}
        with torch.inference_mode():
            pool = transport.input_pool()
            result["export_audit"] = []

            def pooled_inputs(case_, seed_, device_):
                if pool is None:
                    return make_inputs(case_, seed_, device_)
                with torch.cuda.use_mem_pool(pool):
                    return make_inputs(case_, seed_, device_)

            def audited(inputs_):
                record = transport.audit_exports(inputs_, pool, device)     # D-I5: only input segments are exported
                result["export_audit"].append(record)
                return record

            for trial in range(trials):
                seed = shared_seed()
                inputs, own_failure = phase("inputs", pooled_inputs, case, seed, device)
                agree(own_failure)
                originals, own_failure = phase("snapshot", snapshot, inputs, key)
                agree(own_failure)
                expected, own_failure = phase("reference", reference, case, inputs)
                agree(own_failure)
                _, own_failure = phase("export_audit", audited, inputs)
                agree(own_failure)
                first_start = time.perf_counter() if trial == 0 else None
                run, own_failure = phase("candidate", candidate.run, inputs, deadlines.trial_s)
                if own_failure is None:
                    actual, _timing = run
                    if first_start is not None:
                        result["first_call_ms_including_jit"] = (time.perf_counter() - first_start) * 1000
                    _, own_failure = phase("immutability", _immutability, inputs, originals, key, f"in trial {trial}")
                if own_failure is None:
                    _, own_failure = phase("structure", structural_check, case, expected, actual, device)
                agree(own_failure)
                details, own_failure = phase("grade", _grade, case, inputs, expected, actual, device)
                agree(own_failure)
                result["correctness"]["trials"].append({"seed": seed, "outputs": details})
                del inputs, originals, expected, actual
            # Timing on a fresh seed; the last timed outputs are graded (poisoned buffers would fail).
            seed = shared_seed()
            perf_inputs, own_failure = phase("inputs", pooled_inputs, case, seed, device)
            agree(own_failure)
            originals, own_failure = phase("snapshot", snapshot, perf_inputs, key)
            agree(own_failure)
            expected, own_failure = phase("reference", reference, case, perf_inputs)
            agree(own_failure)
            _, own_failure = phase("export_audit", audited, perf_inputs)
            agree(own_failure)
            dist.barrier()
            timed, own_failure = phase("timing", measure_candidate, candidate, perf_inputs, warmup, reps,
                                       rep_timeout_s=deadlines.rep_s, nvtx_prefix=f"megabench:rank{rank}")
            if own_failure is None:
                result["candidate_timing"], last_outputs = timed
                _, own_failure = phase("immutability", _immutability, perf_inputs, originals, key, "during timing")
            if own_failure is None:
                _, own_failure = phase("structure", structural_check, case, expected, last_outputs, device)
            agree(own_failure)
            details, own_failure = phase("grade_after_timing", _grade, case, perf_inputs, expected, last_outputs, device)
            agree(own_failure)
            result["timed_outputs_grade"] = details
            dist.barrier()
            reference_reps = options.get("reference_reps", 0)
            if reference_reps:
                timing, own_failure = phase("reference_timing", _measure, lambda values: reference(case, values),
                                            perf_inputs, device, min(warmup, 1), reference_reps)
                if own_failure is None:
                    result["reference_timing"] = timing
                agree(own_failure)
            dist.barrier()
            audit, own_failure = phase("audit", candidate.profile, perf_inputs, deadlines.audit_s)
            if own_failure is None:
                outputs, counts = audit
                result["launch_audit_advisory"] = counts | {"note": "in-process count from the candidate process; "
                                                                    "the audit of record is the nsys trace (nsys_audit.py)"}
                _, own_failure = phase("immutability", _immutability, perf_inputs, originals, key, "during the audit")
            agree(own_failure)
            del perf_inputs, originals, expected
        if device_kind == "cuda":
            result["hardware"] = {"gpu_name": torch.cuda.get_device_name(device),
                                  "evaluator_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "evaluator_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    except GroupAbort as abort:
        result["status"] = own_failure["status"] if own_failure else "aborted_by_group"
        if own_failure:
            result["failure"] = own_failure
        result["group_failures"] = abort.failures
    except EvaluationFailure as failure:
        result["status"] = failure.status
        result["failure"] = failure.record() | {"traceback": traceback.format_exc(limit=8)}
    except Exception as exc:  # noqa: BLE001 - last resort, typed as infrastructure/evaluator by classify
        failure = classify(exc, phase="setup", rank=rank)
        result["status"] = failure.status
        result["failure"] = failure.record() | {"traceback": traceback.format_exc(limit=8)}
    finally:
        if candidate is not None:
            result["candidate_stderr_tail"] = candidate.stderr_tail()
            candidate.close()
        result["deadlines_s"] = deadlines.to_dict()
        result["measurement"] = measurement_validity(result, reps)
        with (Path(output_dir) / f"rank-{rank}.json").open("x") as file:
            json.dump(result, file)
        if initialized:
            try:
                gathered: list = [None] * case.gpus
                dist.all_gather_object(gathered, result)
                if rank == 0:
                    (Path(output_dir) / "all-ranks.json").write_text(json.dumps(gathered))
            except Exception as exc:  # noqa: BLE001 - a broken group only loses the remote records
                (Path(output_dir) / f"gather-failed-rank{rank}.txt").write_text(f"{type(exc).__name__}: {exc}")
            dist.destroy_process_group()


def measurement_validity(result: dict, reps: int) -> dict:
    """Contamination policy (DECISIONS-LOG L22): a timing figure is reported only when this rank completed
    every timed repetition, the outputs of the last repetition graded correct, the inputs were unchanged,
    the audit ran, and nothing failed on the rank. Otherwise the timing is kept as evidence under
    `invalid_timing` and never as `candidate_timing`."""
    reasons = []
    timing = result.get("candidate_timing")
    if result.get("status") != "pass":
        reasons.append(f"rank status {result.get('status')}")
    if result.get("failure"):
        reasons.append(f"failure in phase {result['failure'].get('phase')}")
    if timing is None:
        reasons.append("no timing")
    elif timing.get("reps", reps) != reps:
        reasons.append(f"{timing.get('reps')} of {reps} repetitions")
    if result.get("timed_outputs_grade") is None:
        reasons.append("timed outputs not graded")
    if timing is not None and "launch_audit_advisory" not in result:
        reasons.append("audit did not run after timing")
    valid = not reasons
    if not valid and timing is not None:
        result["invalid_timing"] = result.pop("candidate_timing")
    return {"valid": valid, "reasons": reasons}


def worst_rank_timing(ranks: list[dict], key: str) -> dict:
    """Use the slowest rank on each repetition, including its device completion."""
    import statistics

    result = {}
    present = [rank for rank in ranks if key in rank]
    for prefix in ("host", "cuda_event"):
        series = [rank[key][prefix + "_ms"] for rank in present]
        if not series or any(len(s) != len(series[0]) for s in series) or not series[0]:
            result[prefix + "_ms"], result[prefix + "_p50_ms"], result[prefix + "_p95_ms"] = [], None, None
            continue
        values = [max(row) for row in zip(*series, strict=True)]
        ordered = sorted(values)
        at = (len(values) - 1) * 0.95
        lo = int(at)
        result[prefix + "_ms"] = values
        result[prefix + "_p50_ms"] = statistics.median(values)
        result[prefix + "_p95_ms"] = ordered[lo] + (ordered[min(lo + 1, len(values) - 1)] - ordered[lo]) * (at - lo)
    reps = {r[key].get("reps") for r in ranks if key in r}
    result["reps"] = reps.pop() if len(reps) == 1 else None          # None: ranks disagree (invalid by L22 anyway)
    return result


def evaluate_distributed(case: Case, submission: Path, *, device: str, trials: int, warmup: int, reps: int,
                         reference_reps: int | None = None) -> dict:
    selected = torch.device(device)
    first_device = selected.index or 0
    topo = topology_from_env(case.gpus)
    if selected.type == "cuda" and torch.cuda.device_count() < first_device + topo.local_world:
        return {"status": "unavailable", "reason": f"requires {topo.local_world} visible GPUs starting at {device}"}
    deadlines = for_case(case, multi_node=topo.multi_node)
    if reference_reps is None:
        reference_reps = 0 if case.params.get("checkpoint") else 3
    options = {"reference_reps": reference_reps}
    with tempfile.TemporaryDirectory(prefix="megabench-ranks-") as directory:
        mp.spawn(_rank_worker, nprocs=topo.local_world, join=True,
                 args=(case.to_dict(), str(submission), selected.type, first_device, trials, warmup, reps,
                       topo.to_dict(), directory, deadlines.to_dict(), options))
        local = [json.loads((Path(directory) / f"rank-{topo.global_rank(r)}.json").read_text()) for r in topo.local_ranks]
        gathered_path = Path(directory) / "all-ranks.json"
        gathered = json.loads(gathered_path.read_text()) if gathered_path.exists() else None
        stderr_files = {f"rank{topo.global_rank(r)}": (Path(directory) / f"candidate-rank{topo.global_rank(r)}.stderr")
                        for r in topo.local_ranks}
        stderr_tails = {k: v.read_bytes()[-4000:].decode("utf-8", errors="replace") for k, v in stderr_files.items() if v.exists()}
    topo_record = topo.describe()
    if topo.multi_node and topo.node_rank != 0:
        return {"status": "node_done", "topology": topo_record, "local_ranks": local,
                "note": "records of this tray were gathered on global rank 0; this node writes no grade"}
    ranks = gathered if gathered and len(gathered) == case.gpus and all(gathered) else None
    if ranks is None:
        ranks = sorted(local, key=lambda r: r["rank"])
        missing = sorted(set(range(case.gpus)) - {r["rank"] for r in ranks})
        for rank in missing:
            ranks.append({"rank": rank, "status": "infrastructure",
                          "failure": {"status": "infrastructure", "message": "no record from this rank (process or gather failed)"}})
        ranks.sort(key=lambda r: r["rank"])
    group = merge_group(ranks)
    invalid = [r["rank"] for r in ranks if not r.get("measurement", {}).get("valid")]
    result = {"ranks": ranks, "topology": topo_record, "deadlines_s": deadlines.to_dict(),
              "candidate_stderr_tail": stderr_tails,
              "measurement": {"valid": not invalid, "ranks_invalid": invalid,
                              "policy": "every rank must complete all repetitions with correct, unmutated outputs; "
                                        "otherwise no candidate_timing is reported (L22)"}}
    if group["status"] != "pass":
        return result | {"status": group["status"], "reason": group.get("message", ""), "group_failure": group}
    if invalid:
        return result | {"status": "evaluator_error", "reason": f"ranks {invalid} passed without a valid measurement",
                         "group_failure": {"status": "evaluator_error", "ranks_with_status": invalid}}
    result |= {"status": "correctness_only" if selected.type == "cpu" else "ok_provisional",
               "build_ms": max(r["build_ms"] for r in ranks),
               "correctness": {"status": "pass", "per_rank": [r["correctness"] for r in ranks]},
               "first_call_ms_including_jit": max(r.get("first_call_ms_including_jit", 0) for r in ranks),
               "candidate_timing": worst_rank_timing(ranks, "candidate_timing"),
               "launch_audit_advisory": {"per_rank": [r.get("launch_audit_advisory") for r in ranks],
                                         "note": "advisory in-process counts; the audit of record is the nsys trace"}}
    if any("reference_timing" in r for r in ranks):
        result["reference_timing"] = worst_rank_timing(ranks, "reference_timing")
        if result["reference_timing"]["host_p50_ms"] and result["candidate_timing"]["host_p50_ms"]:
            result["speedup_vs_eager_host"] = result["reference_timing"]["host_p50_ms"] / result["candidate_timing"]["host_p50_ms"]
        if selected.type == "cuda" and result["reference_timing"]["cuda_event_p50_ms"]:
            result["speedup_vs_eager_cuda_event"] = (result["reference_timing"]["cuda_event_p50_ms"] /
                                                     result["candidate_timing"]["cuda_event_p50_ms"])
    return result
