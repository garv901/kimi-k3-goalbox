"""CPU tests of the hardened evaluator: typed statuses, keyed-hash immutability (collision regression),
source/artifact scan, nsys trace audit, two-tray topology, and gloo end-to-end controls on tiny geometry."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from ..cases import select_cases
from ..harness import correctness, launch, nsys_audit, scan, status
from ..harness.benchmark import classify_events
from ..harness.runner import evaluate_case

CANDIDATES = Path(__file__).resolve().parent / "candidates"
TINY = {"layers": 5, "hidden": 64, "latent_hidden": 64, "intermediate": 64, "dense_intermediate": 128,
        "q_heads": 4, "head_dim": 16, "q_lora_rank": 32, "kv_lora_rank": 32, "qk_nope_dim": 8, "qk_rope_dim": 8,
        "v_head_dim": 16, "vocab": 128, "experts": 8, "topk": 2, "context": 4, "attn_res_block_size": 2}


def tiny(case_id: str, batch: int = 2):
    case = select_cases("all", [case_id])[0]
    return replace(case, params=case.params | TINY | {"batch": batch})


class StatusTests(unittest.TestCase):
    def test_exit_codes_cover_every_status_and_are_distinct_for_failures(self):
        self.assertEqual(set(status.EXIT_CODES), set(status.STATUSES))
        failures = [status.EXIT_CODES[s] for s in status.STATUSES if s not in ("ok_provisional", "correctness_only")]
        self.assertEqual(len(failures), len(set(failures)))
        self.assertNotIn(0, failures)

    def test_classify_by_phase(self):
        self.assertEqual(status.classify(RuntimeError("boom"), phase="candidate").status, "candidate_error")
        self.assertEqual(status.classify(AssertionError("logits: off"), phase="grade").status, "incorrect")
        self.assertEqual(status.classify(AssertionError("x"), phase="immutability").status, "input_mutation")
        self.assertEqual(status.classify(ValueError("bad geometry"), phase="inputs").status, "evaluator_error")
        self.assertEqual(status.classify(RuntimeError("NCCL error: unhandled system error"), phase="reference").status,
                         "infrastructure")
        self.assertEqual(status.classify(TimeoutError("slow"), phase="candidate").status, "timeout")
        self.assertEqual(status.classify(FileNotFoundError("no converted weights"), phase="inputs").status, "infrastructure")
        typed = status.Incorrect("k", phase="grade")
        self.assertIs(status.classify(typed, phase="other"), typed)
        self.assertEqual(status.classify(AssertionError("logits: off"), phase="grade").message, "logits: off")

    def test_merge_group_precedence_and_culprits(self):
        ranks = [{"rank": 0, "status": "aborted_by_group"},
                 {"rank": 1, "status": "candidate_error", "failure": {"message": "boom", "phase": "candidate"}},
                 {"rank": 2, "status": "incorrect", "failure": {"message": "logits", "phase": "grade"}},
                 {"rank": 3, "status": "aborted_by_group"}]
        merged = status.merge_group(ranks)
        self.assertEqual(merged["status"], "incorrect")
        self.assertEqual(merged["ranks_with_status"], [2])
        ranks[2]["status"] = "aborted_by_group"
        self.assertEqual(status.merge_group(ranks)["status"], "candidate_error")
        self.assertEqual(status.merge_group([{"rank": 0, "status": "pass"}])["status"], "pass")
        self.assertEqual(status.merge_group([{"rank": 0, "status": "pass"}, {"rank": 1, "status": "timeout",
                                             "failure": {"message": "t"}}])["status"], "timeout")


class KeyedHashTests(unittest.TestCase):
    def test_sum_preserving_pair_is_detected_on_the_hash_path(self):
        """Regression for ISSUES item 17: +d at word i and -d at word i+65521 passed the old linear checksums."""
        torch.manual_seed(0)
        t = torch.randn(1 << 26, dtype=torch.bfloat16)      # 128 MB: hash path
        inputs = {"x": t}
        key = correctness.new_key()
        before = correctness.snapshot(inputs, key)
        self.assertEqual(before["x"][0], "hash")
        self.assertEqual(correctness.changed_inputs(inputs, before, key), [])
        words = t.view(torch.uint8).view(torch.int32)
        for i, d in [(100, 1), (7_000_000, 12345), (0, -3)]:
            words[i] += d
            words[i + 65521] -= d
            self.assertEqual(correctness.changed_inputs(inputs, before, key), ["x"], f"pair {i},{d} not detected")
            words[i] -= d
            words[i + 65521] += d
        self.assertEqual(correctness.changed_inputs(inputs, before, key), [])
        words[12345] += 1
        self.assertEqual(correctness.changed_inputs(inputs, before, key), ["x"])
        words[12345] -= 1
        other = correctness.snapshot(inputs, correctness.new_key())
        self.assertFalse(torch.equal(other["x"][1], before["x"][1]), "different keys must give different hashes")

    def test_small_and_fp8_tensors_take_the_exact_path(self):
        small = torch.randn(1000, dtype=torch.bfloat16)
        fp8 = torch.randn(64, 32).to(torch.float8_e4m3fn)
        flag = torch.tensor([True, False])
        inputs = {"s": small, "f": fp8, "b": flag}
        before = correctness.snapshot(inputs)             # no key needed: nothing is large
        self.assertEqual({v[0] for k, v in before.items() if not k.startswith("_")}, {"bytes"})
        self.assertEqual(correctness.changed_inputs(inputs, before), [])
        fp8.view(torch.uint8).view(-1)[5] ^= 1
        self.assertEqual(correctness.changed_inputs(inputs, before), ["f"])
        with self.assertRaises(status.InputMutation):
            correctness.check_unchanged(inputs, before)
        with self.assertRaises(ValueError):
            correctness.snapshot({"big": torch.zeros(1 << 25, dtype=torch.int32)})


class ScanTests(unittest.TestCase):
    def write(self, root: Path, name: str, text: str):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write(root, "submission.py", "import torch\n\ndef build(case):\n    return lambda x: x\n")
            self.write(root, "kernel.cu", "__global__ void k(float* x) { x[0] = 1; }\n")
            self.write(root, "reports/old.py", "import megabench\n")   # ignored directory
            self.assertEqual(scan.scan_submission(root / "submission.py")["status"], "pass")
            cases = {
                "oracle_import": "from megabench.tasks import kimi\n",
                "file_tensors": "w = torch.load('/work/w.pt')\n",
                "cuda_graph": "g = torch.cuda.CUDAGraph()\n",
                "outside_reads": "open('/mnt/shared/secret')\n",
                "timer_macro": "FLAGS = '-O3 -DMEGATIMER'\n",
            }
            for rule, line in cases.items():
                with self.subTest(rule=rule):
                    self.write(root, "extra.py", "import torch\n" + line)
                    record = scan.scan_submission(root / "submission.py")
                    self.assertEqual(record["status"], "fail")
                    self.assertIn(rule, record["rules_hit"])
            (root / "extra.py").unlink()
            self.write(root, "kernel.cu", "__global__ void parent() { child<<<1, 1>>>(); }\n")
            self.assertIn("device_launch", scan.scan_submission(root / "submission.py")["rules_hit"])
            self.write(root, "kernel.cu", "// child<<<1,1>>>() in a comment only\n__global__ void k() {}\n")
            self.assertEqual(scan.scan_submission(root / "submission.py")["status"], "pass")
            self.write(root, "k.ptx", ".extern .func (.param .b32 r) cudaLaunchDevice\n")
            self.assertIn("device_launch", scan.scan_submission(root / "submission.py")["rules_hit"])
            self.write(root, "k.ptx", ".global .u64 megatimer_records[1024];\n")
            record = scan.scan_submission(root / "submission.py")
            self.assertIn("timer_macro", record["rules_hit"])
            self.assertEqual(len(record["artifacts"]), 1)
            self.assertEqual(len(record["artifacts"][0]["sha256"]), 64)


class EventClassificationTests(unittest.TestCase):
    def test_split(self):
        events = [("ncclDevKernel_AllReduce_Sum_f32_RING_LL", True), ("nccl:all_reduce", True),
                  ("Memcpy DtoD (Device -> Device)", True), ("Memset (Device)", True),
                  ("my_megakernel(float*)", True), ("cudaLaunchKernel", False)]
        split = classify_events(events, 1)
        self.assertEqual((split["compute_kernels"], split["nccl_kernels"], split["memcpy_events"], split["memset_events"],
                          split["collective_records"]), (1, 1, 1, 1, 1))
        self.assertEqual(split["status"], "launch_budget_failed")        # 2 device kernels against a budget of 1
        self.assertEqual(classify_events([("k", True), ("Memcpy HtoD", True)], 1)["status"], "within_budget")
        self.assertEqual(classify_events([("k", True), ("cudaGraphLaunch", False)], 1)["status"], "launch_budget_failed")
        self.assertEqual(classify_events([("Memcpy HtoD", True)], 1)["status"], "launch_budget_failed")   # zero kernels


class NsysAuditTests(unittest.TestCase):
    def make_db(self, path: Path, kernels, ranges, runtime=(), copies=(), base_ns=None, processes=(), stopped_ns=None):
        conn = sqlite3.connect(path)
        if base_ns is not None:
            conn.executescript("""
                create table TARGET_INFO_SESSION_START_TIME (utcEpochNs integer, utcTime text, localTime text, systemClockNs integer);
                create table PROCESSES (globalPid integer, pid integer, name text);
                create table DIAGNOSTIC_EVENT (timestamp integer, timestampType integer, source integer, severity integer, text text, globalPid integer);
            """)
            conn.execute("insert into TARGET_INFO_SESSION_START_TIME values (?, 'x', 'x', ?)", (1_795_000_000_000_000_000, base_ns))
            if stopped_ns is not None:
                conn.execute("insert into DIAGNOSTIC_EVENT values (?, 1, 3, 1, 'Profiling has stopped.', 0)", (stopped_ns,))
            conn.executemany("insert into PROCESSES values (?, ?, ?)", [((pid << 24) | 0x1000000000000, pid, name) for pid, name in processes])
        conn.executescript("""
            create table StringIds (id integer primary key, value text not null);
            create table NVTX_EVENTS (start integer not null, end integer, eventType integer not null, text text,
                                      globalTid integer, textId integer, domainId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, contextId integer,
                                                     streamId integer, correlationId integer, globalPid integer,
                                                     shortName integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, deviceId integer, bytes integer, copyKind integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, globalTid integer, correlationId integer, nameId integer);
        """)
        names = {}
        for name in [k[3] for k in kernels] + [r[2] for r in runtime]:
            names.setdefault(name, len(names) + 1)
        conn.executemany("insert into StringIds values (?, ?)", [(i, n) for n, i in names.items()])
        conn.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 1, 7, 0, ?, ?, ?)",
                         [(k[0], k[1], k[2], (k[4] if len(k) > 4 else 1), names[k[3]], names[k[3]]) for k in kernels])
        conn.executemany("insert into NVTX_EVENTS values (?, ?, 59, ?, 1, null, 0)", ranges)
        conn.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, 1, 0, ?)", [(s, e, names[n]) for s, e, n in runtime])
        conn.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, 1024, 8)", copies)
        conn.commit()
        conn.close()

    def test_counts_per_rep_and_device(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.sqlite"
            self.make_db(db,
                         kernels=[(100, 200, 0, "mega"), (110, 190, 1, "mega"),        # rep 0: one kernel per device
                                  (300, 350, 0, "mega"), (360, 400, 0, "mega2")],      # rep 1 on device 0: two kernels
                         ranges=[(90, 210, "megabench:rank0:rep0"), (95, 205, "megabench:rank1:rep0"),
                                 (290, 410, "megabench:rank0:rep1"), (1000, 1100, "megabench:rank1:rep1")],
                         copies=[(120, 130, 0)])
            report = nsys_audit.audit_trace(db, budget=1, device_of_rank={0: 0, 1: 1})
            r0, r1 = report["ranks"]["0"]["reps"], report["ranks"]["1"]["reps"]
            self.assertEqual([r["device_kernels"] for r in r0], [1, 2])
            self.assertEqual(r0[0]["memcpy_events"], 1)
            self.assertEqual(r0[0]["kernels_on_other_devices_in_window"], 1)
            self.assertAlmostEqual(r0[0]["gpu_span_ms"], 100 / 1e6)
            self.assertTrue(r0[0]["within_budget"] and not r0[1]["within_budget"])
            self.assertEqual([r["device_kernels"] for r in r1], [1, 0])
            self.assertEqual(report["status"], "launch_budget_failed")
            only_rank1_rep0 = nsys_audit.audit_trace(db, budget=1, device_of_rank={1: 1}, ranks=[1])
            self.assertEqual(only_rank1_rep0["status"], "launch_budget_failed")   # rep 1 of rank 1 had zero kernels

    def test_graph_launch_fails_and_clean_trace_passes(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.sqlite"
            self.make_db(db, kernels=[(100, 200, 0, "mega")], ranges=[(90, 210, "megabench:rank0:rep0")])
            self.assertEqual(nsys_audit.audit_trace(db, budget=1, device_of_rank={0: 0})["status"], "within_budget")
            db2 = Path(d) / "g.sqlite"
            self.make_db(db2, kernels=[(100, 200, 0, "mega")], ranges=[(90, 210, "megabench:rank0:rep0")],
                         runtime=[(95, 96, "cudaGraphLaunch_v10000")])
            report = nsys_audit.audit_trace(db2, budget=1, device_of_rank={0: 0})
            self.assertEqual(report["status"], "launch_budget_failed")
            self.assertEqual(report["ranks"]["0"]["reps"][0]["graph_launches"], 1)
            db3 = Path(d) / "n.sqlite"
            self.make_db(db3, kernels=[(100, 200, 0, "mega")], ranges=[(90, 210, "other:range")])
            self.assertEqual(nsys_audit.audit_trace(db3, budget=1, device_of_rank={0: 0})["status"], "unverified")


    def test_window_audit_is_invalid_unless_complete(self):
        """L24: the audit of record for remote candidates uses the evaluator's own clock windows. Complete and
        within budget -> within_budget; two kernels -> launch_budget_failed; a window without a kernel, a kernel
        from an unexpected process, no clock base, or a missing rank -> audit_invalid, never a pass."""
        from megabench.harness.nsys_audit import audit_windows
        base = 5_000_000_000_000
        w = lambda rank, rep, s, e: {"rep": rep, "start": {"mono_ns": base + s}, "end": {"mono_ns": base + e}}
        windows = {0: [w(0, 0, 10_000_000, 20_000_000), w(0, 1, 30_000_000, 40_000_000)]}
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "w.sqlite"
            self.make_db(db, kernels=[(12_000_000, 13_000_000, 0, "mega", 77), (33_000_000, 34_000_000, 0, "mega", 77)],
                         ranges=[], base_ns=base, processes=[(77, "python")])
            r = audit_windows(db, windows=windows, budget=1, device_of_rank={0: 0}, expected_pids={0: {77}})
            self.assertEqual(r["status"], "within_budget", r); self.assertEqual([x["device_kernels"] for x in r["ranks"]["0"]["reps"]], [1, 1])
            self.assertEqual(audit_windows(db, windows=windows, budget=1, device_of_rank={0: 0}, expected_pids={0: {78}})["status"], "audit_invalid")
            self.assertEqual(audit_windows(db, windows={0: windows[0], 1: []}, budget=1, device_of_rank={0: 0, 1: 1})["status"], "audit_invalid")
            db2 = Path(d) / "two.sqlite"
            self.make_db(db2, kernels=[(12_000_000, 13_000_000, 0, "a", 77), (14_000_000, 15_000_000, 0, "b", 77),
                                       (33_000_000, 34_000_000, 0, "a", 77)], ranges=[], base_ns=base)
            self.assertEqual(audit_windows(db2, windows=windows, budget=1, device_of_rank={0: 0})["status"], "launch_budget_failed")
            db3 = Path(d) / "gap.sqlite"
            self.make_db(db3, kernels=[(12_000_000, 13_000_000, 0, "a", 77)], ranges=[], base_ns=base)   # rep 1 has no kernel
            r = audit_windows(db3, windows=windows, budget=1, device_of_rank={0: 0})
            self.assertEqual(r["status"], "audit_invalid"); self.assertTrue(any("no kernel" in x for x in r["invalid_reasons"]))
            db4 = Path(d) / "nobase.sqlite"
            self.make_db(db4, kernels=[(12_000_000, 13_000_000, 0, "a", 77)], ranges=[])
            self.assertEqual(audit_windows(db4, windows=windows, budget=1, device_of_rank={0: 0})["status"], "audit_invalid")
            db5 = Path(d) / "truncated.sqlite"
            self.make_db(db5, kernels=[(12_000_000, 13_000_000, 0, "mega", 77), (33_000_000, 34_000_000, 0, "mega", 77)],
                         ranges=[], base_ns=base, stopped_ns=35_000_000)                      # profiler stopped inside rep 1
            r = audit_windows(db5, windows=windows, budget=1, device_of_rank={0: 0})
            self.assertEqual(r["status"], "audit_invalid"); self.assertTrue(any("truncated" in x for x in r["invalid_reasons"]))

class TopologyTests(unittest.TestCase):
    def test_single_host_and_two_trays(self):
        single = launch.topology_from_env(4, {})
        self.assertEqual((single.nnodes, single.local_world, single.multi_node), (1, 4, False))
        self.assertTrue(single.rendezvous("/tmp/x").startswith("file:///tmp/x/rendezvous0"))
        env = {"GOALBOX_NNODES": "2", "GOALBOX_NODE_RANK": "1", "GOALBOX_MASTER_ADDR": "goalbox-gpu-a",
               "GOALBOX_MASTER_PORT": "29500", "GOALBOX_JOB_ID": "j1"}
        two = launch.topology_from_env(8, env)
        self.assertEqual([two.global_rank(r) for r in two.local_ranks], [4, 5, 6, 7])
        self.assertEqual(two.rendezvous("/tmp/x", offset=1), "tcp://goalbox-gpu-a:29501")
        self.assertEqual(launch.topology_from_env(8, env | {"GOALBOX_NODE_RANK": "0"}).describe()["ranks_here"], [0, 1, 2, 3])
        for bad in ({"GOALBOX_NNODES": "3"}, {"GOALBOX_NNODES": "2"}, env | {"GOALBOX_MASTER_ADDR": "a b"},
                    env | {"GOALBOX_MASTER_PORT": "65535"}, env | {"GOALBOX_NODE_RANK": "2"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                launch.topology_from_env(8, bad)


@unittest.skipIf(os.environ.get("MEGABENCH_SKIP_E2E") == "1", "end-to-end controls skipped")
class GlooEndToEndTests(unittest.TestCase):
    """The evaluator on the CPU with gloo and tiny geometry: positive control and one control per new check."""

    def evaluate(self, candidate: str, case, **env):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""} | env):
            return evaluate_case(case, CANDIDATES / candidate, device="cpu", trials=2, warmup=1, reps=2, scan=False,
                                 reference_reps=1)

    def test_oracle_passes_tp1_and_tp2(self):
        for case_id in ("kimi-k3-dev-l4-tp1-b8-s8192", "kimi-k3-dev-l8-tp2-b8-s8192"):
            with self.subTest(case=case_id):
                record = self.evaluate("oracle_candidate.py", tiny(case_id))
                self.assertEqual(record["status"], "correctness_only", record.get("reason"))
                self.assertEqual(len(record["ranks"]), 1 if "tp1" in case_id else 2)
                self.assertEqual(len(record["candidate_timing"]["host_ms"]), 2)
                self.assertIn("timed_outputs_grade", record["ranks"][0])
                self.assertIn("launch_audit_advisory", record)
                self.assertEqual(record["topology"]["ranks_here"], list(range(len(record["ranks"]))))
                for key in ("submission_sha256", "hardware", "timestamp_utc", "deadlines_s"):
                    self.assertIn(key, record)

    def test_mutation_is_input_mutation_on_every_rank(self):
        record = self.evaluate("mutate_candidate.py", tiny("kimi-k3-dev-l8-tp2-b8-s8192"))
        self.assertEqual(record["status"], "input_mutation")
        self.assertEqual(record["group_failure"]["ranks_with_status"], [0, 1])
        self.assertIn("l3_kv_cache", record["reason"])
        self.assertIn("submission_sha256", record)

    def test_rank_local_structural_failure_names_the_rank(self):
        record = self.evaluate("ranklocal_shape_candidate.py", tiny("kimi-k3-dev-l8-tp2-b8-s8192"))
        self.assertEqual(record["status"], "incorrect")
        self.assertEqual(record["group_failure"]["ranks_with_status"], [1])
        self.assertEqual(record["group_failure"]["phase"], "structure")
        self.assertEqual(record["ranks"][0]["status"], "aborted_by_group")
        self.assertIn("logits", record["reason"])

    def test_candidate_exception_and_build_failure(self):
        record = self.evaluate("raise_candidate.py", tiny("kimi-k3-dev-l4-tp1-b8-s8192"))
        self.assertEqual(record["status"], "candidate_error")
        self.assertEqual(record["group_failure"]["phase"], "candidate")
        self.assertIn("deliberate failure inside the candidate", record["reason"])
        self.assertIn("deliberate failure", record["ranks"][0]["candidate_stderr_tail"])
        record = self.evaluate("build_fail_candidate.py", tiny("kimi-k3-dev-l4-tp1-b8-s8192"))
        self.assertEqual((record["status"], record["group_failure"]["phase"]), ("candidate_error", "build"))

    def test_wrong_keys_is_incorrect(self):
        record = self.evaluate("wrong_keys_candidate.py", tiny("kimi-k3-dev-l4-tp1-b8-s8192"))
        self.assertEqual(record["status"], "incorrect")
        self.assertIn("cache_length", record["reason"])

    def test_deadline_gives_timeout(self):
        record = self.evaluate("hang_candidate.py", tiny("kimi-k3-dev-l4-tp1-b8-s8192"), GOALBOX_DEADLINE_TRIAL_S="3")
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["deadlines_s"]["trial_s"], 3.0)

    def test_scan_rejects_an_oracle_import_before_running(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            record = evaluate_case(tiny("kimi-k3-dev-l4-tp1-b8-s8192"), CANDIDATES / "oracle_candidate.py", device="cpu",
                                   trials=1, warmup=0, reps=1)
        self.assertEqual(record["status"], "scan_failed")
        self.assertIn("oracle_import", record["scan"]["rules_hit"])


if __name__ == "__main__":
    unittest.main()
