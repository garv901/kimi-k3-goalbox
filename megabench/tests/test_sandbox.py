"""CPU tests of the sandbox pieces that do not need Kubernetes or GPUs: trajectory capture and
recovery, the records verifier, the transcript compiler fallback, the k8s broker/executor spool
protocol, the manifest checker, and the conversion sizing."""

from __future__ import annotations

import importlib.util
import json
import shutil
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load(REPO / "sandbox/lib/capture.py", "capture")
verify_records = load(REPO / "sandbox/lib/verify_records.py", "verify_records")
compiler = load(REPO / "tools/compile_transcript.py", "compile_transcript")


class CaptureTests(unittest.TestCase):
    def test_append_only_copy_integrity_and_rewrite_detection(self):
        with tempfile.TemporaryDirectory() as d:
            home, records = Path(d) / "home", Path(d) / "records"
            proj = home / ".claude/projects/-work/sess"
            proj.mkdir(parents=True)
            main = proj / "main.jsonl"
            main.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
            cap = capture.Capture(home / ".claude/projects", records)
            self.assertEqual(cap.pass_once()["copied"], 1)
            main.write_text(main.read_text() + '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}\n')
            (proj / "subagents").mkdir()
            (proj / "subagents/agent-1.jsonl").write_text('{"type":"assistant","message":{"role":"assistant","content":[]}}\n')
            summary = cap.pass_once()
            self.assertEqual((summary["copied"], summary["unchanged"]), (2, 0))
            copy = records / "transcripts/-work/sess/main.jsonl"
            # a second source (the stream-json emptyDir) goes under its own name
            stream_dir = Path(d) / "stream"; stream_dir.mkdir()
            (stream_dir / "agent-1.jsonl").write_text('{"type":"system"}\n{"type":"result"}\n')
            stream_cap = capture.Capture(stream_dir, records, "stream")
            self.assertEqual(stream_cap.pass_once()["copied"], 1)
            self.assertTrue((records / "stream/agent-1.jsonl").exists())
            self.assertEqual(copy.read_bytes(), main.read_bytes())
            # A rewrite of already-copied bytes is kept as a separate file, never lost.
            main.write_text("X" + main.read_text()[1:])
            self.assertEqual(cap.pass_once()["rewritten"], 1)
            self.assertTrue(list(copy.parent.glob("main.jsonl.rewritten-*")))
            self.assertEqual(copy.read_bytes(), main.read_bytes())
            final = cap.final()
            self.assertEqual(final["mismatched"], [])
            self.assertEqual(stream_cap.final()["mismatched"], [])
            lines = [json.loads(l) for l in (records / "transcripts/MANIFEST.jsonl").read_text().splitlines()]
            self.assertTrue(all(l["copy"].startswith("transcripts/") for l in lines))
            self.assertTrue(all(len(l["sha256"]) == 64 for l in lines if "sha256" in l))
            self.assertEqual(sum(1 for l in lines if l.get("event") == "rewritten"), 1)
            result = verify_records.check(records, home, {"stream": stream_dir})
            self.assertTrue(result["ok"], result["problems"])
            self.assertEqual(sorted(result["sources"]), ["stream", "transcripts"])
            # Tampering with a copy is detected.
            copy.write_bytes(copy.read_bytes()[:-1] + b"?")
            self.assertFalse(verify_records.check(records)["ok"])

    def test_recovery_after_restart_continues_from_state(self):
        with tempfile.TemporaryDirectory() as d:
            home, records = Path(d) / "home", Path(d) / "records"
            proj = home / ".claude/projects/p"
            proj.mkdir(parents=True)
            src = proj / "s.jsonl"
            src.write_text("a" * 1000 + "\n")
            capture.Capture(home / ".claude/projects", records).pass_once()
            src.write_text(src.read_text() + "b" * 500 + "\n")
            again = capture.Capture(home / ".claude/projects", records)       # fresh process, state from STATE.json
            self.assertEqual(again.pass_once()["copied"], 1)
            self.assertEqual((records / "transcripts/p/s.jsonl").read_bytes(), src.read_bytes())


class CompilerTests(unittest.TestCase):
    def test_compiles_and_redacts(self):
        events = [
            {"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": ["Bash"], "permissionMode": "auto", "version": "2.1.322"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Connecting to pod4-gb300-10-tray14 at 10.13.84.11 with sk-ant-oat01-ABCDEFGHIJKLMNOP"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls /mnt/shared/home/kevin"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}},
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x\n" * 10}]}},
            {"type": "assistant", "parent_tool_use_id": "t1", "message": {"role": "assistant", "content": [{"type": "text", "text": "sub"}]}},
            {"type": "result", "subtype": "success", "num_turns": 2, "total_cost_usd": 1.5},
        ]
        redact = compiler.Redactor()
        compiled = compiler.compile_events(events, redact)
        text = json.dumps(compiled)
        for secret in ("sk-ant", "10.13.84.11", "pod4-gb300", "/home/kevin"):
            self.assertNotIn(secret, text)
        self.assertEqual(compiled["tool_calls"], {"Bash": 1})
        self.assertEqual(compiled["usage"]["input_tokens"], 10)
        self.assertEqual(compiled["result"]["total_cost_usd"], 1.5)
        self.assertTrue(any(t.get("parent_tool_use_id") == "t1" for t in compiled["turns"]))
        md = compiler.to_markdown("t", compiled, redact.counts)
        self.assertIn("<host>", md)
        self.assertGreaterEqual(sum(redact.counts.values()), 4)


class SpoolProtocolTests(unittest.TestCase):
    """broker.py (k8s mode) and executor.py exchange jobs through files only; run both for real on the CPU."""

    def test_two_node_dispatch_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            (run / "home/.gpu/req").mkdir(parents=True)
            ctl = root / "ctl"
            env = dict(os.environ, GOALBOX_GPU_MODE="k8s", GOALBOX_CTL=str(ctl), GOALBOX_GPU_NODES="node-a,node-b",
                       GOALBOX_MAX_GPUS="8", GOALBOX_MASTER_PORT="29500")
            broker = subprocess.Popen([PY, str(REPO / "sandbox/lib/broker.py"), str(run)], env=env)
            execs = []
            for n in ("node-a", "node-b"):            # grader + spawner per node, sharing one "emptyDir"
                job_dir = root / f"job-{n}"; job_dir.mkdir()
                execs.append(subprocess.Popen([PY, str(REPO / "sandbox/lib/executor.py")],
                                              env=dict(os.environ, GOALBOX_CTL_REQ=str(ctl / "req"), GOALBOX_CTL_RESULTS=str(ctl / "results"),
                                                       GOALBOX_NODE_NAME=n, GOALBOX_GPUS_HERE="4", GOALBOX_JOB_DIR=str(job_dir),
                                                       GOALBOX_GRADER_TMP=str(root / f"tmp-{n}"), GOALBOX_BENCH=str(REPO), PYTHONPATH=str(REPO),
                                                       GOALBOX_LEFTOVER_CHECK="0")))
                execs.append(subprocess.Popen([PY, str(REPO / "sandbox/lib/spawner.py")],
                                              env=dict(os.environ, GOALBOX_JOB_DIR=str(job_dir), PYTHONPATH=str(REPO))))
            # An unrelated process of the same uid, in its own session, must survive the job cleanup (C1 test 7).
            bystander = subprocess.Popen(["sleep", "600"], start_new_session=True)
            try:
                work = root / "work"; work.mkdir()
                rid = "20261118-000000-1"
                (run / "home/.gpu/req" / f"{rid}.sh").write_text(
                    "echo node_rank=$GOALBOX_NODE_RANK nnodes=$GOALBOX_NNODES master=$GOALBOX_MASTER_ADDR:$GOALBOX_MASTER_PORT "
                    "gpus=$CUDA_VISIBLE_DEVICES job=$GOALBOX_JOB_ID\nset -m; sleep 300 &\nexit 3\n")   # set -m: the stray gets its own process group
                (run / "home/.gpu/req" / f"{rid}.json").write_text(json.dumps({"minutes": 1, "gpus": 8, "cwd": "/work"}))
                status = run / "home/.gpu/status" / rid
                for _ in range(300):
                    if status.exists() and " done " in status.read_text():
                        break
                    time.sleep(0.2)
                else:
                    self.fail("job did not finish: " + (status.read_text() if status.exists() else "no status"))
                text = status.read_text()
                self.assertIn("done 3", text)
                out = (run / "home/.gpu/out" / f"{rid}.log").read_text()
                self.assertIn("node_rank=0 nnodes=2 master=node-a:29500 gpus=0,1,2,3", out)
                self.assertIn("node_rank=1 nnodes=2 master=node-a:29500 gpus=0,1,2,3", out)
                self.assertIn("killed 1 leftover", out)
                self.assertIn("candidate-authored output", out)                     # exec output is labelled
                self.assertIn("candidate-reported", text)                             # and so is its exit code
                self.assertIsNone(bystander.poll(), "the cleanup killed an unrelated same-uid process")
                # a 6-GPU request is rejected; a 2-GPU request goes to node-a only
                for rid2, gpus, expect in (("20261118-000000-2", 6, "rejected"), ("20261118-000000-3", 2, "done 0")):
                    (run / "home/.gpu/req" / f"{rid2}.sh").write_text("echo single=$GOALBOX_NNODES gpus=$CUDA_VISIBLE_DEVICES\n")
                    (run / "home/.gpu/req" / f"{rid2}.json").write_text(json.dumps({"minutes": 1, "gpus": gpus, "cwd": "/work"}))
                    st = run / "home/.gpu/status" / rid2
                    for _ in range(300):
                        if st.exists() and " done " in st.read_text():
                            break
                        time.sleep(0.2)
                    self.assertIn(expect, st.read_text())
                self.assertIn("single= gpus=0,1", (run / "home/.gpu/out/20261118-000000-3.log").read_text())
                self.assertFalse((ctl / "req/gpu/node-b/req/20261118-000000-3.json").exists())
                self.assertTrue((ctl / "results/gpu/node-a/done/20261118-000000-3.json").exists())
            finally:
                for p in [broker, bystander] + execs:
                    p.kill(); p.wait()


class ManifestTests(unittest.TestCase):
    def test_static_checks_pass(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        checker = load(REPO / "sandbox/k8s/check_manifests.py", "check_manifests")
        problems = checker.check(REPO / "sandbox/k8s/manifests")
        self.assertEqual(problems, [])

    def test_checker_catches_regressions(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        checker = load(REPO / "sandbox/k8s/check_manifests.py", "check_manifests")
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d)
            for f in (REPO / "sandbox/k8s/manifests").glob("*.yaml"):
                (dst / f.name).write_text(f.read_text())
            agent = dst / "70-agent.yaml"
            doc = list(yaml.safe_load_all(agent.read_text()))[0]
            doc["spec"]["containers"][1]["volumeMounts"][0]["readOnly"] = False      # capture gets rw home
            doc["spec"]["containers"][0]["securityContext"]["capabilities"]["add"] = ["SYS_ADMIN"]
            agent.write_text(yaml.safe_dump(doc))
            problems = checker.check(dst)
            self.assertTrue(any("read-write" in p for p in problems), problems)
            self.assertTrue(any("SYS_ADMIN" in p for p in problems), problems)
            rbac = dst / "10-rbac.yaml"
            rbac.write_text(rbac.read_text() + "\n---\napiVersion: rbac.authorization.k8s.io/v1\nkind: RoleBinding\n"
                            "metadata: {name: leak, namespace: goalbox-k3}\nroleRef: {apiGroup: rbac.authorization.k8s.io, kind: Role, name: goalbox-controller}\n"
                            "subjects: [{kind: ServiceAccount, name: wideep-session, namespace: inference-wideep}]\n")
            self.assertTrue(any("outside the namespace" in p for p in checker.check(dst)), "foreign subject not caught")


class SizingTests(unittest.TestCase):
    def test_tp8_totals(self):
        sys.path.insert(0, str(REPO))
        from tools.conversion_sizing import sizing
        report = sizing(8, 93, 4)
        self.assertGreater(report["output_bytes_total"], 1.5e12)
        self.assertLess(report["output_bytes_total"], 1.8e12)
        self.assertAlmostEqual(report["output_bytes_total"], 8 * report["output_bytes_per_rank"])
        self.assertLess(report["peak_host_rss_bytes_per_worker_estimate"], 100e9)
        small = sizing(4, 4, 4)
        self.assertLess(small["output_bytes_total"], report["output_bytes_total"])


if __name__ == "__main__":
    unittest.main()


budget = load(REPO / "sandbox/lib/budget.py", "budget")


class BudgetLedgerTests(unittest.TestCase):
    """C1 test 9: the 48 h are run time; interruptions and controller gaps do not count; events fire once."""

    def test_interruptions_and_gaps_are_not_counted_and_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"
            ledger = budget.Ledger(path, budget_s=3600, warn_before_s=600, max_gap_s=100)
            t = 1000.0
            self.assertIsNone(ledger.observe(t, running=False, reason="not started"))   # nothing counted before the first run
            self.assertEqual(ledger.run_s, 0)
            t += 30
            self.assertIsNone(ledger.observe(t, running=True))
            self.assertEqual(ledger.run_s, 0)                                           # previous tick was not running
            for _ in range(40):                                                         # 40 x 30 s running = 1200 s
                t += 30
                ledger.observe(t, running=True)
            self.assertAlmostEqual(ledger.run_s, 1200)
            # infrastructure interruption: 10 min not running
            t += 30; ledger.observe(t, running=False, reason="agent=Pending workers_ready=1")
            self.assertAlmostEqual(ledger.run_s, 1230)                                  # the last running interval counted
            for _ in range(20):
                t += 30; ledger.observe(t, running=False, reason="agent=Pending workers_ready=1")
            t += 30; ledger.observe(t, running=True)
            self.assertAlmostEqual(ledger.run_s, 1230)
            self.assertEqual(len(ledger.state["interruptions"]), 1)
            self.assertAlmostEqual(ledger.state["interruptions"][0]["s"], 630)
            # controller restart with a long gap: not counted, recorded
            again = budget.Ledger(path, budget_s=3600, warn_before_s=600, max_gap_s=100)
            self.assertAlmostEqual(again.run_s, 1230)
            t += 500; again.observe(t, running=True)
            self.assertAlmostEqual(again.run_s, 1230)
            self.assertEqual(len(again.state["controller_gaps"]), 1)
            # warn once at budget - 600, stop once at budget
            events = []
            while again.run_s < 3600 + 60:
                t += 30
                events.append(again.observe(t, running=True))
            self.assertEqual([e for e in events if e], ["warn", "stop"])
            warn_at = next(i for i, e in enumerate(events) if e == "warn")
            stop_at = next(i for i, e in enumerate(events) if e == "stop")
            self.assertAlmostEqual((stop_at - warn_at) * 30, 600, delta=30)
            summary = again.summary()
            self.assertTrue(summary["stopped"] and summary["warned"])
            self.assertGreaterEqual(summary["interrupted_s"], 600)


class GradeProtocolTests(unittest.TestCase):
    """A grade request end to end on the CPU: broker -> grader (executor.py) -> spawner -> remote candidate
    process -> trusted verdict. Then a hostile and a killer candidate (B-C1-1 regressions, CPU part)."""

    def _wait_done(self, status_path: Path, timeout_s: float = 240) -> str:
        for _ in range(int(timeout_s / 0.25)):
            if status_path.exists() and " done " in status_path.read_text():
                return status_path.read_text()
            time.sleep(0.25)
        self.fail("job did not finish: " + (status_path.read_text() if status_path.exists() else "no status"))

    def test_grade_verdicts_are_the_graders_not_the_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"; (run / "home/.gpu/req").mkdir(parents=True)
            ctl = root / "ctl"; work = root / "work"; work.mkdir(); job_dir = root / "job"; job_dir.mkdir()
            candidates = REPO / "megabench/tests/candidates"
            for name in ("oracle_candidate.py", "hostile_candidate.py", "killer_candidate.py"):
                shutil.copy(candidates / name, work / name)
            grade_args = "--trials 1 --warmup 0 --reps 1 --timeout 300 --no-scan --tiny --reference-reps 0"
            broker = subprocess.Popen([PY, str(REPO / "sandbox/lib/broker.py"), str(run)],
                                      env=dict(os.environ, GOALBOX_GPU_MODE="k8s", GOALBOX_CTL=str(ctl), GOALBOX_GPU_NODES="node-a",
                                               GOALBOX_GRADE_ARGS=grade_args, GOALBOX_GRADE_DEVICE="cpu", GOALBOX_WORK_ROOT=str(work),
                                               GOALBOX_BENCH=str(REPO), CUDA_VISIBLE_DEVICES=""))
            grader_env = dict(os.environ, GOALBOX_CTL_REQ=str(ctl / "req"), GOALBOX_CTL_RESULTS=str(ctl / "results"),
                              GOALBOX_NODE_NAME="node-a", GOALBOX_GPUS_HERE="1", GOALBOX_JOB_DIR=str(job_dir),
                              GOALBOX_GRADER_TMP=str(root / "gtmp"), GOALBOX_BENCH=str(REPO), PYTHONPATH=str(REPO),
                              GOALBOX_RUNNER_STALE_S="6", CUDA_VISIBLE_DEVICES="", GOALBOX_LEFTOVER_CHECK="0")
            spawner_env = dict(os.environ, GOALBOX_JOB_DIR=str(job_dir), PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="",
                               HOSTILE_RESULTS=str(ctl / "results/gpu/node-a"))
            grader = subprocess.Popen([PY, str(REPO / "sandbox/lib/executor.py")], env=grader_env)
            spawner = subprocess.Popen([PY, str(REPO / "sandbox/lib/spawner.py")], env=spawner_env)
            try:
                def request(rid, submission, kind="grade", case="kimi-k3-dev-l4-tp1-b8-s8192"):
                    (run / "home/.gpu/req" / f"{rid}.sh").write_text("# grade\n")
                    (run / "home/.gpu/req" / f"{rid}.json").write_text(json.dumps(
                        {"kind": kind, "case": case, "submission": str(work / submission), "minutes": 3, "gpus": 0, "cwd": str(work)}))
                    return self._wait_done(run / "home/.gpu/status" / rid)

                # 1. oracle: a trusted pass
                text = request("20261119-000001-1", "oracle_candidate.py")
                self.assertIn("done 0 correctness_only", text)
                self.assertNotIn("candidate-reported", text)
                done = json.loads((ctl / "results/gpu/node-a/done/20261119-000001-1.json").read_text())
                self.assertTrue(done["trusted"]); self.assertEqual(done["kind"], "grade")
                verdict = json.loads((ctl / "results/gpu/node-a/verdict/20261119-000001-1.json").read_text())
                self.assertEqual(verdict["record"]["status"], "correctness_only")
                self.assertTrue(verdict["record"].get("tiny_self_test"))
                self.assertTrue(verdict["record"]["measurement"]["valid"])                   # L22 on a clean run
                self.assertEqual(verdict["record"]["candidate_timing"]["reps"], 1)                 # --reps 1 in grade_args
                self.assertEqual(verdict["record"]["ranks"][0]["export_audit"][0]["checked"], False)   # CPU: nothing exported
                self.assertEqual(verdict["harness_args"], grade_args.split())           # the broker's options, not the agent's
                self.assertEqual(verdict["graded_snapshot"]["files"], 3)              # the submission's directory, as scanned
                out = (run / "home/.gpu/out/20261119-000001-1.log").read_text()
                self.assertIn("trusted grader verdict", out); self.assertIn("verdict: correctness_only", out)

                # 2. hostile: writes everywhere it can; the verdict is still computed from its outputs, and its
                #    attempts are preserved as candidate-authored evidence
                text = request("20261119-000001-2", "hostile_candidate.py")
                self.assertIn("done 0 correctness_only", text)
                stderr_copies = list((ctl / "results/gpu/node-a/candidate/20261119-000001-2").glob("*.stderr"))
                self.assertEqual(len(stderr_copies), 1)
                attacks = stderr_copies[0].read_text()
                self.assertIn("ATTACK forge_done_record", attacks)
                # the done record the broker read is the grader's (its finished_utc is real, its state is the status)
                done = json.loads((ctl / "results/gpu/node-a/done/20261119-000001-2.json").read_text())
                self.assertEqual(done["state"], "correctness_only"); self.assertLess(done["finished_utc"], time.time() + 60)
                # same-uid host: the forged other-job.json may exist; the broker never dispatched such a job, so it is inert
                self.assertFalse((run / "home/.gpu/status/other-job").exists())

                # 3. an agent-supplied harness option or a submission outside /work is rejected by the broker
                (run / "home/.gpu/req/20261119-000001-3.json").write_text(json.dumps(
                    {"kind": "grade", "case": "kimi-k3-dev-l4-tp1-b8-s8192", "submission": "/etc/passwd", "minutes": 3, "gpus": 0, "cwd": str(work)}))
                self.assertIn("rejected", self._wait_done(run / "home/.gpu/status/20261119-000001-3"))
                (run / "home/.gpu/req/20261119-000001-4.json").write_text(json.dumps(
                    {"kind": "grade", "case": "no-such-case", "submission": str(work / "oracle_candidate.py"), "minutes": 3, "gpus": 0, "cwd": str(work)}))
                self.assertIn("rejected", self._wait_done(run / "home/.gpu/status/20261119-000001-4"))

                # 4. killer: kills what it can see (its spawner); never a pass; the next job is an infrastructure
                #    failure with a runner restart requested; a restarted spawner is picked up again
                text = request("20261119-000001-5", "killer_candidate.py")
                self.assertNotIn("done 0", text)                                        # never a pass
                self.assertRegex(text, r" done [1-9][0-9]* ")
                self.assertIsNotNone(spawner.poll(), "the killer did not reach its spawner (test setup)")
                text = request("20261119-000001-6", "oracle_candidate.py")
                self.assertIn("INFRA_RUNNER_DOWN", text)
                self.assertTrue((job_dir / "RESTART-RUNNER").exists())
                spawner = subprocess.Popen([PY, str(REPO / "sandbox/lib/spawner.py")], env=spawner_env)
                for _ in range(100):
                    if not (job_dir / "RESTART-RUNNER").exists():
                        break
                    time.sleep(0.2)
                self.assertFalse((job_dir / "RESTART-RUNNER").exists(), "restart flag not cleared after the runner came back")
                text = request("20261119-000001-7", "oracle_candidate.py")
                self.assertIn("done 0 correctness_only", text)
            finally:
                for p in (broker, grader, spawner):
                    p.kill(); p.wait()


class TransportTests(unittest.TestCase):
    def test_channel_and_cpu_codec(self):
        import torch
        transport = load(REPO / "megabench/harness/transport.py", "transport")
        with tempfile.TemporaryDirectory() as d:
            spawn = transport.make_spawn_dir(Path(d))
            self.assertEqual(oct((spawn / "to_child").stat().st_mode & 0o777), "0o644")
            self.assertEqual(oct((spawn / "from_child").stat().st_mode & 0o777), "0o622")
            with self.assertRaises(TimeoutError):
                transport.connect_parent(spawn, 0.3)                      # nobody connects
            import threading
            got = {}
            def child():
                c = transport.connect_child(spawn)
                msg = transport.decode(c.recv())
                got["msg"] = msg
                keep = transport.KeepAlive()
                c.send(transport.encode(("done", {"y": msg[1]["x"] * 2}), keep))
                c.close()
            t = threading.Thread(target=child); t.start()
            parent = transport.connect_parent(spawn, 5)
            keep = transport.KeepAlive()
            x = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            parent.send(transport.encode(("run", {"x": x, "n": 3, "t": (1, [2, torch.tensor([1.5])])}), keep))
            self.assertTrue(parent.poll(5))
            reply = transport.decode(parent.recv(5))
            t.join(5)
            self.assertEqual(got["msg"][0], "run"); self.assertEqual(got["msg"][1]["n"], 3)
            self.assertTrue(torch.equal(got["msg"][1]["x"], x)); self.assertEqual(got["msg"][1]["x"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(reply[1]["y"], x * 2))
            parent.close()

    def test_untrusted_frames_are_validated(self):
        """Messages from the runner side are untrusted: no globals in pickles, no oversized frames, no CUDA
        tensor description that could map memory outside its allocation or alias a dtype (L21)."""
        import pickle, torch
        transport = load(REPO / "megabench/harness/transport.py", "transport")
        with self.assertRaises(pickle.UnpicklingError):
            transport.safe_loads(pickle.dumps(os.system))                   # a global: code execution vector
        with self.assertRaises(pickle.UnpicklingError):
            transport.safe_loads(pickle.dumps(torch.tensor([1])))           # torch reductions are globals too
        self.assertEqual(transport.safe_loads(pickle.dumps(("run", {"a": (1, 2.5, b"x", None, True)}))),
                         ("run", {"a": (1, 2.5, b"x", None, True)}))
        good = {"handle": b"\0" * 64, "device": 0, "offset": 0, "nbytes": 12, "dtype": "float32", "shape": (3,)}
        self.assertEqual(transport._validated(good), good)
        for bad in ({**good, "nbytes": 16}, {**good, "dtype": "complex64"}, {**good, "shape": (-1,)},
                    {**good, "handle": b"\0" * 63}, {**good, "offset": -8}, {**good, "shape": "3"}):
            with self.assertRaises(transport.TransportError, msg=str(bad)):
                transport._validated(bad)
        with self.assertRaises(transport.TransportError):
            transport.decode(("__tensor_cpu__", b"anything"), allow_cpu_tensors=False)      # the GPU grader refuses by-value tensors
        # no export audit on the CPU, but the pool/audit API exists and reports so
        self.assertIsNone(transport.input_pool()) if not torch.cuda.is_available() else None
        self.assertEqual(transport.audit_exports({"x": torch.zeros(4)}, None, "cpu")["checked"], False)

    def test_measurement_validity_and_contamination_rules(self):
        """L22: timing is reported only from complete, correct, unmutated runs; GPU counter changes or foreign
        processes during a grade invalidate it on the grader's side."""
        import importlib
        distributed = importlib.import_module("megabench.harness.distributed")
        ok = {"status": "pass", "candidate_timing": {"reps": 10}, "timed_outputs_grade": {}, "launch_audit_advisory": {}}
        self.assertEqual(distributed.measurement_validity(dict(ok), 10), {"valid": True, "reasons": []})
        short = dict(ok, candidate_timing={"reps": 7})
        m = distributed.measurement_validity(short, 10)
        self.assertFalse(m["valid"]); self.assertIn("7 of 10 repetitions", m["reasons"])
        self.assertNotIn("candidate_timing", short); self.assertEqual(short["invalid_timing"], {"reps": 7})
        failed = dict(ok, status="candidate_error", failure={"phase": "timing"})
        m = distributed.measurement_validity(failed, 10)
        self.assertFalse(m["valid"]); self.assertNotIn("candidate_timing", failed)
        self.assertFalse(distributed.measurement_validity({"status": "pass"}, 10)["valid"])
        os.environ["GOALBOX_JOB_DIR"] = tempfile.mkdtemp()
        executor = load(REPO / "sandbox/lib/executor.py", "executor_rules")
        before = {"0": {"ecc.errors.uncorrected.volatile.total": "0", "retired_pages.pending": "No", "clocks.sm": "1800"}}
        self.assertEqual(executor.contamination(before, {"0": dict(before["0"], **{"clocks.sm": "900"})}, [2, 2], 2), [])
        self.assertEqual(len(executor.contamination(before, {"0": dict(before["0"], **{"ecc.errors.uncorrected.volatile.total": "1"})}, [], 2)), 1)
        self.assertEqual(len(executor.contamination(before, {}, [], 2)), 1)
        self.assertIn("3 compute processes", executor.contamination(before, before, [1, 3, 2], 2)[0])
        self.assertEqual(executor.contamination(before, before, [1, 3, 2], 0), [])        # unknown expectation: no claim
