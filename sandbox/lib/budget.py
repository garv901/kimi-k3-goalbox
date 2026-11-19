#!/usr/bin/env python3
"""Run-time budget accounting for the goal run (DECISIONS section 16; C1 test 9).

The 48 hours are *run time*: time during which the agent could work. Time during an infrastructure
interruption (agent pod not Running, a GPU worker not Ready, the broker down) does not count, and
the run resumes afterwards (section 16: resume only after an infrastructure outage). So the budget
cannot be a wall-clock deadline such as the pod's activeDeadlineSeconds alone; the controller keeps
this ledger, persisted on the records volume so a controller restart continues it.

    ledger = Ledger(path, budget_s=48*3600, warn_before_s=1800)
    every tick:  ledger.observe(now, running=<agent pod Running and workers Ready>)
                 -> one of None | "warn" | "stop" (each returned once)

Rules: the first observation starts the ledger (no time counted before the agent pod first runs);
between two observations the elapsed wall time counts only if the previous observation was `running`;
a gap longer than `max_gap_s` between observations (controller itself was down) is treated as an
interruption and not counted, and is recorded; every interruption interval is recorded with its
duration. The ledger file is rewritten atomically on every observation.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Ledger:
    def __init__(self, path: Path, *, budget_s: float = 48 * 3600, warn_before_s: float = 1800, max_gap_s: float = 300):
        self.path = Path(path)
        self.budget_s, self.warn_before_s, self.max_gap_s = budget_s, warn_before_s, max_gap_s
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"run_s": 0.0, "last_t": None, "last_running": False, "interruptions": [], "open_interruption": None,
                          "warned": False, "stopped": False, "started_t": None, "controller_gaps": []}

    @property
    def run_s(self) -> float:
        return self.state["run_s"]

    @property
    def remaining_s(self) -> float:
        return self.budget_s - self.run_s

    def observe(self, now: float, *, running: bool, reason: str = "") -> str | None:
        st = self.state
        if st["last_t"] is not None:
            gap = now - st["last_t"]
            if gap > self.max_gap_s:
                st["controller_gaps"].append({"from": st["last_t"], "to": now, "s": gap})      # controller was away: not counted
            elif st["last_running"]:
                st["run_s"] += gap
        if st["started_t"] is None and running:
            st["started_t"] = now
        if running and st["open_interruption"] is not None:
            st["open_interruption"]["to"] = now
            st["open_interruption"]["s"] = now - st["open_interruption"]["from"]
            st["interruptions"].append(st["open_interruption"])
            st["open_interruption"] = None
        if not running and st["open_interruption"] is None and st["started_t"] is not None:
            st["open_interruption"] = {"from": now, "reason": reason}
        st["last_t"], st["last_running"] = now, running
        event = None
        if not st["stopped"] and st["run_s"] >= self.budget_s:
            st["stopped"] = True
            event = "stop"
        elif not st["warned"] and st["run_s"] >= self.budget_s - self.warn_before_s:
            st["warned"] = True
            event = "warn"
        self._save()
        return event

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)

    def summary(self) -> dict:
        return {"run_s": self.run_s, "remaining_s": self.remaining_s, "budget_s": self.budget_s,
                "interruptions": len(self.state["interruptions"]),
                "interrupted_s": sum(i["s"] for i in self.state["interruptions"]),
                "controller_gaps": len(self.state["controller_gaps"]), "warned": self.state["warned"], "stopped": self.state["stopped"]}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="one observation for the controller loop")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--running", choices=["yes", "no"], required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--budget-hours", type=float, default=float(os.environ.get("GOALBOX_RUN_HOURS", "48")))
    args = parser.parse_args()
    ledger = Ledger(args.ledger, budget_s=args.budget_hours * 3600)
    event = ledger.observe(time.time(), running=args.running == "yes", reason=args.reason)
    print(json.dumps({"event": event} | ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
