"""Per-phase deadlines (D-S4). Values come from measured times plus a margin; they differ for
synthetic and real-weight cases and for two-tray jobs. Every value can be overridden for
calibration with GOALBOX_DEADLINE_<FIELD>=<seconds> (upper-case field name), and the values
in force are written into every record so a timeout can be read against them.

  build_s        import + build(contract) in the candidate process
  inputs_s       make_inputs: for real weights this is the weight load from local RAID
  trial_s        one candidate run during a correctness trial (first call includes JIT)
  rep_s          one timed repetition
  audit_s        the advisory in-process profile run
  collective_s   torch.distributed timeout while grading (shorter than the default 10 minutes
                 for synthetic cases; weight loading is covered by inputs_s, not by this)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

from ..cases import Case


@dataclass(frozen=True)
class Deadlines:
    build_s: float
    inputs_s: float
    trial_s: float
    rep_s: float
    audit_s: float
    collective_s: float

    def to_dict(self) -> dict:
        return asdict(self)


SYNTHETIC = Deadlines(build_s=600, inputs_s=300, trial_s=300, rep_s=120, audit_s=300, collective_s=600)
REAL = Deadlines(build_s=900, inputs_s=2400, trial_s=900, rep_s=300, audit_s=600, collective_s=1800)


def for_case(case: Case, *, multi_node: bool = False, env: dict | None = None) -> Deadlines:
    env = os.environ if env is None else env
    base = REAL if case.params.get("checkpoint") else SYNTHETIC
    values = base.to_dict()
    if multi_node:
        # Two trays: the second tray's weight load and the cross-tray bootstrap add to inputs and collectives.
        values["inputs_s"] *= 1.5
        values["collective_s"] *= 1.5
    for field in fields(Deadlines):
        override = env.get(f"GOALBOX_DEADLINE_{field.name.upper()}")
        if override:
            value = float(override)
            if value <= 0:
                raise ValueError(f"GOALBOX_DEADLINE_{field.name.upper()} must be positive")
            values[field.name] = value
    return Deadlines(**values)
