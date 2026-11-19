"""Two-tray launch topology for TP8 (DECISIONS section 2) and the single-host path.

The evaluator starts one process per local GPU. On one host (TP1, TP2, TP4, or an 8-GPU host)
all ranks are local and rendezvous through a file. For a two-tray TP8 job the broker starts
the same command on both trays with these variables:

  GOALBOX_NNODES=2  GOALBOX_NODE_RANK=0|1  GOALBOX_MASTER_ADDR=<tray A>  GOALBOX_MASTER_PORT=<p>
  GOALBOX_JOB_ID=<id>   (for log names only)

Rank mapping: global rank = node_rank * local_world + local_rank, so ranks 0..3 run on the first
tray and 4..7 on the second. The evaluator ranks use port p and the candidate processes use
port p+1 for their own process group (isolation.py). The same variables are visible to the
agent's own scripts in a `gpu-run -g 8` job (ENVIRONMENT.md).
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


@dataclass(frozen=True)
class Topology:
    world_size: int
    nnodes: int
    node_rank: int
    local_world: int
    master_addr: str | None
    master_port: int | None
    job_id: str

    @property
    def local_ranks(self) -> list[int]:
        return list(range(self.local_world))

    def global_rank(self, local_rank: int) -> int:
        if not 0 <= local_rank < self.local_world:
            raise ValueError(f"local rank {local_rank} outside 0..{self.local_world - 1}")
        return self.node_rank * self.local_world + local_rank

    @property
    def multi_node(self) -> bool:
        return self.nnodes > 1

    def rendezvous(self, directory: str, *, offset: int = 0) -> str:
        """init_method for the evaluator ranks (offset 0) or the candidate ranks (offset 1)."""
        if not self.multi_node:
            return f"file://{directory}/rendezvous{offset}"
        return f"tcp://{self.master_addr}:{self.master_port + offset}"

    def to_dict(self) -> dict:
        """Constructor fields only (round-trips through Topology(**d))."""
        return asdict(self)

    def describe(self) -> dict:
        return self.to_dict() | {"rank_mapping": "rank = node_rank * local_world + local_rank",
                                 "ranks_here": [self.global_rank(r) for r in self.local_ranks]}


def topology_from_env(world_size: int, env: dict | None = None) -> Topology:
    env = os.environ if env is None else env
    nnodes = int(env.get("GOALBOX_NNODES", "1"))
    node_rank = int(env.get("GOALBOX_NODE_RANK", "0"))
    if nnodes < 1 or not 0 <= node_rank < nnodes:
        raise ValueError(f"invalid GOALBOX_NNODES={nnodes} / GOALBOX_NODE_RANK={node_rank}")
    if world_size % nnodes:
        raise ValueError(f"{world_size} ranks do not split over {nnodes} nodes")
    local_world = world_size // nnodes
    master_addr = env.get("GOALBOX_MASTER_ADDR") or None
    master_port = env.get("GOALBOX_MASTER_PORT") or None
    if nnodes > 1:
        if not master_addr or not HOST_RE.match(master_addr):
            raise ValueError("GOALBOX_MASTER_ADDR must be a host name or address for a multi-node job")
        if master_port is None:
            raise ValueError("GOALBOX_MASTER_PORT is required for a multi-node job")
        port = int(master_port)
        if not 1024 <= port <= 65533:
            raise ValueError(f"GOALBOX_MASTER_PORT {port} outside 1024..65533 (two ports are used)")
    else:
        port = int(master_port) if master_port else None
    return Topology(world_size, nnodes, node_rank, local_world, master_addr if nnodes > 1 else None,
                    port if nnodes > 1 else None, env.get("GOALBOX_JOB_ID", "local"))
