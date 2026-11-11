"""Whole-model inference challenges for agent-written megakernels.

Only cases with an implemented oracle are ready for evaluation. Planned cases
stay visible without entering a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    model: str
    phase: str
    params: dict[str, int]
    category: str
    suite: str
    backend: str = "cuda"
    gpus: int = 1
    tp: int = 1
    ep: int = 1
    max_gpu_launches: int = 1
    atol: float = 0.0
    rtol: float = 0.0
    bf16_rtol: float = 0.008
    ready: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.family or not self.model or not self.phase:
            raise ValueError(f"invalid case {self.id}")
        if any(v < 1 for v in self.params.values()):
            raise ValueError(f"invalid dimensions in {self.id}")
        if min(self.gpus, self.tp, self.ep, self.max_gpu_launches) < 1:
            raise ValueError(f"invalid execution geometry in {self.id}")
        if min(self.atol, self.rtol, self.bf16_rtol) < 0:
            raise ValueError(f"invalid tolerance in {self.id}")
        if self.suite not in ("p0", "p1", "p2", "p3"):
            raise ValueError(f"invalid priority in {self.id}")

    def to_dict(self) -> dict:
        return asdict(self)


KIMI = {"layers": 93, "experts": 896, "topk": 16, "hidden": 7168,
        "latent_hidden": 3584, "intermediate": 3072,
        "dense_intermediate": 33792, "first_dense": 1,
        "shared_experts": 2, "q_heads": 96, "head_dim": 128,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_dim": 128, "qk_rope_dim": 64,
        "v_head_dim": 128, "vocab": 163840,
        "conv_kernel": 4, "attn_res_block_size": 12, "context": 8192}


def _kimi(id: str, batch: int, *, layers: int = 93, gpus: int = 8,
          checkpoint: bool = True, suite: str = "p0", note: str = "") -> Case:
    params = KIMI | {"batch": batch, "layers": layers}
    if checkpoint:
        params["checkpoint"] = 1
    return Case(id, "kimi_k3_step", "moonshotai/Kimi-K3", "decode", params,
                "frontier_hybrid", suite, gpus=gpus, tp=gpus, ready=True, note=note)


CASES: tuple[Case, ...] = (
    # Graded: the full 93-layer model with the real checkpoint, TP8 on 8 GPUs.
    _kimi("kimi-k3-tp8-b1-s8192", 1, note="Real weights, latency regime."),
    _kimi("kimi-k3-tp8-b8-s8192", 8, note="Real weights, small serving batch."),
    _kimi("kimi-k3-tp8-b64-s8192", 64, note="Real weights, the largest batch the vLLM latency path serves."),
    # Graded: fresh synthetic weights on every trial, 8 layers at full width (anti-caching).
    _kimi("kimi-k3-syn-l8-tp8-b1-s8192", 1, layers=8, checkpoint=False, suite="p1",
          note="Synthetic weights; one MLA block every 4 layers as in the full model."),
    _kimi("kimi-k3-syn-l8-tp8-b8-s8192", 8, layers=8, checkpoint=False, suite="p1"),
    _kimi("kimi-k3-syn-l8-tp8-b64-s8192", 64, layers=8, checkpoint=False, suite="p1"),
    # Development only (not graded): one GPU, 4 layers, no communication.
    _kimi("kimi-k3-dev-l4-tp1-b8-s8192", 8, layers=4, gpus=1, checkpoint=False, suite="p2",
          note="Development slice: dense layer 0, KDA layers 1-2, MLA layer 3."),
    _kimi("kimi-k3-dev-l8-tp2-b8-s8192", 8, layers=8, gpus=2, checkpoint=False, suite="p2",
          note="Development slice with two-rank communication."),
)

GRADED = tuple(case.id for case in CASES if case.suite in ("p0", "p1"))


def select_cases(suite: str = "core", ids: list[str] | None = None) -> list[Case]:
    by_id = {case.id: case for case in CASES}
    if ids:
        missing = sorted(set(ids) - by_id.keys())
        if missing:
            raise ValueError(f"unknown cases: {', '.join(missing)}")
        return [by_id[id] for id in ids]
    if suite == "all":
        return list(CASES)
    if suite == "core":
        return [case for case in CASES if case.ready]
    if suite == "planned":
        return [case for case in CASES if not case.ready]
    if suite in ("p0", "p1", "p2", "p3"):
        return [case for case in CASES if case.suite == suite]
    raise ValueError(f"unknown suite {suite}")
