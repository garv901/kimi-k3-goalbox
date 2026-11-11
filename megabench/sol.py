"""Optimistic per-GPU HBM traffic floor for one Kimi K3 decode step at TP.

This is a useful-byte bound, not a prediction of kernel latency. For the
slowest rank it counts each weight byte the rank holds and needs once, the old
MLA cache and KDA state it must read, the KDA state it must write, and its
outputs. It assumes peak HBM bandwidth and omits scratch traffic, NCCL
communication, synchronization and compute. Routed experts count the expected
number of distinct experts under uniform routing, E*(1-(1-k/E)^B); real routing
is skewed, so the real number of distinct experts is usually smaller.
"""

from __future__ import annotations

import math

from .cases import Case

DEFAULT_B200_BANDWIDTH_TB_S = 8.0   # B200 and B300 HBM3e peak per GPU
DEFAULT_B200_BANDWIDTH_SOURCE = "https://www.nvidia.com/en-us/data-center/dgx-b300/"
BF16, FP32, INT64 = 2, 4, 8


def _mxfp4_bytes(rows: int, cols: int) -> int:
    return rows*cols//2+rows*cols//32


def estimate_case(case: Case, bandwidth_tb_s: float = DEFAULT_B200_BANDWIDTH_TB_S,
                  measured_ms: float | None = None) -> dict:
    if not math.isfinite(bandwidth_tb_s) or bandwidth_tb_s <= 0:
        raise ValueError("bandwidth_tb_s must be finite and positive")
    result: dict = {"case_id": case.id, "status": "estimated", "per": "slowest rank"}
    if case.family != "kimi_k3_step":
        return result | {"status": "unavailable", "reason": f"no traffic model for {case.family}"}
    from .tasks.kimi import full_layer
    p, tp = case.params, case.tp
    B, h, lh, d, H = p["batch"], p["hidden"], p["latent_hidden"], p["head_dim"], p["q_heads"]
    qdim, ctx, E, k = H*d, p["context"], p["experts"], p["topk"]
    qr, kr, nr, rd, vd = p["q_lora_rank"], p["kv_lora_rank"], p["qk_nope_dim"], p["qk_rope_dim"], p["v_head_dim"]
    inter, shared = p["intermediate"]//tp, p["intermediate"]*p["shared_experts"]//tp
    mla = [layer for layer in range(p["layers"]) if full_layer(case, layer)]
    kda = [layer for layer in range(p["layers"]) if not full_layer(case, layer)]
    moe = p["layers"]-p["first_dense"]
    distinct = E*(1-(1-k/E)**B)
    resets = 0  # every cached row is read (rows that restart read less)
    comps: list[dict] = []

    def add(name, nbytes, detail):
        comps.append({"name": name, "bytes": int(nbytes), "detail": detail})

    add("kda_weights", len(kda)*BF16*(5*qdim//tp*h+d*h+qdim//tp*d+H//tp*h) + len(kda)*FP32*3*qdim//tp*p["conv_kernel"],
        f"{len(kda)} KDA layers: q,k,v,g,o shards, f_a (replicated), f_b, b, conv taps")
    add("mla_weights", len(mla)*BF16*(qr*h+H//tp*(nr+rd)*qr+(kr+rd)*h+H//tp*(nr+vd)*kr+2*H//tp*vd*h),
        f"{len(mla)} MLA layers: q_a and kv_a (replicated), q_b, kv_b, g, o shards")
    add("dense_ffn", p["first_dense"]*3*BF16*p["dense_intermediate"]//tp*h, "layer 0 dense FFN shard")
    add("moe_dense_parts", moe*BF16*(E*h+2*lh*h+3*shared*h), "router, latent down/up (replicated), shared expert shards")
    add("routed_experts", moe*distinct*(2*_mxfp4_bytes(inter, lh)+_mxfp4_bytes(lh, inter)),
        f"{distinct:.1f} distinct experts per layer (uniform routing), MXFP4 gate/up/down shards")
    add("norms_and_residual_mixers", p["layers"]*6*h*BF16, "layer norms and attention-residual vectors")
    add("lm_head", p["vocab"]//tp*h*BF16, "vocabulary shard of the LM head")
    add("embedding_rows", B*h*BF16, "token embedding rows")
    add("mla_cache_reads", (B-resets)*len(mla)*ctx*(kr+rd)*BF16,
        "old latent+rope cache, replicated on every rank (a context split across ranks can lower this)")
    add("kda_state", len(kda)*B*(2*H//tp*d*d*FP32+2*3*qdim//tp*p["conv_kernel"]*BF16),
        "FP32 recurrent state and BF16 conv state, read and written")
    add("outputs", B*(p["vocab"]//tp*FP32+INT64*(2+p["layers"]*k)+len(mla)*(kr+rd)*BF16), "logits shard, tokens, routes, cache writes")
    total = sum(c["bytes"] for c in comps)
    floor_ms = total/(bandwidth_tb_s*1e12)*1e3
    result.update(minimum_bytes=total, minimum_gb=total/1e9, hbm_floor_ms=floor_ms, components=comps,
                  communication_note=f"{2*p['layers']} TP all-reduces of B x {h} per step are not counted")
    if measured_ms is not None:
        result.update(measured_ms=measured_ms, gap_to_hbm_floor_x=measured_ms/floor_ms,
                      percent_of_hbm_sol=100*floor_ms/measured_ms)
    return result
