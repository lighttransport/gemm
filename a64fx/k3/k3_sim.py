#!/usr/bin/env python3
"""Kimi K3 memory and performance model for A64FX.

Only safetensor headers are read. Tensor payloads are never mapped or copied.
The model distinguishes measured/calibrated quantities from assumptions; use
--json for downstream sweeps.
"""
import argparse
import glob
import json
import math
import os
import re
import random
import struct
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple, NamedTuple

HIDDEN = 7168
HEADS = 96
HEAD_DIM = 128
LAYERS = 93
KDA_LAYERS = 69
MLA_LAYERS = 24
MOE_LAYERS = 92
EXPERTS = 896
TOP_K = 16
LATENT = 3584
EXPERT_INTER = 3072
USABLE_GB = 27.0
ATTENTION_GB = 72.404  # header-scan total; small head-TP slices use a separate decode rate
ROUTER_DOWN_GB = 5.909055488  # 92 BF16 router + routed-latent down matrices
ROUTER_GB = 1.181745664
ROUTED_DOWN_GB = 4.726980608
ROUTED_UP_GB = 4.726980608
SHARED_EXPERT_GB = 24.310185984  # TP-sharded shared experts, full stack

# Header-scan fallback, decimal GB, from the release checkpoint.
FALLBACK = {
    "experts": 1446.456,
    "replicated": 6.12,       # router + latent-down + small norms/projections
    "shardable": 108.28,      # text tensors assigned to TP/vocab sharding
    "ignored": 0.894,         # vision/projector, text-v1 excludes these
    "total_text": 1560.856 - 0.894,
}


class WeightSplit(NamedTuple):
    experts: float
    replicated: float
    shardable: float
    ignored: float
    total_text: float
    tensors: int = 0
    shards: int = 0
    source: str = "fallback"


class Memory(NamedTuple):
    weights_gb: float
    expert_gb: float
    replicated_gb: float
    tp_gb: float
    kda_state_gb: float
    mla_cache_gb: float
    attnres_scratch_gb: float
    total_gb: float
    usable_gb: float
    fits: bool


def tensor_bytes(info: dict) -> int:
    begin, end = info["data_offsets"]
    return end - begin


def tensor_policy(name: str) -> str:
    """Return v1 placement policy for a checkpoint tensor."""
    if name.startswith("vision_tower.") or "multi_modal_projector" in name:
        return "ignored"
    if ".experts." in name and ("weight_packed" in name or "weight_scale" in name):
        return "experts"
    # These are deliberately replicated latency-sensitive operands.
    if ("block_sparse_moe.gate." in name
            or "routed_expert_down_proj" in name
            or "routed_expert_norm" in name
            or name.endswith("layernorm.weight")
            or "_res_norm.weight" in name
            or "_res_proj.weight" in name
            or ".self_attn.f_a_proj." in name
            or name.endswith(".self_attn.A_log")
            or name.endswith(".self_attn.dt_bias")
            or "conv1d.weight" in name):
        return "replicated"
    return "shardable"


def iter_headers(model_dir: Path) -> Iterable[Tuple[str, dict]]:
    paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no *.safetensors under {model_dir}")
    for path in paths:
        with path.open("rb", buffering=0) as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"short safetensor header: {path}")
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 512 * 1024 * 1024:
                raise ValueError(f"implausible header length {n}: {path}")
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                yield name, info


def scan_weights(model_dir: Path) -> WeightSplit:
    totals: Dict[str, int] = {k: 0 for k in ("experts", "replicated", "shardable", "ignored")}
    tensors = 0
    for name, info in iter_headers(model_dir):
        totals[tensor_policy(name)] += tensor_bytes(info)
        tensors += 1
    shards = len(list(model_dir.glob("*.safetensors")))
    gb = {k: v / 1e9 for k, v in totals.items()}
    return WeightSplit(**gb, total_text=gb["experts"] + gb["replicated"] + gb["shardable"],
                       tensors=tensors, shards=shards, source=str(model_dir))


def fallback_weights() -> WeightSplit:
    return WeightSplit(**FALLBACK)


def fullest_expert_gb(split: WeightSplit, nodes: int) -> float:
    # e % nodes ownership; all experts have the same packed shapes.
    return split.experts * math.ceil(EXPERTS / nodes) / EXPERTS


def memory(split: WeightSplit, nodes: int, context: int, batch: int,
           kv_bytes: int = 2, usable_gb: float = USABLE_GB,
           expert_tp: bool = False, dense_q8: bool = False,
           fused_moe_ar: bool = False, q8_up_only: bool = False,
           q8w16_up: bool = False, q8w16_dense: bool = False) -> Memory:
    local_heads = math.ceil(HEADS / nodes)
    expert = split.experts / nodes if expert_tp else fullest_expert_gb(split, nodes)
    if dense_q8 or q8_up_only or q8w16_up or q8w16_dense:
        q8_ratio = .625
        if not (q8w16_up or q8w16_dense):
            q8_ratio = .5006
        replicated = split.replicated + q8_ratio * ROUTED_UP_GB
        if dense_q8 or q8w16_dense:
            replicated += (q8_ratio - 1.0) * ROUTED_DOWN_GB
        if q8w16_dense:
            replicated += (.75 - 1.0) * ROUTER_GB
        tp = max(0.0, split.shardable - ROUTED_UP_GB) / nodes
    elif fused_moe_ar:
        replicated = split.replicated + ROUTED_UP_GB
        tp = max(0.0, split.shardable - ROUTED_UP_GB) / nodes
    else:
        replicated = split.replicated
        tp = split.shardable / nodes
    weights = expert + replicated + tp
    # KDA recurrent state is FP32 [layer,stream,local_head,value,key].
    kda = KDA_LAYERS * batch * local_heads * HEAD_DIM * HEAD_DIM * 4 / 1e9
    # Exact expanded MLA K/V, local heads only: 192 key + 128 value BF16 elements.
    mla = MLA_LAYERS * batch * context * local_heads * (192 + 128) * kv_bytes / 1e9
    # 12 saved residuals plus current, FP32; ping-pong and projection workspaces.
    scratch = batch * 15 * HIDDEN * 4 / 1e9 + 0.35
    total = weights + kda + mla + scratch
    return Memory(weights, expert, replicated, tp, kda, mla, scratch,
                  total, usable_gb, total <= usable_gb)


def allreduce_seconds(nodes: int, payload_bytes: int, calls: int,
                      latency_us: float, link_gbps: float) -> float:
    # Recursive doubling latency and a bidirectional-ring payload lower bound.
    steps = math.ceil(math.log2(nodes)) if nodes > 1 else 0
    latency = calls * steps * latency_us * 1e-6
    wire = calls * 2.0 * (nodes - 1) / nodes * payload_bytes / (link_gbps * 1e9)
    return latency + wire


def hierarchical_ar_speedup(elements: int) -> float:
    """12-node measured 3x4/flat speedup, log-interpolated by payload."""
    points = ((3584, 1.16), (7168, 1.19), (10752, 1.24),
              (114688, 1.30), (229376, 1.31))
    if elements <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if elements <= x1:
            f = math.log(elements / x0) / math.log(x1 / x0)
            return y0 + f * (y1 - y0)
    return points[-1][1]


def active_expert_gb(split: WeightSplit, nodes: int, batch: int,
                     imbalance: float) -> float:
    # Distinct experts touched by batch independent top-k draws, per MoE layer.
    distinct = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** batch)
    return split.experts * distinct / EXPERTS / nodes * imbalance


def expert_service_ms(count: int, single_ms: float) -> float:
    """Measured real-expert latency curve, linearly interpolated by bucket M."""
    curve = ((0, 0.0), (1, 0.1665), (2, 0.2975), (4, 0.512),
             (8, 0.946), (16, 1.690), (32, 2.6915))
    scale = single_ms / curve[1][1]
    for (m0, t0), (m1, t1) in zip(curve, curve[1:]):
        if count <= m1:
            return scale * (t0 + (t1 - t0) * (count - m0) / (m1 - m0))
    m0, t0 = curve[-2]
    m1, t1 = curve[-1]
    return scale * (t1 + (count - m1) * (t1 - t0) / (m1 - m0))


def rank_expert_service_ms(rank_buckets: dict, single_ms: float) -> float:
    service = sum(expert_service_ms(m, single_ms)
                  for m in rank_buckets.values())
    # Four-CMG sparse scheduler, measured with 1/2/4/8/16 distinct real experts
    # on every node of the 12-node allocation. Interpolate by active buckets.
    points = ((1, 1.0), (2, 0.874), (4, 0.861),
              (8, 0.775), (16, 0.760))
    count = len(rank_buckets)
    factor = points[-1][1]
    for (n0, f0), (n1, f1) in zip(points, points[1:]):
        if count <= n1:
            factor = f0 + (f1 - f0) * (count - n0) / (n1 - n0)
            break
    return service * factor


@lru_cache(maxsize=None)
def critical_expert_ms(nodes: int, batch: int, single_ms: float,
                       samples: int) -> Tuple[float, float]:
    """Mean/p95 slowest-rank expert service for one MoE layer.

    Each token chooses 16 distinct experts. Experts are assigned by id modulo
    rank, and same-expert tokens use the measured batched kernel curve.
    """
    rng = random.Random(0x4B334D4F45 + nodes * 257 + batch)
    maxima = []
    for _ in range(samples):
        buckets = [dict() for _ in range(nodes)]
        for _token in range(batch):
            for expert in rng.sample(range(EXPERTS), TOP_K):
                rank = expert % nodes
                buckets[rank][expert] = buckets[rank].get(expert, 0) + 1
        maxima.append(max(rank_expert_service_ms(rank_buckets, single_ms)
                          for rank_buckets in buckets))
    maxima.sort()
    return sum(maxima) / samples, maxima[min(samples - 1, int(.95 * samples))]


def collective_seconds(nodes: int, batch: int, moe_collectives: int,
                       latency_us: float, link_gbps: float,
                       hierarchical_ar: bool = False) -> Tuple[float, float, int]:
    # Attention produces one hidden collective per layer. The dense FFN adds
    # one. Current exact LatentMoE has latent-reduce + hidden-output (2).
    hidden_calls = LAYERS + 1 + MOE_LAYERS * min(moe_collectives, 1)
    latent_calls = MOE_LAYERS if moe_collectives >= 2 else 0
    hidden_calls += MOE_LAYERS * max(0, moe_collectives - 2)
    hidden_s = allreduce_seconds(nodes, HIDDEN * 2 * batch, hidden_calls,
                                 latency_us, link_gbps)
    latent_s = allreduce_seconds(nodes, LATENT * 2 * batch, latent_calls,
                                 latency_us, link_gbps)
    if hierarchical_ar:
        hidden_s /= hierarchical_ar_speedup(HIDDEN * batch)
        latent_s /= hierarchical_ar_speedup(LATENT * batch)
    return hidden_s, latent_s, hidden_calls + latent_calls


def decode(split: WeightSplit, nodes: int, context: int, batch: int,
           bw_gbps: float, mla_bw_gbps: float, router_down_gbps: float, head_bw_gbps: float, mxfp4_gbps: float,
           kda_gops: float, latency_us: float,
           link_gbps: float, imbalance: float, expert_ms: float,
           expert_samples: int, moe_collectives: int,
           latent_overlap: bool, hierarchical_ar: bool,
           lean_collective_speedup: float = 1.105,
           expert_tp: bool = False, expert_tp_layer_ms: float = .063,
           fused_moe_ar: bool = False, dense_q8: bool = False,
           q8_down_gbps: float = 140.0, q8_up_gbps: float = 173.0,
           attention_rsag: bool = False, q8_up_only: bool = False,
           q8w16_up: bool = False, q8w16_up_gbps: float = 300.0,
           q8w16_dense: bool = False, q8w16_pair_gbps: float = 280.0,
           moe_prefetch_mib: float = 16.0, prefetch_gbps: float = 248.0,
           prefetch_launch_us: float = 51.0) -> dict:
    expert = active_expert_gb(split, nodes, batch, imbalance)
    attention = min(ATTENTION_GB, split.shardable) / nodes
    other_tp = max(0.0, split.shardable - ATTENTION_GB -
                   (ROUTED_UP_GB if fused_moe_ar else 0.0)) / nodes
    dense = split.replicated + attention + other_tp
    cache = MLA_LAYERS * context * math.ceil(HEADS / nodes) * (192 + 128) * 2 / 1e9
    router_down = min(ROUTER_DOWN_GB, split.replicated)
    if q8w16_dense:
        weight_s = (.75 * ROUTER_GB + .625 * ROUTED_DOWN_GB) / q8w16_pair_gbps
        weight_s += (split.replicated - ROUTER_DOWN_GB + other_tp) / bw_gbps
        weight_s += .625 * ROUTED_UP_GB / q8w16_up_gbps
    elif dense_q8:
        weight_s = ROUTER_GB / router_down_gbps
        weight_s += (split.replicated - ROUTER_DOWN_GB + other_tp) / bw_gbps
        weight_s += .5006 * ROUTED_DOWN_GB / q8_down_gbps
        weight_s += .5006 * ROUTED_UP_GB / q8_up_gbps
    elif q8_up_only or q8w16_up:
        weight_s = router_down / router_down_gbps
        weight_s += (split.replicated - router_down + other_tp) / bw_gbps
        weight_s += ((.625 * ROUTED_UP_GB / q8w16_up_gbps) if q8w16_up else
                     (.5006 * ROUTED_UP_GB / q8_up_gbps))
    else:
        weight_s = router_down / router_down_gbps
        weight_s += (split.replicated - router_down + other_tp) / bw_gbps
        if fused_moe_ar:
            weight_s += ROUTED_UP_GB / bw_gbps
    weight_s += attention / head_bw_gbps
    if expert_tp:
        # All ranks execute the same 16 expert slices.  M growth is sublinear
        # once the native group-32 weights are reused across token rows.
        expert_layer_ms = expert_tp_layer_ms * batch ** .65
        expert_p95_ms = expert_layer_ms
    else:
        expert_layer_ms, expert_p95_ms = critical_expert_ms(
            nodes, batch, expert_ms, expert_samples)
    expert_s = MOE_LAYERS * expert_layer_ms * 1e-3
    cache_s = cache / mla_bw_gbps
    # decay, prediction dot, delta update and output dot; measured single-head kernel.
    kda_ops = batch * KDA_LAYERS * math.ceil(HEADS / nodes) * HEAD_DIM * HEAD_DIM * 6
    kda_s = kda_ops / (kda_gops * 1e9)
    moe_comm = 0.0
    if fused_moe_ar:
        # 93 attention + one dense output, then one concatenated
        # [routed-latent, shared-hidden] reduction per MoE layer.
        attention_comm = allreduce_seconds(nodes, HIDDEN * 2 * batch,
                                           LAYERS, latency_us, link_gbps)
        if attention_rsag:
            # 12-node real BF16 scatter+SVE-sum+gather is 1.03x the tree.
            # Keep the switch as a measured rejected alternative, not a win.
            attention_comm *= 1.03
        elif hierarchical_ar:
            attention_comm /= hierarchical_ar_speedup(HIDDEN * batch)
        dense_comm = allreduce_seconds(nodes, HIDDEN * 2 * batch, 1,
                                       latency_us, link_gbps)
        moe_comm = allreduce_seconds(nodes, (LATENT + HIDDEN) * 2 * batch,
                                     MOE_LAYERS, latency_us, link_gbps)
        if hierarchical_ar:
            dense_comm /= hierarchical_ar_speedup(HIDDEN * batch)
            moe_comm /= hierarchical_ar_speedup((LATENT + HIDDEN) * batch)
        hidden_comm_s = attention_comm + dense_comm + moe_comm
        latent_comm_s = 0.0
        collective_calls = LAYERS + 1 + MOE_LAYERS
    else:
        hidden_comm_s, latent_comm_s, collective_calls = collective_seconds(
            nodes, batch, moe_collectives, latency_us, link_gbps, hierarchical_ar)
    effective_lean = lean_collective_speedup if hierarchical_ar else 1.0
    hidden_comm_s /= effective_lean
    latent_comm_s /= effective_lean
    moe_comm /= effective_lean
    # Only the TP shard of the shared expert is independent of the routed
    # latent reduction.  It is 0.253 GB/rank over the whole 96-node stack, so
    # this overlap is intentionally capped and cannot hide arbitrary comm.
    overlap_s = min(latent_comm_s, SHARED_EXPERT_GB / nodes / bw_gbps) \
        if latent_overlap else 0.0
    comm_s = hidden_comm_s + latent_comm_s - overlap_s
    prefetch_s = 0.0
    if fused_moe_ar and (q8w16_up or q8w16_dense) and moe_prefetch_mib > 0:
        requested_gb = moe_prefetch_mib * (1 << 20) * MOE_LAYERS / 1e9
        available_gb = .625 * ROUTED_UP_GB
        comm_capacity_gb = prefetch_gbps * moe_comm
        prefetched_gb = min(requested_gb, available_gb, comm_capacity_gb)
        prefetch_s = prefetched_gb / q8w16_up_gbps
        comm_s += MOE_LAYERS * prefetch_launch_us * 1e-6
    lower = weight_s + expert_s + cache_s + kda_s - prefetch_s
    predicted = lower + comm_s
    return {
        "context": context, "batch": batch,
        "dense_gb": dense, "attention_tp_gb": attention,
        "active_expert_gb": expert, "mla_scan_gb": cache,
        "weight_ms": (weight_s - prefetch_s) * 1e3,
        "raw_weight_ms": weight_s * 1e3, "expert_ms": expert_s * 1e3,
        "expert_layer_ms": expert_layer_ms, "expert_layer_p95_ms": expert_p95_ms,
        "cache_ms": cache_s * 1e3,
        "kda_ms": kda_s * 1e3, "comm_ms": comm_s * 1e3,
        "hidden_comm_ms": hidden_comm_s * 1e3,
        "latent_comm_ms": latent_comm_s * 1e3,
        "overlap_ms": overlap_s * 1e3,
        "prefetch_overlap_ms": prefetch_s * 1e3,
        "collective_calls": collective_calls,
        "lower_bound_ms": lower * 1e3, "predicted_ms": predicted * 1e3,
        "tokens_per_second": batch / predicted,
    }


def prefill(split: WeightSplit, nodes: int, tokens: int, chunk: int,
            gemm_tflops: float, mla_tflops: float, kda_gops: float, latency_us: float,
            link_gbps: float, moe_collectives: int,
            hierarchical_ar: bool, expert_tp: bool = False,
            fused_moe_ar: bool = False,
            expert_prefill_ms=(1.783, 6.742, 23.929)) -> dict:
    # Every dense/shardable byte corresponds approximately to one matrix weight used once/token.
    # MXFP4 expert bytes are 0.53125 B/weight including scales.
    dense_weights = (split.replicated + split.shardable / nodes) * 1e9 / 2.0
    expert_weights = split.experts * (TOP_K / EXPERTS) / nodes * 1e9 / 0.53125
    dense_flops = tokens * 2.0 * dense_weights
    dense_gemm_s = dense_flops / (gemm_tflops * 1e12)
    chunks = math.ceil(tokens / chunk)
    if expert_tp:
        points = ((0, 0.0), (64, expert_prefill_ms[0]),
                  (256, expert_prefill_ms[1]), (1024, expert_prefill_ms[2]))
        layer_ms = points[-1][1] * chunk / points[-1][0]
        for (m0, t0), (m1, t1) in zip(points, points[1:]):
            if chunk <= m1:
                layer_ms = t0 + (t1 - t0) * (chunk - m0) / (m1 - m0)
                break
        expert_s = MOE_LAYERS * chunks * layer_ms * 1e-3
    else:
        expert_s = tokens * 2.0 * expert_weights / (gemm_tflops * 1e12)
    gemm_s = dense_gemm_s + expert_s
    kda_ops = tokens * KDA_LAYERS * math.ceil(HEADS / nodes) * HEAD_DIM * HEAD_DIM * 6
    kda_s = kda_ops / (kda_gops * 1e9)
    # Causal MLA: sum_{t=1}^T t dot/update work, distributed by head.
    pairs = tokens * (tokens + 1) / 2.0
    mla_ops = MLA_LAYERS * math.ceil(HEADS / nodes) * pairs * 2.0 * (192 + 128)
    attention_s = mla_ops / (mla_tflops * 1e12)
    if fused_moe_ar:
        attention_comm = allreduce_seconds(nodes,HIDDEN*2*chunk,LAYERS,latency_us,link_gbps)
        dense_comm = allreduce_seconds(nodes,HIDDEN*2*chunk,1,latency_us,link_gbps)
        moe_comm = allreduce_seconds(nodes,(LATENT+HIDDEN)*2*chunk,MOE_LAYERS,latency_us,link_gbps)
        if hierarchical_ar:
            attention_comm /= hierarchical_ar_speedup(HIDDEN*chunk)
            dense_comm /= hierarchical_ar_speedup(HIDDEN*chunk)
            moe_comm /= hierarchical_ar_speedup((LATENT+HIDDEN)*chunk)
        comm_s=(attention_comm+dense_comm+moe_comm)*chunks
        calls_per_chunk=LAYERS+1+MOE_LAYERS
    else:
        hidden_chunk_s, latent_chunk_s, calls_per_chunk = collective_seconds(
            nodes, chunk, moe_collectives, latency_us, link_gbps, hierarchical_ar)
        comm_s = (hidden_chunk_s + latent_chunk_s) * chunks
    total = gemm_s + kda_s + attention_s + comm_s
    scratch_gb = (chunk*TOP_K*LATENT*4 + 2*chunk*TOP_K*32*4 +
                  chunk*LATENT*4) / 1e9 if expert_tp else 0.0
    return {
        "tokens": tokens, "chunk": chunk, "gemm_s": gemm_s,
        "dense_gemm_s": dense_gemm_s, "expert_s": expert_s,
        "expert_layer_ms": (layer_ms if expert_tp else None),
        "scratch_gb": scratch_gb, "kda_s": kda_s,
        "mla_attention_s": attention_s, "comm_s": comm_s,
        "collective_calls": calls_per_chunk * chunks, "total_s": total,
        "tokens_per_second": tokens / total,
    }


def fmt_ctx(n: int) -> str:
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n // (1024*1024)}m"


def report(args: argparse.Namespace, split: WeightSplit) -> dict:
    tp_nodes = args.tp_nodes
    replicas = args.nodes // tp_nodes
    result = {"assumptions": vars(args).copy(), "weights": split._asdict(), "memory": [],
              "decode": [], "prefill": []}
    print("Kimi K3 / A64FX performance model")
    print("=" * 72)
    print(f"manifest: {split.source} ({split.shards} shards, {split.tensors:,} tensors)")
    print(f"text weights: {split.total_text:.3f} GB = experts {split.experts:.3f} + "
          f"replicated {split.replicated:.3f} + TP {split.shardable:.3f}")
    print(f"parallelism: physical={args.nodes} TP={tp_nodes} independent-contexts={replicas}")
    print(f"calibration: dense-BW={args.bw_gbps:g} GB/s, MLA-BW={args.mla_bw_gbps:g} GB/s, router/down={args.router_down_gbps:g} GB/s, "
          f"head-slice-BW={args.head_bw_gbps:g} GB/s, "
          f"MXFP4-BW={args.mxfp4_gbps:g} GB/s, "
          f"expert-M1={args.expert_ms:g} ms, "
          f"KDA={args.kda_gops:g} GOP/s, "
          f"dense-GEMM={args.gemm_tflops:g} TF/s, MLA={args.mla_tflops:g} TF/s, "
          f"collective={args.latency_us:g} us/step, "
          f"hierarchical={'on' if args.hierarchical_ar else 'off'}, "
          f"lean-decode={(args.lean_collective_speedup if args.hierarchical_ar else 1.0):g}x, "
          f"latent-overlap={'on' if args.latent_overlap else 'off'}")
    if args.expert_tp or args.fused_moe_ar or args.dense_q8 or args.q8_up_only or args.q8w16_up or args.q8w16_dense or args.attention_rsag:
        print(f"architecture: expert-TP={args.expert_tp} ({args.expert_tp_layer_ms:g} ms/layer), "
              f"fused-MoE-AR={args.fused_moe_ar}, attention-RSAG={args.attention_rsag}, "
              f"dense-Q8={args.dense_q8}, Q8-up-only={args.q8_up_only}, "
              f"Q8W16-up={args.q8w16_up}, Q8W16-dense={args.q8w16_dense} "
              f"({args.q8_down_gbps:g}/{args.q8_up_gbps:g}/{args.q8w16_up_gbps:g}/"
              f"{args.q8w16_pair_gbps:g} GB/s down/up/W16/pair), "
              f"MoE-prefetch={args.moe_prefetch_mib:g} MiB/layer at {args.prefetch_gbps:g} GB/s "
              f"(+{args.prefetch_launch_us:g} us launch), "
              f"expert-prefill-ms@64/256/1024={','.join('%g' % x for x in args.expert_prefill_ms)}")
    print("\nMemory (fullest rank, expanded BF16 MLA cache)")
    print(f"{'ctx':>5} {'M':>3} {'weights':>8} {'KDA':>7} {'MLA-KV':>8} {'total':>8} {'fit27':>6}")
    for ctx in args.contexts:
        for batch in args.batches:
            m = memory(split, tp_nodes, ctx, batch, args.kv_bytes, args.usable_gb,
                       args.expert_tp, args.dense_q8, args.fused_moe_ar,
                       args.q8_up_only, args.q8w16_up, args.q8w16_dense)
            md = m._asdict()
            md.update({"context": ctx, "batch": batch})
            result["memory"].append(md)
            print(f"{fmt_ctx(ctx):>5} {batch:3d} {m.weights_gb:8.2f} {m.kda_state_gb:7.2f} "
                  f"{m.mla_cache_gb:8.2f} {m.total_gb:8.2f} {'yes' if m.fits else 'NO':>6}")
    if args.fused_moe_ar:
        calls = LAYERS + 1 + MOE_LAYERS
    else:
        _, _, calls = collective_seconds(tp_nodes, 1, args.moe_collectives,
                                         args.latency_us, args.link_gbps,
                                         args.hierarchical_ar)
    print(f"\nDecode (batch tokens/s; {calls} collectives/layer-stack)")
    print(f"{'ctx':>5} {'M':>3} {'W ms':>7} {'Exp ms':>7} {'KV ms':>7} {'KDA':>7} {'comm':>7} {'tok/s':>9} {'agg tok/s':>10}")
    target_decode = None
    for ctx in args.contexts:
        for batch in args.batches:
            d = decode(split,tp_nodes,ctx,batch,args.bw_gbps,args.mla_bw_gbps,args.router_down_gbps,args.head_bw_gbps,
                       args.mxfp4_gbps,args.kda_gops,
                       args.latency_us,args.link_gbps,args.imbalance,
                       args.expert_ms,args.expert_samples,args.moe_collectives,
                       args.latent_overlap,args.hierarchical_ar,args.lean_collective_speedup,args.expert_tp,
                       args.expert_tp_layer_ms,args.fused_moe_ar,args.dense_q8,
                       args.q8_down_gbps,args.q8_up_gbps,args.attention_rsag,
                       args.q8_up_only,args.q8w16_up,args.q8w16_up_gbps,
                       args.q8w16_dense,args.q8w16_pair_gbps,
                       args.moe_prefetch_mib,args.prefetch_gbps,args.prefetch_launch_us)
            d["physical_nodes"] = args.nodes
            d["tp_nodes"] = tp_nodes
            d["independent_contexts"] = replicas
            d["aggregate_tokens_per_second"] = d["tokens_per_second"] * replicas
            result["decode"].append(d)
            print(f"{fmt_ctx(ctx):>5} {batch:3d} {d['weight_ms']:7.1f} {d['expert_ms']:7.1f} "
                  f"{d['cache_ms']:7.1f} {d['kda_ms']:7.1f} {d['comm_ms']:7.1f} {d['tokens_per_second']:9.2f} "
                  f"{d['aggregate_tokens_per_second']:10.2f}")
            if ctx == 4096 and batch == 1:
                target_decode = d
    if target_decode is not None:
        budget_ms = 1000.0 / args.decode_target_tps
        predicted_ms = 1000.0 / target_decode["tokens_per_second"]
        margin_ms = budget_ms - predicted_ms
        tolerance_tps = args.decode_target_tps * args.decode_target_tolerance_pct / 100.0
        target = {"tokens_per_second": args.decode_target_tps,
                  "budget_ms": budget_ms, "predicted_ms": predicted_ms,
                  "margin_ms": margin_ms, "tolerance_tps": tolerance_tps,
                  "passes": target_decode["tokens_per_second"] + tolerance_tps >= args.decode_target_tps}
        result["decode_target_4k_m1"] = target
        print(f"Decode target {args.decode_target_tps:.2f} tok/s at 4k M=1: "
              f"{'PASS' if target['passes'] else 'FAIL'} "
              f"({budget_ms:.2f} ms budget, {predicted_ms:.2f} ms predicted, "
              f"{margin_ms:+.2f} ms margin, {args.decode_target_tolerance_pct:g}% calibration tolerance)")
    print("\nPrefill (tokens/s)")
    print(f"{'prompt':>7} {'chunk':>6} {'dense s':>9} {'expert s':>9} {'KDA s':>9} {'MLA s':>9} {'comm s':>9} {'tok/s':>9} {'agg tok/s':>10}")
    for tokens in args.prompts:
        for chunk in args.chunks:
            p = prefill(split,tp_nodes,tokens,chunk,args.gemm_tflops,args.mla_tflops,args.kda_gops,
                        args.latency_us,args.link_gbps,args.moe_collectives,
                        args.hierarchical_ar,args.expert_tp,args.fused_moe_ar,
                        args.expert_prefill_ms)
            p["physical_nodes"] = args.nodes
            p["tp_nodes"] = tp_nodes
            p["independent_contexts"] = replicas
            p["aggregate_tokens_per_second"] = p["tokens_per_second"] * replicas
            result["prefill"].append(p)
            print(f"{fmt_ctx(tokens):>7} {chunk:6d} {p['dense_gemm_s']:9.1f} {p['expert_s']:9.1f} {p['kda_s']:9.1f} "
                  f"{p['mla_attention_s']:9.1f} {p['comm_s']:9.1f} {p['tokens_per_second']:9.1f} "
                  f"{p['aggregate_tokens_per_second']:10.1f}")
    print(f"\nCaveat: memory and latency are modeled per TP{tp_nodes} context; aggregate throughput assumes "
          f"{replicas} independent context group(s) with no cross-group contention. "
          "128k M=1 lacks a full-model run. "
          "Exact 1m BF16 KV exceeds HBM even with perfect context sharding; it needs cache compression or weight-memory reduction. "
          "Predictions are engineering estimates, not measured end-to-end results.")
    return result


def csv_ints(text: str):
    return [int(x) for x in text.split(",")]

def csv_floats(text: str):
    return tuple(float(x) for x in text.split(","))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path.home() / "models/kimi-k3")
    p.add_argument("--no-manifest", action="store_true", help="use documented fallback byte totals")
    p.add_argument("--nodes", type=int, default=96)
    p.add_argument("--tp-nodes", type=int, default=0,
                   help="tensor-parallel ranks per independent context (default min(nodes,96))")
    p.add_argument("--contexts", type=csv_ints, default=csv_ints("4096,131072,1048576"))
    p.add_argument("--batches", type=csv_ints, default=csv_ints("1,8,32"))
    p.add_argument("--decode-target-tps", type=float, default=18.0,
                   help="report the 4k M=1 latency margin against this stable target")
    p.add_argument("--decode-target-tolerance-pct",type=float,default=.5,
                   help="calibration tolerance for accepting the decode target")
    p.add_argument("--prompts", type=csv_ints, default=csv_ints("1024,8192,131072,1048576"))
    p.add_argument("--chunks", type=csv_ints, default=csv_ints("64,256,1024"))
    p.add_argument("--usable-gb", type=float, default=USABLE_GB)
    p.add_argument("--kv-bytes", type=int, choices=(1,2), default=2)
    p.add_argument("--bw-gbps", type=float, default=336.0, help="effective per-rank decode bandwidth")
    p.add_argument("--mla-bw-gbps", type=float, default=148.3,
                   help="effective exact-BF16 MLA scan rate from the 12-node 128K sweep")
    p.add_argument("--router-down-gbps", type=float, default=283.7,
                   help="measured fused BF16 router+routed-down bandwidth")
    p.add_argument("--head-bw-gbps", type=float, default=172.5,
                   help="measured cache-evicted one-head BF16 projection bandwidth")
    p.add_argument("--mxfp4-gbps", type=float, default=180.0,
                   help="48-core MXFP4 bandwidth; extrapolated from the measured 3.75 GB/s/core")
    p.add_argument("--expert-ms", type=float, default=0.1665,
                   help="measured real MXFP4 M=1 expert latency")
    p.add_argument("--expert-samples", type=int, default=1000,
                   help="deterministic routing Monte Carlo samples per batch")
    p.add_argument("--moe-collectives", type=int, choices=(0,1,2,3), default=2,
                   help="MoE collectives/layer: current exact path is 2")
    p.add_argument("--latent-overlap", action="store_true",
                   help="overlap latent reduce with TP-sharded shared expert (bounded)")
    p.add_argument("--hierarchical-ar", action="store_true",
                   help="apply measured 12-node 3x4 hierarchical AR speedup curve")
    p.add_argument("--kda-gops", type=float, default=14.73,
                   help="12-node mean one-head KDA rate at the optimal 8 threads")
    p.add_argument("--gemm-tflops", type=float, default=2.0,
                   help="conservative per-rank dense BF16 GEMM floor (real K3 probe: 2.0--4.7 TFLOP/s)")
    p.add_argument("--mla-tflops",type=float,default=.4375,
                   help="independent per-rank prefill MLA rate; not inferred from dense GEMM")
    p.add_argument("--expert-prefill-ms",type=csv_floats,default=csv_floats("1.783,6.742,23.929"),
                   help="measured TP96 expert-layer milliseconds at chunks 64,256,1024")
    p.add_argument("--latency-us", type=float, default=20.0, help="assumed allreduce latency per log2 step")
    p.add_argument("--link-gbps", type=float, default=8.0)
    p.add_argument("--imbalance", type=float, default=1.20, help="critical-rank routed-expert traffic factor")
    p.add_argument("--expert-tp", action="store_true",
                   help="shard every expert over its group-32 intermediate blocks")
    p.add_argument("--expert-tp-layer-ms", type=float, default=.063,
                   help="measured fused M=1 time for 16 native TP=96 slices")
    p.add_argument("--fused-moe-ar", action="store_true",
                   help="one concatenated latent+shared hidden reduction per MoE layer")
    p.add_argument("--dense-q8", action="store_true",
                   help="row-Q8 routed down and replicated routed up")
    p.add_argument("--q8-up-only", action="store_true",
                   help="keep router/down BF16 and quantize only replicated routed-up")
    p.add_argument("--q8w16-up", action="store_true",
                   help="quality-gated group-16 Q8 weights with FP32 routed-up activation")
    p.add_argument("--q8w16-dense", action="store_true",
                   help="quality-gated Q8W16 routed down+up with BF16 router")
    p.add_argument("--q8-down-gbps", type=float, default=140.0)
    p.add_argument("--q8-up-gbps", type=float, default=173.0)
    p.add_argument("--q8w16-up-gbps", type=float, default=300.0,
                   help="conservative read-cold p95 bandwidth for Q8W16 routed-up")
    p.add_argument("--q8w16-pair-gbps", type=float, default=280.0,
                   help="conservative p95 Q8W8-router + Q8W16-down bandwidth")
    p.add_argument("--moe-prefetch-mib", type=float, default=16.0,
                   help="routed-up bytes prefetched per layer during the fused MoE reduction")
    p.add_argument("--prefetch-gbps", type=float, default=248.0,
                   help="p95 cache-line prefetch bandwidth measured on A64FX")
    p.add_argument("--prefetch-launch-us", type=float, default=51.0,
                   help="measured async-helper overhead per MoE layer")
    p.add_argument("--lean-collective-speedup", type=float, default=1.105,
                   help="robust=2 decode collective speedup measured against eager polling")
    p.add_argument("--attention-rsag", action="store_true",
                   help="model measured real RSAG (1.03x flat tree; rejected)")
    p.add_argument("--json", type=Path, help="also write full results as JSON")
    args = p.parse_args()
    if args.dense_q8 and not args.fused_moe_ar:
        p.error("--dense-q8 requires --fused-moe-ar (replicated routed-up)")
    if args.q8_up_only and not args.fused_moe_ar:
        p.error("--q8-up-only requires --fused-moe-ar")
    if args.q8w16_up and not args.fused_moe_ar:
        p.error("--q8w16-up requires --fused-moe-ar")
    if args.q8w16_dense and not args.fused_moe_ar:
        p.error("--q8w16-dense requires --fused-moe-ar")
    if sum((args.dense_q8,args.q8_up_only,args.q8w16_up,args.q8w16_dense)) > 1:
        p.error("choose one routed projection quantization mode")
    if args.nodes < 1 or args.nodes > 512:
        p.error("--nodes must be in [1,512]")
    if args.tp_nodes == 0:
        args.tp_nodes = min(args.nodes, HEADS)
    if args.tp_nodes < 1 or args.tp_nodes > HEADS or args.nodes % args.tp_nodes:
        p.error("--tp-nodes must be in [1,96] and divide --nodes")
    if args.decode_target_tps <= 0:
        p.error("--decode-target-tps must be positive")
    if args.decode_target_tolerance_pct < 0:
        p.error("--decode-target-tolerance-pct must be nonnegative")
    if len(args.expert_prefill_ms)!=3 or any(x<=0 for x in args.expert_prefill_ms):
        p.error("--expert-prefill-ms expects three positive values for chunks 64,256,1024")
    if args.gemm_tflops<=0 or args.mla_tflops<=0:
        p.error("--gemm-tflops and --mla-tflops must be positive")
    if args.moe_prefetch_mib < 0 or args.prefetch_gbps <= 0 or args.prefetch_launch_us < 0 or args.lean_collective_speedup <= 0:
        p.error("prefetch size/overhead and lean speedup must be nonnegative/positive")
    split = fallback_weights() if args.no_manifest else scan_weights(args.model_dir)
    result = report(args, split)
    if args.json:
        result["assumptions"]["model_dir"] = str(args.model_dir)
        result["assumptions"]["json"] = str(args.json)
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
