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
           kv_bytes: int = 2, usable_gb: float = USABLE_GB) -> Memory:
    local_heads = math.ceil(HEADS / nodes)
    expert = fullest_expert_gb(split, nodes)
    tp = split.shardable / nodes
    weights = expert + split.replicated + tp
    # KDA recurrent state is FP32 [layer,stream,local_head,value,key].
    kda = KDA_LAYERS * batch * local_heads * HEAD_DIM * HEAD_DIM * 4 / 1e9
    # Exact expanded MLA K/V, local heads only: 192 key + 128 value BF16 elements.
    mla = MLA_LAYERS * batch * context * local_heads * (192 + 128) * kv_bytes / 1e9
    # 12 saved residuals plus current, FP32; ping-pong and projection workspaces.
    scratch = batch * 15 * HIDDEN * 4 / 1e9 + 0.35
    total = weights + kda + mla + scratch
    return Memory(weights, expert, split.replicated, tp, kda, mla, scratch,
                  total, usable_gb, total <= usable_gb)


def allreduce_seconds(nodes: int, payload_bytes: int, calls: int,
                      latency_us: float, link_gbps: float) -> float:
    # Recursive doubling latency and a bidirectional-ring payload lower bound.
    steps = math.ceil(math.log2(nodes)) if nodes > 1 else 0
    latency = calls * steps * latency_us * 1e-6
    wire = calls * 2.0 * (nodes - 1) / nodes * payload_bytes / (link_gbps * 1e9)
    return latency + wire


def active_expert_gb(split: WeightSplit, nodes: int, batch: int,
                     imbalance: float) -> float:
    # Distinct experts touched by batch independent top-k draws, per MoE layer.
    distinct = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** batch)
    return split.experts * distinct / EXPERTS / nodes * imbalance


def expert_service_ms(count: int, single_ms: float) -> float:
    """Measured real-expert latency curve, linearly interpolated by bucket M."""
    curve = ((0, 0.0), (1, 0.220), (2, 0.317), (4, 0.555),
             (8, 1.041), (16, 1.884), (32, 3.467))
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
    # Four-CMG sparse scheduler, measured with distinct real experts on 12 nodes.
    # Two/three use a shared workshare; exactly four pin one bucket per CMG.
    factors = {2: 0.952, 3: 0.95, 4: 0.875}
    return service * factors.get(len(rank_buckets), 1.0)


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
                       latency_us: float, link_gbps: float) -> Tuple[float, int]:
    # Attention produces one hidden collective per layer. The dense FFN adds
    # one. Current exact LatentMoE has latent-reduce + hidden-output (2).
    hidden_calls = LAYERS + 1 + MOE_LAYERS * min(moe_collectives, 1)
    latent_calls = MOE_LAYERS if moe_collectives >= 2 else 0
    hidden_calls += MOE_LAYERS * max(0, moe_collectives - 2)
    seconds = allreduce_seconds(nodes, HIDDEN * 2 * batch, hidden_calls,
                                latency_us, link_gbps)
    seconds += allreduce_seconds(nodes, LATENT * 2 * batch, latent_calls,
                                 latency_us, link_gbps)
    return seconds, hidden_calls + latent_calls


def decode(split: WeightSplit, nodes: int, context: int, batch: int,
           bw_gbps: float, head_bw_gbps: float, mxfp4_gbps: float,
           kda_gops: float, latency_us: float,
           link_gbps: float, imbalance: float, expert_ms: float,
           expert_samples: int, moe_collectives: int) -> dict:
    expert = active_expert_gb(split, nodes, batch, imbalance)
    attention = min(ATTENTION_GB, split.shardable) / nodes
    other_tp = max(0.0, split.shardable - ATTENTION_GB) / nodes
    dense = split.replicated + attention + other_tp
    cache = MLA_LAYERS * context * math.ceil(HEADS / nodes) * (192 + 128) * 2 / 1e9
    weight_s = (split.replicated + other_tp) / bw_gbps
    weight_s += attention / head_bw_gbps
    expert_layer_ms, expert_p95_ms = critical_expert_ms(
        nodes, batch, expert_ms, expert_samples)
    expert_s = MOE_LAYERS * expert_layer_ms * 1e-3
    cache_s = cache / bw_gbps
    # decay, prediction dot, delta update and output dot; measured single-head kernel.
    kda_ops = batch * KDA_LAYERS * math.ceil(HEADS / nodes) * HEAD_DIM * HEAD_DIM * 6
    kda_s = kda_ops / (kda_gops * 1e9)
    comm_s, collective_calls = collective_seconds(
        nodes, batch, moe_collectives, latency_us, link_gbps)
    lower = weight_s + expert_s + cache_s + kda_s
    predicted = lower + comm_s
    return {
        "context": context, "batch": batch,
        "dense_gb": dense, "attention_tp_gb": attention,
        "active_expert_gb": expert, "mla_scan_gb": cache,
        "weight_ms": weight_s * 1e3, "expert_ms": expert_s * 1e3,
        "expert_layer_ms": expert_layer_ms, "expert_layer_p95_ms": expert_p95_ms,
        "cache_ms": cache_s * 1e3,
        "kda_ms": kda_s * 1e3, "comm_ms": comm_s * 1e3,
        "collective_calls": collective_calls,
        "lower_bound_ms": lower * 1e3, "predicted_ms": predicted * 1e3,
        "tokens_per_second": batch / predicted,
    }


def prefill(split: WeightSplit, nodes: int, tokens: int, chunk: int,
            gemm_tflops: float, kda_gops: float, latency_us: float,
            link_gbps: float, moe_collectives: int) -> dict:
    # Every dense/shardable byte corresponds approximately to one matrix weight used once/token.
    # MXFP4 expert bytes are 0.53125 B/weight including scales.
    dense_weights = (split.replicated + split.shardable / nodes) * 1e9 / 2.0
    expert_weights = split.experts * (TOP_K / EXPERTS) / nodes * 1e9 / 0.53125
    matmul_flops = tokens * 2.0 * (dense_weights + expert_weights)
    gemm_s = matmul_flops / (gemm_tflops * 1e12)
    kda_ops = tokens * KDA_LAYERS * math.ceil(HEADS / nodes) * HEAD_DIM * HEAD_DIM * 6
    kda_s = kda_ops / (kda_gops * 1e9)
    # Causal MLA: sum_{t=1}^T t dot/update work, distributed by head.
    pairs = tokens * (tokens + 1) / 2.0
    mla_ops = MLA_LAYERS * math.ceil(HEADS / nodes) * pairs * 2.0 * (192 + 128)
    attention_s = mla_ops / (gemm_tflops * 0.35 * 1e12)
    chunks = math.ceil(tokens / chunk)
    comm_chunk_s, calls_per_chunk = collective_seconds(
        nodes, chunk, moe_collectives, latency_us, link_gbps)
    comm_s = comm_chunk_s * chunks
    total = gemm_s + kda_s + attention_s + comm_s
    return {
        "tokens": tokens, "chunk": chunk, "gemm_s": gemm_s, "kda_s": kda_s,
        "mla_attention_s": attention_s, "comm_s": comm_s,
        "collective_calls": calls_per_chunk * chunks, "total_s": total,
        "tokens_per_second": tokens / total,
    }


def fmt_ctx(n: int) -> str:
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n // (1024*1024)}m"


def report(args: argparse.Namespace, split: WeightSplit) -> dict:
    result = {"assumptions": vars(args).copy(), "weights": split._asdict(), "memory": [],
              "decode": [], "prefill": []}
    print("Kimi K3 / A64FX performance model")
    print("=" * 72)
    print(f"manifest: {split.source} ({split.shards} shards, {split.tensors:,} tensors)")
    print(f"text weights: {split.total_text:.3f} GB = experts {split.experts:.3f} + "
          f"replicated {split.replicated:.3f} + TP {split.shardable:.3f}")
    print(f"calibration: dense-BW={args.bw_gbps:g} GB/s, head-slice-BW={args.head_bw_gbps:g} GB/s, "
          f"MXFP4-BW={args.mxfp4_gbps:g} GB/s, "
          f"expert-M1={args.expert_ms:g} ms, "
          f"KDA={args.kda_gops:g} GOP/s, "
          f"GEMM={args.gemm_tflops:g} TF/s, collective={args.latency_us:g} us/step")
    print("\nMemory (fullest rank, expanded BF16 MLA cache)")
    print(f"{'ctx':>5} {'M':>3} {'weights':>8} {'KDA':>7} {'MLA-KV':>8} {'total':>8} {'fit27':>6}")
    for ctx in args.contexts:
        for batch in args.batches:
            m = memory(split, args.nodes, ctx, batch, args.kv_bytes, args.usable_gb)
            md = m._asdict()
            md.update({"context": ctx, "batch": batch})
            result["memory"].append(md)
            print(f"{fmt_ctx(ctx):>5} {batch:3d} {m.weights_gb:8.2f} {m.kda_state_gb:7.2f} "
                  f"{m.mla_cache_gb:8.2f} {m.total_gb:8.2f} {'yes' if m.fits else 'NO':>6}")
    _, calls = collective_seconds(args.nodes, 1, args.moe_collectives,
                                  args.latency_us, args.link_gbps)
    print(f"\nDecode (batch tokens/s; {calls} collectives/layer-stack)")
    print(f"{'ctx':>5} {'M':>3} {'W ms':>7} {'Exp ms':>7} {'KV ms':>7} {'KDA':>7} {'comm':>7} {'tok/s':>9}")
    for ctx in args.contexts:
        for batch in args.batches:
            d = decode(split,args.nodes,ctx,batch,args.bw_gbps,args.head_bw_gbps,
                       args.mxfp4_gbps,args.kda_gops,
                       args.latency_us,args.link_gbps,args.imbalance,
                       args.expert_ms,args.expert_samples,args.moe_collectives)
            result["decode"].append(d)
            print(f"{fmt_ctx(ctx):>5} {batch:3d} {d['weight_ms']:7.1f} {d['expert_ms']:7.1f} "
                  f"{d['cache_ms']:7.1f} {d['kda_ms']:7.1f} {d['comm_ms']:7.1f} {d['tokens_per_second']:9.2f}")
    print("\nPrefill (tokens/s)")
    print(f"{'prompt':>7} {'chunk':>6} {'GEMM s':>9} {'KDA s':>9} {'MLA s':>9} {'comm s':>9} {'tok/s':>9}")
    for tokens in args.prompts:
        for chunk in args.chunks:
            p = prefill(split,args.nodes,tokens,chunk,args.gemm_tflops,args.kda_gops,
                        args.latency_us,args.link_gbps,args.moe_collectives)
            result["prefill"].append(p)
            print(f"{fmt_ctx(tokens):>7} {chunk:6d} {p['gemm_s']:9.1f} {p['kda_s']:9.1f} "
                  f"{p['mla_attention_s']:9.1f} {p['comm_s']:9.1f} {p['tokens_per_second']:9.1f}")
    print("\nCaveat: 128k/1m exact runtime needs context-parallel MLA; v1 only models it. "
          "Predictions are engineering estimates, not measured end-to-end results.")
    return result


def csv_ints(text: str):
    return [int(x) for x in text.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path.home() / "models/kimi-k3")
    p.add_argument("--no-manifest", action="store_true", help="use documented fallback byte totals")
    p.add_argument("--nodes", type=int, default=96)
    p.add_argument("--contexts", type=csv_ints, default=csv_ints("4096,131072,1048576"))
    p.add_argument("--batches", type=csv_ints, default=csv_ints("1,8,32"))
    p.add_argument("--prompts", type=csv_ints, default=csv_ints("1024,8192,131072,1048576"))
    p.add_argument("--chunks", type=csv_ints, default=csv_ints("64,256,1024"))
    p.add_argument("--usable-gb", type=float, default=USABLE_GB)
    p.add_argument("--kv-bytes", type=int, choices=(1,2), default=2)
    p.add_argument("--bw-gbps", type=float, default=336.0, help="effective per-rank decode bandwidth")
    p.add_argument("--head-bw-gbps", type=float, default=83.6,
                   help="measured cache-evicted one-head BF16 projection bandwidth")
    p.add_argument("--mxfp4-gbps", type=float, default=180.0,
                   help="48-core MXFP4 bandwidth; extrapolated from the measured 3.75 GB/s/core")
    p.add_argument("--expert-ms", type=float, default=0.220,
                   help="measured real MXFP4 M=1 expert latency")
    p.add_argument("--expert-samples", type=int, default=1000,
                   help="deterministic routing Monte Carlo samples per batch")
    p.add_argument("--moe-collectives", type=int, choices=(0,1,2,3), default=2,
                   help="MoE collectives/layer: current exact path is 2")
    p.add_argument("--kda-gops", type=float, default=11.35,
                   help="12-node mean one-head KDA rate at the optimal 8 threads")
    p.add_argument("--gemm-tflops", type=float, default=1.25, help="assumed per-rank BF16-equivalent GEMM")
    p.add_argument("--latency-us", type=float, default=20.0, help="assumed allreduce latency per log2 step")
    p.add_argument("--link-gbps", type=float, default=8.0)
    p.add_argument("--imbalance", type=float, default=1.20, help="critical-rank routed-expert traffic factor")
    p.add_argument("--json", type=Path, help="also write full results as JSON")
    args = p.parse_args()
    if args.nodes < 1 or args.nodes > HEADS:
        p.error("--nodes must be in [1,96] for head TP")
    split = fallback_weights() if args.no_manifest else scan_weights(args.model_dir)
    result = report(args, split)
    if args.json:
        result["assumptions"]["model_dir"] = str(args.model_dir)
        result["assumptions"]["json"] = str(args.json)
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
