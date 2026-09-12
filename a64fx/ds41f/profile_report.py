#!/usr/bin/env python3
"""Report the 12-rank decode critical path from bounded runner samples.

Owner spans exclude other ranks' producer waits. Parallel stages use the
maximum local compute+reduction interval; their remainder includes rendezvous
and rank skew, not just wire time. Compare the reconstructed path with TOKEN
to assess this approximation. Nested operator spans are reported separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def report(directory, start, stop):
    meta = [json.loads((directory / f"profile.rank{r:02d}.json").read_text())
            for r in range(12)]
    first = meta[0]
    for rank, item in enumerate(meta):
        if item != dict(first, rank=rank):
            raise ValueError(f"inconsistent metadata for rank {rank}")
    if first["version"] != 1 or first["layers"] != 41 or first["dtype"] != "<f8":
        raise ValueError("unsupported profile format")
    positions = np.arange(first["start"], first["start"] + first["positions"])
    mask = (positions >= start) & (positions < stop)
    if not mask.any():
        raise ValueError("no samples in requested position range")
    shape = (first["positions"], 41, len(first["phases"]))
    data = np.stack([np.fromfile(directory / f"profile.rank{r:02d}.bin", dtype="<f8")
                     .reshape(shape)[mask] for r in range(12)])
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("nonfinite or negative measurements")
    index = {name: i for i, name in enumerate(first["phases"])}

    def phase(name):
        return data[..., index[name]]

    def all_ranks(name):
        return phase(name).sum(axis=(0, 2))

    def owner(name):
        x = phase(name)
        return sum(x[layer % 12, :, layer] for layer in range(40))

    def mean_ms(x):
        return float(np.mean(x) * 1000)

    def stats(x):
        return dict(mean_ms=mean_ms(x), p50_ms=float(np.percentile(x, 50) * 1000),
                    p95_ms=float(np.percentile(x, 95) * 1000))

    token = phase("TOKEN")[0, :, 40]
    critical = {}
    for name in ("EMBED", "ENGRAM_PROJECT", "HC_ATTN_MIX", "HC_ATTN_PRE", "ATTENTION",
                 "HC_ATTN_POST", "HC_FFN_MIX", "HC_FFN_PRE", "GATE", "SHARED_EXPERT",
                 "HC_FFN_POST", "HEAD_PRE", "HEAD_LINEAR", "HEAD_SELECT"):
        critical[name] = all_ranks(name)
    for name in ("ATTN_SYNC", "FFN_BCAST", "RESIDUAL_BCAST"):
        critical[name] = owner(name)
    critical["EMBED_BCAST"] = phase("EMBED_BCAST")[0, :, 0]
    critical["NEXT_BCAST"] = phase("NEXT_BCAST")[11, :, 40]
    for compute, combine in (("EXPERTS", "EXPERT_SUM"), ("ENGRAM_IO", "ENGRAM_SUM")):
        work = phase(compute)[:, :, :40]
        label = compute
        if compute == "EXPERTS" and "SHARED_OVERLAP" in index:
            shared = phase("SHARED_OVERLAP")[:, :, :40]
            if shared.any():
                work = work + shared
                label = "EXPERTS_AND_SHARED"
        collective = phase(combine)[:, :, :40]
        critical[label] = work.max(axis=0).sum(axis=1)
        critical[combine + "_RENDEZVOUS"] = ((work + collective).max(axis=0)
                                               - work.max(axis=0)).sum(axis=1)
    critical["UNATTRIBUTED"] = token - sum(critical.values())
    attention = {name: stats(phase(name).max(axis=0).sum(axis=1)) for name in first["phases"]
                 if name.startswith("ATTN_") and name != "ATTN_SYNC"}
    indexer = {name: stats(all_ranks(name)) for name in ("INDEX_QUERY", "INDEX_SCORE", "INDEX_SELECT")}
    kernels = {}
    for kind, duration in (("FP8", all_ranks("LINEAR_FP8")), ("BF16", all_ranks("LINEAR_BF16")),
                           ("F32", all_ranks("LINEAR_F32")),
                           ("FP4", all_ranks("EXPERT_W13") + all_ranks("EXPERT_W2"))):
        size = all_ranks(kind + "_BYTES")
        kernels[kind] = dict(aggregate_ms=mean_ms(duration),
                             weight_GB_per_token=float(size.mean() / 1e9),
                             effective_GB_s=(float(size.sum() / duration.sum() / 1e9)
                                             if duration.sum() else None))
    if "LINEAR_INT8" in index:
        duration, size = all_ranks("LINEAR_INT8"), all_ranks("INT8_BYTES")
        kernels["INT8"] = dict(aggregate_ms=mean_ms(duration),
                               weight_GB_per_token=float(size.mean() / 1e9),
                               effective_GB_s=(float(size.sum() / duration.sum() / 1e9)
                                               if duration.sum() else None))
    experts = phase("EXPERTS")[:, :, :40]
    slowest = experts.argmax(axis=0)
    expert_parts = {}
    for name in ("EXPERT_QUANT", "EXPERT_W13", "EXPERT_SWIGLU", "EXPERT_W2", "EXPERT_ROUND"):
        x = np.take_along_axis(phase(name)[:, :, :40], slowest[None], axis=0)[0]
        expert_parts[name] = stats(x.sum(axis=1))
    counts = phase("EXPERT_COUNT")[:, :, :40]
    if not np.all(counts.sum(axis=0) == 6):
        raise ValueError("expected six routed experts per token/layer")
    engram_parts = {}
    slowest_engram = phase("ENGRAM_IO")[:, :, :40].argmax(axis=0)
    for name in ("ENGRAM_READ", "ENGRAM_DECODE"):
        if name in index:
            x = np.take_along_axis(phase(name)[:, :, :40], slowest_engram[None], axis=0)[0]
            engram_parts[name] = stats(x.sum(axis=1))
    result = dict(directory=str(directory), first_position=int(positions[mask][0]),
                  last_position=int(positions[mask][-1]), samples=int(mask.sum()),
                  token=dict(stats(token), tokens_per_second=float(1 / token.mean())),
                  critical_path={name: dict(stats(x), percent=float(x.mean() / token.mean() * 100))
                                 for name, x in critical.items()},
                  attention_inclusive=attention, indexer_inclusive=indexer,
                  sparse_components={name: stats(phase(name).max(axis=0).sum(axis=1)) for name in
                                       ("SPARSE_QK", "SPARSE_SOFTMAX", "SPARSE_PV") if name in index},
                  expert_slowest_rank_components=expert_parts,
                  engram_slowest_rank_components=engram_parts,
                  engram_prefetch_overlap=(stats(phase("ENGRAM_PREFETCH").max(axis=0).sum(axis=1))
                                           if "ENGRAM_PREFETCH" in index else None),
                  kernels_aggregate=kernels,
                  expert_placement=dict(mean_max_experts_per_layer=float(counts.max(axis=0).mean()),
                                        mean_active_ranks_per_layer=float((counts > 0).sum(axis=0).mean()),
                                        counts_per_rank_per_token=counts.sum(axis=2).mean(axis=1).tolist()),
                  nested_other={name: stats(all_ranks(name)) for name in
                                ("LINEAR_QUANT", "LINEAR_ROUND", "INT8_INPUT_QUANT", "NORM", "HC_NORM", "HC_MATVEC", "HC_SPLIT")
                                if name in index},
                  tp_group=(int(phase("TP_GROUP").max()) if "TP_GROUP" in index else 1),
                  tp_communication_owner=(stats(owner("TP_COMM")) if "TP_COMM" in index else None),
                  tp_communication=(stats(phase("TP_COMM").max(axis=0).sum(axis=1)) if "TP_COMM" in index else None),
                  caveat="Reduction remainders include rendezvous/skew; nested spans overlap. Sparse baseline components are maximum worker work durations; tiled components include phase barriers. TP attention components use the maximum shard duration per layer.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--stop", type=int, default=2**63 - 1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.start, args.stop)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Positions {result['first_position']}..{result['last_position']}: {result['samples']} samples")
    print(json.dumps(result["token"], sort_keys=True))
    print("Critical path (ms/token, percent; reduction includes rendezvous/skew):")
    for name, value in sorted(result["critical_path"].items(), key=lambda item: -item[1]["mean_ms"]):
        print(f"  {name:28s} {value['mean_ms']:9.3f} {value['percent']:7.2f}%")
    for category in ("attention_inclusive", "sparse_components", "indexer_inclusive", "expert_slowest_rank_components",
                     "engram_slowest_rank_components", "engram_prefetch_overlap", "kernels_aggregate", "expert_placement", "nested_other"):
        print(category + ": " + json.dumps(result[category], sort_keys=True))


if __name__ == "__main__":
    main()
