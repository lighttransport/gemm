#!/usr/bin/env python3
"""Stage intermediate-channel TP slices for GGUF MoE expert tensors.

GGUF expert tensors use [input_columns, intermediate_rows, experts].  A
TP-rank owns rows of w1/w3 and columns of w2 from every expert.  This keeps
the compressed representation intact and emits a manifest that a runner can
consume without reconstructing complete experts.
"""
from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path

from k3_gguf_stage import ALIGN, CHUNK, discover, copy_range, split_dim

EXPERT_TENSORS = {
    "w1": "ffn_up_exps.weight",
    "w2": "ffn_down_exps.weight",
    "w3": "ffn_gate_exps.weight",
}
ROUTER_NAME = "ffn_gate_inp.weight"
ROUTED_DOWN_NAME = "ffn_routed_down.weight"
ROUTED_NORM_NAME = "ffn_routed_norm.weight"
ROUTED_UP_NAME = "ffn_routed_up.weight"


def _expert_records(records, layer):
    prefix = "blk.%d." % layer
    found = {}
    router_matches = [r for r in records
                      if r["name"] == "blk.%d.%s" % (layer, ROUTER_NAME)]
    if len(router_matches) != 1:
        raise ValueError("expected one router tensor for layer %d" % layer)
    found["router"] = router_matches[0]
    down_matches = [r for r in records
                    if r["name"] == "blk.%d.%s" % (layer, ROUTED_DOWN_NAME)]
    if len(down_matches) != 1:
        raise ValueError("expected one routed-down tensor for layer %d" % layer)
    found["routed_down"] = down_matches[0]
    for role, suffix in EXPERT_TENSORS.items():
        matches = [r for r in records
                   if r["name"] == prefix + suffix]
        if len(matches) != 1:
            raise ValueError("expected one %s tensor for layer %d" % (role, layer))
        rec = matches[0]
        if len(rec["dims"]) != 3:
            raise ValueError("%s must be 3-D, got %s" % (rec["name"], rec["dims"]))
        found[role] = rec
    expert_values = [found[k] for k in ("w1", "w2", "w3")]
    if any(len(r["dims"]) != 3 or r["dims"][2] != 896 for r in expert_values):
        raise ValueError("expert axis must contain 896 experts")
    if found["w1"]["dims"][1] != found["w3"]["dims"][1]:
        raise ValueError("w1/w3 intermediate dimensions differ")
    if found["w2"]["dims"][0] != found["w1"]["dims"][1]:
        raise ValueError("w2 input dimension does not match w1/w3 rows")
    if len(found["router"]["dims"]) != 2 or found["router"]["dims"][1] != 896:
        raise ValueError("router must be [hidden,896]")
    if (len(found["routed_down"]["dims"]) != 2 or
            found["routed_down"]["dims"] != [7168, 3584]):
        raise ValueError("routed-down must be [7168,3584]")
    norm_matches = [r for r in records
                    if r["name"] == prefix + ROUTED_NORM_NAME]
    up_matches = [r for r in records
                  if r["name"] == prefix + ROUTED_UP_NAME]
    if len(norm_matches) != 1 or len(up_matches) != 1:
        raise ValueError("expected routed norm/up tensors for layer %d" % layer)
    found["routed_norm"], found["routed_up"] = norm_matches[0], up_matches[0]
    if found["routed_norm"]["dims"] != [3584]:
        raise ValueError("routed norm must be [3584]")
    if found["routed_up"]["dims"] != [3584, 7168]:
        raise ValueError("routed-up must be [3584,7168]")
    return found


def _segment(rec, role, rank, nodes, expert):
    cols, rows, experts = rec["dims"]
    if role in ("w1", "w3"):
        start, count = split_dim(rows, rank, nodes)
        yield {
            "role": role, "expert": expert, "kind": "rows",
            "row_start": start, "row_count": count,
            "col_start": 0, "col_count": cols,
            "source": rec["source"],
            "source_offset": rec["data_start"] + expert * rows * rec["row_bytes"] +
                             start * rec["row_bytes"],
            "nbytes": count * rec["row_bytes"],
            "type": rec["type"], "shape": [cols, count],
            "name": rec["name"],
        }
        return
    start, count = split_dim(cols, rank, nodes)
    block = 32 if rec["type"] == "Q8_0" else 256
    if start % block or count % block:
        raise ValueError("%s TP column slice is not quant-block aligned: %d+%d" %
                         (rec["name"], start, count))
    bytes_per_block = rec["row_bytes"] // (cols // block)
    slice_row_bytes = (count // block) * bytes_per_block
    yield {
        "role": role, "expert": expert, "kind": "cols",
        "row_start": 0, "row_count": rows,
        "col_start": start, "col_count": count,
        "source": rec["source"],
        "source_offset": rec["data_start"] + expert * rows * rec["row_bytes"] +
                         (start // block) * bytes_per_block,
        "source_row_stride": rec["row_bytes"],
        "source_prefix_bytes": (start // block) * bytes_per_block,
        "blob_row_bytes": slice_row_bytes,
        "nbytes": rows * slice_row_bytes,
        "type": rec["type"], "shape": [count, rows],
        "name": rec["name"],
    }


def make_plan(records, layer, rank, nodes, experts):
    if nodes <= 0 or rank < 0 or rank >= nodes:
        raise ValueError("rank must be in [0,nodes)")
    found = _expert_records(records, layer)
    out = []
    router = found["router"]
    start, count = split_dim(router["dims"][1], rank, nodes)
    out.append({
        "role": "router", "expert": -1, "kind": "rows",
        "row_start": start, "row_count": count,
        "col_start": 0, "col_count": router["dims"][0],
        "source": router["source"], "source_offset": router["data_start"] +
                   start * router["row_bytes"],
        "nbytes": count * router["row_bytes"], "type": router["type"],
        "shape": [router["dims"][0], count], "name": router["name"],
    })
    norm = found["routed_norm"]
    out.append({
        "role": "routed_norm", "expert": -1, "kind": "rows",
        "row_start": 0, "row_count": norm["dims"][0],
        "col_start": 0, "col_count": 1,
        "source": norm["source"], "source_offset": norm["data_start"],
        "nbytes": norm["dims"][0] * norm["row_bytes"], "type": norm["type"],
        "shape": norm["dims"], "name": norm["name"],
    })
    down = found["routed_down"]
    start, count = split_dim(down["dims"][1], rank, nodes)
    out.append({
        "role": "routed_down", "expert": -1, "kind": "rows",
        "row_start": start, "row_count": count,
        "col_start": 0, "col_count": down["dims"][0],
        "source": down["source"], "source_offset": down["data_start"] +
                   start * down["row_bytes"],
        "nbytes": count * down["row_bytes"], "type": down["type"],
        "shape": [down["dims"][0], count], "name": down["name"],
    })
    up = found["routed_up"]
    start, count = split_dim(up["dims"][1], rank, nodes)
    out.append({
        "role": "routed_up", "expert": -1, "kind": "rows",
        "row_start": start, "row_count": count,
        "col_start": 0, "col_count": up["dims"][0],
        "source": up["source"], "source_offset": up["data_start"] +
                   start * up["row_bytes"], "nbytes": count * up["row_bytes"],
        "type": up["type"], "shape": [up["dims"][0], count],
        "name": up["name"],
    })
    for expert in experts:
        if expert < 0 or expert >= 896:
            raise ValueError("expert must be in [0,896)")
        for role in ("w1", "w2", "w3"):
            out.extend(_segment(found[role], role, rank, nodes, expert))
    return out, found


def stage(plan, output_dir, rank, nodes, layer, force):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    blob_tmp = outdir / ("rank%03d.expert_tp.blob.tmp" % rank)
    blob = outdir / ("rank%03d.expert_tp.blob" % rank)
    manifest_tmp = outdir / ("rank%03d.expert_tp.manifest.tmp" % rank)
    manifest = outdir / ("rank%03d.expert_tp.manifest" % rank)
    if not force and blob.exists() and manifest.exists():
        return blob.stat().st_size
    pos = 0
    handles = {}
    entries = []
    try:
        with open(str(blob_tmp), "wb", buffering=0) as dst:
            for item in plan:
                pos = (pos + ALIGN - 1) // ALIGN * ALIGN
                if dst.tell() < pos:
                    dst.write(b"\0" * (pos - dst.tell()))
                src = item["source"]
                if src not in handles:
                    handles[src] = open(src, "rb", buffering=0)
                item = dict(item)
                item["blob_offset"] = pos
                if item["kind"] == "cols":
                    source = item["source_offset"] - item["source_prefix_bytes"]
                    total = item["row_count"] * item["source_row_stride"]
                    handles[src].seek(source)
                    plane = handles[src].read(total)
                    if len(plane) != total:
                        raise IOError("short read from %s" % src)
                    for row in range(item["row_count"]):
                        begin = row * item["source_row_stride"] + item["source_prefix_bytes"]
                        dst.write(plane[begin:begin + item["blob_row_bytes"]])
                    try:
                        libc = __import__("ctypes").CDLL(None)
                        libc.posix_fadvise(handles[src].fileno(),
                                          __import__("ctypes").c_longlong(source),
                                          __import__("ctypes").c_longlong(total), 4)
                    except Exception:
                        pass
                else:
                    copy_range(handles[src], dst, item["source_offset"],
                               item["nbytes"])
                entries.append(item)
                pos += item["nbytes"]
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    with open(str(manifest_tmp), "w") as f:
        f.write("# K3GGUFIQTP1 layer=%d rank=%d nodes=%d segments=%d blob_bytes=%d\n" %
                (layer, rank, nodes, len(entries), pos))
        f.write("# role expert kind row_start row_count col_start col_count "
                "blob_offset nbytes blob_row_bytes type\n")
        for item in entries:
            f.write("%s %d %s %d %d %d %d %d %d %d %s\n" % (
                item["role"], item["expert"], item["kind"],
                item["row_start"], item["row_count"], item["col_start"],
                item["col_count"], item["blob_offset"], item["nbytes"],
                item.get("blob_row_bytes", 0), item["type"]))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(blob_tmp), str(blob))
    os.replace(str(manifest_tmp), str(manifest))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--format", choices=("iq1", "q2"), default=None)
    ap.add_argument("--experts", default="all")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    experts = list(range(896)) if args.experts == "all" else [int(x) for x in args.experts.split(",")]
    _, records, _ = discover(Path(args.model_dir))
    if args.format:
        allowed = {
            "iq1": {"IQ1_S", "IQ2_XXS", "IQ3_XXS", "Q8_0", "F32", "BF16"},
            "q2": {"IQ2_XS", "IQ3_XXS", "Q8_0", "F32", "BF16"},
        }[args.format]
        bad = sorted({r["type"] for r in records if r["type"] not in allowed})
        if bad:
            raise ValueError("unexpected %s package tensor types: %s" %
                             (args.format, bad))
    plan, found = make_plan(records, args.layer, args.rank, args.nodes, experts)
    summary = {
        "format": "K3GGUFIQTP1", "layer": args.layer, "rank": args.rank,
        "nodes": args.nodes, "experts": len(experts), "segments": len(plan),
        "types": sorted({item["type"] for item in plan}),
        "w1_shape": found["w1"]["dims"], "w2_shape": found["w2"]["dims"],
        "w3_shape": found["w3"]["dims"],
        "routed_up_shape": found["routed_up"]["dims"],
    }
    if args.plan_only:
        print(json.dumps(summary, sort_keys=True))
        return 0
    summary["blob_bytes"] = stage(plan, args.output_dir, args.rank,
                                   args.nodes, args.layer, args.force)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, IOError) as exc:
        print("k3_gguf_expert_tp_stage: %s" % exc, file=sys.stderr)
        sys.exit(2)
