#!/usr/bin/env python3
"""Materialize one GGUF MLA layer's K/V compressed projections as BF16.

GGUF dimensions are [contiguous, rows, heads].  K is stored as
[128,512,96] and therefore needs a per-head transpose; V is [512,128,96]
and is already in native matvec order.  The operation is bounded to one
layer and one head plane at a time.
"""
from __future__ import print_function

import argparse
import os
import struct
from pathlib import Path

from k3_gguf_stage import discover

def f16(b):
    return struct.unpack("<e", b)[0]

def bf16(x):
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    # round-to-nearest-even before truncating the low 16 bits
    u += 0x7fff + ((u >> 16) & 1)
    return (u >> 16) & 0xffff

def q8_row(data):
    scale = f16(data[:2])
    return [scale * struct.unpack("<b", data[2 + i:3 + i])[0]
            for i in range(32)]

def read_q8_row(data, off, cols):
    out = []
    for b in range(0, cols, 32):
        out.extend(q8_row(data[off + (b // 32) * 34:off + (b // 32 + 1) * 34]))
    return out

def materialize(rec, out_path, transpose):
    cols, rows, heads = rec["dims"]
    if rec["type"] != "Q8_0":
        raise ValueError("%s is %s, expected Q8_0" % (rec["name"], rec["type"]))
    if transpose and (cols, rows) != (128, 512):
        raise ValueError("K shape %s is not [128,512,heads]" % (rec["dims"],))
    if not transpose and (cols, rows) != (512, 128):
        raise ValueError("V shape %s is not [512,128,heads]" % (rec["dims"],))
    row_bytes = rec["row_bytes"]
    with open(rec["source"], "rb", buffering=0) as src, open(out_path, "wb") as dst:
        for h in range(heads):
            base = rec["data_start"] + h * rows * row_bytes
            src.seek(base)
            plane_data = src.read(rows * row_bytes)
            if len(plane_data) != rows * row_bytes:
                raise IOError("short read for %s head=%d" % (rec["name"], h))
            if transpose:
                plane = [read_q8_row(plane_data, i * row_bytes, cols)
                         for i in range(rows)]
                for o in range(cols):
                    dst.write(struct.pack("<%dH" % rows,
                                          *(bf16(plane[i][o]) for i in range(rows))))
            else:
                for i in range(rows):
                    row = read_q8_row(plane_data, i * row_bytes, cols)
                    dst.write(struct.pack("<%dH" % cols,
                                          *(bf16(x) for x in row)))

def split_range(total, rank, nodes):
    base, rem = divmod(total, nodes)
    return rank * base + min(rank, rem), base + (rank < rem)

def materialize_combined_into(krec, vrec, dst, head_start=0, head_count=None,
                              dtype="BF16"):
    """Write native [heads*256, 512] rows: K-head then V-head."""
    if krec["type"] != "Q8_0" or vrec["type"] != "Q8_0":
        raise ValueError("MLA K/V materialization currently requires Q8_0")
    if tuple(krec["dims"]) != (128, 512, 96) or tuple(vrec["dims"]) != (512, 128, 96):
        raise ValueError("unexpected MLA K/V shapes")
    kcols, krows, heads = krec["dims"]
    vcols, vrows, vheads = vrec["dims"]
    if heads != vheads:
        raise ValueError("K/V head count mismatch")
    if head_count is None:
        head_count = heads - head_start
    if head_start < 0 or head_count < 0 or head_start + head_count > heads:
        raise ValueError("invalid head range")
    if dtype not in ("BF16", "F32"):
        raise ValueError("unsupported MLA materialization dtype %s" % dtype)

    def write_row(values):
        if dtype == "BF16":
            dst.write(struct.pack("<%dH" % len(values),
                                  *(bf16(x) for x in values)))
        else:
            dst.write(struct.pack("<%df" % len(values), *values))

    with open(krec["source"], "rb", buffering=0) as ksrc, \
         open(vrec["source"], "rb", buffering=0) as vsrc:
        for h in range(head_start, head_start + head_count):
            kb = krec["data_start"] + h * krows * krec["row_bytes"]
            vb = vrec["data_start"] + h * vrows * vrec["row_bytes"]
            ksrc.seek(kb)
            vsrc.seek(vb)
            kdata = ksrc.read(krows * krec["row_bytes"])
            vdata = vsrc.read(vrows * vrec["row_bytes"])
            if len(kdata) != krows * krec["row_bytes"] or len(vdata) != vrows * vrec["row_bytes"]:
                raise IOError("short MLA plane read at head=%d" % h)
            kplane = [read_q8_row(kdata, i * krec["row_bytes"], kcols)
                      for i in range(krows)]
            for o in range(kcols):
                write_row([kplane[i][o] for i in range(krows)])
            for i in range(vrows):
                row = read_q8_row(vdata, i * vrec["row_bytes"], vcols)
                write_row(row)

def materialize_combined(krec, vrec, out_path, head_start=0, head_count=None,
                         dtype="BF16"):
    """Materialize native combined K/V rows to a standalone file."""
    with open(out_path, "wb") as dst:
        materialize_combined_into(krec, vrec, dst, head_start, head_count,
                                  dtype)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("layer", type=int)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="also retain separate all-head K/V BF16 debug files")
    args = ap.parse_args()
    if args.layer not in {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
                          51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}:
        raise ValueError("layer %d is not MLA" % args.layer)
    if args.nodes <= 0 or args.rank < 0 or args.rank >= args.nodes or 96 % args.nodes:
        raise ValueError("rank/nodes must describe a partition of 96 heads")
    _, records, _ = discover(args.model_dir)
    by_name = {r["name"]: r for r in records}
    k = by_name["blk.%d.attn_k_b.weight" % args.layer]
    v = by_name["blk.%d.attn_v_b.weight" % args.layer]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_start, head_count = split_range(96, args.rank, args.nodes)
    kp = args.output_dir / ("layer%02d.kv_k.bf16" % args.layer)
    vp = args.output_dir / ("layer%02d.kv_v.bf16" % args.layer)
    cp = args.output_dir / ("layer%02d.kv_b.native.bf16" % args.layer)
    mp = args.output_dir / ("layer%02d.kv_b.native.manifest" % args.layer)
    if args.keep_intermediates:
        materialize(k, kp, True)
        materialize(v, vp, False)
    materialize_combined(k, v, cp, head_start, head_count)
    with open(mp, "w") as mf:
        mf.write("# K3FULLV2 mode=mla-materialized rank=%d nodes=%d layer_index=%d tensors=1 blob_bytes=%d\n" %
                 (args.rank, args.nodes, args.layer, cp.stat().st_size))
        mf.write("0 %d BF16 2 %d 512 self_attn.kv_b_proj.weight\n" %
                 (cp.stat().st_size, head_count * 256))
    print("K3_GGUF_MLA_MATERIALIZE PASS layer=%d rank=%d/%d heads=%d+%d k=%s bytes=%d v=%s bytes=%d combined=%s bytes=%d shape=[%d,512]" %
          (args.layer, args.rank, args.nodes, head_start, head_count,
           kp if args.keep_intermediates else "-",
           kp.stat().st_size if args.keep_intermediates else 0,
           vp if args.keep_intermediates else "-",
           vp.stat().st_size if args.keep_intermediates else 0,
           cp, cp.stat().st_size, head_count * 256))

if __name__ == "__main__":
    main()
