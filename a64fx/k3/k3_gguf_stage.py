#!/usr/bin/env python3
"""Plan and stream-stage K3 GGUF shards.

GGUF stores a matrix as [contiguous_columns, rows, ...].  The emitted
manifest keeps that storage order and records the local row/expert slice;
the C runner therefore sees one contiguous quantized row at a time.  No
payload is mmap'ed or retained in Python memory.
"""
from __future__ import print_function

import argparse
import ctypes
import json
import os
import struct
import sys
from pathlib import Path

ALIGN = 256
CHUNK = 8 * 1024 * 1024
TYPE_INFO = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    30: ("BF16", 1, 2),
}


def read_u(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, b)[0]


def read_str(f):
    n = read_u(f, "<Q")
    if n > 1024 * 1024:
        raise ValueError("unreasonable GGUF string length")
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF string")
    return b.decode("utf-8")


def skip_value(f, typ):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
             7: 1, 10: 8, 11: 8, 12: 8}
    if typ in sizes:
        f.seek(sizes[typ], os.SEEK_CUR)
    elif typ == 8:
        f.seek(read_u(f, "<Q"), os.SEEK_CUR)
    elif typ == 9:
        elem = read_u(f, "<I")
        count = read_u(f, "<Q")
        for _ in range(count):
            skip_value(f, elem)
    else:
        raise ValueError("unsupported GGUF metadata type %d" % typ)


def read_header(path):
    """Return metadata and tensor records, with absolute file offsets."""
    with open(str(path), "rb", buffering=0) as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version = read_u(f, "<I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version %d" % version)
        nt = read_u(f, "<Q")
        nk = read_u(f, "<Q")
        if nt > 1000000 or nk > 100000:
            raise ValueError("unreasonable GGUF counts")
        meta = {}
        for _ in range(nk):
            key = read_str(f)
            typ = read_u(f, "<I")
            if key == "general.alignment" and typ == 10:
                meta[key] = read_u(f, "<Q")
            else:
                if key == "split.count" and typ == 10:
                    # Retain this common field for split validation.
                    meta[key] = None
                    meta[key] = read_u(f, "<Q")
                else:
                    skip_value(f, typ)
        infos = []
        for _ in range(nt):
            name = read_str(f)
            nd = read_u(f, "<I")
            if nd > 4:
                raise ValueError("unsupported tensor rank %d: %s" % (nd, name))
            dims = [read_u(f, "<Q") for _ in range(nd)]
            typ = read_u(f, "<I")
            off = read_u(f, "<Q")
            if typ not in TYPE_INFO:
                raise ValueError("unsupported GGUF tensor type %d: %s" % (typ, name))
            infos.append({"name": name, "dims": dims, "type_id": typ, "offset": off})
        alignment = int(meta.get("general.alignment", 32))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        size = path.stat().st_size
        for rec in infos:
            block, type_bytes = TYPE_INFO[rec["type_id"]][1:]
            if rec["dims"]:
                if rec["dims"][0] % block:
                    raise ValueError("dimension is not quantization-block aligned: %s" % rec["name"])
                row_bytes = (rec["dims"][0] + block - 1) // block * type_bytes
                rows = 1
                for d in rec["dims"][1:]:
                    rows *= d
                nbytes = row_bytes * rows
            else:
                row_bytes, nbytes = type_bytes, type_bytes
            rec.update(source=str(path), data_start=data_start + rec["offset"],
                       row_bytes=row_bytes, nbytes=nbytes,
                       type=TYPE_INFO[rec["type_id"]][0])
            if rec["data_start"] + nbytes > size:
                raise ValueError("tensor extends past shard: %s" % rec["name"])
        return meta, infos


def discover(model_dir):
    paths = sorted(Path(model_dir).glob("*.gguf"))
    if not paths:
        raise ValueError("no GGUF shards in %s" % model_dir)
    all_records = []
    split_counts = set()
    for path in paths:
        meta, records = read_header(path)
        if "split.count" in meta:
            split_counts.add(meta["split.count"])
        all_records.extend(records)
    if len(split_counts) > 1:
        raise ValueError("inconsistent GGUF split.count values")
    names = [r["name"] for r in all_records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate tensor name across GGUF shards")
    return paths, all_records, next(iter(split_counts), None)


def split_dim(total, rank, size):
    base, rem = divmod(total, size)
    return rank * base + min(rank, rem), base + (rank < rem)


def select_records(records, layer, include_global):
    prefix = "blk.%d." % layer
    out = [r for r in records if r["name"].startswith(prefix)]
    if include_global:
        out += [r for r in records if r["name"] in
                ("token_embd.weight", "output.weight", "output_norm.weight")]
    if not out:
        raise ValueError("no tensors found for layer %d" % layer)
    return sorted(out, key=lambda r: r["name"])


def segments_for(rec, rank, nodes):
    dims = rec["dims"]
    if len(dims) <= 1:
        return [(rec["data_start"], rec["nbytes"], list(dims))]
    if len(dims) == 2:
        first, count = split_dim(dims[1], rank, nodes)
        return [(rec["data_start"] + first * rec["row_bytes"],
                 count * rec["row_bytes"], [dims[0], count])]
    if len(dims) == 3:
        first, count = split_dim(dims[2], rank, nodes)
        plane = rec["row_bytes"] * dims[1]
        return [(rec["data_start"] + first * plane, count * plane,
                 [dims[0], dims[1], count])]
    raise ValueError("cannot shard rank-%d tensor %s" % (len(dims), rec["name"]))


def copy_range(src, dst, offset, length):
    src.seek(offset)
    left = length
    while left:
        n = min(left, CHUNK)
        buf = src.read(n)
        if len(buf) != n:
            raise IOError("short read from %s" % src.name)
        dst.write(buf)
        left -= n
    try:
        libc = ctypes.CDLL(None)
        libc.posix_fadvise(src.fileno(), ctypes.c_longlong(offset),
                           ctypes.c_longlong(length), 4)
    except Exception:
        pass


def stage(records, output_dir, rank, nodes, layer, include_global, force):
    selected = select_records(records, layer, include_global)
    entries = []
    blob_tmp = Path(output_dir) / ("rank%03d.blob.tmp" % rank)
    blob = Path(output_dir) / ("rank%03d.blob" % rank)
    manifest_tmp = Path(output_dir) / ("rank%03d.manifest.tmp" % rank)
    manifest = Path(output_dir) / ("rank%03d.manifest" % rank)
    if not force and blob.exists() and manifest.exists():
        return len(selected), blob.stat().st_size
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pos = 0
    with open(str(blob_tmp), "wb", buffering=0) as out:
        handles = {}
        try:
            for rec in selected:
                for source, length, shape in segments_for(rec, rank, nodes):
                    pos = (pos + ALIGN - 1) // ALIGN * ALIGN
                    if out.tell() < pos:
                        out.write(b"\0" * (pos - out.tell()))
                    if rec["source"] not in handles:
                        handles[rec["source"]] = open(rec["source"], "rb", buffering=0)
                    copy_range(handles[rec["source"]], out, source, length)
                    entries.append((pos, length, rec["type"], shape, rec["name"]))
                    pos += length
        finally:
            for f in handles.values():
                f.close()
        out.flush()
        os.fsync(out.fileno())
    entries.sort(key=lambda e: e[4])
    with open(str(manifest_tmp), "w") as f:
        f.write("# K3GGUFV1 mode=layer%d rank=%d nodes=%d layer_index=%d tensors=%d blob_bytes=%d\n" %
                (layer, rank, nodes, layer, len(entries), pos))
        for off, nbytes, typ, shape, name in entries:
            f.write("%d %d %s %d %s %s\n" %
                    (off, nbytes, typ, len(shape), " ".join(str(x) for x in shape), name))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(blob_tmp), str(blob))
    os.replace(str(manifest_tmp), str(manifest))
    return len(entries), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--format", choices=("iq1", "q2"), required=True)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.rank < args.nodes or args.nodes <= 0:
        ap.error("rank must be in [0,nodes)")
    paths, records, split_count = discover(args.model_dir)
    allowed = {"iq1": {"IQ1_S", "IQ2_XXS", "IQ3_XXS", "Q8_0", "F32", "BF16"},
               "q2": {"IQ2_XS", "IQ3_XXS", "Q8_0", "F32", "BF16"}}[args.format]
    bad = sorted({r["type"] for r in records if r["type"] not in allowed})
    if bad:
        raise ValueError("unexpected %s package tensor types: %s" % (args.format, bad))
    selected = select_records(records, args.layer_index, not args.no_global)
    summary = {"shards": [str(p) for p in paths], "split_count": split_count,
               "layer": args.layer_index, "rank": args.rank, "nodes": args.nodes,
               "tensors": len(selected), "types": sorted({r["type"] for r in selected})}
    if args.plan_only:
        print(json.dumps(summary, sort_keys=True))
        return 0
    n, size = stage(records, args.output_dir, args.rank, args.nodes,
                    args.layer_index, not args.no_global, args.force)
    summary.update(entries=n, blob_bytes=size, output_dir=args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, IOError) as exc:
        print("k3_gguf_stage: %s" % exc, file=sys.stderr)
        sys.exit(2)
