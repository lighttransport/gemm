#!/usr/bin/env python3
"""Stage one complete Kimi K3 TP/EP rank image.

The output is a single bounded blob plus a text manifest understood by the
C11 full runner.  Only safetensor headers are retained in Python memory; all
payload copies use bounded pread/write loops and source page-cache eviction.

Ownership is deliberately explicit:
  * expert E in every MoE layer belongs to rank E % TP;
  * KDA/MLA attention heads are tensor-parallel by head;
  * shared-expert and dense-FFN intermediate rows are tensor-parallel;
  * the embedding is replicated for arbitrary token lookup;
  * the LM head is vocabulary-row sharded.
"""
from __future__ import print_function

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

ALIGN = 256
CHUNK = 8 * 1024 * 1024
HIDDEN = 7168
HEADS = 96
KDA_HEADS = 96
HEAD_DIM = 128
LAYERS = 93
EXPERTS = 896
MOE_INTER = 3072
LATENT = 3584
SHARED_INTER = 6144
DENSE_INTER = 33792
VOCAB = 163840

EXPERT_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$"
)


def available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent")


def read_header(path):
    with open(str(path), "rb", buffering=0) as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensor header: %s" % path)
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 512 * 1024 * 1024:
            raise ValueError("invalid safetensor header length %d: %s" % (n, path))
        payload = f.read(n)
        if len(payload) != n:
            raise ValueError("short safetensor JSON header: %s" % path)
    return 8 + n, json.loads(payload.decode("utf-8"))


def shards(model_dir):
    paths = sorted(model_dir.glob("model-*.safetensors"))
    if len(paths) != 96:
        raise ValueError("expected 96 model shards, found %d" % len(paths))
    out = []
    for path in paths:
        data_start, header = read_header(path)
        size = path.stat().st_size
        records = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            offsets = info.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError("bad data_offsets for %s" % name)
            begin, end = offsets
            if begin < 0 or end < begin or data_start + end > size:
                raise ValueError("out-of-file tensor offsets for %s" % name)
            records[name] = {
                "source": str(path),
                "source_offset": data_start + begin,
                "nbytes": end - begin,
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
            }
        out.append(records)
    return out


def locate(all_headers, name):
    found = None
    for records in all_headers:
        rec = records.get(name)
        if rec is not None:
            if found is not None:
                raise ValueError("tensor appears in multiple shards: %s" % name)
            found = dict(rec)
    if found is None:
        raise ValueError("missing checkpoint tensor: %s" % name)
    found["name"] = name
    return found


def split_dim(total, rank, size):
    base, rem = divmod(total, size)
    first = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return first, count


def split_group_dim(total, rank, size, group):
    """Partition total elements without splitting fixed-size kernel groups."""
    if total % group:
        raise ValueError("dimension %d is not divisible by group %d" % (total, group))
    first_group, group_count = split_dim(total // group, rank, size)
    return first_group * group, group_count * group


def copy_record(rec, mode, rank, size):
    """Return one or more source ranges for a logical row/column slice."""
    shape = rec["shape"]
    dtype_size = {"BF16": 2, "F32": 4, "F16": 2, "U8": 1}.get(rec["dtype"])
    if dtype_size is None:
        raise ValueError("unsupported dtype %s in %s" % (rec["dtype"], rec["name"]))
    if mode == "full":
        return [dict(rec, segments=[(rec["source_offset"], rec["nbytes"])])]
    if mode == "rows":
        if not shape:
            raise ValueError("cannot row-slice scalar %s" % rec["name"])
        r0, nr = split_dim(shape[0], rank, size)
        row_elems = 1
        for dim in shape[1:]:
            row_elems *= dim
        row_bytes = row_elems * dtype_size
        out = dict(rec)
        out["shape"] = [nr] + list(shape[1:])
        out["segments"] = [(rec["source_offset"] + r0 * row_bytes, nr * row_bytes)]
        out["nbytes"] = nr * row_bytes
        return [out]
    if mode == "rows-group":
        if not shape or shape[0] % 8:
            raise ValueError("group-row slicing requires rows divisible by 8: %s" % rec["name"])
        r0, nr = split_group_dim(shape[0], rank, size, 8)
        row_elems = 1
        for dim in shape[1:]:
            row_elems *= dim
        row_bytes = row_elems * dtype_size
        out = dict(rec)
        out["shape"] = [nr] + list(shape[1:])
        out["segments"] = [(rec["source_offset"] + r0 * row_bytes, nr * row_bytes)]
        out["nbytes"] = nr * row_bytes
        return [out]
    if mode == "cols":
        if len(shape) != 2:
            raise ValueError("column slicing requires a matrix: %s" % rec["name"])
        rows, cols = shape
        c0, nc = split_dim(cols, rank, size)
        row_bytes = cols * dtype_size
        out = dict(rec)
        out["shape"] = [rows, nc]
        out["segments"] = [
            (rec["source_offset"] + r * row_bytes + c0 * dtype_size, nc * dtype_size)
            for r in range(rows)
        ]
        out["nbytes"] = rows * nc * dtype_size
        return [out]
    raise ValueError("unknown slice mode %s" % mode)


def add_record(all_headers, records, name, mode="full", rank=0, size=1):
    rec = locate(all_headers, name)
    records.extend(copy_record(rec, mode, rank, size))


def copy_head_rows(rec, rank, size, head_dim, heads=HEADS):
    """Slice matrix/vector leading rows without splitting attention heads."""
    first_head, local_heads = split_dim(heads, rank, size)
    first = first_head * head_dim
    count = local_heads * head_dim
    shape = rec["shape"]
    row_elems = 1
    for dim in shape[1:]:
        row_elems *= dim
    row_bytes = row_elems * {"BF16": 2, "F32": 4}[rec["dtype"]]
    out = dict(rec)
    out["shape"] = [count] + list(shape[1:])
    out["segments"] = [(rec["source_offset"] + first * row_bytes,
                         count * row_bytes)]
    out["nbytes"] = count * row_bytes
    return out


def copy_head_cols(rec, rank, size, head_dim, heads=HEADS):
    """Slice matrix columns in complete attention-head units."""
    rows, cols = rec["shape"]
    first_head, local_heads = split_dim(heads, rank, size)
    first = first_head * head_dim
    count = local_heads * head_dim
    dtype_size = {"BF16": 2, "F32": 4}[rec["dtype"]]
    row_bytes = cols * dtype_size
    out = dict(rec)
    out["shape"] = [rows, count]
    # Use the blocked rectangular-copy descriptor.  A tuple per row would
    # reopen the same shard thousands of times for every projection.
    out["segments"] = ("rows", rec["source_offset"], rows, row_bytes,
                       first * dtype_size, count * dtype_size)
    out["nbytes"] = rows * count * dtype_size
    return out


def is_mla(layer):
    # Config lists are one-based; checkpoint layer names are zero-based.
    return (layer + 1) in set((4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
                               44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
                               84, 88, 92, 93))


def expert_tp_records(all_headers, prefix, rank, size, expert):
    """Return one intermediate-channel slice for one expert.

    MXFP4 payloads are stored with two packed values per byte and one scale per
    32 logical values.  The TP boundary is therefore always a 32-value block.
    """
    out = []
    ep = prefix + "block_sparse_moe.experts.%d." % expert
    first, local = split_dim(MOE_INTER, rank, size)
    if first % 32 or local % 32:
        raise ValueError("expert TP size must partition 32-channel blocks: rank=%d size=%d" %
                         (rank, size))
    for suffix in ("w1.weight_packed", "w1.weight_scale",
                   "w2.weight_packed", "w2.weight_scale",
                   "w3.weight_packed", "w3.weight_scale"):
        raw = locate(all_headers, ep + suffix)
        rec = dict(raw)
        if suffix.startswith("w1.") or suffix.startswith("w3."):
            row_bytes = raw["shape"][1]
            rec["shape"] = [local, raw["shape"][1]]
            rec["segments"] = [(raw["source_offset"] + first * row_bytes,
                                 local * row_bytes)]
            rec["nbytes"] = local * row_bytes
        elif suffix == "w2.weight_packed":
            packed_first = first // 2
            packed_local = local // 2
            row_bytes = raw["shape"][1]
            rec["shape"] = [raw["shape"][0], packed_local]
            # Keep a compact row-slice descriptor.  Materializing one tuple
            # per row for all 896 experts would consume many GiB before the
            # first byte is staged.
            rec["segments"] = ("rows", raw["source_offset"], raw["shape"][0],
                                row_bytes, packed_first, packed_local)
            rec["nbytes"] = raw["shape"][0] * packed_local
        else:
            scale_first = first // 32
            scale_local = local // 32
            row_bytes = raw["shape"][1]
            rec["shape"] = [raw["shape"][0], scale_local]
            rec["segments"] = ("rows", raw["source_offset"], raw["shape"][0],
                                row_bytes, scale_first, scale_local)
            rec["nbytes"] = raw["shape"][0] * scale_local
        rec["name"] = ep + suffix
        out.append(rec)
    return out


def make_plan(all_headers, rank, size, layer_indices=None, include_global=True,
              expert_tp=False, moe_shard_layout="replicated"):
    records = []

    if include_global:
        # Token lookup is collective: only the rank owning the vocabulary row keeps it.
        add_record(all_headers, records, "language_model.model.embed_tokens.weight",
                   "rows", rank, size)
        add_record(all_headers, records, "language_model.model.norm.weight")
        # Kimi applies one final model-level attention residual after the last
        # decoder layer and before the final RMSNorm.  These small full tensors
        # must be present on every rank.
        add_record(all_headers, records,
                   "language_model.model.output_attn_res_norm.weight")
        add_record(all_headers, records,
                   "language_model.model.output_attn_res_proj.weight")
        head = locate(all_headers, "language_model.lm_head.weight")
        records.extend(copy_record(head, "rows", rank, size))

    kda_full = {
        "self_attn.A_log", "self_attn.o_norm.weight",
    }
    kda_row_head = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.g_proj.weight",
        "self_attn.f_b_proj.weight", "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
    }
    kda_col_head = {"self_attn.o_proj.weight"}
    mla_full = {
        "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
    }
    mla_row_head = {"self_attn.q_b_proj.weight", "self_attn.kv_b_proj.weight",
                    "self_attn.g_proj.weight"}
    mla_col_head = {"self_attn.o_proj.weight"}

    if layer_indices is None:
        layer_indices = range(LAYERS)
    for layer in layer_indices:
        prefix = "language_model.model.layers.%d." % layer
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                       "self_attention_res_norm.weight", "self_attention_res_proj.weight",
                       "mlp_res_norm.weight", "mlp_res_proj.weight"):
            add_record(all_headers, records, prefix + suffix)

        if is_mla(layer):
            for suffix in sorted(mla_full):
                add_record(all_headers, records, prefix + suffix)
            # q_b has 192 rows/head; kv_b has 256 rows/head; g has 128 rows/head.
            for suffix in sorted(mla_row_head):
                rec = locate(all_headers, prefix + suffix)
                if suffix == "self_attn.q_b_proj.weight":
                    rec["shape"] = [192, rec["shape"][1]]
                    # q_b is row sliced in 192-row head units.
                    r0, nr = split_dim(96, rank, size)
                    rec["shape"] = [nr * 192, rec["shape"][1]]
                    raw = locate(all_headers, prefix + suffix)
                    row_bytes = raw["shape"][1] * 2
                    rec["segments"] = [(raw["source_offset"] + (r0 * 192) * row_bytes,
                                         nr * 192 * row_bytes)]
                    rec["nbytes"] = nr * 192 * row_bytes
                    records.append(rec)
                elif suffix == "self_attn.kv_b_proj.weight":
                    raw = locate(all_headers, prefix + suffix)
                    r0, nr = split_dim(96, rank, size)
                    row_bytes = raw["shape"][1] * 2
                    rec["shape"] = [nr * 256, raw["shape"][1]]
                    rec["segments"] = [(raw["source_offset"] + (r0 * 256) * row_bytes,
                                         nr * 256 * row_bytes)]
                    rec["nbytes"] = nr * 256 * row_bytes
                    records.append(rec)
                else:
                    raw = locate(all_headers, prefix + suffix)
                    records.extend(copy_record(raw, "rows", rank, size))
            for suffix in sorted(mla_col_head):
                add_record(all_headers, records, prefix + suffix, "cols", rank, size)
        else:
            for suffix in sorted(kda_full):
                add_record(all_headers, records, prefix + suffix)
            for suffix in sorted(kda_row_head):
                raw = locate(all_headers, prefix + suffix)
                records.append(copy_head_rows(raw, rank, size, HEAD_DIM,
                                              KDA_HEADS))
            add_record(all_headers, records, prefix + "self_attn.f_a_proj.weight")
            records.append(copy_head_rows(locate(all_headers,
                prefix + "self_attn.dt_bias"), rank, size, HEAD_DIM,
                KDA_HEADS))
            records.append(copy_head_rows(locate(all_headers,
                prefix + "self_attn.b_proj.weight"), rank, size, 1,
                KDA_HEADS))
            records.append(copy_head_cols(locate(all_headers,
                prefix + "self_attn.o_proj.weight"), rank, size, HEAD_DIM,
                KDA_HEADS))

        if layer == 0:
            for suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                add_record(all_headers, records, prefix + suffix, "rows", rank, size)
            add_record(all_headers, records, prefix + "mlp.down_proj.weight", "cols", rank, size)
        else:
            for suffix in ("block_sparse_moe.gate.weight",
                           "block_sparse_moe.gate.e_score_correction_bias",
                           "block_sparse_moe.routed_expert_down_proj.weight",
                           "block_sparse_moe.routed_expert_norm.weight",
                           "block_sparse_moe.routed_expert_up_proj.weight"):
                mode = "rows" if suffix == "block_sparse_moe.routed_expert_down_proj.weight" else "full"
                mode = "cols" if suffix == "block_sparse_moe.routed_expert_up_proj.weight" else mode
                if expert_tp and suffix in (
                        "block_sparse_moe.routed_expert_down_proj.weight",
                        "block_sparse_moe.routed_expert_up_proj.weight"):
                    mode = "rows-group" if moe_shard_layout == "row-aligned" else "full"
                add_record(all_headers, records, prefix + suffix, mode, rank, size)
            for suffix in ("block_sparse_moe.shared_experts.gate_proj.weight",
                           "block_sparse_moe.shared_experts.up_proj.weight"):
                add_record(all_headers, records, prefix + suffix, "rows", rank, size)
            add_record(all_headers, records,
                       prefix + "block_sparse_moe.shared_experts.down_proj.weight",
                       "cols", rank, size)
            for expert in range(EXPERTS):
                if expert_tp:
                    records.extend(expert_tp_records(all_headers, prefix,
                                                     rank, size, expert))
                    continue
                if expert % size != rank:
                    continue
                ep = prefix + "block_sparse_moe.experts.%d." % expert
                for suffix in ("w1.weight_packed", "w1.weight_scale",
                               "w2.weight_packed", "w2.weight_scale",
                               "w3.weight_packed", "w3.weight_scale"):
                    add_record(all_headers, records, ep + suffix)
    return records


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]


def copy_ranges(dst, rec, chunk):
    descriptor = rec["segments"]
    if (isinstance(descriptor, tuple) and len(descriptor) == 6 and
            descriptor[0] == "rows"):
        _, base, rows, row_bytes, first, count = descriptor
        src = os.open(rec["source"], os.O_RDONLY)
        try:
            # W2 and its scale matrix are row-strided in the source file.
            # Read a bounded rectangular source block, compact the selected
            # columns into one output block, then write once.  This avoids a
            # tiny pread/write pair for every row without retaining a tensor.
            rows_per_block = max(1, min(256, chunk // max(1, row_bytes)))
            output_buf = bytearray(rows_per_block * count)
            for row0 in range(0, rows, rows_per_block):
                nr = min(rows_per_block, rows - row0)
                source_bytes = nr * row_bytes
                data = os.pread(src, source_bytes, base + row0 * row_bytes)
                if len(data) != source_bytes:
                    raise IOError("short read %s at %d" %
                                  (rec["name"], base + row0 * row_bytes))
                for row in range(nr):
                    begin = row * row_bytes + first
                    output_buf[row * count:(row + 1) * count] = \
                        data[begin:begin + count]
                write_all(dst, memoryview(output_buf)[:nr * count])
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    os.posix_fadvise(src, base + row0 * row_bytes, source_bytes,
                                     os.POSIX_FADV_DONTNEED)
        finally:
            os.close(src)
        return
    for source, count in descriptor:
        src = os.open(rec["source"], os.O_RDONLY)
        try:
            done = 0
            while done < count:
                want = min(chunk, count - done)
                data = os.pread(src, want, source + done)
                if len(data) != want:
                    raise IOError("short read %s at %d" % (rec["name"], source + done))
                write_all(dst, data)
                done += want
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    os.posix_fadvise(src, source + done - want, want,
                                     os.POSIX_FADV_DONTNEED)
        finally:
            os.close(src)


def stage(records, output_dir, rank, size, chunk, force, mode, layer_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    blob = output_dir / ("rank%03d.blob" % rank)
    manifest = output_dir / ("rank%03d.manifest" % rank)
    if not force and (blob.exists() or manifest.exists()):
        raise FileExistsError("output exists: %s" % output_dir)
    blob_tmp = Path(str(blob) + ".tmp.%d" % os.getpid())
    manifest_tmp = Path(str(manifest) + ".tmp.%d" % os.getpid())
    offset = 0
    fd = None
    entries = []
    try:
        fd = os.open(str(blob_tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        for rec in records:
            aligned = (offset + ALIGN - 1) & ~(ALIGN - 1)
            if aligned > offset:
                pad = b"\0" * min(CHUNK, aligned - offset)
                remain = aligned - offset
                while remain:
                    take = min(len(pad), remain)
                    write_all(fd, pad[:take])
                    remain -= take
            rec = dict(rec)
            rec["offset"] = aligned
            copy_ranges(fd, rec, chunk)
            # Re-read the destination only in bounded chunks for the manifest CRC.
            offset = aligned + rec["nbytes"]
            entries.append(rec)
        os.fsync(fd)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            # The next process uploads this blob into anonymous HBM.  Drop the
            # just-written clean file pages first so staging and model upload
            # do not transiently compete for the node's 32 GiB.
            os.posix_fadvise(fd, 0, offset, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        fd = None
        with open(str(manifest_tmp), "w") as f:
            f.write("# K3FULLV2 mode=%s rank=%d nodes=%d layer_index=%d "
                    "tensors=%d blob_bytes=%d\n" %
                    (mode, rank, size, layer_index, len(entries), offset))
            for e in entries:
                shape = e["shape"]
                f.write("%d %d %s %d %s %s\n" %
                        (e["offset"], e["nbytes"], e["dtype"], len(shape),
                         " ".join(str(x) for x in shape), e["name"]))
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(blob_tmp), str(blob))
        os.replace(str(manifest_tmp), str(manifest))
    except Exception:
        if fd is not None:
            os.close(fd)
        for path in (blob_tmp, manifest_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return blob, manifest, offset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "models/kimi-k3")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--mode", choices=("full96", "layer12"), default="full96")
    ap.add_argument("--nodes", type=int, default=96)
    ap.add_argument("--layer-index", type=int)
    ap.add_argument("--chunk-mib", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--expert-tp", action="store_true",
                    help="stage one intermediate-channel slice of every expert")
    ap.add_argument("--moe-shard-layout", choices=("replicated", "row-aligned"),
                    default="replicated",
                    help="expert-TP routed projection layout")
    args = ap.parse_args()
    if not (0 <= args.rank < args.nodes):
        ap.error("rank must be within [0,nodes)")
    if args.mode == "full96" and args.nodes != 96:
        ap.error("full96 staging requires nodes=96")
    if args.mode == "layer12" and args.nodes != 12:
        ap.error("layer12 staging requires nodes=12")
    if args.mode == "layer12" and not (0 <= args.layer_index < LAYERS):
        ap.error("layer12 staging requires --layer-index in [0,92]")
    if args.mode == "full96" and args.layer_index is not None:
        ap.error("--layer-index is only valid with --mode layer12")
    if args.expert_tp and args.mode not in ("full96", "layer12"):
        ap.error("--expert-tp is only supported with full96/layer12")
    if args.chunk_mib < 1 or args.chunk_mib > 64:
        ap.error("--chunk-mib must be in [1,64]")
    if available_kb() < 6 * 1024 * 1024:
        raise MemoryError("MemAvailable is below the 6 GiB staging guard")
    all_headers = shards(args.model_dir)
    records = make_plan(all_headers, args.rank, args.nodes,
                        [args.layer_index] if args.mode == "layer12" else None,
                        include_global=args.mode != "layer12",
                        expert_tp=args.expert_tp,
                        moe_shard_layout=args.moe_shard_layout)
    stage_mode = args.mode + ("-expert-tp" if args.expert_tp else "")
    if args.expert_tp and args.moe_shard_layout != "replicated":
        stage_mode += "-" + args.moe_shard_layout
    total = sum(r["nbytes"] for r in records)
    print("K3 full stage plan: mode=%s rank=%d/%d layer_index=%s tensors=%d "
          "bytes=%d (%.3f GiB)" %
          (stage_mode, args.rank, args.nodes,
           args.layer_index if args.layer_index is not None else "all",
           len(records), total, total / float(1 << 30)))
    if args.plan_only:
        return
    blob, manifest, size = stage(records, args.output_dir, args.rank, args.nodes,
                                 args.chunk_mib * 1024 * 1024, args.force,
                                 stage_mode, args.layer_index if args.layer_index is not None else -1)
    print("staged blob=%s manifest=%s bytes=%d" % (blob, manifest, size))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("k3_full_stage: %s" % exc, file=sys.stderr)
        sys.exit(2)
