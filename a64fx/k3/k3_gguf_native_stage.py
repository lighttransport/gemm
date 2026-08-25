#!/usr/bin/env python3
"""Stage GGUF tensors in the native K3 runner's row/head-sharded layout."""
from __future__ import print_function

import argparse
import os
import re
from pathlib import Path

from k3_gguf_stage import ALIGN, TYPE_INFO, discover
from k3_gguf_mla_materialize import materialize_combined_into

LAYERS = 93
EXPERT_INTER = 3072
IQ_BLOCK = 256
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
       51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}

def native_name(layer, suffix):
    return "language_model.model.layers.%d.%s" % (layer, suffix)

def add(out, records, gguf, native, mode, detail):
    if gguf not in records:
        raise ValueError("missing %s" % gguf)
    rec = records[gguf]
    out.append((native, gguf, mode, tuple(rec["dims"]), rec["type"], detail))

def plan(model_dir, nodes):
    _, all_records, split_count = discover(model_dir)
    records = {r["name"]: r for r in all_records}
    out = []
    # Full-model native images keep vocabulary rows local to the owning rank;
    # the final norm is replicated.  These names match k3_full_runner.c.
    for g, n, mode in (("token_embd.weight", "language_model.model.embed_tokens.weight", "head-rows"),
                       ("output.weight", "language_model.lm_head.weight", "head-rows"),
                       ("output_norm.weight", "language_model.model.norm.weight", "full"),
                       ("output_res_score.weight", "language_model.model.output_attn_res_norm.weight", "full")):
        add(out, records, g, n, mode, "global")
    for name in ("token_embd.weight", "output.weight", "output_norm.weight",
                 "output_res_score.weight"):
        if name not in records:
            raise ValueError("missing global tensor %s" % name)
    for l in range(LAYERS):
        p = "blk.%d." % l
        for g, n in (("attn_norm.weight", "input_layernorm.weight"),
                     ("ffn_norm.weight", "post_attention_layernorm.weight"),
                     ("attn_res_score.weight", "self_attention_res_norm.weight"),
                     ("ffn_res_score.weight", "mlp_res_norm.weight")):
            add(out, records, p + g, native_name(l, n), "full", "norm")
        if l in MLA:
            for g, n, mode in (
                ("attn_q_a.weight", "self_attn.q_a_proj.weight", "full"),
                ("attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight", "full"),
                ("attn_q_b.weight", "self_attn.q_b_proj.weight", "head-rows"),
                ("attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight", "full"),
                ("attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight", "full"),
                # GGUF keeps K and V compressed projections as separate 3-D
                # tensors.  They cannot be renamed to one native tensor: the
                # native graph must concatenate their per-head outputs.
                ("attn_k_b.weight", "self_attn.kv_b_k_proj.weight", "mla-transpose-dequant"),
                ("attn_v_b.weight", "self_attn.kv_b_v_proj.weight", "mla-transpose-dequant"),
                ("attn_gate.weight", "self_attn.g_proj.weight", "head-rows"),
                ("attn_output.weight", "self_attn.o_proj.weight", "head-cols"),
            ):
                add(out, records, p + g, native_name(l, n), mode, "mla")
        else:
            for g, n, mode in (
                ("attn_q.weight", "self_attn.q_proj.weight", "head-rows"),
                ("attn_k.weight", "self_attn.k_proj.weight", "head-rows"),
                ("attn_v.weight", "self_attn.v_proj.weight", "head-rows"),
                ("ssm_g.weight", "self_attn.g_proj.weight", "head-rows"),
                ("ssm_f_a.weight", "self_attn.f_a_proj.weight", "full"),
                ("ssm_f_b.weight", "self_attn.f_b_proj.weight", "head-rows"),
                ("ssm_beta.weight", "self_attn.b_proj.weight", "head-rows"),
                ("ssm_conv1d_q.weight", "self_attn.q_conv1d.weight", "full"),
                ("ssm_conv1d_k.weight", "self_attn.k_conv1d.weight", "full"),
                ("ssm_conv1d_v.weight", "self_attn.v_conv1d.weight", "full"),
                ("ssm_a", "self_attn.A_log", "full"),
                ("ssm_dt.bias", "self_attn.dt_bias", "head-rows"),
                ("ssm_norm.weight", "self_attn.o_norm.weight", "full"),
                ("attn_output.weight", "self_attn.o_proj.weight", "head-cols"),
            ):
                add(out, records, p + g, native_name(l, n), mode, "kda")
        if l == 0:
            for g, n, mode in (("ffn_gate.weight", "mlp.gate_proj.weight", "rows"),
                               ("ffn_up.weight", "mlp.up_proj.weight", "rows"),
                               ("ffn_down.weight", "mlp.down_proj.weight", "cols")):
                add(out, records, p + g, native_name(l, n), mode, "dense")
        else:
            for g, n, mode in (
                ("ffn_gate_inp.weight", "block_sparse_moe.gate.weight", "full"),
                ("exp_probs_b.bias", "block_sparse_moe.gate.e_score_correction_bias", "full"),
                ("ffn_routed_down.weight", "block_sparse_moe.routed_expert_down_proj.weight", "rows"),
                ("ffn_routed_norm.weight", "block_sparse_moe.routed_expert_norm.weight", "full"),
                ("ffn_routed_up.weight", "block_sparse_moe.routed_expert_up_proj.weight", "cols"),
                ("ffn_gate_shexp.weight", "block_sparse_moe.shared_experts.gate_proj.weight", "rows"),
                ("ffn_up_shexp.weight", "block_sparse_moe.shared_experts.up_proj.weight", "rows"),
                ("ffn_down_shexp.weight", "block_sparse_moe.shared_experts.down_proj.weight", "cols"),
            ):
                add(out, records, p + g, native_name(l, n), mode, "moe")
            # Native expert kernels follow the checkpoint convention:
            # w1=gate, w2=down, w3=up.  SiTU is asymmetric, so swapping the
            # equally-shaped gate/up planes is a silent but severe quality bug.
            for g, tag in (("ffn_gate_exps.weight", "w1"),
                           ("ffn_down_exps.weight", "w2"),
                           ("ffn_up_exps.weight", "w3")):
                add(out, records, p + g,
                    native_name(l, "block_sparse_moe.experts.<id>.%s.weight_quant" % tag),
                    "expert-slice", "iq-expert")
    return out, split_count

def split_dim(total, rank, nodes):
    base, rem = divmod(total, nodes)
    return rank * base + min(rank, rem), base + (rank < rem)

def split_groups(total, rank, nodes, group):
    if total % group:
        raise ValueError("dimension %d is not divisible by group %d" %
                         (total, group))
    first, count = split_dim(total // group, rank, nodes)
    return first * group, count * group

def validate_expert_tp_nodes(nodes):
    """Require every IQ W2 column slice to contain complete 256-value blocks."""
    if EXPERT_INTER % nodes or (EXPERT_INTER // nodes) % IQ_BLOCK:
        raise ValueError("IQ expert TP requires 3072/nodes to be a multiple "
                         "of 256 (nodes must divide 12), got nodes=%d" % nodes)

def native_shape(rec):
    dims = tuple(rec["dims"])
    if len(dims) == 2:
        return (dims[1], dims[0])
    return dims

def _copy(src, dst, offset, length):
    src.seek(offset)
    left = length
    while left:
        data = src.read(min(left, 8 * 1024 * 1024))
        if not data:
            raise IOError("short read from %s" % src.name)
        dst.write(data)
        left -= len(data)

def _handle(handles, path):
    if path not in handles:
        handles[path] = open(path, "rb", buffering=0)
    return handles[path]

def stage_layer(model_dir, output_dir, nodes, rank, layer, force=False,
                expert_tp=False):
    _, all_records, _ = discover(model_dir)
    records = {r["name"]: r for r in all_records}
    items, _ = plan(model_dir, nodes)
    if layer is not None:
        prefix = "language_model.model.layers.%d." % layer
        items = [x for x in items if x[0].startswith(prefix)]
        # Layer debug images do not need global tensors.
        items = [x for x in items if ".model.embed_tokens." not in x[0] and
                 ".lm_head." not in x[0] and ".model.norm." not in x[0]]
    if not items:
        raise ValueError("no native items for layer %d" % layer)
    # GGUF has one residual-score vector for each join.  The safetensor graph
    # names its two multiplicands separately; alias the same immutable payload
    # so full_attn_res can execute the native graph without fabricating data.
    aliases = []
    for native, gguf, mode, dims, typ, detail in items:
        if native.endswith("self_attention_res_norm.weight"):
            aliases.append((native.replace("_norm.weight", "_proj.weight"),
                            gguf, mode, dims, typ, detail))
        if native.endswith("mlp_res_norm.weight"):
            aliases.append((native.replace("_norm.weight", "_proj.weight"),
                            gguf, mode, dims, typ, detail))
        if native.endswith("output_attn_res_norm.weight"):
            aliases.append((native.replace("_norm.weight", "_proj.weight"),
                            gguf, mode, dims, typ, detail))
    items += aliases

    output_dir.mkdir(parents=True, exist_ok=True)
    blob = output_dir / ("rank%03d.blob" % rank)
    manifest = output_dir / ("rank%03d.manifest" % rank)
    if not force and blob.exists() and manifest.exists():
        return
    blob_tmp = Path(str(blob) + ".tmp")
    manifest_tmp = Path(str(manifest) + ".tmp")
    entries = []
    handles = {}
    pos = 0
    with open(str(blob_tmp), "wb", buffering=0) as out:
        try:
            mla_layers = sorted(set(int(x[0].split(".")[3]) for x in items
                                    if x[2] == "mla-transpose-dequant"))
            for mla_layer in mla_layers:
                mla_parts = [x for x in items if x[2] == "mla-transpose-dequant" and
                             x[0].startswith("language_model.model.layers.%d." % mla_layer)]
                if len(mla_parts) != 2:
                    raise ValueError("MLA staging requires exactly one K/V pair")
                by_gguf = {x[1]: records[x[1]] for x in mla_parts}
                kname = "blk.%d.attn_k_b.weight" % mla_layer
                vname = "blk.%d.attn_v_b.weight" % mla_layer
                if kname not in by_gguf or vname not in by_gguf:
                    raise ValueError("MLA staging could not identify K/V tensors")
                first, count = split_dim(96, rank, nodes)
                pos = (pos + ALIGN - 1) // ALIGN * ALIGN
                if out.tell() < pos:
                    out.write(b"\0" * (pos - out.tell()))
                begin = pos
                materialize_combined_into(by_gguf[kname], by_gguf[vname], out,
                                          first, count, "F32")
                pos = out.tell()
                name = native_name(mla_layer, "self_attn.kv_b_proj.weight")
                entries.append((begin, pos - begin, "F32",
                                (count * 256, 512), name))
            for native, gguf, mode, _, _, detail in items:
                if mode == "mla-transpose-dequant":
                    continue
                rec = records[gguf]
                if mode == "expert-slice":
                    cols, rows, experts = rec["dims"]
                    plane_bytes = rec["row_bytes"] * rows
                    which = native.split(".")[-2]
                    owned = range(experts) if expert_tp else range(rank, experts, nodes)
                    for expert in owned:
                        name = native.replace("<id>", str(expert))
                        plane = rec["data_start"] + expert * plane_bytes
                        pos = (pos + ALIGN - 1) // ALIGN * ALIGN
                        if out.tell() < pos:
                            out.write(b"\0" * (pos - out.tell()))
                        src = _handle(handles, rec["source"])
                        begin = pos
                        if not expert_tp:
                            _copy(src, out, plane, plane_bytes)
                            pos += plane_bytes
                            shape = (rows, cols)
                        elif which in ("w1", "w3"):
                            first, count = split_dim(rows, rank, nodes)
                            length = count * rec["row_bytes"]
                            _copy(src, out,
                                  plane + first * rec["row_bytes"], length)
                            pos += length
                            shape = (count, cols)
                        elif which == "w2":
                            block, type_bytes = TYPE_INFO[rec["type_id"]][1:]
                            first, count = split_dim(cols, rank, nodes)
                            if first % block or count % block:
                                raise ValueError("expert-TP column slice is not block aligned")
                            delta = first // block * type_bytes
                            piece = count // block * type_bytes
                            # Read one expert plane sequentially, then emit the
                            # narrow quant-block slice from each output row.
                            # This avoids millions of tiny filesystem reads.
                            src.seek(plane)
                            payload = src.read(plane_bytes)
                            if len(payload) != plane_bytes:
                                raise IOError("short read from %s" % src.name)
                            for r in range(rows):
                                row = r * rec["row_bytes"] + delta
                                out.write(payload[row:row + piece])
                            pos += rows * piece
                            shape = (rows, count)
                        else:
                            raise ValueError("unexpected expert tensor %s" % which)
                        entries.append((begin, pos - begin, rec["type"],
                                        shape, name))
                    continue

                dims = rec["dims"]
                segments = []
                shape = native_shape(rec)
                actual_mode = mode
                if len(dims) == 3 and "conv1d" in native:
                    actual_mode = "channel-3d"
                if actual_mode in ("full", "norm"):
                    segments = [(rec["data_start"], rec["nbytes"])]
                elif actual_mode in ("rows", "head-rows"):
                    if len(dims) == 1:
                        first, count = split_dim(dims[0], rank, nodes)
                        _, type_bytes = TYPE_INFO[rec["type_id"]][1:]
                        segments = [(rec["data_start"] + first * type_bytes,
                                     count * type_bytes)]
                        shape = (count,)
                    else:
                        if "routed_expert_down_proj" in native:
                            first, count = split_groups(dims[1], rank, nodes, 32)
                        else:
                            first, count = split_dim(dims[1], rank, nodes)
                        segments = [(rec["data_start"] + first * rec["row_bytes"],
                                     count * rec["row_bytes"])]
                        shape = (count, dims[0])
                elif actual_mode in ("cols", "head-cols"):
                    cols, rows = dims
                    block, type_bytes = TYPE_INFO[rec["type_id"]][1:]
                    if "routed_expert_up_proj" in native:
                        first, count = split_groups(cols, rank, nodes, block)
                    else:
                        first, count = split_dim(cols, rank, nodes)
                    if first % block or count % block:
                        raise ValueError("column shard is not block aligned: %s" % gguf)
                    piece = count // block * type_bytes
                    delta = first // block * type_bytes
                    segments = [(rec["data_start"] + r * rec["row_bytes"] + delta,
                                 piece) for r in range(rows)]
                    shape = (rows, count)
                elif actual_mode == "channel-3d":
                    width, one, channels = dims
                    if one != 1 or rec["type"] != "F32":
                        raise ValueError("unexpected conv tensor %s" % gguf)
                    first, count = split_dim(channels, rank, nodes)
                    piece = width * 4
                    segments = [(rec["data_start"] + first * piece,
                                 count * piece)]
                    shape = (count, width)
                else:
                    raise ValueError("unsupported executable mode %s for %s" %
                                     (actual_mode, gguf))
                pos = (pos + ALIGN - 1) // ALIGN * ALIGN
                if out.tell() < pos:
                    out.write(b"\0" * (pos - out.tell()))
                src = _handle(handles, rec["source"])
                begin = pos
                for off, length in segments:
                    _copy(src, out, off, length)
                    pos += length
                entries.append((begin, pos - begin, rec["type"], shape, native))
        finally:
            for src in handles.values():
                src.close()
        out.flush()
        os.fsync(out.fileno())
    entries.sort(key=lambda x: x[4])
    with open(str(manifest_tmp), "w") as out:
        stage_mode = (("layer12-iq-expert-tp" if expert_tp else "layer12-iq")
                      if layer is not None else
                      ("full96-iq-expert-tp" if expert_tp else "full96-iq"))
        out.write("# K3FULLV3 mode=%s rank=%s nodes=%s layer_index=%s "
                  "tensors=%s blob_bytes=%s expert_roles=gate-w1-up-w3\n" %
                  (stage_mode, rank, nodes, -1 if layer is None else layer,
                   len(entries), pos))
        for off, size, typ, shape, name in entries:
            out.write("%d %d %s %d %s %s\n" %
                      (off, size, typ, len(shape),
                       " ".join(str(x) for x in shape), name))
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(blob_tmp), str(blob))
    os.replace(str(manifest_tmp), str(manifest))

def fix_expert_roles(output_dir, rank):
    """Repair pre-fix native manifests without rewriting their large blobs.

    Old stages put the up plane under w1 and gate under w3.  Their shape,
    dtype, and byte size are identical, so swapping only the two manifest
    names is exact and leaves all blob offsets valid.
    """
    manifest = output_dir / ("rank%03d.manifest" % rank)
    if not manifest.exists():
        raise ValueError("missing existing manifest for rank %d" % rank)
    lines = manifest.read_text().splitlines()
    if not lines or not lines[0].startswith("# K3FULLV3"):
        raise ValueError("unrecognized manifest %s" % manifest)
    marker = "expert_roles=gate-w1-up-w3"
    if marker in lines[0]:
        return False
    w1 = ".w1.weight_quant"
    w3 = ".w3.weight_quant"
    seen1 = sum(w1 in line for line in lines[1:])
    seen3 = sum(w3 in line for line in lines[1:])
    if not seen1 or seen1 != seen3:
        raise ValueError("manifest has inconsistent expert roles: w1=%d w3=%d" %
                         (seen1, seen3))
    repaired = [lines[0] + " " + marker]
    for line in lines[1:]:
        if w1 in line:
            line = line.replace(w1, w3)
        elif w3 in line:
            line = line.replace(w3, w1)
        repaired.append(line)
    manifest_tmp = Path(str(manifest) + ".tmp")
    with open(str(manifest_tmp), "w") as out:
        for line in repaired:
            out.write(line + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(manifest_tmp), str(manifest))
    return True

def augment_output_res_score(model_dir, output_dir, rank):
    """Append the final fused residual score to an existing full stage.

    This is deliberately tiny (two 28 KiB F32 payloads) and avoids a costly
    full-model restage when upgrading older native GGUF images.
    """
    _, all_records, _ = discover(model_dir)
    records = {r["name"]: r for r in all_records}
    rec = records.get("output_res_score.weight")
    if rec is None:
        raise ValueError("missing output_res_score.weight")
    blob = output_dir / ("rank%03d.blob" % rank)
    manifest = output_dir / ("rank%03d.manifest" % rank)
    if not blob.exists() or not manifest.exists():
        raise ValueError("missing existing full stage for rank %d" % rank)
    lines = manifest.read_text().splitlines()
    names = {line.rsplit(" ", 1)[-1] for line in lines[1:] if line and not line.startswith("#")}
    wanted = ("language_model.model.output_attn_res_norm.weight",
              "language_model.model.output_attn_res_proj.weight")
    if all(name in names for name in wanted):
        return
    if any(name in names for name in wanted):
        raise ValueError("partial output residual stage for rank %d" % rank)
    pos = blob.stat().st_size
    entries = []
    with open(blob, "ab", buffering=0) as out, open(rec["source"], "rb", buffering=0) as src:
        for name in wanted:
            pos = (pos + ALIGN - 1) // ALIGN * ALIGN
            if out.tell() < pos:
                out.write(b"\0" * (pos - out.tell()))
            begin = pos
            _copy(src, out, rec["data_start"], rec["nbytes"])
            pos += rec["nbytes"]
            entries.append((begin, rec["nbytes"], rec["type"], (7168,), name))
        out.flush()
        os.fsync(out.fileno())
    header = lines[0]
    tensors = re.search(r"tensors=(\d+)", header)
    if not tensors:
        raise ValueError("unrecognized manifest header: %r" % header)
    header = header.replace("tensors=" + tensors.group(1),
                            "tensors=" + str(int(tensors.group(1)) + len(entries)))
    header = re.sub(r"blob_bytes=\d+", "blob_bytes=" + str(pos), header)
    manifest_tmp = Path(str(manifest) + ".tmp")
    with open(manifest_tmp, "w") as out:
        out.write(header + "\n")
        for line in lines[1:]:
            out.write(line + "\n")
        for off, size, typ, shape, name in entries:
            out.write("%d %d %s %d %s %s\n" %
                      (off, size, typ, len(shape), " ".join(map(str, shape)), name))
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(manifest_tmp), str(manifest))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--full", action="store_true",
                    help="stage all globals and all 93 layers in one rank image")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--expert-tp", action="store_true",
                    help="stage every expert with a 1/nodes intermediate slice")
    ap.add_argument("--augment-output-res-score", action="store_true",
                    help="append missing final residual score to an existing full stage")
    ap.add_argument("--fix-expert-roles", action="store_true",
                    help="swap old GGUF-native w1/w3 manifest labels without rewriting blobs")
    args = ap.parse_args()
    if args.nodes <= 0 or 96 % args.nodes:
        raise ValueError("nodes must divide 96 heads")
    if args.expert_tp:
        validate_expert_tp_nodes(args.nodes)
    if args.output_dir is not None:
        if args.rank is None or not 0 <= args.rank < args.nodes:
            raise ValueError("--rank is required and must be in [0,nodes)")
        if args.fix_expert_roles:
            changed = fix_expert_roles(args.output_dir, args.rank)
            print("K3_GGUF_NATIVE_STAGE PASS fix-expert-roles rank=%d changed=%d output=%s" %
                  (args.rank, int(changed), args.output_dir))
            return
        if args.augment_output_res_score:
            if not args.full:
                raise ValueError("--augment-output-res-score requires --full")
            augment_output_res_score(args.model_dir, args.output_dir, args.rank)
            print("K3_GGUF_NATIVE_STAGE PASS augment-output-res-score rank=%d output=%s" %
                  (args.rank, args.output_dir))
            return
        if not 0 <= args.layer < LAYERS:
            raise ValueError("layer is out of range")
        layer = None if args.full else args.layer
        stage_layer(args.model_dir, args.output_dir, args.nodes, args.rank,
                    layer, args.force, args.expert_tp)
        label = "full" if layer is None else "layer=%d" % layer
        print("K3_GGUF_NATIVE_STAGE PASS %s rank=%d nodes=%d output=%s" %
              (label, args.rank, args.nodes, args.output_dir))
        return 0
    items, split_count = plan(args.model_dir, args.nodes)
    counts = {}
    for _, _, mode, _, _, detail in items:
        counts[(detail, mode)] = counts.get((detail, mode), 0) + 1
    print("K3_GGUF_NATIVE_STAGE_PLAN PASS nodes=%d tensors=%d split_count=%s" %
          (args.nodes, len(items), split_count))
    for key in sorted(counts):
        print("  %-8s %-12s %d" % (key[0], key[1], counts[key]))
    print("  native graph: 69 KDA + 24 MLA; MLA K/V planes require transpose+dequant")
    print("  IQ expert tensors remain quantized and need a native expert matvec path")
    return 0

if __name__ == "__main__":
    main()
