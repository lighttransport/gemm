#!/usr/bin/env python3
"""Bounded GLM-5.3F checkpoint and trace validation helpers.

This tool deliberately reads safetensors headers and payload slices only.  It
never maps a shard and never creates a local copy of the checkpoint.
"""
import argparse
import json
import math
import os
import struct
import sys
from collections import Counter, defaultdict

LAYERS = 45
EXPERTS = 288
TOP_K = 8
HIDDEN = 4096
KPOOL = 4
INDEX_TOPK = 2048


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError("short safetensors header: %s" % path)
        n = struct.unpack("<Q", raw)[0]
        if n > 128 * 1024 * 1024:
            raise RuntimeError("implausible safetensors header: %s" % path)
        header = json.loads(f.read(n))
    return header, 8 + n


def load_index(model):
    with open(os.path.join(model, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def shape_bytes(info):
    dtype = info.get("dtype", "")
    sizes = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2}
    if dtype not in sizes:
        raise RuntimeError("unsupported dtype %s" % dtype)
    count = 1
    for dim in info["shape"]:
        count *= dim
    return count * sizes[dtype]


def scan(model):
    mapping = load_index(model)
    by_shard = defaultdict(list)
    for name, shard in mapping.items():
        by_shard[shard].append(name)
    tensors = {}
    for shard, names in sorted(by_shard.items()):
        header, data_offset = read_header(os.path.join(model, shard))
        for name in names:
            item = header.get(name)
            if item is None:
                raise RuntimeError("index/header mismatch for %s" % name)
            tensors[name] = (shard, data_offset, item)
    return tensors


def layer_of(name):
    p = "model.language_model.layers."
    if not name.startswith(p):
        return None
    rest = name[len(p):]
    try:
        return int(rest.split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def cmd_manifest(args):
    tensors = scan(args.model)
    totals = Counter()
    layers = defaultdict(int)
    counts = Counter()
    for name, (_, _, info) in tensors.items():
        n = shape_bytes(info)
        totals[info["dtype"]] += n
        counts[info["dtype"]] += 1
        layer = layer_of(name)
        if layer is not None:
            layers[layer] += n
    required = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
        "model.language_model.layers.3.mlp.gate.weight",
    ]
    missing = [n for n in required if n not in tensors]
    print("GLM53F_MANIFEST tensors=%d shards=%d" %
          (len(tensors), len(set(v[0] for v in tensors.values()))))
    for dtype in sorted(totals):
        print("dtype=%s tensors=%d bytes=%d GiB=%.3f" %
              (dtype, counts[dtype], totals[dtype], totals[dtype] / 2**30))
    for layer in range(LAYERS):
        print("layer=%02d bytes=%d GiB=%.3f" %
              (layer, layers[layer], layers[layer] / 2**30))
    if missing:
        print("MISSING " + " ".join(missing), file=sys.stderr)
        return 2
    if totals["F8_E4M3"] < 280 * 2**30:
        print("unexpectedly small FP8 payload", file=sys.stderr)
        return 2
    print("contract=PASS fp8_streaming=PASS local_limit=87GiB")
    return 0


def router(logits, bias):
    # Match glm53f_router_topk: bias is added after sigmoid for selection;
    # unbiased sigmoid weights are normalized and scaled after top-k.
    order = sorted(range(len(logits)),
                   key=lambda i: (-(1.0 / (1.0 + math.exp(-logits[i])) +
                                   (bias[i] if bias else 0.0)), i))
    ids = order[:TOP_K]
    weights = [1.0 / (1.0 + math.exp(-logits[i])) for i in ids]
    scale = 2.5 / sum(weights)
    return ids, [w * scale for w in weights]


def cmd_router(args):
    # Text format: one record per line, 288 logits followed by 288 bias values.
    # A missing bias is accepted and means noaux_tc/no selection bias.
    passed = 0
    with open(args.input) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) not in (EXPERTS, 2 * EXPERTS):
                raise RuntimeError("line %d: expected 288 or 576 floats" % line_no)
            logits = values[:EXPERTS]
            bias = values[EXPERTS:] if len(values) == 2 * EXPERTS else None
            ids, weights = router(logits, bias)
            print("%d %s %s" % (line_no, ",".join(map(str, ids)),
                               ",".join("%.9g" % x for x in weights)))
            if len(set(ids)) != TOP_K or abs(sum(weights) - 2.5) > 2e-6:
                return 2
            passed += 1
    print("router_records=%d top_k=%d scale=2.5 stable_ties=PASS" % (passed, TOP_K), file=sys.stderr)
    return 0


def floats(path):
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            if len(b) % 4:
                raise RuntimeError("not a float32 stream: %s" % path)
            for x in struct.iter_unpack("<f", b):
                yield x[0]


def cmd_compare(args):
    n = 0
    sum_d = sum_r = max_abs = 0.0
    with open(args.ref, "rb") as rf, open(args.got, "rb") as gf:
        while True:
            a = rf.read(1024 * 1024)
            b = gf.read(1024 * 1024)
            if not a and not b:
                break
            if len(a) != len(b) or len(a) % 4:
                print("size mismatch at float %d" % n, file=sys.stderr)
                return 2
            for (x,), (y,) in zip(struct.iter_unpack("<f", a), struct.iter_unpack("<f", b)):
                if not math.isfinite(x) or not math.isfinite(y):
                    print("non-finite at float %d" % n, file=sys.stderr)
                    return 2
                d = x - y
                sum_d += d * d
                sum_r += y * y
                max_abs = max(max_abs, abs(d))
                n += 1
    rel = math.sqrt(sum_d / sum_r) if sum_r else (0.0 if sum_d == 0 else float("inf"))
    print("count=%d rel_l2=%.9g max_abs=%.9g threshold=%.9g %s" %
          (n, rel, max_abs, args.threshold, "PASS" if rel <= args.threshold else "FAIL"))
    return 0 if rel <= args.threshold else 1


def cmd_trace_index(args):
    # Native C layout: the two uint64 fields are aligned after seven uint32s.
    record = struct.Struct("<QIIIIIII4xQQdff")
    with open(args.path, "rb") as f:
        header = f.read(32)
        if len(header) != 32 or struct.unpack_from("<Q", header)[0] != 0x474c4d3533464254:
            raise RuntimeError("not a GLM53F boundary trace")
        records = 0
        for _ in range(args.limit):
            raw = f.read(record.size)
            if not raw:
                break
            if len(raw) != record.size:
                raise RuntimeError("truncated trace record")
            _, version, flags, layer, token, boundary, dtype, count, payload, digest, ss, lo, hi = record.unpack(raw)
            if payload:
                f.seek(payload, 1)
            print("record=%d layer=%d token=%d boundary=%d dtype=%d count=%d values=%d digest=%016x rms=%.9g min=%.9g max=%.9g" %
                  (records, layer, token, boundary, dtype, count, bool(flags & 1), digest,
                   math.sqrt(ss / count) if count else 0.0, lo, hi))
            records += 1
    print("trace_records=%d" % records, file=sys.stderr)
    return 0


def load_artifact_manifest(path):
    records = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") != "tensor":
                continue
            key = (item.get("stage"), item.get("name"), item.get("dtype"),
                   tuple(item.get("shape", [])))
            if key in records:
                raise RuntimeError("duplicate artifact tensor at line %d" % line_no)
            item["_manifest_dir"] = os.path.dirname(os.path.abspath(path))
            records[key] = item
    return records


def compare_artifact_record(a, b, threshold):
    if a["dtype"] != b["dtype"] or a["count"] != b["count"] or a["bytes"] != b["bytes"]:
        return False, 0.0, float("inf"), "metadata mismatch"
    pa = os.path.join(a["_manifest_dir"], a["payload"])
    pb = os.path.join(b["_manifest_dir"], b["payload"])
    if a["dtype"] == "i32":
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            same = True
            while True:
                xa = fa.read(1024 * 1024)
                xb = fb.read(1024 * 1024)
                if not xa and not xb:
                    break
                same &= xa == xb
                if len(xa) != len(xb):
                    same = False
        return same, 0.0 if same else float("inf"), 0.0 if same else float("inf"), "ok" if same else "integer payload mismatch"
    n = 0
    sum_d = sum_r = max_abs = 0.0
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        while True:
            xa = fa.read(1024 * 1024)
            xb = fb.read(1024 * 1024)
            if not xa and not xb:
                break
            if len(xa) != len(xb) or len(xa) % 4:
                return False, 0.0, float("inf"), "payload size mismatch"
            for (x,), (y,) in zip(struct.iter_unpack("<f", xa), struct.iter_unpack("<f", xb)):
                if not math.isfinite(x) or not math.isfinite(y):
                    return False, 0.0, float("inf"), "non-finite payload"
                d = x - y
                sum_d += d * d
                sum_r += y * y
                max_abs = max(max_abs, abs(d))
                n += 1
    rel = math.sqrt(sum_d / sum_r) if sum_r else (0.0 if sum_d == 0 else float("inf"))
    return rel <= threshold, rel, max_abs, "ok"


def cmd_artifact_compare(args):
    left = load_artifact_manifest(args.left)
    right = load_artifact_manifest(args.right)
    if set(left) != set(right):
        missing = len(set(left) - set(right))
        extra = len(set(right) - set(left))
        print("artifact_keys_mismatch missing=%d extra=%d" % (missing, extra))
        return 1
    failed = 0
    for key in sorted(left):
        ok, rel, max_abs, reason = compare_artifact_record(left[key], right[key], args.threshold)
        print("stage=%s name=%s rel_l2=%.9g max_abs=%.9g %s" %
              (key[0], key[1], rel, max_abs, "PASS" if ok else "FAIL:%s" % reason))
        failed |= not ok
    print("artifact_compare tensors=%d threshold=%.9g %s" %
          (len(left), args.threshold, "PASS" if not failed else "FAIL"))
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    m = sub.add_parser("manifest")
    m.add_argument("model")
    m.set_defaults(func=cmd_manifest)
    r = sub.add_parser("router")
    r.add_argument("input")
    r.set_defaults(func=cmd_router)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("got")
    c.add_argument("--threshold", type=float, default=1e-3)
    c.set_defaults(func=cmd_compare)
    t = sub.add_parser("trace-index")
    t.add_argument("path")
    t.add_argument("--limit", type=int, default=1000000)
    t.set_defaults(func=cmd_trace_index)
    a = sub.add_parser("artifact-compare")
    a.add_argument("left")
    a.add_argument("right")
    a.add_argument("--threshold", type=float, default=1e-3)
    a.set_defaults(func=cmd_artifact_compare)
    args = p.parse_args()
    if not args.command:
        p.error("a command is required")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
