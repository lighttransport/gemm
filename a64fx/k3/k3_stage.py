#!/usr/bin/env python3
"""Memory-safe partial Kimi K3 expert stager.

The tool parses safetensor JSON headers, locates the six MXFP4 tensors for one
layer/expert pair, and copies them with bounded pread calls into an aligned blob.
It never mmap()s a shard or reads an unrelated tensor payload.
"""
import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

ALIGN = 256
MIN_AVAILABLE_KB = 6 * 1024 * 1024
EXPECTED = {
    "w1.weight_packed": ("U8", [3072, 1792]),
    "w1.weight_scale":  ("U8", [3072, 112]),
    "w2.weight_packed": ("U8", [3584, 1536]),
    "w2.weight_scale":  ("U8", [3584, 96]),
    "w3.weight_packed": ("U8", [3072, 1792]),
    "w3.weight_scale":  ("U8", [3072, 112]),
}
EXPERT_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123]\.weight_(?:packed|scale))$")


def mem_available_kb():
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def read_header(path):
    with open(str(path), "rb", buffering=0) as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("short header: %s" % path)
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 0 or header_bytes > 512 * 1024 * 1024:
            raise ValueError("invalid header length %d: %s" % (header_bytes, path))
        payload = f.read(header_bytes)
        if len(payload) != header_bytes:
            raise ValueError("short JSON header: %s" % path)
    return 8 + header_bytes, json.loads(payload.decode("utf-8"))


def target_names(layer, expert):
    prefix = "language_model.model.layers.%d.block_sparse_moe.experts.%d." % (layer, expert)
    return {prefix + suffix: suffix for suffix in EXPECTED}


def locate(model_dir, layer, expert):
    wanted = target_names(layer, expert)
    found = {}
    paths = sorted(model_dir.glob("*.safetensors"))
    if len(paths) != 96:
        raise ValueError("expected 96 safetensor shards, found %d" % len(paths))
    for path in paths:
        data_start, header = read_header(path)
        size = path.stat().st_size
        for name, suffix in wanted.items():
            if name not in header:
                continue
            info = header[name]
            begin, end = info["data_offsets"]
            if begin < 0 or end < begin or data_start + end > size:
                raise ValueError("out-of-file offsets for %s" % name)
            dtype, shape = EXPECTED[suffix]
            if info["dtype"] != dtype or info["shape"] != shape:
                raise ValueError("%s: got %s %s, expected %s %s" %
                                 (name, info["dtype"], info["shape"], dtype, shape))
            found[suffix] = {
                "name": name, "source": str(path), "source_offset": data_start + begin,
                "nbytes": end - begin, "dtype": dtype, "shape": shape,
            }
    missing = sorted(set(EXPECTED) - set(found))
    if missing:
        raise ValueError("missing tensors: %s" % ", ".join(missing))
    return [found[suffix] for suffix in sorted(found)]


def validate_checkpoint(model_dir):
    config_path = model_dir / "config.json"
    with config_path.open("r") as f:
        config = json.load(f)
    config = config.get("text_config", config)
    if config.get("hidden_size") != 7168 or config.get("num_hidden_layers") != 93:
        raise ValueError("unexpected hidden size or layer count in config.json")
    if config.get("num_experts") != 896 or config.get("num_experts_per_token") != 16:
        raise ValueError("unexpected expert configuration")
    # Config attention layer lists are one-based; checkpoint tensor names are zero-based.
    kda_expected = set(x - 1 for x in config["linear_attn_config"]["kda_layers"])
    if len(kda_expected) != 69:
        raise ValueError("expected 69 KDA layers, config has %d" % len(kda_expected))
    expert_bits = {}
    kda_alog = set()
    suffix_bits = {name: 1 << i for i, name in enumerate(sorted(EXPECTED))}
    tensor_count = 0
    paths = sorted(model_dir.glob("*.safetensors"))
    if len(paths) != 96:
        raise ValueError("expected 96 safetensor shards, found %d" % len(paths))
    for path in paths:
        _, header = read_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            match = EXPERT_RE.match(name)
            if match:
                layer, expert, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
                if layer < 1 or layer > 92 or expert < 0 or expert >= 896:
                    raise ValueError("invalid expert coordinates: %s" % name)
                dtype, shape = EXPECTED[suffix]
                if info["dtype"] != dtype or info["shape"] != shape:
                    raise ValueError("shape mismatch: %s" % name)
                key = (layer, expert)
                bit = suffix_bits[suffix]
                if expert_bits.get(key, 0) & bit:
                    raise ValueError("duplicate expert tensor: %s" % name)
                expert_bits[key] = expert_bits.get(key, 0) | bit
            if name.endswith(".self_attn.A_log"):
                match_layer = re.search(r"\.layers\.(\d+)\.", name)
                if not match_layer or info["dtype"] != "F32" or info["shape"] != [128]:
                    raise ValueError("KDA A_log must be F32[128]: %s" % name)
                kda_alog.add(int(match_layer.group(1)))
    full_mask = (1 << len(EXPECTED)) - 1
    expected_pairs = 92 * 896
    if len(expert_bits) != expected_pairs:
        raise ValueError("expected %d layer/expert pairs, found %d" %
                         (expected_pairs, len(expert_bits)))
    incomplete = [key for key, mask in expert_bits.items() if mask != full_mask]
    if incomplete:
        raise ValueError("incomplete expert tensor groups, first=%s" % (incomplete[0],))
    if kda_alog != kda_expected:
        raise ValueError("A_log layer set differs from configured KDA layers")
    print("checkpoint validation: PASS")
    print("  shards=96 tensors=%d layers=93 KDA=69 MLA=24" % tensor_count)
    print("  expert groups=%d tensors=%d shapes=exact" %
          (len(expert_bits), len(expert_bits) * len(EXPECTED)))
    print("  A_log: 69 tensors, F32[128] checkpoint contract")


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]


def copy_range(src_fd, dst_fd, source_offset, nbytes, chunk_bytes):
    done = 0
    while done < nbytes:
        count = min(chunk_bytes, nbytes - done)
        data = os.pread(src_fd, count, source_offset + done)
        if len(data) != count:
            raise IOError("short pread at %d" % (source_offset + done))
        write_all(dst_fd, data)
        done += count
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(src_fd, source_offset + done - count, count,
                             os.POSIX_FADV_DONTNEED)


def stage(records, output_dir, layer, expert, chunk_bytes, force):
    output_dir.mkdir(parents=True, exist_ok=True)
    blob = output_dir / ("layer%02d_expert%03d.blob" % (layer, expert))
    manifest = output_dir / ("layer%02d_expert%03d.manifest" % (layer, expert))
    if not force and (blob.exists() or manifest.exists()):
        raise FileExistsError("output exists; pass --force to replace %s" % output_dir)
    token = ".tmp.%d" % os.getpid()
    blob_tmp = Path(str(blob) + token)
    manifest_tmp = Path(str(manifest) + token)
    entries = []
    dst = None
    try:
        dst = os.open(str(blob_tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        offset = 0
        for record in records:
            aligned = (offset + ALIGN - 1) & ~(ALIGN - 1)
            if aligned != offset:
                write_all(dst, b"\0" * (aligned - offset))
            src = os.open(record["source"], os.O_RDONLY)
            try:
                copy_range(src, dst, record["source_offset"], record["nbytes"], chunk_bytes)
            finally:
                os.close(src)
            entry = dict(record)
            entry["offset"] = aligned
            entries.append(entry)
            offset = aligned + record["nbytes"]
        os.fsync(dst)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(dst, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(dst)
        dst = None
        with open(str(manifest_tmp), "w") as f:
            f.write("# K3EXPERTV1 layer=%d expert=%d tensors=%d blob_bytes=%d\n" %
                    (layer, expert, len(entries), offset))
            for e in entries:
                f.write("%d %d %s %d %s %s\n" %
                        (e["offset"], e["nbytes"], e["dtype"], len(e["shape"]),
                         " ".join(str(x) for x in e["shape"]), e["name"]))
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(blob_tmp), str(blob))
        os.replace(str(manifest_tmp), str(manifest))
        return blob, manifest, offset
    except Exception:
        if dst is not None:
            os.close(dst)
        for path in (blob_tmp, manifest_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "models/kimi-k3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.layer < 1 or args.layer > 92:
        parser.error("expert layers are 1..92")
    if args.nodes < 1 or args.rank < 0 or args.rank >= args.nodes:
        parser.error("invalid rank/nodes")
    if args.expert < 0 or args.expert >= 896:
        parser.error("expert must be in [0,895]")
    if args.expert % args.nodes != args.rank:
        parser.error("expert %d belongs to rank %d for %d nodes" %
                     (args.expert, args.expert % args.nodes, args.nodes))
    if args.chunk_mib < 1 or args.chunk_mib > 64:
        parser.error("--chunk-mib must be in [1,64]")
    if not args.output_dir and not args.validate_all:
        parser.error("--output-dir is required unless --validate-all is used")
    available = mem_available_kb()
    if available < MIN_AVAILABLE_KB:
        raise MemoryError("MemAvailable %.2f GiB is below the 6 GiB guard" %
                          (available / 1048576.0))
    if args.validate_all:
        validate_checkpoint(args.model_dir)
        if not args.output_dir:
            return
    records = locate(args.model_dir, args.layer, args.expert)
    total = sum(r["nbytes"] for r in records)
    print("K3 partial plan: layer=%d expert=%d rank=%d/%d tensors=%d bytes=%d (%.3f MiB)" %
          (args.layer, args.expert, args.rank, args.nodes, len(records), total, total / 1048576.0))
    for r in records:
        print("  %-16s %8.3f MiB  %s" %
              (r["name"].rsplit(".", 2)[-2] + "." + r["name"].rsplit(".", 1)[-1],
               r["nbytes"] / 1048576.0, Path(r["source"]).name))
    if args.plan_only:
        return
    blob, manifest, size = stage(records, args.output_dir, args.layer, args.expert,
                                 args.chunk_mib * 1024 * 1024, args.force)
    print("staged %.3f MiB: %s" % (size / 1048576.0, blob))
    print("manifest: %s" % manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("k3_stage: %s" % exc, file=sys.stderr)
        sys.exit(2)
