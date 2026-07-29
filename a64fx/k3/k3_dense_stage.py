#!/usr/bin/env python3
"""Stage the real K3 router and routed-latent down projection for one layer."""
import argparse
import os
import sys
from pathlib import Path

import k3_stage

ALIGN = 256
EXPECTED = {
    "block_sparse_moe.gate.weight": ("BF16", [896, 7168]),
    "block_sparse_moe.routed_expert_down_proj.weight": ("BF16", [3584, 7168]),
}


def locate(model_dir, layer):
    prefix = "language_model.model.layers.%d." % layer
    wanted = {prefix + suffix: (suffix, dtype, shape)
              for suffix, (dtype, shape) in EXPECTED.items()}
    found = {}
    paths = sorted(model_dir.glob("*.safetensors"))
    if len(paths) != 96:
        raise ValueError("expected 96 shards, found %d" % len(paths))
    for path in paths:
        data_start, header = k3_stage.read_header(path)
        for name in set(wanted).intersection(header):
            suffix, dtype, shape = wanted[name]
            info = header[name]
            if info["dtype"] != dtype or info["shape"] != shape:
                raise ValueError("shape mismatch for %s" % name)
            begin, end = info["data_offsets"]
            found[suffix] = {"name": name, "source": str(path),
                "source_offset": data_start + begin, "nbytes": end - begin,
                "dtype": dtype, "shape": shape}
    missing = sorted(set(EXPECTED) - set(found))
    if missing:
        raise ValueError("missing tensors: %s" % ", ".join(missing))
    return [found[suffix] for suffix in EXPECTED]


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]


def stage(records, output_dir, layer, chunk_bytes):
    output_dir.mkdir(parents=True, exist_ok=True)
    blob = output_dir / ("layer%02d_dense.blob" % layer)
    manifest = output_dir / ("layer%02d_dense.manifest" % layer)
    if blob.exists() or manifest.exists():
        raise FileExistsError("output exists: %s" % output_dir)
    btmp = Path(str(blob) + ".tmp.%d" % os.getpid())
    mtmp = Path(str(manifest) + ".tmp.%d" % os.getpid())
    fd = os.open(str(btmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        entries = []
        for rec in records:
            aligned = (offset + ALIGN - 1) & ~(ALIGN - 1)
            if aligned > offset:
                write_all(fd, b"\0" * (aligned - offset))
            src = os.open(rec["source"], os.O_RDONLY)
            try:
                k3_stage.copy_range(src, fd, rec["source_offset"],
                                    rec["nbytes"], chunk_bytes)
            finally:
                os.close(src)
            ent = dict(rec)
            ent["offset"] = aligned
            entries.append(ent)
            offset = aligned + rec["nbytes"]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        with mtmp.open("w") as f:
            f.write("# K3DENSEV1 layer=%d tensors=%d blob_bytes=%d\n" %
                    (layer, len(entries), offset))
            for ent in entries:
                f.write("%d %d %s %d %s %s\n" %
                    (ent["offset"], ent["nbytes"], ent["dtype"],
                     len(ent["shape"]), " ".join(map(str, ent["shape"])),
                     ent["name"]))
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(btmp), str(blob))
        os.replace(str(mtmp), str(manifest))
        return blob, manifest, offset
    except Exception:
        if fd >= 0:
            os.close(fd)
        for path in (btmp, mtmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path.home() / "models/kimi-k3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.layer < 1 or args.layer > 92:
        parser.error("layer must be in [1,92]")
    if k3_stage.mem_available_kb() < k3_stage.MIN_AVAILABLE_KB:
        raise MemoryError("MemAvailable below 6 GiB")
    records = locate(args.model_dir, args.layer)
    blob, manifest, size = stage(records, args.output_dir, args.layer,
                                 args.chunk_mib * 1048576)
    print("staged K3 dense slice %.3f MiB: %s" % (size / 1048576.0, blob))
    print("manifest: %s" % manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("k3_dense_stage: %s" % exc, file=sys.stderr)
        sys.exit(2)
