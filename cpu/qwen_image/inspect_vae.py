#!/usr/bin/env python3
"""Inspect Qwen-Image VAE safetensors tensor names and shapes."""
import struct, json, sys

def inspect_safetensors(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    metadata = json.loads(header_json)

    keys = [k for k in metadata.keys() if k != "__metadata__"]
    print(f"Safetensors: {path}")
    print(f"Total tensors: {len(keys)}")

    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8}
    total_params = 0

    # Group by prefix
    groups = {}
    for k in sorted(keys):
        info = metadata[k]
        shape = info["shape"]
        dtype = info["dtype"]
        numel = 1
        for s in shape:
            numel *= s
        total_params += numel

        parts = k.split(".")
        # Find group - look for numbered parts
        prefix_parts = []
        for i, p in enumerate(parts):
            if p.isdigit():
                prefix_parts.append("N")
                break
            prefix_parts.append(p)
            if i >= 2:
                break
        prefix = ".".join(prefix_parts)

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((k, shape, dtype, numel))

    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print()

    # Print all tensors
    print("=== All Tensors ===")
    for k in sorted(keys):
        info = metadata[k]
        print(f"  {k}: {info['shape']} ({info['dtype']})")

    # Print groups
    print("\n=== Groups ===")
    for prefix in sorted(groups):
        group = groups[prefix]
        group_params = sum(n for _, _, _, n in group)
        print(f"\n[{prefix}] ({len(group)} tensors, {group_params:,} params)")
        shown = set()
        for name, shape, dtype, numel in group:
            # Generalize numbered parts
            parts = name.split(".")
            generic_parts = []
            for p in parts:
                if p.isdigit():
                    generic_parts.append("N")
                else:
                    generic_parts.append(p)
            generic = ".".join(generic_parts)
            if generic not in shown:
                shown.add(generic)
                print(f"    {generic}: {shape} ({dtype})")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors"
    inspect_safetensors(path)
