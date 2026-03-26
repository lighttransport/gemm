#!/usr/bin/env python3
"""
verify_trellis2.py - Inspect TRELLIS.2 weights and generate reference outputs.

Usage:
  # Inspect Stage 1 flow model weights
  python verify_trellis2.py --inspect <ss_flow_*.safetensors>

  # Inspect structure decoder weights
  python verify_trellis2.py --inspect <ss_dec_*.safetensors>

  # Compare two .npy files
  python verify_trellis2.py --compare ref.npy c_output.npy

  # Generate reference Stage 1 output (requires TRELLIS.2 repo installed)
  python verify_trellis2.py --generate-stage1 \
      --flow-model <ss_flow_*.safetensors> \
      --cond <features.npy> \
      --output stage1_ref.npy

Dependencies:
  pip install numpy safetensors
  pip install torch (for --generate-stage1)
"""

import argparse
import numpy as np
import sys
import os


def inspect_safetensors(path):
    """Print all tensor names, shapes, and dtypes from a safetensors file."""
    from safetensors import safe_open
    import json, struct

    # Read metadata from header directly to handle BF16
    with open(path, "rb") as fp:
        header_size = struct.unpack("<Q", fp.read(8))[0]
        header_json = fp.read(header_size).decode("utf-8")
    metadata = json.loads(header_json)

    keys = [k for k in metadata.keys() if k != "__metadata__"]
    print(f"Safetensors file: {path}")
    print(f"Total tensors: {len(keys)}")

    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8}

    # Calculate total size
    total_params = 0
    total_bytes = 0

    # Group by top-level prefix
    groups = {}
    for k in sorted(keys):
        info = metadata[k]
        shape = info["shape"]
        dtype = info["dtype"]
        numel = 1
        for s in shape:
            numel *= s
        total_params += numel
        total_bytes += numel * dtype_sizes.get(dtype, 2)

        # Find group prefix
        parts = k.split(".")
        if parts[0] == "blocks" or parts[0] == "backbone":
            if len(parts) >= 2 and parts[0] == "backbone":
                prefix = f"backbone.{parts[1]}"
            elif parts[1].isdigit():
                prefix = f"blocks.N"
            else:
                prefix = parts[0]
        else:
            prefix = parts[0]

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((k, shape, dtype, numel))

        print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"Total size: {total_bytes:,} bytes ({total_bytes/1e9:.2f} GB)")
        print()

        for prefix in sorted(groups):
            group = groups[prefix]
            group_params = sum(n for _, _, _, n in group)
            print(f"[{prefix}] ({len(group)} tensors, {group_params:,} params)")
            # For blocks.N, just show block 0 as representative
            shown = set()
            for name, shape, dtype, numel in group:
                display_name = name
                if prefix == "blocks.N":
                    # Replace block number with N for dedup
                    generic = name.replace(name.split(".")[1], "N", 1) if "blocks." in name else name
                    if generic in shown:
                        continue
                    shown.add(generic)
                    display_name = generic
                print(f"  {display_name}: {shape} ({dtype})")
            print()


def compare_outputs(ref_path, c_path):
    """Compare reference and C implementation outputs."""
    ref = np.load(ref_path)
    c_out = np.load(c_path)

    print(f"Reference: shape={ref.shape}, dtype={ref.dtype}")
    print(f"C output:  shape={c_out.shape}, dtype={c_out.dtype}")

    if ref.shape != c_out.shape:
        # Try reshape
        if np.prod(ref.shape) == np.prod(c_out.shape):
            print(f"Reshaping C output from {c_out.shape} to {ref.shape}")
            c_out = c_out.reshape(ref.shape)
        else:
            print(f"ERROR: Shape mismatch! ({np.prod(ref.shape)} vs {np.prod(c_out.shape)} elements)")
            return False

    ref_flat = ref.flatten()
    c_flat = c_out.flatten()

    # Overall stats
    abs_diff = np.abs(ref_flat - c_flat)
    print(f"\nRef stats:    min={ref_flat.min():.4f}, max={ref_flat.max():.4f}, mean={ref_flat.mean():.6f}")
    print(f"C stats:      min={c_flat.min():.4f}, max={c_flat.max():.4f}, mean={c_flat.mean():.6f}")
    print(f"\nAbs diff:     max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")

    # Correlation
    if len(ref_flat) > 1:
        corr = np.corrcoef(ref_flat, c_flat)[0, 1]
        print(f"Correlation:  r={corr:.6f}")

    # Per-channel/token correlation if multidim
    if ref.ndim >= 2:
        n_tokens = ref.shape[0]
        dim = np.prod(ref.shape[1:])
        ref2d = ref.reshape(n_tokens, dim)
        c2d = c_out.reshape(n_tokens, dim)
        corrs = []
        for t in range(n_tokens):
            r = np.corrcoef(ref2d[t], c2d[t])[0, 1]
            corrs.append(r)
        corrs = np.array(corrs)
        print(f"\nPer-token correlation:")
        print(f"  Mean: {np.nanmean(corrs):.6f}")
        print(f"  Min:  {np.nanmin(corrs):.6f}")

    print(f"\n{'PASS' if abs_diff.max() < 0.1 else 'CHECK'}: max abs diff = {abs_diff.max():.6f}")
    return True


def visualize_occupancy(path, threshold=0.0):
    """Visualize 64^3 occupancy grid."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return

    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(64, 64, 64)
    elif data.ndim == 4:
        data = data[0]  # Remove channel dim
    print(f"Occupancy shape: {data.shape}")
    print(f"Stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.6f}")

    occupied = data > threshold
    n_occ = occupied.sum()
    print(f"Occupied voxels (logit > {threshold}): {n_occ}/{data.size} ({100*n_occ/data.size:.1f}%)")

    if n_occ == 0:
        print("No occupied voxels to visualize!")
        return

    # 3D scatter plot
    z, y, x = np.where(occupied)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Occupancy ({n_occ} voxels)')

    out_path = path.replace('.npy', '_vis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 verification tool")
    parser.add_argument("--inspect", type=str,
                        help="Inspect safetensors file")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "C_OUTPUT"),
                        help="Compare two .npy files")
    parser.add_argument("--visualize", type=str,
                        help="Visualize occupancy .npy file")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Occupancy threshold (default: 0.0)")

    args = parser.parse_args()

    if args.inspect:
        inspect_safetensors(args.inspect)
    elif args.compare:
        compare_outputs(args.compare[0], args.compare[1])
    elif args.visualize:
        visualize_occupancy(args.visualize, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
