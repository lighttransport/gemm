#!/usr/bin/env python3
"""
Compare DA3 depth outputs: reference .npy vs our implementation's output.

Supports both .npy (raw float32) and .pgm (normalized uint16) inputs.
For .pgm inputs, uses scale-invariant alignment.

Usage:
    uv run python compare.py \
        --reference output/depth_ref.npy \
        --ours output/depth_cpu.npy \
        --label "CPU vs Reference"
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_pgm(path):
    """Load a 16-bit PGM (P5) file as float32 array."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Not a PGM file: {path}")
        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        if maxval == 65535:
            data = np.frombuffer(f.read(), dtype=np.uint16).byteswap()
            data = data.reshape(h, w).astype(np.float32)
        elif maxval == 255:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(h, w).astype(np.float32)
        else:
            raise ValueError(f"Unsupported PGM maxval: {maxval}")
    return data


def load_depth(path):
    """Load depth map from .npy or .pgm file."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(str(path)).astype(np.float32)
        return data, "npy"
    elif path.suffix == ".pgm":
        data = load_pgm(str(path))
        return data, "pgm"
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def compute_ssim_simple(a, b, win_size=7):
    """Compute SSIM between two 2D arrays (simplified, no skimage dependency)."""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(a, b, data_range=a.max() - a.min())
    except ImportError:
        # Fallback: global SSIM approximation
        C1 = (0.01 * (a.max() - a.min())) ** 2
        C2 = (0.03 * (a.max() - a.min())) ** 2
        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.std(), b.std()
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))
        return float(ssim)


def scale_invariant_align(ref, pred):
    """
    Find optimal a, b such that a*pred + b ≈ ref (least-squares).
    Returns aligned prediction.
    """
    pred_flat = pred.flatten()
    ref_flat = ref.flatten()
    # Solve: [pred, 1] @ [a, b]^T = ref
    A = np.column_stack([pred_flat, np.ones_like(pred_flat)])
    result = np.linalg.lstsq(A, ref_flat, rcond=None)
    coeffs = result[0]
    a, b = coeffs[0], coeffs[1]
    aligned = a * pred + b
    return aligned, a, b


def compute_metrics(ref, pred, label=""):
    """Compute and print comparison metrics."""
    assert ref.shape == pred.shape, \
        f"Shape mismatch: ref={ref.shape} vs pred={pred.shape}"

    n = ref.size
    diff = pred - ref

    # Direct metrics
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_ae = float(np.max(np.abs(diff)))
    pearson_r = float(np.corrcoef(ref.flatten(), pred.flatten())[0, 1])
    ssim = compute_ssim_simple(ref, pred)

    # Relative metrics (avoid division by zero)
    ref_abs = np.abs(ref)
    mask = ref_abs > 1e-6
    if mask.sum() > 0:
        rel_mae = float(np.mean(np.abs(diff[mask]) / ref_abs[mask]))
    else:
        rel_mae = float("inf")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Max AE": max_ae,
        "Rel MAE": rel_mae,
        "Pearson r": pearson_r,
        "SSIM": ssim,
    }


def print_metrics(metrics, label=""):
    """Print metrics table."""
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 60}")
    for name, val in metrics.items():
        if isinstance(val, float):
            if abs(val) < 0.001 or abs(val) > 1e6:
                print(f"  {name:20s}: {val:.6e}")
            else:
                print(f"  {name:20s}: {val:.6f}")
    print(f"{'=' * 60}")


def save_diff_heatmap(ref, pred, output_path, label=""):
    """Save absolute difference as a heatmap PNG."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  (skipping heatmap: Pillow not available)")
        return

    diff = np.abs(pred - ref)
    # Normalize to [0, 255]
    d_max = diff.max()
    if d_max < 1e-10:
        d_max = 1.0
    heatmap = (diff / d_max * 255).astype(np.uint8)

    # Apply a simple colormap (blue=0, red=max)
    h, w = heatmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = heatmap           # Red channel = high error
    rgb[:, :, 2] = 255 - heatmap     # Blue channel = low error

    img = Image.fromarray(rgb)
    img.save(str(output_path))
    print(f"  Saved difference heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DA3 depth outputs")
    parser.add_argument("--reference", required=True,
                        help="Reference depth file (.npy or .pgm)")
    parser.add_argument("--ours", required=True,
                        help="Our depth output file (.npy or .pgm)")
    parser.add_argument("--label", default="",
                        help="Label for this comparison")
    parser.add_argument("--tolerance-r", type=float, default=0.999,
                        help="Minimum Pearson correlation for pass")
    parser.add_argument("--tolerance-ssim", type=float, default=0.99,
                        help="Minimum SSIM for pass")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for heatmap output (default: same as --ours)")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    our_path = Path(args.ours)

    if not ref_path.exists():
        print(f"ERROR: Reference file not found: {ref_path}")
        sys.exit(1)
    if not our_path.exists():
        print(f"ERROR: Our output file not found: {our_path}")
        sys.exit(1)

    ref, ref_fmt = load_depth(ref_path)
    ours, our_fmt = load_depth(our_path)

    print(f"Reference: {ref_path} ({ref_fmt}, {ref.shape}, "
          f"min={ref.min():.4f}, max={ref.max():.4f})")
    print(f"Ours:      {our_path} ({our_fmt}, {ours.shape}, "
          f"min={ours.min():.4f}, max={ours.max():.4f})")

    # Handle shape mismatch by resizing ours to match reference
    if ref.shape != ours.shape:
        print(f"WARNING: Shape mismatch, resizing ours {ours.shape} -> {ref.shape}")
        from PIL import Image
        ours_img = Image.fromarray(ours)
        ours_img = ours_img.resize((ref.shape[1], ref.shape[0]),
                                    Image.BILINEAR)
        ours = np.array(ours_img, dtype=np.float32)

    # Direct comparison
    direct = compute_metrics(ref, ours, args.label)
    print_metrics(direct, f"{args.label} (direct)")

    # Scale-invariant comparison (useful when one is normalized PGM)
    need_alignment = (ref_fmt == "pgm" or our_fmt == "pgm") and \
                     (ref_fmt != our_fmt)
    if need_alignment:
        aligned, a, b = scale_invariant_align(ref, ours)
        print(f"\n  Scale-invariant alignment: a={a:.6f}, b={b:.6f}")
        si_metrics = compute_metrics(ref, aligned, args.label)
        print_metrics(si_metrics, f"{args.label} (scale-invariant)")
        eval_metrics = si_metrics
    else:
        # Even for npy-vs-npy, do alignment for a more robust comparison
        aligned, a, b = scale_invariant_align(ref, ours)
        print(f"\n  Scale alignment check: a={a:.6f}, b={b:.6f}")
        si_metrics = compute_metrics(ref, aligned, args.label)
        print_metrics(si_metrics, f"{args.label} (scale-invariant)")
        # Use direct metrics for pass/fail when both are npy
        eval_metrics = direct

    # Save difference heatmap
    out_dir = Path(args.output_dir) if args.output_dir else our_path.parent
    safe_label = args.label.replace(" ", "_").replace("/", "_").lower()
    if safe_label:
        heatmap_path = out_dir / f"diff_{safe_label}.png"
    else:
        heatmap_path = out_dir / f"diff_{our_path.stem}.png"
    save_diff_heatmap(ref, ours, heatmap_path, args.label)

    # Pass/fail
    r = eval_metrics["Pearson r"]
    ssim = eval_metrics["SSIM"]
    pass_r = r >= args.tolerance_r
    pass_ssim = ssim >= args.tolerance_ssim
    passed = pass_r and pass_ssim

    print(f"\n  Pearson r:  {r:.6f}  (threshold: {args.tolerance_r})  "
          f"{'PASS' if pass_r else 'FAIL'}")
    print(f"  SSIM:       {ssim:.6f}  (threshold: {args.tolerance_ssim})  "
          f"{'PASS' if pass_ssim else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
