#!/usr/bin/env python3
"""Compare all DA3 output modalities: depth, confidence, pose, rays, ray_confidence.

Usage:
    python compare_all.py --ref-dir results/ --our-dir results/cpu_npy/
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def pearson_r(a, b):
    a_flat, b_flat = a.flatten(), b.flatten()
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def metrics_2d(ref, ours, name):
    """Compute metrics for 2D (or flattened) arrays."""
    diff = ours - ref
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_ae = float(np.max(np.abs(diff)))
    r = pearson_r(ref, ours)
    return {"name": name, "Pearson r": r, "MAE": mae, "Max AE": max_ae,
            "RMSE": float(np.sqrt(mse)),
            "ref range": f"[{ref.min():.4f}, {ref.max():.4f}]",
            "our range": f"[{ours.min():.4f}, {ours.max():.4f}]"}


def print_result(m, threshold_r=0.999):
    r = m["Pearson r"]
    status = "PASS" if r >= threshold_r else "FAIL"
    print(f"  {m['name']:20s}  r={r:.6f}  MAE={m['MAE']:.6f}  MaxAE={m['Max AE']:.6f}  "
          f"RMSE={m['RMSE']:.6f}  {status}")
    print(f"    {'':20s}  ref={m['ref range']}  ours={m['our range']}")
    return r >= threshold_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True, help="Dir with official_*.npy files")
    parser.add_argument("--our-dir", required=True, help="Dir with our depth.npy, rays.npy, etc")
    parser.add_argument("--threshold", type=float, default=0.999, help="Pearson r threshold")
    args = parser.parse_args()

    ref = Path(args.ref_dir)
    ours = Path(args.our_dir)
    all_pass = True

    print("=" * 80)
    print("DA3 All-Modality Comparison: Official vs CPU")
    print("=" * 80)

    # --- Depth ---
    ref_depth = ref / "official_depth_2100x1400.npy"
    our_depth = ours / "depth.npy"
    if ref_depth.exists() and our_depth.exists():
        rd = np.load(str(ref_depth))
        od = np.load(str(our_depth))
        m = metrics_2d(rd, od, "depth")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP depth: ref={ref_depth.exists()}, ours={our_depth.exists()}")

    # --- Confidence ---
    # Official confidence is from 518x518 output; ours is upsampled to 2100x1400
    # We need to compare at same resolution. Check if 2100x1400 conf exists.
    our_conf = ours / "confidence.npy"
    if our_conf.exists():
        oc = np.load(str(our_conf))
        # Try to find matching reference
        ref_conf_full = ref / "official_confidence_2100x1400.npy"
        ref_conf_518 = ref / "official_depth_conf_518.npy"
        if ref_conf_full.exists():
            rc = np.load(str(ref_conf_full))
            m = metrics_2d(rc, oc, "confidence")
            if not print_result(m, args.threshold):
                all_pass = False
        elif ref_conf_518.exists():
            print(f"  confidence: ref only at 518x518, ours at {oc.shape} — range comparison only")
            rc518 = np.load(str(ref_conf_518))
            print(f"    ref 518: [{rc518.min():.4f}, {rc518.max():.4f}], mean={rc518.mean():.4f}")
            print(f"    ours:    [{oc.min():.4f}, {oc.max():.4f}], mean={oc.mean():.4f}")
        else:
            print(f"  SKIP confidence: no ref file found")

    # --- Pose ---
    ref_pose = ref / "official_pose.npy"
    our_pose = ours / "pose.npy"
    if ref_pose.exists() and our_pose.exists():
        rp = np.load(str(ref_pose)).flatten()
        op = np.load(str(our_pose)).flatten()
        diff = np.abs(rp - op)
        max_err = float(diff.max())
        print(f"\n  {'pose':20s}  max_abs_err={max_err:.6f}  {'PASS' if max_err < 0.01 else 'FAIL'}")
        print(f"    ref t ={rp[:3]}")
        print(f"    our t ={op[:3]}")
        print(f"    ref q ={rp[3:7]}")
        print(f"    our q ={op[3:7]}")
        print(f"    ref fov={rp[7:]}")
        print(f"    our fov={op[7:]}")
        if max_err >= 0.01:
            all_pass = False
    else:
        print(f"  SKIP pose: ref={ref_pose.exists()}, ours={our_pose.exists()}")

    # --- Rays ---
    ref_rays = ref / "official_rays_2100x1400.npy"
    our_rays = ours / "rays.npy"
    if ref_rays.exists() and our_rays.exists():
        rr = np.load(str(ref_rays))
        orr = np.load(str(our_rays))
        print()
        # Overall
        m = metrics_2d(rr, orr, "rays (all 6ch)")
        if not print_result(m, args.threshold):
            all_pass = False
        # Per-channel
        for ch in range(min(6, rr.shape[0])):
            m = metrics_2d(rr[ch], orr[ch], f"  rays ch{ch}")
            print_result(m, args.threshold)
    else:
        print(f"  SKIP rays: ref={ref_rays.exists()}, ours={our_rays.exists()}")

    # --- Ray confidence ---
    ref_rc = ref / "official_ray_confidence_2100x1400.npy"
    our_rc = ours / "ray_confidence.npy"
    if ref_rc.exists() and our_rc.exists():
        rrc = np.load(str(ref_rc))
        orc = np.load(str(our_rc))
        print()
        m = metrics_2d(rrc, orc, "ray_confidence")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP ray_confidence: ref={ref_rc.exists()}, ours={our_rc.exists()}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 80)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
