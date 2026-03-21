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


def find_file(directory, candidates):
    """Find first existing file from a list of candidate names or glob patterns."""
    for name in candidates:
        if '*' in name:
            matches = sorted(directory.glob(name))
            if matches:
                return matches[0]
        else:
            p = directory / name
            if p.exists():
                return p
    return None


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
    ref_depth = find_file(ref, ["official_depth_2100x1400.npy", "depth_ref.npy",
                                 "depth_*_ref.npy", "depth.npy"])
    our_depth = find_file(ours, ["depth.npy", "depth_ref.npy"])
    if ref_depth and our_depth:
        rd = np.load(str(ref_depth))
        od = np.load(str(our_depth))
        if rd.shape != od.shape:
            print(f"  depth shape mismatch: ref={rd.shape}, ours={od.shape} — resizing ours")
            from scipy.ndimage import zoom
            factors = tuple(r / o for r, o in zip(rd.shape, od.shape))
            od = zoom(od, factors, order=1)
        m = metrics_2d(rd, od, "depth")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP depth: ref={ref_depth}, ours={our_depth}")

    # --- Confidence ---
    ref_conf = find_file(ref, ["official_confidence_2100x1400.npy", "confidence_ref.npy",
                                "confidence_*_ref.npy", "confidence.npy"])
    our_conf = find_file(ours, ["confidence.npy", "confidence_ref.npy"])
    if ref_conf and our_conf:
        rc = np.load(str(ref_conf))
        oc = np.load(str(our_conf))
        if rc.shape != oc.shape:
            print(f"  confidence shape mismatch: ref={rc.shape}, ours={oc.shape} — resizing ours")
            from scipy.ndimage import zoom
            factors = tuple(r / o for r, o in zip(rc.shape, oc.shape))
            oc = zoom(oc, factors, order=1)
        m = metrics_2d(rc, oc, "confidence")
        if not print_result(m, args.threshold):
            all_pass = False
    elif our_conf:
        oc = np.load(str(our_conf))
        ref_conf_518 = ref / "official_depth_conf_518.npy"
        if ref_conf_518.exists():
            print(f"  confidence: ref only at 518x518, ours at {oc.shape} — range comparison only")
            rc518 = np.load(str(ref_conf_518))
            print(f"    ref 518: [{rc518.min():.4f}, {rc518.max():.4f}], mean={rc518.mean():.4f}")
            print(f"    ours:    [{oc.min():.4f}, {oc.max():.4f}], mean={oc.mean():.4f}")
        else:
            print(f"  SKIP confidence: no ref file found")

    # --- Pose ---
    ref_pose = find_file(ref, ["official_pose.npy", "pose_ref.npy", "pose_*_ref.npy", "pose.npy"])
    our_pose = find_file(ours, ["pose.npy", "pose_ref.npy"])
    if ref_pose and our_pose:
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
        print(f"  SKIP pose: ref={ref_pose}, ours={our_pose}")

    # --- Rays ---
    ref_rays = find_file(ref, ["official_rays_2100x1400.npy", "rays_ref.npy",
                                "rays_*_ref.npy", "rays.npy"])
    our_rays = find_file(ours, ["rays.npy", "rays_ref.npy"])
    if ref_rays and our_rays:
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
        print(f"  SKIP rays: ref={ref_rays}, ours={our_rays}")

    # --- Ray confidence ---
    ref_rc = find_file(ref, ["official_ray_confidence_2100x1400.npy", "ray_confidence_ref.npy",
                              "ray_conf_*_ref.npy", "ray_confidence.npy"])
    our_rc = find_file(ours, ["ray_confidence.npy", "ray_confidence_ref.npy"])
    if ref_rc and our_rc:
        rrc = np.load(str(ref_rc))
        orc = np.load(str(our_rc))
        print()
        m = metrics_2d(rrc, orc, "ray_confidence")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP ray_confidence: ref={ref_rc}, ours={our_rc}")

    # --- Gaussians ---
    ref_gauss = find_file(ref, ["gaussians_ref.npy", "gaussians_*_ref.npy", "gaussians.npy"])
    our_gauss = find_file(ours, ["gaussians.npy", "gaussians_ref.npy"])
    if ref_gauss and our_gauss:
        rg = np.load(str(ref_gauss))
        og = np.load(str(our_gauss))
        print()
        m = metrics_2d(rg, og, "gaussians")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP gaussians: ref={ref_gauss}, ours={our_gauss}")

    # --- Metric depth (DA3-Nested) ---
    ref_md = find_file(ref, ["metric_depth_ref.npy", "metric_depth.npy"])
    our_md = find_file(ours, ["metric_depth.npy", "metric_depth_ref.npy"])
    if ref_md and our_md:
        rmd = np.load(str(ref_md))
        omd = np.load(str(our_md))
        if rmd.shape != omd.shape:
            print(f"  metric_depth shape mismatch: ref={rmd.shape}, ours={omd.shape} — resizing ours")
            from scipy.ndimage import zoom
            factors = tuple(r / o for r, o in zip(rmd.shape, omd.shape))
            omd = zoom(omd, factors, order=1)
        print()
        m = metrics_2d(rmd, omd, "metric_depth")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP metric_depth: ref={ref_md}, ours={our_md}")

    # --- Sky segmentation (DA3-Nested) ---
    ref_sky = find_file(ref, ["sky_seg_ref.npy", "sky_seg.npy"])
    our_sky = find_file(ours, ["sky_seg.npy", "sky_seg_ref.npy"])
    if ref_sky and our_sky:
        rsk = np.load(str(ref_sky))
        osk = np.load(str(our_sky))
        if rsk.shape != osk.shape:
            print(f"  sky_seg shape mismatch: ref={rsk.shape}, ours={osk.shape} — resizing ours")
            from scipy.ndimage import zoom
            factors = tuple(r / o for r, o in zip(rsk.shape, osk.shape))
            osk = zoom(osk, factors, order=1)
        print()
        m = metrics_2d(rsk, osk, "sky_seg")
        if not print_result(m, args.threshold):
            all_pass = False
    else:
        print(f"  SKIP sky_seg: ref={ref_sky}, ours={our_sky}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 80)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
