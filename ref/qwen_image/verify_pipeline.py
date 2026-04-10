#!/usr/bin/env python3
"""
Qwen-Image Pipeline Verification — compare our CUDA output against ComfyUI reference.

Usage:
    # Compare all available stages (text encoder, DiT blocks, VAE layers, final image)
    python verify_pipeline.py

    # Specify directories
    python verify_pipeline.py --cuda-dir ../../cuda/qimg/ --ref-dir output/

    # Only check specific components
    python verify_pipeline.py --text --dit-blocks --vae --image

Prerequisites:
    1. Generate ComfyUI reference data:
       python generate_reference.py --all
    2. Generate our CUDA output with debug dumps:
       cd ../../cuda/qimg && ./test_cuda_qimg --generate --verbose 3
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_npy(path):
    """Load .npy file."""
    return np.load(path).astype(np.float32).flatten()


def load_bin_f32(path, shape=None):
    """Load raw float32 binary file."""
    data = np.fromfile(path, dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data.flatten()


def load_ppm(path):
    """Load PPM (P6) image as float32 array [0, 255]."""
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P6'
        w, h = map(int, f.readline().split())
        assert f.readline().strip() == b'255'
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
    return data, h, w


def compare(ref, ours, label):
    """Compare two float arrays. Returns (status, metrics_str)."""
    if ref.size != ours.size:
        n = min(ref.size, ours.size)
        ref, ours = ref[:n], ours[:n]

    diff = np.abs(ref - ours)
    max_diff = diff.max()
    mean_diff = diff.mean()

    if ref.size > 10 and np.std(ref) > 1e-10 and np.std(ours) > 1e-10:
        corr = np.corrcoef(ref, ours)[0, 1]
    else:
        corr = float('nan')

    if corr > 0.999 or max_diff < 0.001:
        status = "PASS"
    elif corr > 0.99 or max_diff < 0.01:
        status = "PASS"
    elif corr > 0.95 or max_diff < 0.1:
        status = "MARGINAL"
    else:
        status = "FAIL"

    return status, f"corr={corr:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}"


def ssim_simple(img1, img2):
    """Simple SSIM for uint8 images (no skimage dependency)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2 = img1.var(), img2.var()
    s12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2*mu1*mu2 + c1) * (2*s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))


def print_row(component, check, status, metric):
    """Print one verification row."""
    status_color = {"PASS": "\033[32m", "MARGINAL": "\033[33m", "FAIL": "\033[31m"}
    reset = "\033[0m"
    color = status_color.get(status, "")
    print(f"  {component:<20s} {check:<25s} {color}{status:<10s}{reset} {metric}")


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image pipeline verification")
    parser.add_argument('--cuda-dir', default='../../cuda/qimg/',
                        help='Directory with CUDA output files')
    parser.add_argument('--ref-dir', default='output/',
                        help='Directory with ComfyUI reference .npy files')
    parser.add_argument('--text', action='store_true', help='Check text encoder')
    parser.add_argument('--dit-blocks', action='store_true', help='Check DiT per-block')
    parser.add_argument('--dit-steps', action='store_true', help='Check DiT per-step latents')
    parser.add_argument('--vae', action='store_true', help='Check VAE layers')
    parser.add_argument('--image', action='store_true', help='Check final image')
    args = parser.parse_args()

    # Default: check everything available
    check_all = not (args.text or args.dit_blocks or args.dit_steps or args.vae or args.image)

    cuda = Path(args.cuda_dir)
    ref = Path(args.ref_dir)
    results = {"PASS": 0, "MARGINAL": 0, "FAIL": 0, "SKIP": 0}

    print("=" * 78)
    print("  Qwen-Image Pipeline Verification")
    print("=" * 78)
    print(f"  CUDA dir: {cuda.resolve()}")
    print(f"  Ref dir:  {ref.resolve()}")
    print("-" * 78)

    # --- Text Encoder ---
    if check_all or args.text:
        for suffix, label in [("comfyui_text_hidden.npy", "pos hidden"),
                              ("comfyui_text_hidden_neg.npy", "neg hidden")]:
            ref_path = ref / suffix
            # Our text encoder saves to same location when using cached path
            if ref_path.exists():
                # We need our output — check if cuda dir has a text_hidden dump
                # For now, compare cached reference availability
                print_row("Text Encoder", label, "SKIP", f"(ref exists: {ref_path.name})")
                results["SKIP"] += 1
            else:
                print_row("Text Encoder", label, "SKIP", "no reference data")
                results["SKIP"] += 1

    # --- DiT Per-Block ---
    if check_all or args.dit_blocks:
        n_checked = 0
        for block_idx in range(60):
            ref_img = ref / f"cf_dit_block{block_idx:02d}_img.npy"
            our_img = cuda / f"cuda_dit_block{block_idx:02d}_img.npy"
            if ref_img.exists() and our_img.exists():
                r = load_npy(ref_img)
                o = load_npy(our_img)
                status, metric = compare(r, o, f"block {block_idx}")
                results[status] += 1
                n_checked += 1
                # Only print first, last, and failures
                if block_idx == 0 or block_idx == 59 or status != "PASS":
                    print_row("DiT Block", f"block {block_idx:02d} img", status, metric)
        if n_checked > 2:
            print_row("DiT Block", f"({n_checked} blocks total)", "---", "")
        elif n_checked == 0:
            print_row("DiT Block", "per-block outputs", "SKIP", "no .npy files (run --verbose 3)")
            results["SKIP"] += 1

    # --- DiT Per-Step ---
    if check_all or args.dit_steps:
        n_checked = 0
        for step in range(100):
            ref_step = ref / f"cf_perstep_{step:02d}.npy"
            our_step = cuda / f"cuda_perstep_{step:02d}.bin"
            if ref_step.exists() and our_step.exists():
                r = load_npy(ref_step)
                o = load_bin_f32(our_step)
                status, metric = compare(r, o, f"step {step}")
                results[status] += 1
                n_checked += 1
                print_row("DiT Step", f"step {step:02d} latent", status, metric)
        if n_checked == 0:
            print_row("DiT Step", "per-step latents", "SKIP",
                      "no data (run --verbose 3 + trace_perstep.py)")
            results["SKIP"] += 1

    # --- VAE Layers ---
    if check_all or args.vae:
        vae_stages = [
            ("post_quant", "cuda_vae_post_quant.npy", "cf_vae_post_quant.npy"),
            ("conv1", "cuda_vae_conv1.npy", "cf_vae_conv1.npy"),
            ("middle_0", "cuda_vae_middle_0.npy", "cf_vae_middle_0.npy"),
            ("middle_1", "cuda_vae_middle_1.npy", "cf_vae_middle_1.npy"),
            ("middle_2", "cuda_vae_middle_2.npy", "cf_vae_middle_2.npy"),
        ]
        # Add upsample blocks
        for i in range(15):
            vae_stages.append((f"upsample_{i}", f"cuda_vae_upsample_{i}.npy",
                               f"cf_vae_upsample_{i}.npy"))

        n_checked = 0
        for label, our_name, ref_name in vae_stages:
            our_path = cuda / our_name
            ref_path = ref / ref_name
            if not ref_path.exists():
                # Try ComfyUI output directory
                ref_path = Path("/mnt/disk01/ComfyUI/output") / ref_name
            if our_path.exists() and ref_path.exists():
                r = load_npy(ref_path)
                o = load_npy(our_path)
                status, metric = compare(r, o, label)
                results[status] += 1
                n_checked += 1
                print_row("VAE", label, status, metric)
        if n_checked == 0:
            print_row("VAE", "layer outputs", "SKIP",
                      "no data (run --verbose 3 + run_vae_reference.py)")
            results["SKIP"] += 1

    # --- Final Image ---
    if check_all or args.image:
        our_ppm = cuda / "cuda_qimg_output.ppm"
        # Try several reference image locations
        ref_candidates = [
            ref / "comfyui_ournoise_256.png",
            ref / "ground_truth_comfyui_256.png",
            Path("/mnt/disk01/ComfyUI/output/comfyui_ournoise_256.png"),
        ]
        ref_img_path = None
        for c in ref_candidates:
            if c.exists():
                ref_img_path = c
                break

        if our_ppm.exists() and ref_img_path:
            our_data, h, w = load_ppm(our_ppm)
            try:
                from PIL import Image
                ref_pil = Image.open(ref_img_path).convert('RGB')
                ref_data = np.array(ref_pil).astype(np.float32).flatten()

                # Pixel stats
                if our_data.size == ref_data.size:
                    diff = np.abs(our_data - ref_data)
                    mean_px = diff.mean()
                    max_px = diff.max()

                    # SSIM
                    ssim_val = ssim_simple(our_data, ref_data)

                    status = "PASS" if ssim_val > 0.90 else \
                             "MARGINAL" if ssim_val > 0.60 else "FAIL"
                    print_row("Final Image", f"{w}x{h} SSIM",
                              status, f"ssim={ssim_val:.4f}  mean_px={mean_px:.1f}  max_px={max_px:.0f}")
                    results[status] += 1
                else:
                    print_row("Final Image", f"{w}x{h}", "FAIL",
                              f"size mismatch: {our_data.size} vs {ref_data.size}")
                    results["FAIL"] += 1
            except ImportError:
                print_row("Final Image", "comparison", "SKIP", "PIL not available")
                results["SKIP"] += 1
        else:
            missing = "our PPM" if not our_ppm.exists() else "reference PNG"
            print_row("Final Image", "comparison", "SKIP", f"no {missing}")
            results["SKIP"] += 1

    # --- Summary ---
    print("-" * 78)
    total = results["PASS"] + results["MARGINAL"] + results["FAIL"]
    skipped = results["SKIP"]
    print(f"  Summary: {results['PASS']} PASS, {results['MARGINAL']} MARGINAL, "
          f"{results['FAIL']} FAIL" + (f", {skipped} skipped" if skipped else ""))

    if results["FAIL"] > 0:
        print("  ** FAILURES DETECTED **")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
