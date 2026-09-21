#!/usr/bin/env python3
"""Decode paired final latents with one native VAE to inspect quantization loss.

This isolates denoiser differences using the same validated F32 decoder on
both sides. It is not a separate PyTorch VAE parity test or an image-quality
acceptance claim: inspect the saved images as well as the reported metrics.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from compare import _cosine_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reference-latents", type=Path, required=True)
    ap.add_argument("--quantized-latents", type=Path, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if min(args.height, args.width) < 32 or args.height % 32 or args.width % 32:
        ap.error("height and width must be positive multiples of 32")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    decoded = []
    for label, source in (("reference", args.reference_latents), ("quantized", args.quantized_latents)):
        latents = np.load(source, allow_pickle=False)
        count = (args.height // 16) * (args.width // 16)
        if latents.shape == (1, count, 64):
            latents = latents[0]
        if latents.shape != (count, 64) or not np.isfinite(latents).all():
            raise ValueError("invalid packed final latents")
        packed = args.out_dir / f"{label}_latents.npy"
        np.save(packed, np.ascontiguousarray(latents, dtype=np.float32))
        out = args.out_dir / f"{label}.npy"
        subprocess.run([str(Path(__file__).with_name("test_cuda_qimg21_vae").resolve()),
                        "--model", str(args.model / "vae"), "--latents", str(packed),
                        "--height-tokens", str(args.height // 16),
                        "--width-tokens", str(args.width // 16), "--out", str(out)], check=True)
        image = np.load(out, allow_pickle=False)
        if image.shape != (4, args.height, args.width) or not np.isfinite(image).all():
            raise ValueError("invalid decoded image")
        pixels = np.clip(image * 0.5 + 0.5, 0, 1)
        Image.fromarray(np.rint(pixels.transpose(1, 2, 0) * 255).astype(np.uint8)).save(
            args.out_dir / f"{label}.png")
        decoded.append(pixels.astype(np.float64))
    ref, quant = decoded
    cosine, relative_l2 = _cosine_error(ref[:3], quant[:3])
    mse = float(np.mean((ref[:3] - quant[:3]) ** 2))
    result = {"decoder": "shared native F32 original-weight VAE",
              "reference_latents": str(args.reference_latents.resolve()),
              "quantized_latents": str(args.quantized_latents.resolve()),
              "finite": True, "rgb_cosine": cosine, "rgb_relative_l2": relative_l2,
              "rgb_mae": float(np.mean(np.abs(ref[:3] - quant[:3]))),
              "rgb_psnr_db": float(-10 * np.log10(mse)) if mse else None,
              "alpha_mae": float(np.mean(np.abs(ref[3] - quant[3]))),
              "visual_review_required": True}
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
