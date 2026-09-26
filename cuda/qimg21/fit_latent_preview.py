#!/usr/bin/env python3
"""Fit the linear latent -> RGB map the web demo uses for per-step previews.

A Qwen-Image 2.1 latent token is a 16x16 pixel patch in 64 channels. Running
the VAE on every denoising step is too slow for a progress view, so the demo
maps each token straight to a PATCHxPATCH RGB patch with one affine map, fitted
here by least squares on pairs of final latents and the pictures they decoded
to. The map reads the token and its eight neighbours: a token alone cannot say
how its patch meets the next one, and a per-token map shows every patch border
as a visible grid. The result is soft, but it shows composition and colour as a
run develops.

    python cuda/qimg21/fit_latent_preview.py --jobs tmp/qimg21-web-jobs \\
        --out cuda/qimg21/latent_preview.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def pairs(jobs: Path):
    """(latents [tokens, 64], image [H, W, 3]) for every finished native run."""
    for job in sorted(jobs.iterdir()):
        for backend in ("cuda", "rocm"):
            latents, image = job / f"{backend}-work" / "native_latents.npy", job / f"{backend}.png"
            if latents.is_file() and image.is_file():
                yield np.load(latents).reshape(-1, 64), np.asarray(Image.open(image).convert("RGB"))


def features(latents: np.ndarray, h: int, w: int) -> np.ndarray:
    """[tokens, 64] -> [tokens, 9 * 64 + 1]: each token's 3x3 neighbourhood
    (edge-replicated) and a bias. The server builds the same features."""
    grid = np.pad(latents.reshape(h, w, 64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    near = [grid[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    flat = np.concatenate(near, axis=2).reshape(h * w, 9 * 64)
    return np.hstack([flat, np.ones((h * w, 1), flat.dtype)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True, help="directory of demo jobs to fit on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch", type=int, default=8, help="preview pixels per latent token side")
    ap.add_argument("--holdout", type=int, default=3, help="runs kept out of the fit to report PSNR on")
    args = ap.parse_args()
    if 16 % args.patch:
        ap.error("--patch must divide 16")
    p = args.patch
    xs, ys = [], []
    for latents, image in pairs(args.jobs):
        h, w = image.shape[0] // 16, image.shape[1] // 16
        if latents.shape[0] != h * w:
            continue
        # Box-filter each 16x16 patch down to p x p, then flatten per token.
        small = image.reshape(h * p, 16 // p, w * p, 16 // p, 3).mean(axis=(1, 3)) / 127.5 - 1.0
        patches = small.reshape(h, p, w, p, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, p * p * 3)
        xs.append(features(latents.astype(np.float32), h, w))
        ys.append(patches)
    if len(xs) <= args.holdout:
        raise SystemExit(f"need more than {args.holdout} runs, found {len(xs)}")
    x, y = np.vstack(xs[args.holdout:]), np.vstack(ys[args.holdout:])
    weights, *_ = np.linalg.lstsq(x.astype(np.float64), y.astype(np.float64), rcond=1e-6)
    for i in range(args.holdout):
        err = np.clip(xs[i] @ weights, -1, 1) - ys[i]
        rmse = float(np.sqrt(np.mean(err * err))) * 127.5
        print(f"holdout {i}: {xs[i].shape[0]} tokens, PSNR {20 * np.log10(255 / max(rmse, 1e-9)):.2f} dB")
    np.save(args.out, weights.astype(np.float16))
    print(f"fitted on {len(xs) - args.holdout} runs, {x.shape[0]} tokens -> {args.out} {weights.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
