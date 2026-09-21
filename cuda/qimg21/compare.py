#!/usr/bin/env python3
"""Compare qimg21 runner/reference image fixtures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


NONQUANTIZED_COSINE_THRESHOLD = 0.99996
# Quantization error depends on the quantizer and calibration set.  Keep this
# as a deliberately explicit provisional gate instead of pretending that the
# non-quantized threshold applies unchanged to reduced-precision weights.
QUANTIZED_COSINE_THRESHOLD = 0.995


def _step_path(directory: Path, name: str) -> Path:
    """Accept both the flat reference layout and native_generate's layout."""
    direct = directory / name
    if direct.exists():
        return direct
    nested = directory / "steps" / name
    if nested.exists():
        return nested
    return direct


def _cosine_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        squeezed_reference = np.squeeze(reference)
        squeezed_candidate = np.squeeze(candidate)
        if squeezed_reference.shape != squeezed_candidate.shape:
            raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
        reference, candidate = squeezed_reference, squeezed_candidate
    ref = reference.astype(np.float64, copy=False).ravel()
    got = candidate.astype(np.float64, copy=False).ravel()
    if not ref.size or not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("empty or non-finite comparison fixture")
    denom = np.linalg.norm(ref) * np.linalg.norm(got)
    cosine = float(np.dot(ref, got) / max(denom, 1e-30))
    rel_l2 = float(np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-30))
    return cosine, rel_l2


def _compare_array(label: str, reference_path: Path, candidate_path: Path) -> tuple[float, float]:
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing candidate fixture: {candidate_path}")
    cosine, rel_l2 = _cosine_error(np.load(reference_path), np.load(candidate_path))
    print(f"{label}: cosine={cosine:.9f} rel_l2={rel_l2:.8g}")
    return cosine, rel_l2


def _step_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("step_*.npy"))


def _prediction_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("pred_*.npy"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--runner-dir", required=True)
    ap.add_argument(
        "--quantized",
        action="store_true",
        help=f"use the provisional quantized cosine gate ({QUANTIZED_COSINE_THRESHOLD:.6f})",
    )
    ap.add_argument(
        "--cosine-threshold",
        type=float,
        help="override the cosine acceptance gate (default: 0.999960 non-quantized)",
    )
    ap.add_argument(
        "--steps-only",
        action="store_true",
        help="compare denoising checkpoints only (for the native C runner, which has no image output)",
    )
    ap.add_argument(
        "--denoiser-only",
        action="store_true",
        help="compare per-step native denoiser predictions only",
    )
    args = ap.parse_args()
    if args.steps_only and args.denoiser_only:
        raise SystemExit("--steps-only and --denoiser-only are mutually exclusive")
    ref_dir, run_dir = Path(args.reference_dir), Path(args.runner_dir)
    threshold = args.cosine_threshold
    if threshold is None:
        threshold = QUANTIZED_COSINE_THRESHOLD if args.quantized else NONQUANTIZED_COSINE_THRESHOLD
    if not 0.0 < threshold <= 1.0:
        raise SystemExit("--cosine-threshold must be in (0, 1]")
    print(f"acceptance cosine threshold={threshold:.9f} ({'quantized' if args.quantized else 'non-quantized'})")

    failures: list[str] = []
    if not args.steps_only and not args.denoiser_only:
        ref_image = ref_dir / "reference_rgba.npy"
        run_image = run_dir / "runner.png"
        if ref_image.exists() and run_image.exists():
            ref = np.load(ref_image).astype(np.float64)
            got = np.asarray(Image.open(run_image).convert("RGBA")).astype(np.float64)
            if ref.shape != got.shape:
                failures.append(f"image shape mismatch: {ref.shape} vs {got.shape}")
            else:
                diff = got - ref
                rmse = float(np.sqrt(np.mean(diff * diff)))
                mae = float(np.mean(np.abs(diff)))
                psnr = 20 * math.log10(255.0 / max(rmse, 1e-12))
                cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
                print(f"image: shape={ref.shape} mae={mae:.5f} rmse={rmse:.5f} psnr={psnr:.3f}dB cosine={cos:.9f}")
        elif ref_image.exists() or run_image.exists():
            failures.append("image fixture is present on only one side")

    fixture_names = () if args.steps_only or args.denoiser_only else ("initial_latents.npy", "prompt_embeds.npy", "prompt_mask.npy", "image_pad_mask.npy")
    for name in fixture_names:
        rp, gp = ref_dir / name, run_dir / name
        if rp.exists() or gp.exists():
            if not rp.exists() or not gp.exists():
                failures.append(f"missing prompt fixture pair: {name}")
                continue
            try:
                cosine, _ = _compare_array(name, rp, gp)
            except (OSError, ValueError) as exc:
                failures.append(f"{name}: {exc}")
            else:
                if name in ("initial_latents.npy", "prompt_embeds.npy") and cosine < threshold:
                    failures.append(f"{name} cosine {cosine:.9f} < {threshold:.9f}")

    ref_steps: list[str] = []
    if not args.denoiser_only:
        ref_steps = _step_names(ref_dir)
        if not ref_steps:
            failures.append(f"no reference step fixtures in {ref_dir}")
        run_steps = sorted(set(_step_names(run_dir)) | set(_step_names(run_dir / "steps")))
        if run_steps != ref_steps:
            failures.append(f"trajectory fixture mismatch: reference={ref_steps} candidate={run_steps}")
        for name in ref_steps:
            rp = ref_dir / name
            gp = _step_path(run_dir, name)
            try:
                cosine, _ = _compare_array(name, rp, gp)
            except (OSError, ValueError, FileNotFoundError) as exc:
                failures.append(f"{name}: {exc}")
            else:
                if not np.isfinite(cosine) or cosine < threshold:
                    failures.append(f"{name} cosine {cosine:.9f} < {threshold:.9f}")

    ref_predictions = []
    if args.denoiser_only:
        ref_predictions = _prediction_names(ref_dir)
        if not ref_predictions:
            failures.append(f"no reference denoiser prediction fixtures in {ref_dir}")
        run_predictions = _prediction_names(run_dir)
        if run_predictions != ref_predictions:
            failures.append(
                f"denoiser prediction fixture mismatch: reference={ref_predictions} candidate={run_predictions}"
            )
        for name in ref_predictions:
            try:
                cosine, _ = _compare_array(name, ref_dir / name, run_dir / name)
            except (OSError, ValueError, FileNotFoundError) as exc:
                failures.append(f"{name}: {exc}")
            else:
                if not np.isfinite(cosine) or cosine < threshold:
                    failures.append(f"{name} cosine {cosine:.9f} < {threshold:.9f}")

    if failures:
        print("PARITY FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if args.denoiser_only:
        print(f"PARITY PASS: {len(ref_predictions)} denoiser predictions meet the cosine gate")
    else:
        print(f"PARITY PASS: {len(ref_steps)} denoising checkpoints meet the cosine gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
