#!/usr/bin/env python3
"""Permanent full native Qwen-Image 2.1 single-image editing regression.

The PyTorch reference must use ``reference.py --prompt-fixture-dir`` with an
official pre-final-RMSNorm text capture.  This avoids Transformers output-
aliasing differences while still checking native vision/text conditioning
against that independently captured boundary.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare import NONQUANTIZED_COSINE_THRESHOLD, _cosine_error


def compare(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim == candidate.ndim + 1 and reference.shape[0] == 1:
        reference = reference[0]
    elif candidate.ndim == reference.ndim + 1 and candidate.shape[0] == 1:
        candidate = candidate[0]
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("comparison contains nonfinite values")
    cosine, relative_l2 = _cosine_error(reference, candidate)
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_absolute_error": float(np.max(np.abs(reference.astype(np.float64) - candidate))),
        "passed": bool(cosine >= NONQUANTIZED_COSINE_THRESHOLD),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--text-reference-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--prompt", default="change the apple color to green")
    ap.add_argument("--reuse-native", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    reference = args.reference_dir.resolve()
    text_reference = args.text_reference_dir.resolve()
    work = args.work_dir.resolve()
    run = json.loads((reference / "run.json").read_text())
    if run.get("sdpa_backend") != "efficient" or not run.get("prompt_fixture_dir"):
        raise ValueError("reference must pin efficient SDPA and an explicit prompt fixture")
    steps = int(run["steps"])
    if steps < 2:
        raise ValueError("full regression requires at least two trajectory checkpoints")
    if not args.reuse_native:
        if work.exists():
            raise ValueError(f"work directory already exists: {work}")
        output = work.with_suffix(".png")
        subprocess.run([
            sys.executable, str(root / "cuda/qimg21/native_generate.py"),
            "--model", str(args.model.resolve()), "--image", str(args.image.resolve()),
            "--prompt", args.prompt, "--height", str(run["height"]),
            "--width", str(run["width"]), "--steps", str(steps),
            "--seed", str(run["seed"]), "--native-attention", "cutlass-efficient",
            "--native-normalization", "vector4", "--native-rope", "host-table-exact",
            "--native-vae", "--work-dir", str(work), "--out", str(output),
        ], cwd=root, check=True)
    if not work.is_dir():
        raise ValueError(f"native work directory does not exist: {work}")

    layout = json.loads((reference / "positive_layout.json").read_text())
    shapes = layout["img_shapes"][0]
    target_tokens = int(layout["target_tokens"])
    condition_tokens = sum(int(h) * int(w) for _, h, w in shapes) - target_tokens
    transformer_input = np.load(reference / "input_000.npy", allow_pickle=False)[0]
    condition_reference = transformer_input[:condition_tokens]
    decoded = np.load(work / "native_decoded.npy", allow_pickle=False)
    decoded_rgba = np.rint(np.clip(decoded * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)

    checks = {
        "condition_latents": compare(
            condition_reference, np.load(work / "condition/latents.npy", allow_pickle=False)),
        "multimodal_prompt": compare(
            np.load(text_reference / "prompt_embeds.npy", allow_pickle=False),
            np.load(work / "prompt/prompt_embeds.npy", allow_pickle=False)[None]),
        "initial_latents": compare(
            np.load(reference / "initial_latents.npy", allow_pickle=False),
            np.load(work / "latents.npy", allow_pickle=False)),
        "decoded_rgba": compare(
            np.load(reference / "reference_rgba.npy", allow_pickle=False), decoded_rgba),
    }
    for index in range(steps):
        checks[f"trajectory_{index:03d}"] = compare(
            np.load(reference / f"step_{index:03d}.npy", allow_pickle=False),
            np.load(work / f"steps/step_{index:03d}.npy", allow_pickle=False),
        )
    results = {
        "threshold": NONQUANTIZED_COSINE_THRESHOLD,
        "reference": str(reference),
        "text_reference": str(text_reference),
        "native_work": str(work),
        "checks": checks,
        "passed": all(value["passed"] for value in checks.values()),
    }
    (work / "e2e_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
