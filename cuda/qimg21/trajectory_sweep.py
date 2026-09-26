#!/usr/bin/env python3
"""Measure fast-runner settings against PyTorch reference trajectories.

Each case directory holds a reference.py dump (`ref/`, run with
--dump-initial-latents) and the native text encoder's `prompt_embeds.npy` for
the same prompt. Every combination runs test_cuda_qimg21_fast from the
reference's own initial noise, so the two trajectories differ only by the
denoiser's arithmetic, and reports the latent cosine at a few steps plus the
measured denoise time.

A denoising trajectory amplifies small differences, so read the numbers
against the reference's own spread: PyTorch with a different SDPA backend
drifts to about cosine 0.9994 by step 19 at 512^2 x 20 steps.

    python cuda/qimg21/trajectory_sweep.py --case DIR [--case DIR ...] \\
        --combo fast12:sage --combo low8-fp4:sage:0,31 --combo fast12:sage::/path/to/package
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {"int8": "/mnt/nvme01/models/qimg-21-fast/int8-smooth-a0.6",
            "nvfp4": "/mnt/nvme01/models/qimg-21-fast/nvfp4-svd-a0.5-m"}
PRESET_WEIGHTS = {"low8": "int8", "low8-fp4": "nvfp4", "fast12": "int8", "accurate": None}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_combo(case: Path, combo: str, model: str, work: Path) -> dict:
    preset, attention, *rest = combo.split(":")
    ref = case / "ref"
    meta = json.loads((ref / "run.json").read_text())
    h, w, steps = meta["height"] // 16, meta["width"] // 16, meta["steps"]
    noise = work / "noise.npy"
    np.save(noise, np.load(ref / "initial_latents.npy").reshape(-1, 64).astype(np.float32))
    command = [str(ROOT / "cuda/qimg21/test_cuda_qimg21_fast"), "--preset", preset, "--attention", attention,
               "--model", model, "--prompt-embeds", str(case / "prompt_embeds.npy"),
               "--height-tokens", str(h), "--width-tokens", str(w), "--steps", str(steps),
               "--latents", str(noise), "--out", str(work / "out.npy"), "--dump-dir", str(work / "steps")]
    if PRESET_WEIGHTS[preset]:
        package = rest[1] if len(rest) > 1 and rest[1] else PACKAGES[PRESET_WEIGHTS[preset]]
        command += ["--quant-package", package]
    if rest and rest[0]:
        command += ["--bf16-blocks", rest[0]]
    log = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if log.returncode:
        raise SystemExit(f"{combo} on {case.name} failed:\n{log.stderr[-2000:]}")
    seconds = float(re.search(r"denoise \d+ steps \(image generation\) ([0-9.]+) s", log.stderr).group(1))
    plan = re.search(r"fast: plan (\d+) resident \+ (\d+) streamed", log.stderr)
    last = steps - 1
    picks = sorted({0, last // 2, last})
    return {"seconds": seconds, "streamed": int(plan.group(2)) if plan else None,
            "cos": {i: cosine(np.load(ref / f"step_{i:03d}.npy"), np.load(work / "steps" / f"step_{i:03d}.npy"))
                    for i in picks}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, action="append", required=True)
    ap.add_argument("--combo", action="append", required=True, help="preset:attention[:bf16-blocks[:package]]")
    ap.add_argument("--model", default="/mnt/nvme01/models/qimg-21")
    ap.add_argument("--json", type=Path, help="also write the results here")
    args = ap.parse_args()
    results = {}
    for combo in args.combo:
        rows = []
        for case in args.case:
            with tempfile.TemporaryDirectory(prefix="qimg21-sweep-") as td:
                rows.append(run_combo(case, combo, args.model, Path(td)))
            last = max(rows[-1]["cos"])
            print(f"{combo:32s} {case.name:8s} cos@{last}={rows[-1]['cos'][last]:.6f} "
                  f"denoise {rows[-1]['seconds']:.2f} s", file=sys.stderr, flush=True)
        last = max(rows[0]["cos"])
        mean_gap = float(np.mean([1 - r["cos"][last] for r in rows]))
        results[combo] = {"mean_1_minus_cos_last": mean_gap,
                          "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
                          "streamed_blocks": rows[0]["streamed"], "cases": rows}
        print(f"== {combo:29s} mean 1-cos@last {mean_gap:.2e}   mean denoise "
              f"{results[combo]['mean_seconds']:.2f} s   streamed blocks {rows[0]['streamed']}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
