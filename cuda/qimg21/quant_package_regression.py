#!/usr/bin/env python3
"""Check a text-to-image INT8 prediction against streaming INT8 and PyTorch."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from compare import QUANTIZED_COSINE_THRESHOLD, _cosine_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--height-tokens", type=int, required=True)
    parser.add_argument("--width-tokens", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, default=Path(__file__).with_name("test_cuda_qimg21_native"))
    args = parser.parse_args()
    if args.step < 0 or min(args.height_tokens, args.width_tokens) < 1:
        raise ValueError("invalid checkpoint or geometry")
    reference = args.reference_dir.resolve()
    if (reference / "negative_prompt_embeds.npy").exists():
        raise ValueError("this bounded package check does not support true CFG")
    latent = np.load(reference / f"input_{args.step:03d}.npy", allow_pickle=False, mmap_mode="r")
    geometry = (args.height_tokens * args.width_tokens, 64)
    if latent.shape not in (geometry, (1, *geometry)):
        raise ValueError("expected target-only text-to-image input matching the requested geometry")
    timestep = np.load(reference / f"timestep_{args.step:03d}.npy", allow_pickle=False)
    if timestep.size != 1 or not np.isfinite(timestep).all() or not 0 <= timestep.item() <= 1:
        raise ValueError("invalid captured timestep")
    target = np.load(reference / f"pred_{args.step:03d}.npy", allow_pickle=False)
    _cosine_error(target, target)  # Reject invalid reference data before starting GPU work.
    args.work_dir.mkdir(parents=True, exist_ok=False)
    result = {"scope": "single matched denoiser prediction", "step": args.step,
              "timestep": timestep.item(), "reference": str(reference),
              "package": str(args.package.resolve()), "model": str(args.model.resolve()),
              "threshold": QUANTIZED_COSINE_THRESHOLD, "paths": {}}
    predictions = []
    for mode, options in (("package", ["--quantized-transformer", str(args.package.resolve())]),
                          ("streaming", ["--quantize-on-load", "int8-row"])):
        output = args.work_dir / mode
        command = [str(args.native_bin.resolve()), "--model", str(args.model.resolve()),
                   "--prompt-embeds", str(reference / "prompt_embeds.npy"),
                   "--latents", str(reference / f"input_{args.step:03d}.npy"),
                   "--height-tokens", str(args.height_tokens), "--width-tokens", str(args.width_tokens),
                   "--steps", "1", "--timestep", repr(timestep.item()),
                   "--attention", "math", "--normalization", "default", "--rope", "default",
                   "--pred-dir", str(output), "--out", str(output / "latents.npy"), *options]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        prediction = np.load(output / "pred_000.npy", allow_pickle=False)
        cosine, relative_l2 = _cosine_error(target, prediction)
        predictions.append(prediction)
        result["paths"][mode] = {"cosine": cosine, "relative_l2": relative_l2,
                                  "passed": cosine >= QUANTIZED_COSINE_THRESHOLD}
    result["package_stream_bit_exact"] = (
        predictions[0].shape == predictions[1].shape and
        predictions[0].dtype == predictions[1].dtype == np.float32 and
        np.array_equal(predictions[0].view(np.uint32), predictions[1].view(np.uint32)))
    result["passed"] = result["package_stream_bit_exact"] and all(x["passed"] for x in result["paths"].values())
    (args.work_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
