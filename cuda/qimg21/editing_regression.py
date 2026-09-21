#!/usr/bin/env python3
"""Native editing predictions and trajectory against captured PyTorch calls.

Requires an official batch-one, unpadded editing capture.
Condition encoding stays in the reference; all transformer/Euler work is native.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from compare import (_cosine_error, _relative_mae, NONQUANTIZED_COSINE_THRESHOLD,
                     QUANTIZED_MRE_THRESHOLD)
from prepare_edit_fixture import prepare


def guidance_scale(reference):
    run = json.loads((reference / "run.json").read_text())
    scale = run.get("true_cfg_scale", 1.0)
    cfg = run.get("use_true_cfg", False)
    if not isinstance(scale, (int, float)) or not np.isfinite(scale):
        raise ValueError("invalid captured CFG scale")
    has_negative = (reference / "negative_prompt_embeds.npy").exists()
    if bool(cfg) != has_negative or (cfg and scale <= 1):
        raise ValueError("inconsistent CFG capture metadata")
    return float(scale) if cfg else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--native-attention", choices=("math", "reverse64", "mma64", "mma64-flash", "mma64-mixed", "mma64-forward-flash", "mma128-efficient", "cutlass-efficient"), default="math")
    ap.add_argument("--native-normalization", choices=("default", "vector4"), default="default")
    ap.add_argument("--native-rope", choices=("default", "host-table", "host-table-vector4", "host-table-exact"), default="default")
    quant = ap.add_mutually_exclusive_group()
    quant.add_argument("--quantized-transformer", type=Path)
    quant.add_argument("--quantize-on-load", choices=("int8-row",))
    args = ap.parse_args()
    ref = args.reference_dir.resolve()
    predictions = sorted(ref.glob("pred_*.npy"))
    steps = len(predictions)
    if steps < 2 or [p.name for p in predictions] != [f"pred_{i:03d}.npy" for i in range(steps)]:
        raise ValueError("requires a contiguous capture with at least two steps")
    scale = guidance_scale(ref)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    binary = str(Path(__file__).with_name("test_cuda_qimg21_native").resolve())
    is_quantized = bool(args.quantized_transformer or args.quantize_on_load)
    cosine_threshold = None if is_quantized else NONQUANTIZED_COSINE_THRESHOLD
    results = {"threshold": cosine_threshold,
               "mre_threshold": QUANTIZED_MRE_THRESHOLD if is_quantized else None,
               "quantized": is_quantized,
               "reference": str(ref), "model": str(args.model.resolve()),
               "true_cfg_scale": scale, "predictions": [], "trajectory": [],
               "attention": args.native_attention, "normalization": args.native_normalization,
               "rope": args.native_rope}
    first_condition = None
    first_fixture = None

    def command(fixture, metadata):
        cmd = [binary, "--model", str(args.model.resolve()),
                "--attention", args.native_attention, "--normalization", args.native_normalization,
                "--rope", args.native_rope,
                "--prompt-embeds", str(fixture / "prompt_embeds.npy"),
                "--latents", str(fixture / "target_latents.npy"),
                "--condition-latents", str(fixture / "condition_latents.npy"),
                "--editing-layout", str(fixture / "layout.txt"),
                "--height-tokens", str(metadata["height_tokens"]),
                "--width-tokens", str(metadata["width_tokens"])]
        if args.quantized_transformer:
            cmd.extend(["--quantized-transformer", str(args.quantized_transformer.resolve())])
        elif args.quantize_on_load:
            cmd.extend(["--quantize-on-load", args.quantize_on_load])
        if scale > 1:
            negative = fixture / "negative"
            cmd.extend(["--negative-editing-layout", str(negative / "layout.txt"),
                        "--negative-prompt-embeds", str(negative / "prompt_embeds.npy"),
                        "--guidance-scale", repr(scale)])
        return cmd

    def compare(reference, candidate):
        reference_array = np.load(reference, allow_pickle=False)
        candidate_array = np.load(candidate, allow_pickle=False)
        cosine, relative_l2 = _cosine_error(reference_array, candidate_array)
        mre = _relative_mae(reference_array, candidate_array)
        passed = mre <= QUANTIZED_MRE_THRESHOLD if is_quantized else cosine >= cosine_threshold
        return {"cosine": cosine, "relative_l2": relative_l2, "mre": mre,
                "passed": passed}

    for i in range(steps):
        fixture = work / f"fixture-{i:03d}"
        metadata = prepare(ref, fixture, i)
        if scale > 1:
            negative = fixture / "negative"
            negative_metadata = prepare(ref, negative, i, "negative")
            for key in ("height_tokens", "width_tokens", "condition_tokens", "target_tokens", "timestep"):
                if metadata[key] != negative_metadata[key]:
                    raise ValueError(f"CFG branch mismatch: {key}")
            for name in ("condition_latents.npy", "target_latents.npy"):
                if not np.array_equal(np.load(fixture / name), np.load(negative / name)):
                    raise ValueError(f"CFG branch latent mismatch: {name}")
        condition = np.load(fixture / "condition_latents.npy")
        if first_condition is None:
            first_condition, first_fixture = condition, fixture
            first_metadata = metadata
        elif not np.array_equal(first_condition, condition):
            raise ValueError("condition latents changed across reference calls")
        elif ((fixture / "layout.txt").read_bytes() != (first_fixture / "layout.txt").read_bytes() or
              not np.array_equal(np.load(fixture / "prompt_embeds.npy"),
                                 np.load(first_fixture / "prompt_embeds.npy"))):
            raise ValueError("layout or prompt changed across reference calls")
        output = work / f"prediction-{i:03d}"
        subprocess.run(command(fixture, metadata) + ["--steps", "1", "--timestep",
                       repr(metadata["timestep"]), "--pred-dir", str(output),
                       "--out", str(output / "latents.npy")], check=True)
        results["predictions"].append(compare(predictions[i], output / "pred_000.npy"))
        (work / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    output = work / "trajectory"
    subprocess.run(command(first_fixture, first_metadata) + ["--steps", str(steps),
                   "--dump-dir", str(output), "--out", str(output / "latents.npy")], check=True)
    for i in range(steps):
        results["trajectory"].append(compare(ref / f"step_{i:03d}.npy", output / f"step_{i:03d}.npy"))
    results["passed"] = all(x["passed"] for group in ("predictions", "trajectory") for x in results[group])
    (work / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
