#!/usr/bin/env python3
"""Native editing predictions and trajectory against captured PyTorch calls.

Requires an official batch-one, unpadded, positive-only editing capture.
Condition encoding stays in the reference; all transformer/Euler work is native.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from compare import _cosine_error, NONQUANTIZED_COSINE_THRESHOLD
from prepare_edit_fixture import prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    args = ap.parse_args()
    ref = args.reference_dir.resolve()
    predictions = sorted(ref.glob("pred_*.npy"))
    steps = len(predictions)
    if steps < 2 or [p.name for p in predictions] != [f"pred_{i:03d}.npy" for i in range(steps)]:
        raise ValueError("requires a contiguous capture with at least two steps")
    if (ref / "negative_prompt_embeds.npy").exists():
        raise ValueError("native editing CFG is not yet supported")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    binary = str(Path(__file__).with_name("test_cuda_qimg21_native").resolve())
    results = {"threshold": NONQUANTIZED_COSINE_THRESHOLD, "quantized": False,
               "reference": str(ref), "model": str(args.model.resolve()),
               "predictions": [], "trajectory": []}
    first_condition = None
    first_fixture = None

    def command(fixture, metadata):
        return [binary, "--model", str(args.model.resolve()),
                "--prompt-embeds", str(fixture / "prompt_embeds.npy"),
                "--latents", str(fixture / "target_latents.npy"),
                "--condition-latents", str(fixture / "condition_latents.npy"),
                "--editing-layout", str(fixture / "layout.txt"),
                "--height-tokens", str(metadata["height_tokens"]),
                "--width-tokens", str(metadata["width_tokens"])]

    def compare(reference, candidate):
        cosine, relative_l2 = _cosine_error(np.load(reference, allow_pickle=False),
                                           np.load(candidate, allow_pickle=False))
        return {"cosine": cosine, "relative_l2": relative_l2,
                "passed": cosine >= NONQUANTIZED_COSINE_THRESHOLD}

    for i in range(steps):
        fixture = work / f"fixture-{i:03d}"
        metadata = prepare(ref, fixture, i)
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
