#!/usr/bin/env python3
"""Report the approved CUDA/ROCm reference-floor tier alongside the strict gate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda" / "qimg21"))
from compare import NONQUANTIZED_COSINE_THRESHOLD, _cosine_error  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def compare(reference, candidate):
    return _cosine_error(np.load(reference, allow_pickle=False),
                         np.load(candidate, allow_pickle=False))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--cuda-free-run", required=True, type=Path,
                        help="original CUDA pipeline capture used to start the ROCm free run")
    parser.add_argument("--rocm-reference", required=True, type=Path)
    parser.add_argument("--native-regression", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cuda, cuda_free, rocm, native = (path.resolve() for path in
                                     (args.cuda_reference, args.cuda_free_run,
                                      args.rocm_reference, args.native_regression))
    cuda_run, rocm_run = read_json(cuda / "run.json"), read_json(rocm / "run.json")
    cuda_free_run = read_json(cuda_free / "run.json")
    native_run = read_json(native / "results.json")
    if Path(cuda_run.get("capture", "")).resolve() != rocm:
        raise ValueError("CUDA oracle must capture the exact ROCm free-running input directory")
    if (cuda_run.get("free_run") is not False or rocm_run.get("free_run") is not True or
        "+cu" not in cuda_run.get("torch", "") or
        "+rocm" not in rocm_run.get("torch", "")):
        raise ValueError("expected a pinned CUDA replay and a free-running ROCm oracle")
    if (Path(rocm_run.get("capture", "")).resolve() != cuda_free or
        "+cu" not in cuda_free_run.get("torch", "")):
        raise ValueError("ROCm oracle must start from the pinned CUDA pipeline capture")
    for field in ("model", "prompt", "height", "width", "steps", "seed",
                  "sdpa_backend", "use_true_cfg", "true_cfg_scale"):
        if cuda_run.get(field) != rocm_run.get(field) or cuda_free_run.get(field) != rocm_run.get(field):
            raise ValueError(f"CUDA/ROCm reference metadata differs: {field}")
    if not np.array_equal(np.load(cuda_free / "input_000.npy", allow_pickle=False),
                          np.load(rocm / "input_000.npy", allow_pickle=False)):
        raise ValueError("CUDA/ROCm free runs do not start from identical transformer inputs")
    if not np.array_equal(np.load(cuda_free / "initial_latents.npy", allow_pickle=False),
                          np.load(rocm / "initial_latents.npy", allow_pickle=False)):
        raise ValueError("CUDA/ROCm free runs do not start from identical target noise")
    if (cuda_run.get("sdpa_backend") != "efficient" or
        Path(native_run.get("reference", "")).resolve() != rocm or
        native_run.get("quantized") is not False or
        native_run.get("threshold") != NONQUANTIZED_COSINE_THRESHOLD):
        raise ValueError("reference backend or native regression provenance differs")
    steps = rocm_run["steps"]
    if steps < 2 or len(native_run["predictions"]) != steps or len(native_run["trajectory"]) != steps:
        raise ValueError("incomplete two-step or longer editing regression")

    predictions, trajectory = [], []
    for step in range(steps):
        name = f"pred_{step:03d}.npy"
        floor = compare(rocm / name, cuda / name)
        native_cosine = compare(rocm / name,
                                native / f"prediction-{step:03d}" / "pred_000.npy")
        if abs(native_cosine - native_run["predictions"][step]["cosine"]) > 1e-12:
            raise ValueError(f"native saved prediction metric differs at step {step}")
        predictions.append({"step": step, "cuda_rocm_same_input_cosine": floor,
                            "native_rocm_cosine": native_cosine,
                            "margin_above_reference_floor": native_cosine - floor,
                            "strict_pass": native_cosine >= NONQUANTIZED_COSINE_THRESHOLD,
                            "reference_floor_pass": native_cosine >= floor})
        name = f"step_{step:03d}.npy"
        checkpoint = compare(rocm / name, native / "trajectory" / name)
        trajectory_floor = compare(rocm / name, cuda_free / name)
        if abs(checkpoint - native_run["trajectory"][step]["cosine"]) > 1e-12:
            raise ValueError(f"native saved trajectory metric differs at step {step}")
        trajectory.append({"step": step, "native_rocm_cosine": checkpoint,
                           "cuda_rocm_free_run_cosine": trajectory_floor,
                           "margin_above_reference_floor": checkpoint - trajectory_floor,
                           "strict_pass": checkpoint >= NONQUANTIZED_COSINE_THRESHOLD,
                           "reference_floor_pass": checkpoint >= trajectory_floor})
    report = {
        "status": "user-approved separate reference-floor tier; strict editing gate unchanged",
        "strict_threshold": NONQUANTIZED_COSINE_THRESHOLD,
        "cuda_reference": str(cuda), "cuda_free_run": str(cuda_free),
        "rocm_reference": str(rocm),
        "native_regression": str(native),
        "cuda_torch": cuda_run["torch"], "rocm_torch": rocm_run["torch"],
        "predictions": predictions, "trajectory": trajectory,
        "strict_existing_gate_pass": all(row["strict_pass"] for row in predictions + trajectory),
        "reference_floor_tier_pass": all(row["reference_floor_pass"]
                                         for row in predictions + trajectory),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
