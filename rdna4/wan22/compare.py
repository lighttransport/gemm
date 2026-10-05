"""Compare HIP/PyTorch generation captures on identical pipeline settings."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hip", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    hip = json.loads((args.hip / "manifest.json").read_text())
    reference = json.loads((args.reference / "manifest.json").read_text())
    if hip["backend"] != "wan22_rocm_hip" or reference["backend"] != "wan22_rocm_pytorch":
        raise ValueError("Expected repository HIP and independent PyTorch GEMM backends")
    for key in ("weights", "prompt", "negative_prompt", "image", "width", "height", "frames",
                "steps", "seed", "fps", "guidance_scale", "torch", "rocm"):
        if hip[key] != reference[key]:
            raise ValueError(f"Mismatched generation setting: {key}")
    if hip.get("latent_only", False) != reference.get("latent_only", False):
        raise ValueError("Mismatched latent-only setting")
    results = []
    for step in range(hip["steps"]):
        name = f"latent_{step:03d}.npy"
        actual = np.load(args.hip / name).astype(np.float64)
        expected = np.load(args.reference / name).astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"Invalid capture: {name}")
        denom = np.linalg.norm(expected)
        relative = float(np.linalg.norm(actual - expected) / max(denom, 1e-30))
        cosine = float(np.sum(actual * expected) / max(np.linalg.norm(actual) * denom, 1e-30))
        results.append({"step": step, "relative_l2": relative, "cosine": cosine,
                        "passed": relative <= .02 and cosine >= .9999})
    def frames(directory):
        data = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(directory / "video.mp4"),
                                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        expected_size = hip["frames"] * hip["height"] * hip["width"] * 3
        if len(data) != expected_size:
            raise ValueError("Decoded MP4 frame count or dimensions do not match")
        return np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    latent_only = hip.get("latent_only", False)
    frame_mae = None if latent_only else float(np.mean(np.abs(frames(args.hip) - frames(args.reference))))
    result = {"scope": "captured schedule only" if latent_only else "captured schedule and encoded MP4; not raw-frame parity",
              "updates": results, "decoded_mp4_mae_255": frame_mae,
              "passed": all(r["passed"] for r in results) and (latent_only or frame_mae <= 2.0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
