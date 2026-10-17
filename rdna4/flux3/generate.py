#!/usr/bin/env python3
"""CPU Qwen3-VL prompt encoding followed by native ROCm FLUX3 DiT/VAE."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("/mnt/nvme02/model/flux3-klein"))
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", type=Path, required=True, help="PPM image")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--vae-wmma", type=int, choices=(0, 1), default=0)
    p.add_argument("--reserve-mib", type=int, default=1024)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    state = root / "tmp" / "flux3"
    state.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    import hashlib
    key = hashlib.sha256((str(args.model_dir.resolve()) + "\0" + args.prompt).encode()).hexdigest()[:24]
    features = state / (key + ".f32")
    if not features.exists() or not features.with_suffix(".f32.json").exists():
        subprocess.run([sys.executable, str(here / "prepare_text.py"), "--model-dir", str(args.model_dir),
                        "--prompt", args.prompt, "--output", str(features), "--threads", str(args.threads)], check=True)
    manifest = json.loads(features.with_suffix(".f32.json").read_text())
    subprocess.run(["make", "-C", str(here)], check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(here / "test_hip_flux3"), "--model-dir", str(args.model_dir),
                    "--text", str(features), "--ntxt", str(manifest["tokens"]),
                    "--width", str(args.width), "--height", str(args.height), "--steps", str(args.steps),
                    "--seed", str(args.seed), "--device", str(args.device), "--vae-wmma", str(args.vae_wmma), "--reserve-mib", str(args.reserve_mib),
                    "--dump", str(args.output.with_suffix(".latent.f32")), "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
