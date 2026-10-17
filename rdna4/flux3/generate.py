#!/usr/bin/env python3
"""CPU Qwen3-VL prompt encoding followed by native ROCm FLUX3 DiT/VAE."""
import argparse
import fcntl
import tempfile
if __package__:
    from .prompt_cache import identity, key, cached, publish
else:
    from prompt_cache import identity, key, cached, publish
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
    descriptor = identity(args.model_dir, args.prompt, args.threads)
    features = state / (key(descriptor) + ".f32")
    with features.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = cached(features, descriptor)
        if manifest is None:
            with tempfile.TemporaryDirectory(prefix='text-', dir=state) as directory:
                prepared = Path(directory)/'text.f32'
                subprocess.run([sys.executable, str(here / "prepare_text.py"), "--model-dir", str(args.model_dir),
                                "--prompt", args.prompt, "--output", str(prepared), "--threads", str(args.threads)], check=True)
                manifest = publish(prepared, features, descriptor)
    subprocess.run(["make", "-C", str(here)], check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(here / "test_hip_flux3"), "--model-dir", str(args.model_dir),
                    "--text", str(features), "--ntxt", str(manifest["tokens"]),
                    "--width", str(args.width), "--height", str(args.height), "--steps", str(args.steps),
                    "--seed", str(args.seed), "--device", str(args.device), "--vae-wmma", str(args.vae_wmma), "--reserve-mib", str(args.reserve_mib),
                    "--dump", str(args.output.with_suffix(".latent.f32")), "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
