#!/usr/bin/env python3
"""Check packed latent unpatching and HIP VAE against BFL's CPU autoencoder."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("/mnt/nvme02/model/flux3-klein"))
    p.add_argument("--output-dir", type=Path, default=Path("tmp/flux3/verify-vae"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("flux3_official_ae", args.reference_dir / "src/flux3/autoencoder.py")
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)
    model = ae.AutoEncoder().eval()
    model.load_state_dict(load_file(str(args.model_dir / "ae.safetensors")))
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randn(4 * 4, 128, generator=generator) * .1
    expected = model.decode(tokens, ae.latent_ids(4, 4)).float().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent = args.output_dir / "latent.f32"
    tokens.numpy().astype("<f4").tofile(latent)
    output = args.output_dir / "rgb.f32"
    subprocess.run([str(Path(__file__).resolve().parent / "test_hip_flux3"), "--vae-only",
                    "--model-dir", str(args.model_dir), "--latent", str(latent), "--width", "64", "--height", "64",
                    "--output", str(args.output_dir / "native.ppm"), "--dump-rgb", str(output)], check=True)
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    err = actual - expected
    relative = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(expected ** 2)))
    result = {"relative_rmse": relative, "max_abs": float(np.max(np.abs(err))), "pass": relative < .02}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
