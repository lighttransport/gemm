#!/usr/bin/env python3
"""Compare the native single-frame F32 decoder with the official PyTorch VAE."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("/mnt/nvme01/models/qimg-21"))
    ap.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-vae-regression"))
    ap.add_argument("--native-bin", type=Path,
                    default=Path(__file__).with_name("test_cuda_qimg21_vae"))
    ap.add_argument("--case", action="append", help="HEIGHTxWIDTH:SEED; repeatable")
    args = ap.parse_args()
    cases = args.case or ["128x128:17", "256x256:42", "128x256:123"]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = []
    # Finish native subprocesses before loading the PyTorch model onto the GPU.
    for case in cases:
        resolution, seed = case.split(":")
        height, width = map(int, resolution.split("x"))
        if min(height, width) <= 0 or height % 32 or width % 32:
            raise ValueError("positive dimensions divisible by 32 required")
        folder = args.work_dir / case.replace(":", "-seed")
        folder.mkdir(exist_ok=True)
        latent = np.random.default_rng(int(seed)).standard_normal(
            (height // 16 * (width // 16), 64), dtype=np.float32)
        np.save(folder / "latents.npy", latent)
        start = time.perf_counter()
        subprocess.run([
            str(args.native_bin.resolve()), "--model", str(args.model / "vae"),
            "--latents", str(folder / "latents.npy"),
            "--height-tokens", str(height // 16), "--width-tokens", str(width // 16),
            "--out", str(folder / "native.npy"), "--quiet",
        ], check=True)
        fixtures.append((case, folder, height, width, time.perf_counter() - start))

    import torch
    from diffusers import AutoencoderKLQwenImage21

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    vae = AutoencoderKLQwenImage21.from_pretrained(
        str(args.model / "vae"), torch_dtype=torch.float32, local_files_only=True,
    ).eval().to("cuda")
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 64, 1, 1, 1)
    results = []
    for case, folder, height, width, elapsed in fixtures:
        latent = torch.from_numpy(np.load(folder / "latents.npy")).to("cuda")
        latent = latent.T.reshape(1, 64, 1, height // 16, width // 16).contiguous()
        with torch.inference_mode():
            target = vae.decode(latent * std + mean, return_dict=False)[0][0, :, 0]
        reference = target.cpu().numpy()
        np.save(folder / "reference.npy", reference)
        actual = np.load(folder / "native.npy")
        if actual.shape != reference.shape:
            raise ValueError(f"{case}: shape mismatch")
        a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        mae = float(np.mean(abs(a - b)))
        result = dict(case=case, cosine=cosine, mae=mae, finite=finite,
                      native_seconds=elapsed, passed=finite and cosine >= 0.99996)
        results.append(result)
        (args.work_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del latent, target
        torch.cuda.empty_cache()
    (args.work_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
