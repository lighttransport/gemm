#!/usr/bin/env python3
"""Compare native single-frame F32 VAE posterior moments with official PyTorch."""
import argparse
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline

from compare import NONQUANTIZED_COSINE_THRESHOLD, _cosine_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="VAE directory")
    parser.add_argument("--case", action="append", help="HEIGHTxWIDTH:SEED")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, default=Path(__file__).with_name("test_cuda_qimg21_vae_encode"))
    args = parser.parse_args()
    cases = []
    for case in args.case or ["64x64:17", "128x128:42", "64x128:123"]:
        geometry, seed = case.split(":")
        h, w = map(int, geometry.split("x"))
        if min(h, w) < 16 or max(h, w) > 1024 or h % 16 or w % 16:
            raise ValueError("expected dimensions divisible by 16, within 16..1024")
        cases.append((h, w, int(seed)))
    args.work_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {"scope": "single-frame F32 posterior moments and normalized mode tokens", "threshold": NONQUANTIZED_COSINE_THRESHOLD,
               "torch": torch.__version__, "model": str(args.model.resolve()), "cases": []}
    for h, w, seed in cases:
        directory = args.work_dir / f"{h}x{w}-seed{seed}"
        directory.mkdir()
        x = np.random.default_rng(seed).uniform(-1, 1, (4, h, w)).astype(np.float32)
        np.save(directory / "input.npy", x)
        model = AutoencoderKLQwenImage21.from_pretrained(str(args.model), torch_dtype=torch.float32,
                                                       local_files_only=True).eval().cuda()
        with torch.inference_mode():
            posterior = model.encode(torch.from_numpy(x)[None, :, None].cuda()).latent_dist
            expected = posterior.parameters[0, :, 0].float().cpu().numpy().copy()
            mean = torch.tensor(model.config.latents_mean, device="cuda", dtype=torch.float32).view(1,64,1,1,1)
            std = torch.tensor(model.config.latents_std, device="cuda", dtype=torch.float32).view(1,64,1,1,1)
            latent = (posterior.mode() - mean) / std
            expected_latent = QwenImage21Pipeline._pack_latents(latent,1,64,h//16,w//16)[0].cpu().numpy().copy()
        np.save(directory / "reference.npy", expected)
        np.save(directory / "reference_latents.npy", expected_latent)
        del posterior, model, latent, mean, std
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run([str(args.native_bin.resolve()), "--model", str(args.model.resolve()),
                        "--image", str(directory / "input.npy"), "--out", str(directory / "native.npy"),
                        "--normalized-latents", str(directory / "native_latents.npy")], check=True)
        actual = np.load(directory / "native.npy", allow_pickle=False)
        cosine, relative_l2 = _cosine_error(expected, actual)
        mean_cosine, _ = _cosine_error(expected[:64], actual[:64])
        actual_latent = np.load(directory / "native_latents.npy", allow_pickle=False)
        latent_cosine, latent_relative_l2 = _cosine_error(expected_latent, actual_latent)
        result = {"height": h, "width": w, "seed": seed, "cosine": cosine,
                  "mean_cosine": mean_cosine, "relative_l2": relative_l2,
                  "latent_cosine": latent_cosine, "latent_relative_l2": latent_relative_l2,
                  "max_absolute_error": float(np.max(np.abs(actual-expected))),
                  "passed": min(cosine, mean_cosine, latent_cosine) >= NONQUANTIZED_COSINE_THRESHOLD}
        results["cases"].append(result)
        results["passed"] = all(case["passed"] for case in results["cases"])
        (args.work_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
