#!/usr/bin/env python3
"""Create the small .npy inputs consumed by test_cuda_qimg21_native."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--prompt-embeds", required=True, help="F32 prompt_embeds.npy from --test-text")
ap.add_argument("--height-tokens", type=int, default=16)
ap.add_argument("--width-tokens", type=int, default=16)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
ap.add_argument("--latents", type=Path,
                help="reuse a finite F32 [height_tokens*width_tokens, 64] noise tensor")
ap.add_argument(
    "--torch-rng",
    action="store_true",
    help="use the same CUDA torch.randn layout as QwenImage21Pipeline",
)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
if a.latents and a.torch_rng:
    ap.error("--latents and --torch-rng are mutually exclusive")
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
np.save(out / "prompt_embeds.npy", np.load(a.prompt_embeds).astype(np.float32, copy=False))
if a.latents:
    latents = np.load(a.latents, allow_pickle=False)
    shape = (a.height_tokens * a.width_tokens, 64)
    if latents.dtype != np.float32 or latents.shape != shape or not np.isfinite(latents).all():
        raise ValueError(f"--latents must be finite F32 with shape {shape}")
    np.save(out / "latents.npy", latents)
elif a.torch_rng:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("--torch-rng requires CUDA")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    generator = torch.Generator(device="cuda").manual_seed(a.seed)
    latents = torch.randn(
        (1, 1, 64, a.height_tokens, a.width_tokens),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    # Qwen-Image 2.1 uses a plain spatial flatten: [B,C,H,W] -> [B,HW,C].
    latents = latents.view(1, 64, a.height_tokens * a.width_tokens).transpose(1, 2)[0]
    np.save(out / "latents.npy", np.ascontiguousarray(latents.float().cpu().numpy()))
else:
    g = np.random.default_rng(a.seed)
    latents = g.standard_normal((a.height_tokens * a.width_tokens, 64), dtype=np.float32)
    if a.dtype == "bf16":
        bits = latents.view(np.uint32)
        bits += np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
        bits &= np.uint32(0xffff0000)
    else:
        latents = latents.astype(np.float16).astype(np.float32)
    np.save(out / "latents.npy", latents)
print(out)
