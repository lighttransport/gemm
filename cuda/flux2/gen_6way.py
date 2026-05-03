#!/usr/bin/env python3
"""6-variant comparison for a single prompt:

  1. ours BF16 (cuBLAS-LT off, v7 kernel)
  2. ours BF16 (cuBLAS-LT on)
  3. ours FP8  (cuBLAS-LT off, v7 kernel)
  4. ours FP8  (cuBLAS-LT on)
  5. PyTorch ComfyUI BF16 compute
  6. PyTorch ComfyUI FP8  compute

Outputs a 2x3 composite (compare_6way.png) plus per-variant PNGs.

Usage:
    python gen_6way.py [--prompt "..."] [--seed N] [--height H] [--width W]
                       [--steps N]
"""
import sys
import os
import math
import time
import argparse
import subprocess

CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = "/mnt/disk01/ComfyUI"
MODEL_DIR = "/mnt/disk01/models/klein2-4b"
TOK_PATH = "/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf"

sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
import numpy as np  # noqa: E402

# Reuse helpers from gen_compare.py
sys.path.insert(0, CUDA_DIR)
from gen_compare import (  # noqa: E402
    load_ppm, our_prng_noise, heatmap, add_label, save_png,
)

VARIANTS_OURS = [
    # (label, env vars dict)
    ("OURS BF16 LT-OFF", {
        "FLUX2_BF16_GEMM": "1", "FLUX2_BF16_ATTN": "1",
        "FLUX2_CUBLASLT_BF16": "0",
    }),
    ("OURS BF16 LT-ON", {
        "FLUX2_BF16_GEMM": "1", "FLUX2_BF16_ATTN": "1",
    }),
    ("OURS FP8 LT-OFF", {
        "FLUX2_FP8_GEMM": "1", "FLUX2_FP8_V7": "1", "FLUX2_FP8_ATTN": "1",
        "FLUX2_CUBLASLT_FP8": "0",
    }),
    ("OURS FP8 LT-ON", {
        "FLUX2_FP8_GEMM": "1", "FLUX2_FP8_V7": "1", "FLUX2_FP8_ATTN": "1",
    }),
]


def run_ours(label, env_extra, prompt, seed, height, width, steps):
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [
        "./test_cuda_flux2",
        "--generate", "--gpu-enc",
        "--enc", os.path.join(MODEL_DIR, "text_encoder"),
        "--tok", TOK_PATH,
        "--height", str(height), "--width", str(width),
        "--steps", str(steps), "--seed", str(seed),
        "--prompt", prompt,
    ]
    print(f"  [{label}] running ...")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=CUDA_DIR, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"{label} failed")
    dt = time.time() - t0
    print(f"  [{label}] done in {dt:.1f}s")
    img = load_ppm(os.path.join(CUDA_DIR, "cuda_flux2_output.ppm"))
    return img


def load_comfy(weight_dtype):
    """Load ComfyUI DiT + diffusers VAE with given compute dtype.
    weight_dtype: torch.bfloat16 or 'fp8' (uses fp8_optimizations).
    """
    sys.path.insert(0, COMFYUI_DIR)
    os.chdir(COMFYUI_DIR)
    import comfy.sd
    import comfy.model_management
    from safetensors import safe_open
    from diffusers import AutoencoderKL
    import torch

    device = comfy.model_management.get_torch_device()
    dit_path = os.path.join(MODEL_DIR, "diffusion_models",
                            "flux-2-klein-4b-fp8.safetensors")
    if weight_dtype == "fp8":
        opts = {"fp8_optimizations": True}
    else:
        opts = {"dtype": weight_dtype}
    print(f"  loading DiT (mode={weight_dtype})")
    mp = comfy.sd.load_diffusion_model(dit_path, model_options=opts)

    vae_path = os.path.join(MODEL_DIR, "vae", "flux2-vae.safetensors")
    vae_sd = {}
    with safe_open(vae_path, framework="pt") as f:
        for key in f.keys():
            vae_sd[key] = f.get_tensor(key)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2, norm_num_groups=32,
        latent_channels=32, scaling_factor=1.0,
    )
    vae.load_state_dict(vae_sd, strict=False)
    vae = vae.to(device, torch.float32).eval()
    return mp, vae, device


def run_comfy(mp, vae, device, hidden_path, seed, height, width, steps):
    import comfy.samplers
    import torch
    lat_h, lat_w = height // 8, width // 8
    ps, lc = 2, 32
    ph, pw = lat_h // ps, lat_w // ps
    n_lat = lc * lat_h * lat_w

    noise_np = our_prng_noise(seed, n_lat).reshape(lc, lat_h, lat_w)
    patchified = np.zeros((lc * ps * ps, ph, pw), dtype=np.float32)
    for c in range(lc):
        for pr in range(ps):
            for pc in range(ps):
                patchified[c * ps * ps + pr * ps + pc] = \
                    noise_np[c, pr::ps, pc::ps]
    noise = torch.tensor(
        patchified.reshape(1, lc * ps * ps, ph, pw), dtype=torch.float32)

    text_hidden = np.load(hidden_path)
    text_cond = torch.tensor(text_hidden, dtype=torch.float32).unsqueeze(0)
    pooled = torch.zeros(1, 3072, dtype=torch.float32)
    positive = [[text_cond, {"pooled_output": pooled}]]
    negative = [[torch.zeros_like(text_cond), {"pooled_output": pooled}]]

    sampler = comfy.samplers.KSampler(
        model=mp, steps=steps, device=device,
        sampler="euler", scheduler="simple", denoise=1.0,
    )
    latent_image = torch.zeros_like(noise)
    t0 = time.time()
    result = sampler.sample(
        noise=noise, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent_image,
        start_step=0, last_step=steps, force_full_denoise=True,
    )
    dt = time.time() - t0
    print(f"  [comfy] sample done in {dt:.1f}s")

    patchified_out = result[0].float().cpu().numpy()
    latent_out = np.zeros((lc, lat_h, lat_w), dtype=np.float32)
    for c in range(lc):
        for pr in range(ps):
            for pc in range(ps):
                latent_out[c, pr::ps, pc::ps] = \
                    patchified_out[c * ps * ps + pr * ps + pc]

    with torch.no_grad():
        lat_t = torch.tensor(
            latent_out, dtype=torch.float32, device=device).unsqueeze(0)
        rgb = vae.decode(lat_t).sample[0].cpu().numpy()
    img = ((rgb.transpose(1, 2, 0) + 1.0) * 0.5 * 255) \
        .clip(0, 255).astype(np.uint8)
    return img


def compose_grid(images, labels, cols=3, gap=8):
    """Stack images into a `cols`-wide grid with labels above each cell."""
    labeled = [add_label(img, lab) for img, lab in zip(images, labels)]
    h, w, _ = labeled[0].shape
    rows = []
    for r in range((len(labeled) + cols - 1) // cols):
        cells = labeled[r * cols:(r + 1) * cols]
        while len(cells) < cols:
            cells.append(np.full_like(cells[0], 30))
        gap_v = np.full((h, gap, 3), 30, dtype=np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, gap_v, c], axis=1)
        rows.append(row)
    gap_h = np.full((gap, rows[0].shape[1], 3), 30, dtype=np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, gap_h, r], axis=0)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default="compare_6way.png")
    args = ap.parse_args()

    print("=" * 70)
    print(f'Prompt: "{args.prompt}"  seed={args.seed}  '
          f'{args.width}x{args.height}  {args.steps} steps')
    print("=" * 70)

    # Phase 1: ours, 4 variants
    print("\nPhase 1: our CUDA runs")
    ours_imgs = []
    for label, env in VARIANTS_OURS:
        img = run_ours(label, env, args.prompt, args.seed,
                       args.height, args.width, args.steps)
        ours_imgs.append(img)
        # Slugify label for filename
        slug = label.lower().replace(" ", "_")
        save_png(img, os.path.join(CUDA_DIR, f"out_6way_{slug}.png"))

    # Cache hidden states (last run wrote them)
    src_hidden = os.path.join(CUDA_DIR, "cuda_flux2_text_hidden.npy")
    cached_hidden = os.path.join(CUDA_DIR, "hidden_6way.npy")
    subprocess.run(["cp", src_hidden, cached_hidden], check=True)

    # Phase 2: ComfyUI, 2 variants. Load + run each separately to free VRAM.
    print("\nPhase 2: ComfyUI runs")
    import torch
    comfy_imgs = []
    for mode_name, dt in [("BF16", torch.bfloat16), ("FP8", "fp8")]:
        print(f"  loading comfy ({mode_name}) ...")
        mp, vae, device = load_comfy(dt)
        img = run_comfy(mp, vae, device, cached_hidden, args.seed,
                        args.height, args.width, args.steps)
        comfy_imgs.append(img)
        save_png(img, os.path.join(CUDA_DIR,
                 f"out_6way_pytorch_{mode_name.lower()}.png"))
        # Free
        del mp, vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 3: composite
    print("\nPhase 3: building composite")
    images = ours_imgs + comfy_imgs
    labels = [v[0] for v in VARIANTS_OURS] + ["PYTORCH BF16", "PYTORCH FP8"]
    grid = compose_grid(images, labels, cols=3, gap=8)
    out_path = os.path.join(CUDA_DIR, args.out)
    save_png(grid, out_path)
    print(f"  saved {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    # Print pairwise diff stats vs ours-FP8-LToff (the "reference fast path")
    print("\nPairwise max-channel diffs vs PYTORCH BF16:")
    ref = comfy_imgs[0]  # pytorch bf16 as reference
    for img, lab in zip(images, labels):
        d = np.abs(img.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
        print(f"  {lab:20s} max={int(d.max()):4d} mean={float(d.mean()):.2f}")


if __name__ == "__main__":
    main()
