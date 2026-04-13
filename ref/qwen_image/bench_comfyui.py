#!/usr/bin/env python3
"""Time ComfyUI qwen-image sampling for the same configs as our CUDA runner."""
import os, sys, math, time
import numpy as np
import torch


def our_prng_noise(seed, n):
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10:
            u1 = 1e-10
        r = math.sqrt(-2.0 * math.log(u1))
        vals[i] = r * math.cos(2.0 * math.pi * u2); i += 1
        if i < n:
            vals[i] = r * math.sin(2.0 * math.pi * u2); i += 1
    return vals


sys.path.insert(0, '/mnt/disk01/ComfyUI')
os.chdir('/mnt/disk01/ComfyUI')
import comfy.sd, comfy.model_management, comfy.model_sampling, comfy.sample, comfy.utils

model_dir = '/mnt/disk01/models/qwen-image-st'

print("Loading DiT + CLIP + VAE...", flush=True)
t0 = time.time()
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
mp = comfy.sd.load_diffusion_model(dit_path)

sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
sampling_type = comfy.model_sampling.CONST
class MSA(sampling_base, sampling_type):
    pass
ms = MSA(mp.model.model_config)
ms.set_parameters(shift=3.1, multiplier=1.0)
mp.add_object_patch("model_sampling", ms)

clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)

tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
tokens_neg = clip.tokenize(" ")
negative = clip.encode_from_tokens_scheduled(tokens_neg)

vae_path = os.path.join(model_dir, 'vae', 'qwen_image_vae.safetensors')
vae_sd = comfy.utils.load_torch_file(vae_path)
vae = comfy.sd.VAE(sd=vae_sd)
print(f"Load + encode done in {time.time()-t0:.1f}s", flush=True)


def bench(h, w, n_steps, label):
    lat_h, lat_w = h // 8, w // 8
    noise_np = our_prng_noise(42, 16 * lat_h * lat_w).reshape(1, 16, 1, lat_h, lat_w)
    noise = torch.from_numpy(noise_np)
    latent_image = torch.zeros_like(noise)

    # Warm-up run to eat JIT/cache costs
    print(f"\n[{label}] warm-up...", flush=True)
    _ = comfy.sample.sample(
        mp, noise, n_steps, 2.5, "euler", "simple",
        positive, negative, latent_image, denoise=1.0, seed=42)
    torch.cuda.synchronize()

    # Timed run
    print(f"[{label}] timed run...", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    samples = comfy.sample.sample(
        mp, noise, n_steps, 2.5, "euler", "simple",
        positive, negative, latent_image, denoise=1.0, seed=42)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_step = elapsed / n_steps
    print(f"[{label}] {w}x{h}, {n_steps} steps: total={elapsed:.2f}s  per-step={per_step:.3f}s", flush=True)
    return elapsed, per_step


r1 = bench(256, 256, 10, "256-10")
r2 = bench(512, 512, 4,  "512-4")

print("\n=== summary ===")
print(f"256x256 10-step:  {r1[0]:.2f}s total, {r1[1]:.3f}s/step")
print(f"512x512  4-step:  {r2[0]:.2f}s total, {r2[1]:.3f}s/step")
