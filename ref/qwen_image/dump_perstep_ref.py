#!/usr/bin/env python3
"""Dump per-step latents from ComfyUI for qwen-image using our_prng noise.

Matches the config used by cuda/qimg --generate --steps 10 --seed 42:
  256x256, 10 euler steps, cfg=2.5, shift=3.1, simple scheduler,
  prompt='a red apple on a white table', negative=' '.

Output: /tmp/ref_step_00.npy .. /tmp/ref_step_10.npy  (F32 (16,32,32))
"""
import os, sys, math
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

import comfy.sd
import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils

model_dir = '/mnt/disk01/models/qwen-image-st'
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
print("Loading DiT...")
mp = comfy.sd.load_diffusion_model(dit_path)

sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
sampling_type = comfy.model_sampling.CONST
class MSA(sampling_base, sampling_type):
    pass
ms = MSA(mp.model.model_config)
ms.set_parameters(shift=3.1, multiplier=1.0)
mp.add_object_patch("model_sampling", ms)

clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
print("Loading CLIP...")
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
tokens_neg = clip.tokenize(" ")
negative = clip.encode_from_tokens_scheduled(tokens_neg)

# Matching our_prng_noise layout (1, 16, 1, 32, 32)
noise_np = our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32)
noise = torch.from_numpy(noise_np)
latent_image = torch.zeros_like(noise)

step_latents = {}

def callback(step, denoised, x, total_steps):
    # x is the latent at the START of step (pre-Euler); denoised is the predicted x0.
    # For flow matching + CONST: we want to track x over time.
    step_latents[step] = x.detach().cpu().float().numpy().copy()
    print(f"  step {step+1}/{total_steps}: x std={float(x.std()):.4f} denoised std={float(denoised.std()):.4f}")

print("Sampling 10 steps, cfg=2.5, 'a red apple on a white table'...")
samples = comfy.sample.sample(
    mp, noise,
    10, 2.5,
    "euler", "simple",
    positive, negative,
    latent_image, denoise=1.0,
    seed=42,
    callback=callback,
)

print(f"Final samples: std={float(samples.std()):.4f} shape={tuple(samples.shape)}")

# Save trajectory
import numpy as np
os.makedirs('/tmp/refsteps', exist_ok=True)
for k, v in sorted(step_latents.items()):
    p = f'/tmp/refsteps/ref_step_{k:02d}.npy'
    np.save(p, v)
    print(f"  wrote {p} shape={v.shape} std={v.std():.4f}")
# Final
np.save('/tmp/refsteps/ref_final.npy', samples.cpu().float().numpy())
print("Done.")
