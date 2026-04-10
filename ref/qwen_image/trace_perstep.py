#!/usr/bin/env python3
"""Save latent at each sampling step using ComfyUI's sample API with callback."""
import os, sys, math
import numpy as np
import torch

def our_prng_noise(seed, n):
    state = seed; vals = np.zeros(n, dtype=np.float32); i = 0
    while i < n:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        r = math.sqrt(-2.0 * math.log(u1))
        vals[i] = r * math.cos(2.0 * math.pi * u2); i += 1
        if i < n: vals[i] = r * math.sin(2.0 * math.pi * u2); i += 1
    return vals

sys.path.insert(0, '/mnt/disk01/ComfyUI')
os.chdir('/mnt/disk01/ComfyUI')
import comfy.sd, comfy.model_management, comfy.model_sampling, comfy.sample

model_dir = '/mnt/disk01/models/qwen-image-st'
out_dir = '/mnt/disk01/ComfyUI/output'

dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
print("Loading...")
model_patcher = comfy.sd.load_diffusion_model(dit_path)
sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
sampling_type = comfy.model_sampling.CONST
class MSA(sampling_base, sampling_type): pass
ms = MSA(model_patcher.model.model_config)
ms.set_parameters(shift=3.1, multiplier=1.0)
model_patcher.add_object_patch("model_sampling", ms)

clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
tokens_neg = clip.tokenize(" ")
negative = clip.encode_from_tokens_scheduled(tokens_neg)

noise = torch.from_numpy(our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32))
latent_image = torch.zeros_like(noise)

# Callback to save per-step latent (ComfyUI passes: step, denoised, x, total_steps)
def step_callback(step, denoised, x, total_steps):
    lat = x.float().cpu().numpy().reshape(16, 32, 32)
    np.save(os.path.join(out_dir, f'cf_perstep_{step:02d}.npy'), lat)
    print(f"  step {step}: std={lat.std():.4f}")

# Run with CFG=2.5
samples = comfy.sample.sample(model_patcher, noise, 10, 2.5,
    "euler", "simple", positive, negative, latent_image,
    denoise=1.0, seed=42, callback=step_callback)
lat_final = samples.float().cpu().numpy().reshape(16, 32, 32)
np.save(os.path.join(out_dir, 'cf_perstep_final.npy'), lat_final)
print(f"Final: std={lat_final.std():.4f}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
