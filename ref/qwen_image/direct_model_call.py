#!/usr/bin/env python3
"""Directly call the ComfyUI model and capture raw output."""
import os, sys, struct, math
import numpy as np
import torch

def our_prng_noise(seed, n):
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        vals[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return vals

sys.path.insert(0, '/mnt/disk01/ComfyUI')
os.chdir('/mnt/disk01/ComfyUI')

import comfy.sd, comfy.model_management

model_dir = '/mnt/disk01/models/qwen-image-st'
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')

print("Loading DiT...")
model_patcher = comfy.sd.load_diffusion_model(dit_path)

# Load text
clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
print("Loading CLIP...")
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
txt_hs = positive[0][0]  # [1, 12, 3584]
print(f"Text: {txt_hs.shape}, std={txt_hs.std():.2f}")

# Generate noise
noise = our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32)
x = torch.from_numpy(noise).float()

# Load model to GPU
comfy.model_management.load_models_gpu([model_patcher])
device = comfy.model_management.get_torch_device()

# Call model directly
model = model_patcher.model
ms = model.model_sampling
sigma = torch.tensor([1.0], device=device)
timestep = ms.timestep(sigma)
print(f"sigma={sigma.item()}, timestep={timestep.item()}")

with torch.no_grad():
    x_in = x.to(device)

    # apply_model handles calculate_input + forward + calculate_denoised
    # But we want the RAW model output BEFORE calculate_denoised
    # So let's call the components separately

    # 1. calculate_input
    x_input = ms.calculate_input(sigma, x_in)
    print(f"x_input: std={x_input.std().item():.4f}")

    # 2. Call the diffusion model directly
    # Build context
    context = txt_hs.to(device)
    model_output = model.diffusion_model(x_input, timestep, context=context)

    print(f"\nRaw model output: {model_output.shape}")
    out = model_output.float().cpu().numpy()
    print(f"  min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.6f}, std={out.std():.4f}")

    # 3. calculate_denoised
    denoised = ms.calculate_denoised(sigma, model_output, x_input)
    den = denoised.float().cpu().numpy()
    print(f"Denoised (x - v*sigma): min={den.min():.4f}, max={den.max():.4f}, std={den.std():.4f}")

    # Save raw model output
    np.save('output/raw_model_output.npy', out)

    # Compare with our C code output
    # Our C code saves dit_output_step1.bin which is [256, 64] in token space
    # The model output here is [1, 16, 1, 32, 32] in latent space
    # Patchify to compare
    out_lat = out[0, :, 0, :, :]  # [16, 32, 32]
    ps = 2
    hp, wp = 16, 16
    tok = np.zeros((256, 64), dtype=np.float32)
    for py in range(hp):
        for px in range(wp):
            t = py * wp + px
            idx = 0
            for c in range(16):
                for dy in range(ps):
                    for dx in range(ps):
                        tok[t, idx] = out_lat[c, py*ps+dy, px*ps+dx]
                        idx += 1
    print(f"\nPatchified model output: {tok.shape}, std={tok.std():.4f}")
    np.save('output/comfyui_tokens_raw.npy', tok)
    print("Saved output/comfyui_tokens_raw.npy")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
