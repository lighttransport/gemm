#!/usr/bin/env python3
"""Trace DiT block-by-block intermediates in ComfyUI F32 for comparison with CUDA.
Uses our PRNG noise, sigma=first_step (matching CUDA's first denoising step).
Saves: img_in, txt_in, temb, after each block's img/txt, final output."""
import os, sys, math
import numpy as np
import torch

def our_prng_noise(seed, n):
    """Match C code's Box-Muller with pair caching (cos+sin per u1,u2 pair)."""
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        r = math.sqrt(-2.0 * math.log(u1))
        vals[i] = r * math.cos(2.0 * math.pi * u2)
        i += 1
        if i < n:
            vals[i] = r * math.sin(2.0 * math.pi * u2)
            i += 1
    return vals

sys.path.insert(0, '/mnt/disk01/ComfyUI')
os.chdir('/mnt/disk01/ComfyUI')
import comfy.sd, comfy.model_management, comfy.model_sampling

model_dir = '/mnt/disk01/models/qwen-image-st'
out_dir = '/mnt/disk01/ComfyUI/output'
os.makedirs(out_dir, exist_ok=True)

# Load model
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
print("Loading DiT...")
model_patcher = comfy.sd.load_diffusion_model(dit_path)

# Apply AuraFlow shift=3.1
sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
sampling_type = comfy.model_sampling.CONST
class ModelSamplingAdvanced(sampling_base, sampling_type):
    pass
model_sampling = ModelSamplingAdvanced(model_patcher.model.model_config)
model_sampling.set_parameters(shift=3.1, multiplier=1.0)
model_patcher.add_object_patch("model_sampling", model_sampling)

comfy.model_management.load_models_gpu([model_patcher])
device = comfy.model_management.get_torch_device()
dm = model_patcher.model.diffusion_model

# Text encoder
clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
print("Loading CLIP...")
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
txt_hs = positive[0][0].to(device)
print(f"  txt_hs: {txt_hs.shape} dtype={txt_hs.dtype}")

# Save text hidden states
np.save(os.path.join(out_dir, 'cf_dit_txt_input.npy'),
        txt_hs.float().cpu().numpy()[0])

# Noise (our PRNG)
noise = our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32)
x = torch.from_numpy(noise).to(device).to(txt_hs.dtype)

# Compute sigma for first step (10 steps, shift=3.1)
n_steps = 10
es = math.exp(3.1)
sigma_raw = 1.0 - 0.0 / n_steps  # first step: i=0
sigma = es * sigma_raw / (1.0 + (es - 1.0) * sigma_raw)
print(f"  First step sigma={sigma:.6f}")

# Get timestep from model_sampling
ms = model_patcher.model.model_sampling
sigma_t = torch.tensor([sigma], device=device)
timestep = ms.timestep(sigma_t)
print(f"  timestep={timestep.item():.6f}")

with torch.no_grad():
    # Process image
    hidden_states, img_ids, orig_shape = dm.process_img(x)
    np.save(os.path.join(out_dir, 'cf_dit_img_input.npy'),
            hidden_states.float().cpu().numpy()[0])
    print(f"  hidden_states (patches): {hidden_states.shape} std={hidden_states.float().std():.6f}")

    # img_in
    img_proj = dm.img_in(hidden_states)
    np.save(os.path.join(out_dir, 'cf_dit_img_projected.npy'),
            img_proj.float().cpu().numpy()[0])
    print(f"  img_projected: {img_proj.shape} std={img_proj.float().std():.6f}")

    # txt_in
    txt_normed = dm.txt_norm(txt_hs)
    txt_proj = dm.txt_in(txt_normed)
    np.save(os.path.join(out_dir, 'cf_dit_txt_projected.npy'),
            txt_proj.float().cpu().numpy()[0])
    print(f"  txt_projected: {txt_proj.shape} std={txt_proj.float().std():.6f}")

    # Timestep embedding
    temb = dm.time_text_embed(timestep, img_proj)
    np.save(os.path.join(out_dir, 'cf_dit_temb.npy'),
            temb.float().cpu().numpy()[0])
    print(f"  temb: {temb.shape} std={temb.float().std():.6f}")

    # RoPE
    ids = torch.cat((
        torch.arange(8, 20, device=device).reshape(1,-1,1).repeat(1,1,3).to(txt_hs.dtype),
        img_ids
    ), dim=1)
    pe = dm.pe_embedder(ids).to(txt_hs.dtype).contiguous()

    # Run all blocks, save intermediates
    h = img_proj.clone()
    e = txt_proj.clone()

    for i, blk in enumerate(dm.transformer_blocks):
        e, h = blk(
            hidden_states=h,
            encoder_hidden_states=e,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=pe,
        )
        # Save every block's output
        h_np = h.float().cpu().numpy()[0]
        e_np = e.float().cpu().numpy()[0]
        np.save(os.path.join(out_dir, f'cf_dit_block{i:02d}_img.npy'), h_np)
        np.save(os.path.join(out_dir, f'cf_dit_block{i:02d}_txt.npy'), e_np)
        if i < 3 or i == 59:
            print(f"  block {i:2d}: img std={h.float().std():.6f}  txt std={e.float().std():.6f}")

    # Final: norm_out + proj_out
    h_final = dm.norm_out(h, temb)
    np.save(os.path.join(out_dir, 'cf_dit_norm_out.npy'),
            h_final.float().cpu().numpy()[0])
    print(f"  norm_out: std={h_final.float().std():.6f}")

    h_proj = dm.proj_out(h_final)
    np.save(os.path.join(out_dir, 'cf_dit_proj_out.npy'),
            h_proj.float().cpu().numpy()[0])
    print(f"  proj_out: {h_proj.shape} std={h_proj.float().std():.6f}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"\nSaved to {out_dir}/cf_dit_*.npy")
