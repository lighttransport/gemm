#!/usr/bin/env python3
"""Generate ComfyUI output using our PRNG noise (with pair caching) for comparison."""
import os, sys, math, time
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
import comfy.sd, comfy.model_management, comfy.model_sampling, comfy.sample, comfy.utils

model_dir = '/mnt/disk01/models/qwen-image-st'
out_dir = '/mnt/disk01/ComfyUI/output'

# Load model
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
print("Loading DiT...")
model_patcher = comfy.sd.load_diffusion_model(dit_path)

sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
sampling_type = comfy.model_sampling.CONST
class ModelSamplingAdvanced(sampling_base, sampling_type):
    pass
model_sampling = ModelSamplingAdvanced(model_patcher.model.model_config)
model_sampling.set_parameters(shift=3.1, multiplier=1.0)
model_patcher.add_object_patch("model_sampling", model_sampling)

# Text encoder
clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
print("Loading CLIP...")
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)

tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
tokens_neg = clip.tokenize(" ")
negative = clip.encode_from_tokens_scheduled(tokens_neg)

# VAE
vae_path = os.path.join(model_dir, 'vae', 'qwen_image_vae.safetensors')
print("Loading VAE...")
vae_sd = comfy.utils.load_torch_file(vae_path)
vae = comfy.sd.VAE(sd=vae_sd)

# Our noise (with pair caching)
noise_np = our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32)
noise = torch.from_numpy(noise_np)
latent_image = torch.zeros_like(noise)

for n_steps in [10, 20]:
    print(f"\nGenerating {n_steps} steps, cfg=2.5...")
    t0 = time.time()
    samples = comfy.sample.sample(
        model_patcher, noise,
        n_steps, 2.5,
        "euler", "simple",
        positive, negative,
        latent_image, denoise=1.0,
        seed=42,
    )
    elapsed = time.time() - t0
    print(f"  Sampled in {elapsed:.1f}s")

    # Save latent
    lat_path = os.path.join(out_dir, f'cf_ournoise2_{n_steps}step_latent.npy')
    np.save(lat_path, samples.cpu().float().numpy())
    print(f"  Saved latent: {lat_path}")

    # VAE decode
    vae.process_output = lambda image: (image + 1.0) / 2.0
    images = vae.decode(samples).clamp(0.0, 1.0)
    img_np = images[0].detach().cpu().float().numpy()
    while img_np.ndim > 3:
        img_np = img_np[0]
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    from PIL import Image
    img_path = os.path.join(out_dir, f'cf_ournoise2_{n_steps}step.png')
    Image.fromarray(img_np).save(img_path)
    print(f"  Saved: {img_path}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Done.")
