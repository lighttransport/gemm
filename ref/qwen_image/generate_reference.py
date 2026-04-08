#!/usr/bin/env python3
"""
Generate ComfyUI reference data for pipeline verification.

Produces all .npy files needed by verify_pipeline.py in one shot.

Usage:
    # Generate everything (requires ComfyUI + GPU)
    python generate_reference.py --all

    # Only text encoder hidden states (fastest)
    python generate_reference.py --text

    # Only per-block DiT outputs
    python generate_reference.py --dit-blocks

    # Only per-step latents
    python generate_reference.py --dit-steps

    # Only VAE layer outputs (needs a latent .npy as input)
    python generate_reference.py --vae --latent output/comfyui_latent.npy

    # Custom paths
    python generate_reference.py --all \
        --comfyui-dir /mnt/disk01/ComfyUI \
        --model-dir /mnt/disk01/models/qwen-image-st \
        --output-dir output/
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch


# ---- Our PRNG (matches C code exactly) ----

def our_prng_noise(seed, n):
    """Box-Muller with pair caching, matching C code's LCG PRNG."""
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
        vals[i] = r * math.cos(2.0 * math.pi * u2)
        i += 1
        if i < n:
            vals[i] = r * math.sin(2.0 * math.pi * u2)
            i += 1
    return vals


# ---- ComfyUI setup ----

def setup_comfyui(comfyui_dir):
    """Add ComfyUI to path and chdir."""
    sys.path.insert(0, comfyui_dir)
    os.chdir(comfyui_dir)


def load_model(model_dir):
    """Load DiT model with AuraFlow shift=3.1 scheduler."""
    import comfy.sd
    import comfy.model_sampling

    dit_path = os.path.join(model_dir, 'diffusion_models',
                            'qwen_image_fp8_e4m3fn.safetensors')
    print(f"Loading DiT: {dit_path}")
    model_patcher = comfy.sd.load_diffusion_model(dit_path)

    # Set scheduler: AuraFlow shift=3.1, simple mode
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    ms = ModelSamplingAdvanced(model_patcher.model.model_config)
    ms.set_parameters(shift=3.1, multiplier=1.0)
    model_patcher.add_object_patch("model_sampling", ms)
    return model_patcher


def load_clip(model_dir):
    """Load text encoder."""
    import comfy.sd
    clip_path = os.path.join(model_dir, 'text_encoders',
                             'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    print(f"Loading CLIP: {clip_path}")
    return comfy.sd.load_clip(ckpt_paths=[clip_path],
                               clip_type=comfy.sd.CLIPType.QWEN_IMAGE)


def load_vae(model_dir):
    """Load VAE decoder."""
    import comfy.sd
    import comfy.utils
    vae_path = os.path.join(model_dir, 'vae', 'qwen_image_vae.safetensors')
    print(f"Loading VAE: {vae_path}")
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))


# ---- Reference generation functions ----

def generate_text(clip, prompt, negative, output_dir):
    """Generate text encoder hidden states."""
    print(f"\n[Text Encoder] prompt='{prompt}', negative='{negative}'")
    tokens = clip.tokenize(prompt)
    pos = clip.encode_from_tokens_scheduled(tokens)
    pos_hs = pos[0][0]
    pos_path = os.path.join(output_dir, "comfyui_text_hidden.npy")
    np.save(pos_path, pos_hs.cpu().float().numpy())
    print(f"  Positive: {pos_hs.shape} -> {pos_path}")

    tokens_neg = clip.tokenize(negative)
    neg = clip.encode_from_tokens_scheduled(tokens_neg)
    neg_hs = neg[0][0]
    neg_path = os.path.join(output_dir, "comfyui_text_hidden_neg.npy")
    np.save(neg_path, neg_hs.cpu().float().numpy())
    print(f"  Negative: {neg_hs.shape} -> {neg_path}")

    return pos, neg


def generate_dit_blocks(model_patcher, positive, negative, output_dir,
                        height=256, width=256, seed=42, n_steps=10):
    """Generate per-block DiT reference outputs (all 60 blocks, step 0 only)."""
    import comfy.model_management
    print(f"\n[DiT Blocks] {width}x{height}, {n_steps} steps, seed={seed}")

    lat_h, lat_w = height // 8, width // 8
    noise_np = our_prng_noise(seed, 16 * lat_h * lat_w).reshape(1, 16, 1, lat_h, lat_w)
    noise = torch.from_numpy(noise_np)

    # Get model and run single step with block output hooks
    comfy.model_management.load_models_gpu([model_patcher])
    model = model_patcher.model

    # Get sigma for step 0
    ms = model_patcher.get_model_object("model_sampling")
    sigmas = ms.sigma(torch.arange(n_steps + 1, device='cpu') / n_steps).flip(0)

    # Prepare input
    sigma = sigmas[0]
    x = noise.to(comfy.model_management.get_torch_device())
    x_scaled = ms.noise_scaling(sigma, x, torch.zeros_like(x))

    # Get conditioning
    pos_cond = positive[0][0].to(x.device)
    neg_cond = negative[0][0].to(x.device)
    timestep = ms.timestep(sigma).to(x.device)

    # Hook into each block to save output
    block_outputs_img = {}
    block_outputs_txt = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                block_outputs_img[block_idx] = output[0].detach().cpu().float().numpy()
                block_outputs_txt[block_idx] = output[1].detach().cpu().float().numpy()
        return hook_fn

    hooks = []
    dit = model.diffusion_model
    if hasattr(dit, 'blocks'):
        for i, block in enumerate(dit.blocks):
            h = block.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Forward pass (conditional)
    try:
        with torch.no_grad():
            model.apply_model(x_scaled, timestep, c_crossattn=pos_cond)
    finally:
        for h in hooks:
            h.remove()

    # Save block outputs
    for idx in sorted(block_outputs_img.keys()):
        img_path = os.path.join(output_dir, f"cf_dit_block{idx:02d}_img.npy")
        txt_path = os.path.join(output_dir, f"cf_dit_block{idx:02d}_txt.npy")
        np.save(img_path, block_outputs_img[idx])
        np.save(txt_path, block_outputs_txt[idx])
    print(f"  Saved {len(block_outputs_img)} block outputs")


def generate_dit_steps(model_patcher, positive, negative, output_dir,
                       height=256, width=256, seed=42, n_steps=10, cfg_scale=2.5):
    """Generate per-step latent trajectory using ComfyUI sampling."""
    import comfy.sample
    print(f"\n[DiT Steps] {width}x{height}, {n_steps} steps, cfg={cfg_scale}, seed={seed}")

    lat_h, lat_w = height // 8, width // 8
    noise_np = our_prng_noise(seed, 16 * lat_h * lat_w).reshape(1, 16, 1, lat_h, lat_w)
    noise = torch.from_numpy(noise_np)
    latent_image = torch.zeros_like(noise)

    step_latents = {}

    def callback(step, denoised, x, total_steps):
        step_latents[step] = x.detach().cpu().float().numpy()

    samples = comfy.sample.sample(
        model_patcher, noise,
        n_steps, cfg_scale,
        "euler", "simple",
        positive, negative,
        latent_image, denoise=1.0,
        seed=seed, callback=callback,
    )

    # Save per-step latents
    for step_idx in sorted(step_latents.keys()):
        path = os.path.join(output_dir, f"cf_perstep_{step_idx:02d}.npy")
        np.save(path, step_latents[step_idx])
    print(f"  Saved {len(step_latents)} step latents")

    # Save final latent
    final_path = os.path.join(output_dir, "comfyui_latent.npy")
    np.save(final_path, samples.cpu().float().numpy())
    print(f"  Saved final latent: {final_path}")

    return samples


def generate_vae(vae, latent, output_dir, height=256, width=256):
    """Decode latent through VAE and save intermediate + final outputs."""
    print(f"\n[VAE Decode] {width}x{height}")

    vae.process_output = lambda image: (image + 1.0) / 2.0
    images = vae.decode(latent).clamp(0.0, 1.0)

    # Save final image
    img_np = images[0].detach().cpu().float().numpy()
    while img_np.ndim > 3:
        img_np = img_np[0]
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    from PIL import Image
    img_path = os.path.join(output_dir, f"comfyui_ournoise_{height}.png")
    Image.fromarray(img_np).save(img_path)
    print(f"  Saved image: {img_path}")


def generate_full_pipeline(model_patcher, clip, vae, output_dir,
                           prompt="a red apple on a white table", negative=" ",
                           height=256, width=256, seed=42, n_steps=10, cfg_scale=2.5):
    """Generate all reference data in one go."""
    # Text encoder
    pos, neg = generate_text(clip, prompt, negative, output_dir)

    # DiT per-step (includes final latent)
    latent = generate_dit_steps(model_patcher, pos, neg, output_dir,
                                height, width, seed, n_steps, cfg_scale)

    # VAE decode final image
    generate_vae(vae, latent, output_dir, height, width)


def main():
    parser = argparse.ArgumentParser(description="Generate ComfyUI reference data")
    parser.add_argument('--comfyui-dir', default='/mnt/disk01/ComfyUI')
    parser.add_argument('--model-dir', default='/mnt/disk01/models/qwen-image-st')
    parser.add_argument('--output-dir', default='output/')
    parser.add_argument('--prompt', default='a red apple on a white table')
    parser.add_argument('--negative', default=' ')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-scale', type=float, default=2.5)

    # Mode selection
    parser.add_argument('--all', action='store_true', help='Generate everything')
    parser.add_argument('--text', action='store_true', help='Text encoder only')
    parser.add_argument('--dit-blocks', action='store_true', help='DiT per-block outputs')
    parser.add_argument('--dit-steps', action='store_true', help='DiT per-step latents')
    parser.add_argument('--vae', action='store_true', help='VAE decode')
    parser.add_argument('--latent', help='Latent .npy for VAE-only mode')

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not (args.all or args.text or args.dit_blocks or args.dit_steps or args.vae):
        args.all = True

    os.makedirs(args.output_dir, exist_ok=True)

    setup_comfyui(args.comfyui_dir)

    t0 = time.time()

    if args.all:
        model = load_model(args.model_dir)
        clip = load_clip(args.model_dir)
        vae = load_vae(args.model_dir)
        generate_full_pipeline(model, clip, vae, args.output_dir,
                               args.prompt, args.negative,
                               args.height, args.width, args.seed,
                               args.steps, args.cfg_scale)
    else:
        clip = None
        model = None

        if args.text:
            clip = load_clip(args.model_dir)
            generate_text(clip, args.prompt, args.negative, args.output_dir)

        if args.dit_blocks or args.dit_steps:
            model = load_model(args.model_dir)
            if not clip:
                clip = load_clip(args.model_dir)
            tokens = clip.tokenize(args.prompt)
            pos = clip.encode_from_tokens_scheduled(tokens)
            tokens_neg = clip.tokenize(args.negative)
            neg = clip.encode_from_tokens_scheduled(tokens_neg)

            if args.dit_blocks:
                generate_dit_blocks(model, pos, neg, args.output_dir,
                                    args.height, args.width, args.seed, args.steps)
            if args.dit_steps:
                latent = generate_dit_steps(model, pos, neg, args.output_dir,
                                            args.height, args.width, args.seed,
                                            args.steps, args.cfg_scale)

        if args.vae:
            vae = load_vae(args.model_dir)
            if args.latent:
                latent = torch.from_numpy(np.load(args.latent))
            elif 'latent' not in dir():
                print("Error: --vae requires --latent <path> or run with --dit-steps first")
                return 1
            generate_vae(vae, latent, args.output_dir, args.height, args.width)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
