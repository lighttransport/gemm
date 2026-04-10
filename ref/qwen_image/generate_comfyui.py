#!/usr/bin/env python3
"""
Generate ground-truth Qwen-Image output using ComfyUI internals directly.

Uses FP8 safetensor weights with ComfyUI's model loading + sampling.
No ComfyUI server needed.

Usage:
    uv run python generate_comfyui.py \
        --prompt "a red apple on a white table" \
        --width 512 --height 512 --steps 20 --seed 42 \
        --output output/ground_truth_comfyui.png
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfyui-dir', default='/mnt/disk01/ComfyUI')
    parser.add_argument('--model-dir', default='/mnt/disk01/models/qwen-image-st')
    parser.add_argument('--prompt', default='a red apple on a white table')
    parser.add_argument('--negative-prompt', default=' ')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cfg', type=float, default=2.5)
    parser.add_argument('--shift', type=float, default=3.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='output/ground_truth_comfyui.png')
    parser.add_argument('--dump-latent', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Add ComfyUI to path
    sys.path.insert(0, args.comfyui_dir)
    os.chdir(args.comfyui_dir)  # ComfyUI expects to run from its own dir

    print(f"ComfyUI: {args.comfyui_dir}")
    print(f"Models:  {args.model_dir}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory//(1024**3)} GB)")

    # Import ComfyUI
    print("\nImporting ComfyUI...", flush=True)
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import comfy.sample
    import comfy.model_sampling

    model_dir = args.model_dir

    # ---- 1. Load diffusion model (UNET/DiT) ----
    dit_path = os.path.join(model_dir, 'diffusion_models',
                            'qwen_image_fp8_e4m3fn.safetensors')
    print(f"\n[1/4] Loading DiT: {os.path.basename(dit_path)}", flush=True)
    t0 = time.time()
    model_patcher = comfy.sd.load_diffusion_model(dit_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Apply ModelSamplingAuraFlow shift (same as the ComfyUI node)
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model_patcher.model.model_config)
    model_sampling.set_parameters(shift=args.shift, multiplier=1.0)
    model_patcher.add_object_patch("model_sampling", model_sampling)
    print(f"  Applied AuraFlow shift={args.shift}")

    # ---- 2. Load text encoder (CLIP) ----
    clip_path = os.path.join(model_dir, 'text_encoders',
                             'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    print(f"\n[2/4] Loading CLIP: {os.path.basename(clip_path)}", flush=True)
    t0 = time.time()
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ---- 3. Load VAE ----
    vae_path = os.path.join(model_dir, 'vae', 'qwen_image_vae.safetensors')
    print(f"\n[3/4] Loading VAE: {os.path.basename(vae_path)}", flush=True)
    t0 = time.time()
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ---- 4. Generate ----
    print(f"\n[4/4] Generating {args.width}x{args.height}, "
          f"{args.steps} steps, cfg={args.cfg}, seed={args.seed}", flush=True)
    print(f"  Prompt: \"{args.prompt}\"")

    # Encode text
    t0 = time.time()
    tokens = clip.tokenize(args.prompt)
    positive = clip.encode_from_tokens_scheduled(tokens)
    tokens_neg = clip.tokenize(args.negative_prompt)
    negative = clip.encode_from_tokens_scheduled(tokens_neg)
    print(f"  Text encoded in {time.time()-t0:.1f}s")

    # Create noise — Qwen-Image uses 3D VAE, latent has temporal dim
    latent_h = args.height // 8
    latent_w = args.width // 8
    noise = torch.randn(1, 16, 1, latent_h, latent_w,
                        generator=torch.manual_seed(args.seed))
    latent_image = torch.zeros_like(noise)

    # Sample
    t0 = time.time()
    samples = comfy.sample.sample(
        model_patcher, noise,
        args.steps, args.cfg,
        "euler", "simple",
        positive, negative,
        latent_image, denoise=1.0,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Sampled in {elapsed:.1f}s ({elapsed/args.steps:.1f}s/step)")

    if args.dump_latent:
        lat_path = args.output.replace('.png', '_latent.npy')
        np.save(lat_path, samples.cpu().float().numpy())
        print(f"  Saved latent: {lat_path}")

    # VAE decode — patch process_output to avoid inplace ops on inference tensors
    vae.process_output = lambda image: (image + 1.0) / 2.0
    t0 = time.time()
    images = vae.decode(samples)
    images = images.clamp(0.0, 1.0)
    print(f"  VAE decoded in {time.time()-t0:.1f}s")

    # Save image (ComfyUI VAE returns [B, T, H, W, C] or [B, H, W, C] in [0,1])
    img_np = images[0].detach().cpu().float().numpy()
    while img_np.ndim > 3:
        img_np = img_np[0]  # remove batch/temporal dims until [H, W, C]
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(img_np).save(args.output)
    print(f"\nSaved: {args.output} ({img_np.shape[1]}x{img_np.shape[0]})")

    # Save numpy for comparison
    npy_path = args.output.replace('.png', '.npy')
    np.save(npy_path, img_np.astype(np.float32) / 255.0)

    # Return to original dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Done.")


if __name__ == "__main__":
    main()
