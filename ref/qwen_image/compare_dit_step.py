#!/usr/bin/env python3
"""
Compare a single DiT sampling step between ComfyUI and our C implementation.

1. Loads our initial noise from cuda_latent_init.bin (or generates matching noise)
2. Runs one ComfyUI sampling step
3. Saves the resulting latent for comparison with C output

Usage:
    uv run python compare_dit_step.py --noise ../../cuda/qimg/cuda_latent.bin
"""
import argparse, os, sys, struct, time
import numpy as np
import torch

def our_prng_noise(seed, shape):
    """Replicate our LCG PRNG to generate matching noise."""
    state = seed
    n = 1
    for s in shape:
        n *= s
    vals = np.zeros(n, dtype=np.float32)
    for i in range(0, n, 2):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10:
            u1 = 1e-10
        import math
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        vals[i] = z0
        if i + 1 < n:
            vals[i + 1] = z1  # Box-Muller produces pairs but our PRNG only uses z0
    # Actually our PRNG only uses z0 (cos), not z1
    state2 = seed
    for i in range(n):
        state2 = (state2 * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = (state2 >> 11) / (1 << 53)
        state2 = (state2 * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = (state2 >> 11) / (1 << 53)
        if u1 < 1e-10:
            u1 = 1e-10
        vals[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return vals.reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfyui-dir', default='/mnt/disk01/ComfyUI')
    parser.add_argument('--model-dir', default='/mnt/disk01/models/qwen-image-st')
    parser.add_argument('--prompt', default='a red apple on a white table')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shift', type=float, default=3.1)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lat_h, lat_w = args.height // 8, args.width // 8

    # Generate our PRNG noise
    print("Generating noise with our PRNG (seed=42)...")
    our_noise = our_prng_noise(args.seed, (16, lat_h, lat_w))
    print(f"  Noise: shape={our_noise.shape}, min={our_noise.min():.3f}, max={our_noise.max():.3f}, "
          f"mean={our_noise.mean():.4f}, std={our_noise.std():.4f}")

    # Save noise for C code comparison
    np.save(os.path.join(args.output_dir, 'our_noise.npy'), our_noise)

    # Import ComfyUI
    sys.path.insert(0, args.comfyui_dir)
    os.chdir(args.comfyui_dir)

    import comfy.sd, comfy.sample, comfy.model_sampling

    # Load model
    dit_path = os.path.join(args.model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
    print(f"\nLoading DiT: {os.path.basename(dit_path)}")
    model_patcher = comfy.sd.load_diffusion_model(dit_path)

    # Apply AuraFlow shift
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST
    class MSA(sampling_base, sampling_type):
        pass
    ms = MSA(model_patcher.model.model_config)
    ms.set_parameters(shift=args.shift, multiplier=1.0)
    model_patcher.add_object_patch("model_sampling", ms)
    print(f"  AuraFlow shift={args.shift}, multiplier=1.0")

    # Load text encoder
    clip_path = os.path.join(args.model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    print(f"\nLoading CLIP: {os.path.basename(clip_path)}")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)

    # Encode text
    tokens = clip.tokenize(args.prompt)
    positive = clip.encode_from_tokens_scheduled(tokens)
    tokens_neg = clip.tokenize(" ")
    negative = clip.encode_from_tokens_scheduled(tokens_neg)

    # Print text conditioning stats
    pos_hs = positive[0][0]
    neg_hs = negative[0][0]
    print(f"  Positive: shape={pos_hs.shape}, min={pos_hs.min():.2f}, max={pos_hs.max():.2f}")
    print(f"  Negative: shape={neg_hs.shape}, min={neg_hs.min():.2f}, max={neg_hs.max():.2f}")

    # Save text conditioning
    np.save(os.path.join(args.output_dir, 'comfyui_text_hidden.npy'), pos_hs.cpu().float().numpy())
    np.save(os.path.join(args.output_dir, 'comfyui_text_hidden_neg.npy'), neg_hs.cpu().float().numpy())

    # Create noise tensor with our PRNG noise
    noise_5d = torch.from_numpy(our_noise).unsqueeze(0).unsqueeze(2).float()  # [1, 16, 1, H, W]
    print(f"\nNoise tensor: shape={noise_5d.shape}")
    latent_image = torch.zeros_like(noise_5d)

    # Run 1 step of sampling
    print("\nRunning 1 step of ComfyUI sampling...")
    t0 = time.time()
    samples = comfy.sample.sample(
        model_patcher, noise_5d,
        1, 2.5,  # 1 step, cfg=2.5
        "euler", "simple",
        positive, negative,
        latent_image, denoise=1.0,
        seed=args.seed,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Save 1-step latent
    lat_1step = samples.cpu().float().numpy()
    print(f"  Latent after 1 step: shape={lat_1step.shape}, "
          f"min={lat_1step.min():.3f}, max={lat_1step.max():.3f}, "
          f"mean={lat_1step.mean():.4f}, std={lat_1step.std():.4f}")
    np.save(os.path.join(args.output_dir, 'comfyui_latent_1step.npy'), lat_1step)

    # Also run full 20 steps for comparison
    print("\nRunning full 20 steps...")
    t0 = time.time()
    noise_5d_full = torch.from_numpy(our_noise).unsqueeze(0).unsqueeze(2).float()
    samples_full = comfy.sample.sample(
        model_patcher, noise_5d_full,
        20, 2.5,
        "euler", "simple",
        positive, negative,
        latent_image, denoise=1.0,
        seed=args.seed,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    lat_full = samples_full.cpu().float().numpy()
    print(f"  Final latent: min={lat_full.min():.3f}, max={lat_full.max():.3f}, "
          f"mean={lat_full.mean():.4f}, std={lat_full.std():.4f}")
    np.save(os.path.join(args.output_dir, 'comfyui_latent_20step_ournoise.npy'), lat_full)

    # Decode with VAE
    vae_path = os.path.join(args.model_dir, 'vae', 'qwen_image_vae.safetensors')
    import comfy.utils
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    vae.process_output = lambda image: (image + 1.0) / 2.0
    images = vae.decode(samples_full).clamp(0, 1)
    img_np = (images[0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    while img_np.ndim > 3:
        img_np = img_np[0]
    from PIL import Image
    out_path = os.path.join(args.output_dir, f'comfyui_ournoise_{args.height}.png')
    Image.fromarray(img_np).save(out_path)
    print(f"\nSaved: {out_path}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Done.")


if __name__ == "__main__":
    main()
