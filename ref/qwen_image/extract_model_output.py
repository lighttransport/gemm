#!/usr/bin/env python3
"""Extract raw model output at step 1 from ComfyUI for comparison with C code."""
import os, sys, struct, math
import numpy as np
import torch

def our_prng_noise(seed, shape):
    state = seed
    n = 1
    for s in shape: n *= s
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        vals[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return vals.reshape(shape)

def main():
    sys.path.insert(0, '/mnt/disk01/ComfyUI')
    os.chdir('/mnt/disk01/ComfyUI')

    import comfy.sd, comfy.model_sampling, comfy.samplers
    model_dir = '/mnt/disk01/models/qwen-image-st'

    # Load model
    dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
    print(f"Loading DiT...")
    model_patcher = comfy.sd.load_diffusion_model(dit_path)

    # Load text encoder
    clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    print(f"Loading CLIP...")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)

    # Encode text
    tokens = clip.tokenize("a red apple on a white table")
    positive = clip.encode_from_tokens_scheduled(tokens)
    pos_hs = positive[0][0]
    print(f"Text: {pos_hs.shape}")

    # Generate noise with our PRNG
    noise = our_prng_noise(42, (16, 32, 32))
    x = torch.from_numpy(noise).unsqueeze(0).unsqueeze(2).float()  # [1,16,1,32,32]

    # Get the model's sampling info
    ms = model_patcher.model.model_sampling
    print(f"Model sampling type: {type(ms).__name__}")
    print(f"Shift: {ms.shift}")

    # Get sigma for step 0 (from "simple" scheduler with 20 steps)
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", 20)
    print(f"Sigmas[0:3]: {sigmas[:3].tolist()}")
    print(f"dt[0]: {(sigmas[1]-sigmas[0]).item():.6f}")

    sigma0 = sigmas[0]
    print(f"\nsigma0 = {sigma0.item():.6f}")
    print(f"timestep(sigma0) = {ms.timestep(sigma0).item():.6f}")

    # Use ComfyUI's proper sampling for 1 step (handles GPU memory)
    import comfy.sample
    latent_image = torch.zeros_like(x)
    print(f"\nRunning 1-step sampling (no CFG)...")
    samples_1 = comfy.sample.sample(
        model_patcher, x, 1, 1.0,  # 1 step, cfg=1.0 (no guidance)
        "euler", "simple",
        positive, positive,  # same for pos/neg = no CFG
        latent_image, denoise=1.0, seed=42,
    )
    s1 = samples_1.cpu().float().numpy()
    print(f"After 1 step: shape={s1.shape}, min={s1.min():.4f}, max={s1.max():.4f}, std={s1.std():.4f}")

    # The velocity = (samples_1 - x_noisy) / dt
    # where x_noisy = noise_scaling(sigma0, noise, zeros) = sigma0 * noise = noise (sigma0=1)
    x_noisy_np = noise.reshape(1, 16, 1, 32, 32)  # sigma=1.0 * noise
    diff = s1 - x_noisy_np
    dt0 = sigmas[1].item() - sigmas[0].item()
    effective_vel = diff / dt0
    print(f"dt[0] = {dt0:.6f}")
    print(f"Effective velocity: min={effective_vel.min():.4f}, max={effective_vel.max():.4f}, std={effective_vel.std():.4f}")

    # Save
    np.save('output/comfyui_latent_1step_nocfg.npy', s1)
    np.save('output/comfyui_effective_vel.npy', effective_vel)

    # Patchify model output for comparison with C code's token output
    # C code output is [256, 64] = patchified velocity
    # ComfyUI output is [1, 16, 1, 32, 32] = latent-space velocity
    # Need to patchify ComfyUI output to compare with C's dit_output_step1.bin
    out_lat = out_np[0, :, 0, :, :]  # [16, 32, 32]
    ps = 2
    hp, wp = 32//ps, 32//ps  # 16, 16
    tokens = np.zeros((hp*wp, 16*ps*ps), dtype=np.float32)
    for py in range(hp):
        for px in range(wp):
            tok = py * wp + px
            idx = 0
            for c in range(16):
                for dy in range(ps):
                    for dx in range(ps):
                        tokens[tok, idx] = out_lat[c, py*ps+dy, px*ps+dx]
                        idx += 1
    np.save('output/comfyui_tokens_step1.npy', tokens)
    print(f"Saved patchified tokens: {tokens.shape}, std={tokens.std():.4f}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Done.")

if __name__ == "__main__":
    main()
