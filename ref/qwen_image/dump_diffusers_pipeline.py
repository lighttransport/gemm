#!/usr/bin/env python3
"""Dump everything needed to reproduce the Qwen-Image diffusers run on the
HIP runner: text embeddings, init packed latent, sigmas, final packed
latent, and the reference apple PNG.

Mirrors ref/flux2_klein/dump_pipeline.py for the QwenImagePipeline.

Run:
    HF_HOME=/mnt/disk1/hf-cache python dump_diffusers_pipeline.py \
        --size 256 --steps 20 --suffix _256
"""
import argparse
import time

import numpy as np
import torch


DEFAULT_FP8_DIT = "/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--negative", default=" ")
    ap.add_argument("--model", default="Qwen/Qwen-Image",
                    help="HF repo id for text encoder + tokenizer + VAE + scheduler")
    ap.add_argument("--fp8-dit", default=DEFAULT_FP8_DIT,
                    help="Path to ComfyUI FP8 single-file DiT (set to '' to use BF16 multi-shard)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    sfx = args.suffix

    import os
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

    if args.fp8_dit and os.path.exists(args.fp8_dit):
        print(f"loading FP8 DiT from {args.fp8_dit}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            args.fp8_dit, config=args.model, subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = QwenImagePipeline.from_pretrained(
            args.model, transformer=transformer, torch_dtype=torch.bfloat16,
        )
    else:
        pipe = QwenImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    # ---- 1. Text embeddings ----
    # QwenImagePipeline.encode_prompt(prompt, device, num_images_per_prompt,
    #   prompt_embeds, prompt_embeds_mask, max_sequence_length)
    # → (prompt_embeds, prompt_embeds_mask)
    print("encoding prompt")
    out = pipe.encode_prompt(
        prompt=[args.prompt],
        device=torch.device("cuda"),
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    prompt_embeds = out[0] if isinstance(out, tuple) else out
    arr = prompt_embeds[0].detach().to(torch.float32).cpu().numpy()
    arr.tofile(f"apple_text{sfx}.bin")
    print(f"apple_text{sfx}.bin: {arr.shape} mean {arr.mean():.5f} std {arr.std():.5f}")

    # ---- 2. Hook transformer for first input (init noise packed) and per-step latents ----
    captured = {}

    orig_tx_forward = pipe.transformer.forward
    def tx_forward(*a, **kw):
        hs = kw.get("hidden_states")
        if hs is None and len(a) >= 1:
            hs = a[0]
        if hs is not None and "first_input" not in captured:
            captured["first_input"] = hs.detach().to(torch.float32).cpu().numpy()
            print(f"first DiT hidden_states: {hs.shape} dtype {hs.dtype}")
        return orig_tx_forward(*a, **kw)
    pipe.transformer.forward = tx_forward

    def cb(pipe_, step, ts, kws):
        lat = kws.get("latents")
        if lat is not None:
            captured.setdefault("per_step", []).append(
                lat.detach().to(torch.float32).cpu().numpy()
            )
        return kws

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        generator=gen,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    torch.cuda.synchronize()
    print(f"[timing size={args.size} steps={args.steps}] total={time.perf_counter()-t0:.2f}s")

    img = out.images[0]
    img.save(f"apple_ref{sfx}.png")
    print(f"saved apple_ref{sfx}.png ({img.size[0]}x{img.size[1]})")

    if "first_input" in captured:
        first = captured["first_input"]
        print(f"first_input: {first.shape} mean {first.mean():.4f} std {first.std():.4f}")
        first[0].astype(np.float32).tofile(f"init_latent{sfx}.bin")
        print(f"saved init_latent{sfx}.bin")

    if captured.get("per_step"):
        last = captured["per_step"][-1]
        print(f"final latent: {last.shape} mean {last.mean():.4f} std {last.std():.4f}")
        last[0].astype(np.float32).tofile(f"final_latent_packed{sfx}.bin")
        print(f"saved final_latent_packed{sfx}.bin")
        for i, lat in enumerate(captured["per_step"]):
            lat[0].astype(np.float32).tofile(f"latent_step{i}{sfx}.bin")

    # ---- 3. Sigmas ----
    sched = pipe.scheduler
    sigmas = sched.sigmas.detach().cpu().numpy().tolist() if hasattr(sched, "sigmas") else []
    timesteps = sched.timesteps.detach().cpu().numpy().tolist() if hasattr(sched, "timesteps") else []
    if sigmas:
        np.array(sigmas, dtype=np.float32).tofile(f"sigmas{sfx}.bin")
        print(f"saved sigmas{sfx}.bin ({len(sigmas)} floats): {sigmas[:5]}{'...' if len(sigmas) > 5 else ''}")
    if timesteps:
        print(f"timesteps[:5]: {timesteps[:5]}")

    # ---- 4. Save VAE latent normalization constants if present ----
    vae = pipe.vae
    cfg = vae.config
    meta_lines = [f"latent_channels={getattr(cfg, 'latent_channels', None)}"]
    for k in ("latents_mean", "latents_std", "scaling_factor", "shift_factor"):
        v = getattr(cfg, k, None)
        if v is not None:
            if isinstance(v, (list, tuple)):
                arr = np.array(v, dtype=np.float32)
                arr.tofile(f"vae_{k}{sfx}.bin")
                meta_lines.append(f"{k}=[{len(v)}] -> vae_{k}{sfx}.bin")
            else:
                meta_lines.append(f"{k}={v}")
    with open(f"vae_meta{sfx}.txt", "w") as f:
        f.write("\n".join(meta_lines) + "\n")
    print(f"VAE meta -> vae_meta{sfx}.txt")
    for line in meta_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
