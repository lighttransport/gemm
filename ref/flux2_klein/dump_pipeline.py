#!/usr/bin/env python3
"""
Dump every reference signal needed to reproduce / compare against the HIP port.

Produces, for the default prompt at size=256 steps=4 seed=42:
  apple_text.bin     : prompt_embeds  [512, 7680] F32
  init_latent.bin    : initial noise latent after BN packing  [n_patches, 128] F32
                       (i.e. the thing our DiT consumes as img_tokens)
  final_latent.bin   : latent after last Euler step, BEFORE BN de-norm + unpatchify
                       [n_patches, 128] F32  (what the VAE decode consumes after BN/unpack)
  final_latent_chw.bin: latent after unpatchify [32, 32, 32] F32 — VAE decode input
  apple_ref.png      : reference apple image (same pipeline)
  sigmas.txt         : the scheduler's sigma and dt per step

Run:
  HF_HOME=/mnt/disk1/hf-cache python dump_pipeline.py
"""
import argparse
import json
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help="Append to output filenames")
    args = ap.parse_args()
    sfx = args.suffix

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # sequential cpu offload streams individual submodules — needed for 512+ on 16GB VRAM
    if args.size >= 512:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    # ---- 1. Text embeds
    embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        prompt=args.prompt,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    arr = embeds[0].detach().cpu().numpy().astype(np.float32)
    arr.tofile(f"apple_text{sfx}.bin")
    print(f"apple_text{sfx}.bin: {arr.shape} mean {arr.mean():.5f} std {arr.std():.5f}")

    # ---- 2. Hook into denoise loop: capture initial packed latent + final packed latent
    # Strategy: monkey-patch the transformer forward to log inputs, and patch `callback_on_step_end`.
    captured = {}

    orig_tx_forward = pipe.transformer.forward
    def tx_forward(*a, **kw):
        # First call's `hidden_states` arg is the noisy latent going into DiT.
        hs = kw.get("hidden_states")
        if hs is None and len(a) >= 1:
            hs = a[0]
        if hs is not None and "first_input" not in captured:
            captured["first_input"] = hs.detach().to(torch.float32).cpu().numpy()
            print(f"first DiT hidden_states in: {hs.shape} dtype {hs.dtype}")
        return orig_tx_forward(*a, **kw)
    pipe.transformer.forward = tx_forward

    def cb(pipe_, step, ts, kws):
        lat = kws.get("latents")
        if lat is not None:
            captured.setdefault("per_step", []).append(
                lat.detach().to(torch.float32).cpu().numpy()
            )
        return kws

    import time
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        generator=gen,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print(f"[timing size={args.size}x{args.size} steps={args.steps}] total={t_end - t_start:.3f}s (text+DiT+VAE, incl. CPU offload transfers)")
    img = out.images[0]
    img.save(f"apple_ref{sfx}.png")
    print(f"saved apple_ref{sfx}.png ({img.size[0]}x{img.size[1]})")

    # first_input is the FIRST latents the DiT sees — that IS the initial noise (packed).
    # shape: [1, n_patches, 128]
    first = captured["first_input"]
    print(f"first_input: {first.shape} mean {first.mean():.4f} std {first.std():.4f}")
    first[0].astype(np.float32).tofile(f"init_latent{sfx}.bin")  # [n_patches, 128]
    print(f"saved init_latent{sfx}.bin")

    # per_step[-1] is the latent AFTER the last Euler step — still packed [1, n_patches, 128]
    # (that's the "final_latent" before BN de-norm + unpack + VAE decode).
    last = captured["per_step"][-1]
    print(f"last_step latent: {last.shape} mean {last.mean():.4f} std {last.std():.4f}")
    last[0].astype(np.float32).tofile(f"final_latent_packed{sfx}.bin")
    print(f"saved final_latent_packed{sfx}.bin")

    # Also dump per-step progression
    for i, lat in enumerate(captured["per_step"]):
        lat[0].astype(np.float32).tofile(f"latent_step{i}.bin")
    print(f"saved latent_step0..{len(captured['per_step'])-1}.bin")

    # ---- 3. Sigmas
    scheduler = pipe.scheduler
    sigmas = scheduler.sigmas.detach().cpu().numpy().tolist() if hasattr(scheduler, "sigmas") else []
    timesteps = scheduler.timesteps.detach().cpu().numpy().tolist() if hasattr(scheduler, "timesteps") else []
    with open("sigmas.txt", "w") as fp:
        fp.write(f"sigmas = {sigmas}\n")
        fp.write(f"timesteps = {timesteps}\n")
    if sigmas:
        np.array(sigmas, dtype=np.float32).tofile("sigmas.bin")
        print(f"saved sigmas.bin ({len(sigmas)} floats)")
    print(f"sigmas: {sigmas}")
    print(f"timesteps: {timesteps}")

    # ---- 4. Also dump BN stats + config for sanity
    vae = pipe.vae
    bn_mean = vae.bn.running_mean.detach().to(torch.float32).cpu().numpy()
    bn_var  = vae.bn.running_var.detach().to(torch.float32).cpu().numpy()
    bn_eps  = vae.config.batch_norm_eps
    meta = {
        "bn_mean": bn_mean.tolist(),
        "bn_var":  bn_var.tolist(),
        "bn_eps":  bn_eps,
        "n_patches": int(first.shape[1]),
    }
    with open("bn_stats.json", "w") as fp:
        json.dump(meta, fp)
    bn_mean.astype(np.float32).tofile("bn_mean.bin")
    bn_var.astype(np.float32).tofile("bn_var.bin")
    print(f"BN: n_ch={len(bn_mean)}, eps={bn_eps}, mean[0..3]={bn_mean[:4]}, var[0..3]={bn_var[:4]}")
    print("saved bn_mean.bin bn_var.bin")

if __name__ == "__main__":
    main()
