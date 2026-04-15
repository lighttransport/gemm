#!/usr/bin/env python3
"""
Generate a reference Flux.2 Klein 4B image using diffusers + pytorch-rocm.

Usage:
    python gen_reference.py [--size 256] [--steps 4] [--seed 42] \
                            [--prompt "a red apple on a white table"] \
                            [-o apple_ref.png]

Tries Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B").
This will download the text encoder + tokenizer on first run (~10 GB);
DiT and VAE will also be fetched unless HF_HUB_CACHE already has them.
"""

import argparse
import os
import sys
import time

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default="apple_ref.png")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    from diffusers import Flux2KleinPipeline
    print("using diffusers.Flux2KleinPipeline")
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)

    # 16GB VRAM is too small for the whole pipeline resident at once.
    # model_cpu_offload keeps weights on CPU and moves the active submodule to GPU.
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        generator=gen,
    )
    dt = time.time() - t0
    print(f"generation: {dt:.1f}s")

    img = out.images[0]
    img.save(args.out)
    print(f"saved {args.out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
