#!/usr/bin/env python3
"""Generate a Qwen-Image reference image via diffusers + pytorch-rocm.

Mirrors ref/flux2_klein/gen_reference.py for the Qwen-Image pipeline.

Run with:
    HF_HOME=/mnt/disk1/hf-cache python gen_diffusers_reference.py \
        --size 256 --steps 20 --seed 42 -o apple_ref_256.png
"""
import argparse
import time

import torch


DEFAULT_FP8_DIT = "/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--negative", default=" ")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("-o", "--out", default="apple_ref.png")
    ap.add_argument("--model", default="Qwen/Qwen-Image",
                    help="HF repo id (used for text encoder + tokenizer + VAE + scheduler)")
    ap.add_argument("--fp8-dit", default=DEFAULT_FP8_DIT,
                    help="Path to the ComfyUI FP8 single-file DiT (set to '' to use the BF16 multi-shard from --model)")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
    import os
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.fp8_dit and os.path.exists(args.fp8_dit):
        # Apples-to-apples with the HIP runner: load the same ComfyUI FP8
        # single-file via from_single_file, then attach to the pipeline along
        # with text_encoder/tokenizer/vae/scheduler from the HF repo.
        print(f"loading FP8 DiT from {args.fp8_dit}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            args.fp8_dit, config=args.model, subfolder="transformer",
            torch_dtype=dtype,
        )
        print(f"loading rest of pipeline from {args.model}")
        pipe = QwenImagePipeline.from_pretrained(
            args.model, transformer=transformer, torch_dtype=dtype,
        )
    else:
        print(f"loading full BF16 pipeline from {args.model}")
        pipe = QwenImagePipeline.from_pretrained(args.model, torch_dtype=dtype)

    # 16 GB VRAM is too small for Qwen-Image even with model_cpu_offload —
    # the BF16 transformer alone is ~20 GB. Use sequential_cpu_offload which
    # streams individual submodules.
    pipe.enable_sequential_cpu_offload()

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
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[timing size={args.size}x{args.size} steps={args.steps}] total={dt:.2f}s")

    img = out.images[0]
    img.save(args.out)
    print(f"saved {args.out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
