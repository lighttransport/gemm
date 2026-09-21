#!/usr/bin/env python3
"""Qwen-Image 2.1 CUDA runner.

This runner intentionally uses the official Diffusers implementation as the
CUDA execution backend while the native C kernels are brought up.  It keeps
the model on the local Hugging Face snapshot, uses sequential CPU offload for
the 16 GB RTX 5060 Ti, and exposes the same fixture/dump interface used by the
older qimg verification flow.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is missing; run ./setup_reference.sh") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this runner requires an NVIDIA GPU")
    return torch


def _load_pipe(model_dir: Path, dtype_name: str = "bf16"):
    torch = _torch()
    from diffusers import QwenImage21Pipeline

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    print(f"loading Qwen-Image 2.1 from {model_dir}", file=sys.stderr)
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model_dir), dtype=dtype, local_files_only=True
    )
    # The text encoder and transformer each exceed the available free VRAM.
    # Sequential offload keeps only the active leaf module resident.
    pipe.enable_sequential_cpu_offload(device="cuda")
    return pipe


def _save_array(path: Path, value) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    np.save(path, np.ascontiguousarray(np.asarray(value)))


def _dump_prompt(pipe, prompt: str, out_dir: Path, image=None, negative_prompt: str | None = None,
                 dump_text_stages: bool = False) -> None:
    torch = _torch()
    from text_capture import capture_text_encoder

    # Unlike pipeline.__call__, encode_prompt expects an image sequence.
    if image is not None and not isinstance(image, (list, tuple)):
        image = [image]

    with torch.inference_mode(), (capture_text_encoder(pipe, out_dir / "text_positive")
                                  if dump_text_stages else nullcontext()):
        embeds, mask, image_mask = pipe.encode_prompt(
            prompt=prompt,
            image=image,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
        )
    _save_array(out_dir / "prompt_embeds.npy", embeds)
    _save_array(out_dir / "prompt_mask.npy", mask if mask is not None else np.ones(embeds.shape[:2], dtype=np.bool_))
    _save_array(out_dir / "image_pad_mask.npy", image_mask)
    if negative_prompt is not None:
        with torch.inference_mode(), (capture_text_encoder(pipe, out_dir / "text_negative")
                                      if dump_text_stages else nullcontext()):
            negative_embeds, negative_mask, negative_image_mask = pipe.encode_prompt(
                prompt=negative_prompt,
                image=image,
                device=torch.device("cuda"),
                num_images_per_prompt=1,
            )
        _save_array(out_dir / "negative_prompt_embeds.npy", negative_embeds)
        _save_array(
            out_dir / "negative_prompt_mask.npy",
            negative_mask if negative_mask is not None else np.ones(negative_embeds.shape[:2], dtype=np.bool_),
        )
        _save_array(out_dir / "negative_image_pad_mask.npy", negative_image_mask)
    print(f"saved prompt fixtures in {out_dir}")


def generate(args) -> int:
    torch = _torch()
    model = Path(args.model).resolve()
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")
    pipe = _load_pipe(model, args.dtype)
    if max(args.height, args.width) > 1024:
        pipe.vae.enable_tiling()
    image = Image.open(args.image) if args.image else None
    out_dir = Path(args.dump_dir) if args.dump_dir else None

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    if args.init_latents:
        latents = torch.from_numpy(np.load(args.init_latents)).to(
            device="cuda", dtype=torch.bfloat16
        )
        generator = None
    else:
        latents = None
    if args.dump_initial_latents and latents is None:
        latents, _ = pipe.prepare_latents(
            None,
            1,
            pipe.transformer.config.in_channels,
            args.height,
            args.width,
            torch.bfloat16 if args.dtype == "bf16" else torch.float16,
            torch.device("cuda"),
            generator,
            None,
        )
        if out_dir:
            _save_array(out_dir / "initial_latents.npy", latents[0])
        generator = None

    free0, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  free={free0/2**30:.2f}GiB total={total/2**30:.2f}GiB")
    start = time.perf_counter()
    callback = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

        def callback(_pipe, step, _timestep, kwargs):
            value = kwargs.get("latents")
            if value is not None:
                _save_array(out_dir / f"step_{step:03d}.npy", value)
            prompt_embeds = kwargs.get("prompt_embeds")
            if prompt_embeds is not None and step == 0:
                _save_array(out_dir / "prompt_embeds.npy", prompt_embeds)
            return kwargs

    result = pipe(
        prompt=args.prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
        latents=latents,
        output_type="pil",
        use_kv_cache=not args.no_kv_cache,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result.images[0].save(args.out)
    free1, _ = torch.cuda.mem_get_info()
    print(f"saved {args.out} ({args.width}x{args.height})")
    print(f"elapsed={elapsed:.2f}s cuda_used={(free0-free1)/2**30:.2f}GiB")
    if out_dir:
        result.images[0].save(out_dir / "runner.png")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true", help="generate an image")
    ap.add_argument("--test-load", action="store_true", help="load model and exit")
    ap.add_argument("--test-text", action="store_true", help="dump prompt embeddings")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--image", help="optional condition image for editing")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--out", default="qwen_image21.png")
    ap.add_argument("--dump-dir")
    ap.add_argument("--dump-text-stages", action="store_true",
                    help="With --test-text, capture token IDs and pre-final-norm encoder states")
    ap.add_argument("--init-latents")
    ap.add_argument(
        "--dump-initial-latents",
        action="store_true",
        help="save the exact packed PyTorch noise tensor used by the denoising loop",
    )
    ap.add_argument("--no-kv-cache", action="store_true")
    args = ap.parse_args()

    if args.height % 32 or args.width % 32:
        print("warning: dimensions are rounded down to multiples of 32", file=sys.stderr)
        args.height = args.height // 32 * 32
        args.width = args.width // 32 * 32
    if args.test_load or args.test_text:
        if args.test_text:
            if not args.dump_dir:
                raise SystemExit("--test-text requires --dump-dir")
            pipe = _load_pipe(Path(args.model).resolve(), args.dtype)
            image = Image.open(args.image) if args.image else None
            _dump_prompt(pipe, args.prompt, Path(args.dump_dir), image, args.negative_prompt,
                         args.dump_text_stages)
            del pipe
            return 0
        pipe = _load_pipe(Path(args.model).resolve(), args.dtype)
        del pipe
        return 0
    if not args.generate:
        ap.error("choose --generate, --test-load, or --test-text")
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
