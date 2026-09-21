#!/usr/bin/env python3
"""Create deterministic PyTorch/Diffusers fixtures for qimg21 comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--image", help="optional condition image for editing")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument(
        "--dump-initial-latents",
        action="store_true",
        help="save the exact packed PyTorch noise tensor used by the denoising loop",
    )
    ap.add_argument(
        "--dump-pred-dir",
        help="save each transformer denoiser prediction as pred_NNN.npy",
    )
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()

    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise SystemExit("reference requires CUDA")
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image) if args.image else None
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pipe = QwenImage21Pipeline.from_pretrained(
        str(Path(args.model).resolve()), dtype=dtype, local_files_only=True
    )
    pipe.enable_sequential_cpu_offload(device="cuda")
    if max(args.height, args.width) > 1024:
        pipe.vae.enable_tiling()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    pred_dir = Path(args.dump_pred_dir) if args.dump_pred_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_index = [0]
    use_cfg = args.negative_prompt is not None and args.true_cfg_scale > 1.0
    conditional_prediction = [None]
    if pred_dir:
        def dump_timestep(_module, _inputs, kwargs):
            step = pred_index[0] // 2 if use_cfg else pred_index[0]
            is_negative = use_cfg and pred_index[0] % 2 == 1
            embeds = kwargs.get("encoder_hidden_states")
            if step == 0 and embeds is not None:
                name = "negative_prompt_embeds.npy" if is_negative else "prompt_embeds.npy"
                np.save(out / name, embeds.detach().float().cpu().numpy())
                branch = "negative" if is_negative else "positive"
                for key in ("img_mask", "encoder_hidden_states_mask"):
                    value = kwargs.get(key)
                    # An absent key mask means every text key is valid.
                    # Persist that semantic value so native fixtures need
                    # not guess whether a missing file means no padding.
                    if key == "encoder_hidden_states_mask" and value is None:
                        value = torch.ones(embeds.shape[:2], dtype=torch.bool)
                    if value is not None:
                        np.save(pred_dir / f"{branch}_{key}.npy", value.detach().cpu().numpy())
                shapes = kwargs.get("img_shapes")
                if shapes is not None:
                    (pred_dir / f"{branch}_layout.json").write_text(json.dumps({
                        "img_shapes": shapes, "text_slots": int(embeds.shape[1]),
                        "target_tokens": (args.height // 16) * (args.width // 16),
                    }, indent=2) + "\n")
            if is_negative:
                return
            value = kwargs.get("timestep")
            if value is not None:
                np.save(
                    pred_dir / f"timestep_{step:03d}.npy",
                    np.ascontiguousarray(value.detach().float().cpu().numpy()),
                )

        def dump_input(_module, inputs):
            if use_cfg and pred_index[0] % 2:
                return
            if not inputs:
                return
            value = inputs[0]
            np.save(
                pred_dir / f"input_{pred_index[0] // 2 if use_cfg else pred_index[0]:03d}.npy",
                np.ascontiguousarray(value.detach().float().cpu().numpy()),
            )

        def dump_prediction(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            target_tokens = (args.height // 16) * (args.width // 16)
            value = value[:, -target_tokens:]
            step = pred_index[0] // 2 if use_cfg else pred_index[0]
            if use_cfg:
                branch = "negative" if pred_index[0] % 2 else "positive"
                np.save(pred_dir / f"{branch}_{step:03d}.npy", value.detach().float().cpu().numpy())
                if pred_index[0] % 2 == 0:
                    conditional_prediction[0] = value.detach().clone()
                    pred_index[0] += 1
                    return
                # Keep the arithmetic on the same device and dtype as the
                # pipeline, including the BF16 intermediate boundaries.
                value = value + args.true_cfg_scale * (conditional_prediction[0] - value)
                conditional_prediction[0] = None
            np.save(
                pred_dir / f"pred_{step:03d}.npy",
                np.ascontiguousarray(value.detach().float().cpu().numpy()),
            )
            pred_index[0] += 1
        pipe.transformer.register_forward_pre_hook(dump_timestep, with_kwargs=True)
        pipe.transformer.img_in.register_forward_pre_hook(dump_input)
        pipe.transformer.proj_out.register_forward_hook(dump_prediction)
    initial_latents = None
    if args.dump_initial_latents:
        initial_latents, _ = pipe.prepare_latents(
            None,
            1,
            pipe.transformer.config.in_channels,
            args.height,
            args.width,
            dtype,
            torch.device("cuda"),
            gen,
            None,
        )
        np.save(
            out / "initial_latents.npy",
            np.ascontiguousarray(initial_latents[0].detach().float().cpu().numpy()),
        )

    def callback(_pipe, step, _timestep, kwargs):
        value = kwargs.get("latents")
        if value is not None:
            np.save(out / f"step_{step:03d}.npy", value.detach().float().cpu().numpy())
        prompt_embeds = kwargs.get("prompt_embeds")
        if prompt_embeds is not None and step == 0:
            np.save(out / "prompt_embeds.npy", prompt_embeds.detach().float().cpu().numpy())
        return kwargs

    t0 = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=None if initial_latents is not None else gen,
        latents=initial_latents,
        # Native denoiser parity replays every step through the same full
        # prefill path.  The cache path is numerically equivalent but uses a
        # different attention execution route, which would measure cache
        # drift instead of kernel parity in the per-step fixtures.
        use_kv_cache=pred_dir is None,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    torch.cuda.synchronize()
    result.images[0].save(out / "reference.png")
    np.save(out / "reference_rgba.npy", np.asarray(result.images[0].convert("RGBA")))
    (out / "run.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "use_true_cfg": use_cfg,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - t0,
        "torch": torch.__version__,
    }, indent=2) + "\n")
    print(f"saved {out / 'reference.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
