#!/usr/bin/env python3
"""Create deterministic PyTorch/Diffusers fixtures for qimg21 comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()

    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise SystemExit("reference requires CUDA")
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(
        str(Path(args.model).resolve()), dtype=torch.bfloat16, local_files_only=True
    )
    pipe.enable_sequential_cpu_offload(device="cuda")
    if max(args.height, args.width) > 1024:
        pipe.vae.enable_tiling()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
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
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=gen,
        use_kv_cache=True,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    torch.cuda.synchronize()
    result.images[0].save(out / "reference.png")
    np.save(out / "reference_rgba.npy", np.asarray(result.images[0].convert("RGBA")))
    (out / "run.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()),
        "prompt": args.prompt,
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
