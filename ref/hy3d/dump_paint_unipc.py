"""Dump per-step state of UniPCMultistepScheduler with the
Hunyuan3D-2.1 paint pipeline's exact config so the C port in
cuda/hy3d_paint/cuda_paint_unipc.h can be validated bit-for-bit on the
hot path actually used at inference.

Config (from .../hunyuan3d-paintpbr-v2-1/scheduler/scheduler_config.json
+ pipeline override `timestep_spacing="trailing"`; UniPC defaults):
  beta_schedule        = scaled_linear (0.00085 -> 0.012, 1000 steps)
  prediction_type      = v_prediction
  timestep_spacing     = trailing
  rescale_betas_zero_snr = True
  num_inference_steps  = 15
  solver_order         = 2
  predict_x0           = True
  solver_type          = bh2
  lower_order_final    = True
  final_sigmas_type    = zero

We use small randn tensors as model outputs (the actual UNet output
shape doesn't matter for scheduler validation — only the math needs to
match). Latent shape mirrors a tiny paint UNet output: [4, 4, 8, 8].

Outputs (in --outdir):
  timesteps.npy   [15] i64
  sigmas.npy      [16] f32   (includes trailing 0 for final)
  x0.npy          [4,4,8,8] f32  initial sample
  model_out_<i>.npy   per-step model output
  x_after_<i>.npy     sample after step i
  meta.json       config snapshot

Usage:
  uv run --with torch --with diffusers --with numpy \\
      python ref/hy3d/dump_paint_unipc.py --outdir /tmp/hy3d_paint_unipc_ref
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unipc_ref")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
        # UniPC defaults — explicit to lock the dump
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
        final_sigmas_type="zero",
        steps_offset=1,
    )
    sch = UniPCMultistepScheduler(**cfg)
    sch.set_timesteps(args.steps)

    np.save(os.path.join(args.outdir, "timesteps.npy"),
            sch.timesteps.cpu().numpy().astype(np.int64))
    np.save(os.path.join(args.outdir, "sigmas.npy"),
            sch.sigmas.cpu().numpy().astype(np.float32))

    torch.manual_seed(args.seed)
    shape = (4, 4, 8, 8)
    x = torch.randn(shape, dtype=torch.float32)
    np.save(os.path.join(args.outdir, "x0.npy"), x.numpy())

    g = torch.Generator().manual_seed(args.seed + 1)
    for i, t in enumerate(sch.timesteps):
        m = torch.randn(shape, generator=g, dtype=torch.float32)
        np.save(os.path.join(args.outdir, f"model_out_{i}.npy"), m.numpy())
        out = sch.step(m, t, x, return_dict=True)
        x = out.prev_sample
        np.save(os.path.join(args.outdir, f"x_after_{i}.npy"),
                x.cpu().numpy().astype(np.float32))
        print(f"step {i:2d}: t={int(t):4d} sigma={float(sch.sigmas[i]):.6f} "
              f"-> sigma'={float(sch.sigmas[i+1]):.6f}  "
              f"x range=[{float(x.min()):+.3f},{float(x.max()):+.3f}]")

    meta = dict(cfg=cfg, steps=args.steps, seed=args.seed,
                shape=list(shape))
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.steps} steps -> {args.outdir}")


if __name__ == "__main__":
    main()
