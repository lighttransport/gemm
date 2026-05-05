"""Dump a scripted UniPC + 3-way CFG denoising loop for the
Hunyuan3D-2.1 paint pipeline. The model output is *seeded random*
(not a real UNet call) so we can validate the C-side
UniPC+CFG-combine math in isolation from the UNet, which is already
validated end-to-end via dump_paint_unet_wrapper.py.

CFG (matches `pipeline.py` denoising loop):
  noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)
  view_scale[v] = cam_mapping(azim[v])     # 1..2 piecewise
  noise_pred = uncond + g*vs*(ref - uncond) + g*vs*(full - ref)
            = uncond + g*vs*(full - uncond)        (algebraically;
                                                     verified below)
We keep the original two-term form to exactly match diffusers
arithmetic order.

Latent layout: [3*B*N_pbr*N_gen, C, H, W]  (3-way CFG batch).

Outputs:
  azims.npy        [N_gen]    f32   azimuth degrees per view
  view_scales.npy  [N_gen]    f32   cam_mapping(azim) per view
  timesteps.npy    [N]        i64
  sigmas.npy       [N+1]      f32
  x0.npy           [B*N_pbr*N_gen, C, H, W]  initial scheduler latent
  model_out_<i>.npy [3*B*N_pbr*N_gen, C, H, W]  3-way fake noise prediction
  noise_pred_<i>.npy [B*N_pbr*N_gen, C, H, W]   after CFG combine
  x_after_<i>.npy   [B*N_pbr*N_gen, C, H, W]   after scheduler.step
  meta.json
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler


def cam_mapping(azim):
    if 0 <= azim < 90:
        return float(azim) / 90.0 + 1.0
    elif 90 <= azim < 330:
        return 2.0
    else:
        return -float(azim) / 90.0 + 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_cfg_loop_ref")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--batch",   type=int, default=1)
    ap.add_argument("--n-pbr",   type=int, default=2)
    ap.add_argument("--n-gen",   type=int, default=6)
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 6 candidate views from textureGenPipeline.py (pre-extra fan).
    azims_full = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0]
    azims = np.array(azims_full[:args.n_gen], dtype=np.float32)
    view_scales = np.array([cam_mapping(a) for a in azims], dtype=np.float32)
    np.save(os.path.join(args.outdir, "azims.npy"), azims)
    np.save(os.path.join(args.outdir, "view_scales.npy"), view_scales)

    cfg = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
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

    Beff = args.batch * args.n_pbr * args.n_gen
    shape_x = (Beff, args.c, args.h, args.w)
    shape_m = (3 * Beff, args.c, args.h, args.w)

    torch.manual_seed(args.seed)
    x = torch.randn(shape_x, dtype=torch.float32)
    np.save(os.path.join(args.outdir, "x0.npy"), x.numpy())

    # view_scale broadcast: pipeline computes view_scale_tensor with shape
    # [n_pbr * n_gen, 1, 1, 1]. We mirror that.
    vs_t = torch.from_numpy(view_scales).repeat(args.n_pbr).view(-1)[:, None, None, None]
    # extends to Beff rows (B times)
    vs_t = vs_t.repeat(args.batch, 1, 1, 1).to(torch.float32)

    g = torch.Generator().manual_seed(args.seed + 1)
    G = float(args.guidance)
    for i, t in enumerate(sch.timesteps):
        m = torch.randn(shape_m, generator=g, dtype=torch.float32)
        np.save(os.path.join(args.outdir, f"model_out_{i}.npy"), m.numpy())

        m_uncond, m_ref, m_full = m.chunk(3)
        # exact diffusers arithmetic order (first refines uncond->ref, then ref->full)
        np_pred = m_uncond + G * vs_t * (m_ref - m_uncond)
        np_pred = np_pred + G * vs_t * (m_full - m_ref)
        np.save(os.path.join(args.outdir, f"noise_pred_{i}.npy"),
                np_pred.numpy().astype(np.float32))

        out = sch.step(np_pred, t, x, return_dict=True)
        x = out.prev_sample
        np.save(os.path.join(args.outdir, f"x_after_{i}.npy"),
                x.cpu().numpy().astype(np.float32))
        print(f"step {i:2d}: t={int(t):4d} sigma={float(sch.sigmas[i]):.6f}  "
              f"x range=[{float(x.min()):+.3f},{float(x.max()):+.3f}]")

    meta = dict(cfg=cfg, steps=args.steps, seed=args.seed,
                guidance=args.guidance, batch=args.batch,
                n_pbr=args.n_pbr, n_gen=args.n_gen,
                c=args.c, h=args.h, w=args.w,
                azims=azims.tolist(), view_scales=view_scales.tolist(),
                shape_x=list(shape_x), shape_m=list(shape_m))
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.steps} steps -> {args.outdir}")


if __name__ == "__main__":
    main()
