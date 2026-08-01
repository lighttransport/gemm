#!/usr/bin/env python3
"""CFG 12-step reference. Matches HIP --full (without --no-cfg):
  - rescaled t-schedule (5t/(1+4t))
  - CFG scale=7.5, rescale=0.7, interval [0.6, 1.0]
  - zero features as uncond
  - Euler integration
Saves per-step latents to ref_cfg_stepNN.npy.
"""
import argparse, os, sys, time
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cpu', 'trellis2', 'trellis2_repo'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--noise',    required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--cfg-scale',   type=float, default=7.5)
    ap.add_argument('--cfg-rescale', type=float, default=0.7)
    ap.add_argument('--cfg-lo', type=float, default=0.6)
    ap.add_argument('--cfg-hi', type=float, default=1.0)
    ap.add_argument('--sigma-min', type=float, default=1e-5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    sd = {k: v.float() for k, v in load_file(args.stage1).items()}
    model = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval(); del sd

    feat  = torch.from_numpy(np.load(args.features)).to(device).unsqueeze(0).float()
    zero  = torch.zeros_like(feat)
    x     = torch.from_numpy(np.load(args.noise)).to(device).float()

    t_seq = np.linspace(1, 0, args.steps + 1)
    t_seq = 5.0 * t_seq / (1 + 4.0 * t_seq)

    for step in range(args.steps):
        t_cur, t_prev = float(t_seq[step]), float(t_seq[step+1])
        dt = t_cur - t_prev
        t_t = torch.tensor([1000.0 * t_cur], device=device)
        use_cfg = args.cfg_lo <= t_cur <= args.cfg_hi
        st = time.time()
        with torch.no_grad():
            v_cond = model(x, t_t, feat)
            if use_cfg:
                v_uncond = model(x, t_t, zero)
                v_cfg = args.cfg_scale * v_cond + (1.0 - args.cfg_scale) * v_uncond
                # TRELLIS.2 official: rescale in x0 space, not velocity space
                sm = args.sigma_min
                coeff = sm + (1.0 - sm) * t_cur
                x0_pos = (1.0 - sm) * x - coeff * v_cond
                x0_cfg = (1.0 - sm) * x - coeff * v_cfg
                std_pos = x0_pos.std()
                std_cfg = x0_cfg.std()
                x0_rescaled = x0_cfg * (std_pos / std_cfg)
                x0 = args.cfg_rescale * x0_rescaled + (1.0 - args.cfg_rescale) * x0_cfg
                v = ((1.0 - sm) * x - x0) / coeff
            else:
                v = v_cond
        x = x - dt * v
        out = x.cpu().float().numpy()
        np.save(os.path.join(args.output_dir, f'ref_cfg_step{step:02d}.npy'), out)
        print(f'  step {step+1:2d}/{args.steps} t={t_cur:.4f}->{t_prev:.4f} '
              f'{(time.time()-st)*1000:.0f}ms std={out.std():.4f} {"[CFG]" if use_cfg else ""}')

    np.save(os.path.join(args.output_dir, 'ref_cfg_latent.npy'), x.cpu().float().numpy())

if __name__ == '__main__':
    main()
