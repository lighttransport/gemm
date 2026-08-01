#!/usr/bin/env python3
"""No-CFG short reference: reuses existing ref_features.npy + ref_noise_init.npy.
Runs N Euler steps with no classifier-free guidance and saves per-step latents."""
import argparse, os, sys, time
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cpu', 'trellis2', 'trellis2_repo'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--noise',    required=True, help='ref_noise_init.npy [1,8,16,16,16]')
    ap.add_argument('--output-dir', default='/tmp/chair_ref_nocfg')
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--total-steps', type=int, default=12,
                    help='t-schedule resolution (take first --steps entries)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    sd = {k: v.float() for k, v in load_file(args.stage1).items()}
    model = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    del sd

    feat = np.load(args.features)  # [1029, 1024]
    noise = np.load(args.noise)    # [1,8,16,16,16]
    features = torch.from_numpy(feat).to(device).unsqueeze(0).float()
    x = torch.from_numpy(noise).to(device).float()

    t_seq = np.linspace(1, 0, args.total_steps + 1)
    t_seq = 5.0 * t_seq / (1 + 4.0 * t_seq)

    for step in range(args.steps):
        t_cur, t_prev = float(t_seq[step]), float(t_seq[step + 1])
        t_tensor = torch.tensor([1000.0 * t_cur], device=device)
        st = time.time()
        with torch.no_grad():
            pred_v = model(x, t_tensor, features)
        x = x - (t_cur - t_prev) * pred_v
        out = x.cpu().float().numpy()
        np.save(os.path.join(args.output_dir, f'ref_nocfg_step{step}.npy'), out)
        print(f'  step {step+1}/{args.steps} t={t_cur:.4f}->{t_prev:.4f} {(time.time()-st)*1000:.0f}ms '
              f'std={out.std():.4f}')

if __name__ == '__main__':
    main()
