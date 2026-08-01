#!/usr/bin/env python3
"""Single DiT forward at fixed t. Saves velocity output for HIP comparison."""
import argparse, os, sys
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cpu', 'trellis2', 'trellis2_repo'))

ap = argparse.ArgumentParser()
ap.add_argument('--stage1', required=True)
ap.add_argument('--features', required=True)
ap.add_argument('--noise', required=True)
ap.add_argument('--t', type=float, required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

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
model = model.to(device).eval()

feat = torch.from_numpy(np.load(args.features)).to(device).unsqueeze(0).float()
x = torch.from_numpy(np.load(args.noise)).to(device).float()
if x.dim() == 2:
    x = x.t().reshape(1, 8, 16, 16, 16)
t_tensor = torch.tensor([1000.0 * args.t], device=device)
with torch.no_grad():
    v = model(x, t_tensor, feat)
np.save(args.out, v.cpu().float().numpy())
print(f't={args.t} vel: std={v.std().item():.4f} range=[{v.min().item():.3f},{v.max().item():.3f}]')
