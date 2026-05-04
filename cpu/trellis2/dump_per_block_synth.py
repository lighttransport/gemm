#!/usr/bin/env python3
"""Dump per-block hidden states from official TRELLIS.2 DiT using synthetic
inputs that match the HIP runner's --dump-blocks consumption.

Inputs:
  /tmp/feat_synth.npy   [1029, 1024]   features
  /tmp/noise_synth.npy  [4096, 8]      flat noise (token-major, channel last)

Outputs:
  ref_block{00..29}_hidden.npy  per-block hidden states  [4096, 1536]

The PyTorch model expects noise as [1, 8, 16, 16, 16] and reshapes to [1, 4096, 8]
internally. We bypass that and feed [1, 4096, 8] directly into input_layer to
match the byte-for-byte token ordering the HIP runner sees.
"""
import sys, os, argparse
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.modules.utils import manual_cast

ap = argparse.ArgumentParser()
ap.add_argument('--dit', default='/mnt/disk1/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors')
ap.add_argument('--features', default='/tmp/feat_synth.npy')
ap.add_argument('--noise', default='/tmp/noise_synth.npy')
ap.add_argument('--t', type=float, default=500.0, help='timestep in [0,1000]')
ap.add_argument('--out-dir', default='.')
args = ap.parse_args()

print(f"loading DiT from {args.dit}", flush=True)
sd = load_file(args.dit)
sd = {k: v.float() for k, v in sd.items()}

model = SparseStructureFlowModel(
    resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
    out_channels=8, num_blocks=30, num_head_channels=128,
    mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
    share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
)
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
print("model on GPU", flush=True)

noise_flat = torch.from_numpy(np.load(args.noise)).float().cuda()       # [4096, 8]
features   = torch.from_numpy(np.load(args.features)).float().cuda()    # [1029, 1024]
assert noise_flat.shape == (4096, 8),    f"noise shape {noise_flat.shape}"
assert features.shape   == (1029, 1024), f"features shape {features.shape}"
print(f"noise_flat {tuple(noise_flat.shape)}  features {tuple(features.shape)}  t={args.t}", flush=True)

with torch.no_grad():
    h = noise_flat.unsqueeze(0)                # [1, 4096, 8]
    h = model.input_layer(h)                   # [1, 4096, 1536]
    h = manual_cast(h, torch.float32)

    t = torch.tensor([args.t], device='cuda')
    t_emb = model.t_embedder(t)
    t_emb = model.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, torch.float32)

    cond = features.unsqueeze(0)               # [1, 1029, 1024]
    cond = manual_cast(cond, torch.float32)

    print(f"input_emb std={h.std().item():.4f}", flush=True)
    for bi, block in enumerate(model.blocks):
        h = block(h, t_emb, cond, model.rope_phases)
        print(f"  block {bi:2d}: std={h.std().item():10.4f}", flush=True)
        np.save(os.path.join(args.out_dir, f'ref_block{bi:02d}_hidden.npy'),
                h[0].float().cpu().numpy())

print(f"\ndone — wrote ref_block{{00..29}}_hidden.npy to {args.out_dir}", flush=True)
