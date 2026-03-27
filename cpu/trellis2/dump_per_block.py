#!/usr/bin/env python3
"""Dump per-block hidden states from official TRELLIS.2 DiT for CUDA comparison."""
import sys, os
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.modules.utils import manual_cast

path = '/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors'
sd = load_file(path)
sd = {k: v.float() for k, v in sd.items()}

model = SparseStructureFlowModel(
    resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
    out_channels=8, num_blocks=30, num_head_channels=128,
    mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
    share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
)
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()

noise = torch.from_numpy(np.load('pytorch_noise_seed42.npy')).float().cuda()
features = torch.from_numpy(np.load('official_teapot_features.npy')).float().unsqueeze(0).cuda()

# Hook into the forward to capture per-block hidden states
with torch.no_grad():
    # Replicate the forward method manually to capture intermediates
    x = noise
    assert [*x.shape] == [1, 8, 16, 16, 16]
    h = x.view(1, 8, -1).permute(0, 2, 1).contiguous()  # [1, 4096, 8]
    h = model.input_layer(h)  # [1, 4096, 1536]

    t = torch.tensor([1000.0], device='cuda')
    t_emb = model.t_embedder(t)
    t_emb = model.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, torch.float32)
    h = manual_cast(h, torch.float32)
    cond = manual_cast(features, torch.float32)

    print(f"input_emb: h[0,0,:4]={h[0,0,:4].cpu().numpy()}, std={h.std():.4f}")
    print(f"t_emb[:4]={t_emb[0,:4].cpu().numpy()}")

    for bi, block in enumerate(model.blocks):
        h = block(h, t_emb, cond, model.rope_phases)
        s = h.std().item()
        v = h[0, 0, :4].cpu().numpy()
        print(f"block {bi:2d}: std={s:10.4f}  h[0,0,:4]={v}")
        if bi < 5 or bi == 29:
            np.save(f'ref_block{bi}_hidden.npy', h[0].cpu().numpy())

    h = manual_cast(h, torch.float32)
    h = torch.nn.functional.layer_norm(h, h.shape[-1:])
    out = model.out_layer(h)
    out = out.permute(0, 2, 1).view(1, 8, 16, 16, 16).contiguous()
    print(f"\nfinal output: std={out.std():.4f}")
    print(f"out[:8]={out.flatten()[:8].cpu().numpy()}")
