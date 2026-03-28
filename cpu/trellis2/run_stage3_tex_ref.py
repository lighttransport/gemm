#!/usr/bin/env python3
"""Run Stage 3 texture flow (no CFG, cond-only) on CPU."""
import os, sys, time
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np, torch.nn.functional as F, json
from safetensors.torch import load_file

# Patch sparse attention
import trellis2.modules.sparse.attention.full_attn as sfa
import trellis2.modules.sparse.attention.modules as sam
from trellis2.modules.sparse.basic import VarLenTensor

def patched_sdpa(*args, **kwargs):
    if len(args)==1:
        qkv=args[0]; q,k,v=qkv.feats.unbind(1)
        out=F.scaled_dot_product_attention(q[None].transpose(1,2),k[None].transpose(1,2),v[None].transpose(1,2))
        return qkv.replace(out.transpose(1,2)[0])
    elif len(args)==2:
        q_vlt,kv=args; q=q_vlt.feats[None].transpose(1,2)
        k,v2=kv[:,:,0].transpose(1,2),kv[:,:,1].transpose(1,2)
        return q_vlt.replace(F.scaled_dot_product_attention(q,k,v2).transpose(1,2)[0])
    elif len(args)==3:
        q_a,k_a,v_a=args; q=q_a.feats[None].transpose(1,2)
        return q_a.replace(F.scaled_dot_product_attention(q,k_a.transpose(1,2),v_a.transpose(1,2)).transpose(1,2)[0])
sfa.sparse_scaled_dot_product_attention = patched_sdpa
sam.sparse_scaled_dot_product_attention = patched_sdpa

from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.modules.sparse.basic import SparseTensor
import trellis2.modules.sparse as sp

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tex-flow", required=True)
    p.add_argument("--shape-slat", required=True, help="Shape latent [N, 32] .npy")
    p.add_argument("--coords", required=True, help="Sparse coords [N, 4] .npy")
    p.add_argument("--features", required=True, help="DINOv3 features [1029, 1024] .npy")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="ref_stage3")
    args = p.parse_args()

    device = 'cpu'
    coords = torch.from_numpy(np.load(args.coords)).int()
    cond = torch.from_numpy(np.load(args.features)).float().unsqueeze(0)
    shape_slat_raw = torch.from_numpy(np.load(args.shape_slat)).float()
    N = coords.shape[0]
    print(f"N={N}")

    # Normalize shape_slat for concat_cond
    with open('/tmp/t2cfg/pipeline.json') as f:
        cfg = json.load(f)
    shape_mean = torch.tensor(cfg['args']['shape_slat_normalization']['mean'])[None]
    shape_std = torch.tensor(cfg['args']['shape_slat_normalization']['std'])[None]
    shape_slat_norm = (shape_slat_raw - shape_mean) / shape_std
    print(f"shape_slat normalized: std={shape_slat_norm.std():.4f}")

    # Load texture flow model
    t0 = time.time()
    sd = load_file(args.tex_flow)
    sd = {k: v.float() for k, v in sd.items()}
    in_ch = sd['input_layer.weight'].shape[1]  # 64
    out_ch = sd['out_layer.weight'].shape[0]   # 32
    print(f"Tex flow: in_ch={in_ch}, out_ch={out_ch}")

    model = SLatFlowModel(resolution=32, in_channels=in_ch, model_channels=1536,
        cond_channels=1024, out_channels=out_ch, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32', share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True)
    model.load_state_dict(sd, strict=False); del sd
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    # Generate noise (only noise_ch = in_ch - 32 = 32 channels)
    noise_ch = in_ch - shape_slat_norm.shape[1]  # 64 - 32 = 32
    torch.manual_seed(args.seed)
    noise = torch.randn(N, noise_ch)

    # Concat: [noise, shape_slat_norm] -> [N, 64]
    concat_cond = SparseTensor(feats=shape_slat_norm, coords=coords, shape=torch.Size([1,32,32,32]))

    # Sampling: guidance_strength=1.0 means NO CFG (cond only)
    t_seq = np.linspace(1, 0, args.steps + 1)
    t_seq = 3.0 * t_seq / (1 + 2.0 * t_seq)
    x_feats = noise.clone()

    t0 = time.time()
    for step in range(args.steps):
        tc, tp = t_seq[step], t_seq[step+1]
        tt = torch.tensor([1000*tc], dtype=torch.float32)
        x_st = SparseTensor(feats=x_feats, coords=coords, shape=torch.Size([1,32,32,32]))
        with torch.no_grad():
            # Cond-only (guidance_strength=1.0)
            pred = model(x_st, tt, cond, concat_cond=concat_cond).feats
        x_feats = x_feats - (tc - tp) * pred
        print(f"  step {step+1}/{args.steps} t={tc:.4f} std={x_feats.std():.4f} {time.time()-t0:.0f}s")

    # Denormalize
    tex_mean = torch.tensor(cfg['args']['tex_slat_normalization']['mean'])[None]
    tex_std = torch.tensor(cfg['args']['tex_slat_normalization']['std'])[None]
    tex_slat = x_feats * tex_std + tex_mean

    np.save(f"{args.output}_tex_slat.npy", tex_slat.numpy())
    np.save(f"{args.output}_coords.npy", coords.numpy())
    print(f"Saved {args.output}_tex_slat.npy: std={tex_slat.std():.2f}")
    print(f"Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
