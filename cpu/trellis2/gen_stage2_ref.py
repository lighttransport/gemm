#!/usr/bin/env python3
"""Generate Stage 2 DiT single-step reference data for CUDA verification.

Saves: noise, coords, cond, and model output for timestep=1.0 (model receives 1000.0).
"""
import os, sys
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np, torch.nn.functional as F
from safetensors.torch import load_file

# Monkey-patch sparse attention
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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage2", required=True, help="Stage 2 flow model .safetensors")
    p.add_argument("--coords", required=True, help="Sparse coords [N,4] .npy")
    p.add_argument("--features", required=True, help="DINOv3 features [1029,1024] .npy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--N", type=int, default=100, help="Use only first N voxels (for quick test)")
    p.add_argument("--output", default="s2ref")
    args = p.parse_args()

    coords_full = torch.from_numpy(np.load(args.coords)).int()
    cond = torch.from_numpy(np.load(args.features)).float().unsqueeze(0)  # [1, 1029, 1024]

    N = min(args.N, coords_full.shape[0])
    coords = coords_full[:N]
    print(f"Using N={N} voxels (of {coords_full.shape[0]})")
    print(f"Coords range: z=[{coords[:,1].min()},{coords[:,1].max()}] "
          f"y=[{coords[:,2].min()},{coords[:,2].max()}] x=[{coords[:,3].min()},{coords[:,3].max()}]")

    # Load model
    sd = load_file(args.stage2)
    sd = {k: v.float() for k, v in sd.items()}
    in_ch = sd['input_layer.weight'].shape[1]
    print(f"in_channels={in_ch}")

    model = SLatFlowModel(resolution=32, in_channels=in_ch, model_channels=1536,
        cond_channels=1024, out_channels=in_ch, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32', share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True)
    model.load_state_dict(sd, strict=False); del sd
    model.eval()

    # Generate noise
    torch.manual_seed(args.seed)
    noise = torch.randn(N, in_ch)
    print(f"Noise: std={noise.std():.4f}, [:4]={noise[0,:4].tolist()}")

    # Run single step at t=1.0 (model receives 1000*1.0 = 1000.0)
    # This is what run_dit_forward_generic does: timestep * 1000
    # So the caller should pass t=1.0, and the function multiplies by 1000 internally
    t_raw = 1.0  # raw timestep before *1000
    tt = torch.tensor([1000.0 * t_raw], dtype=torch.float32)  # what model receives
    x_st = SparseTensor(feats=noise, coords=coords, shape=torch.Size([1,32,32,32]))

    with torch.no_grad():
        out = model(x_st, tt, cond)

    output = out.feats.numpy()
    print(f"Output: shape={output.shape}, std={output.std():.4f}, [:4]={output[0,:4]}")

    # Save everything
    np.save(f"{args.output}_noise.npy", noise.numpy())
    np.save(f"{args.output}_coords.npy", coords.numpy())
    np.save(f"{args.output}_cond.npy", cond[0].numpy())  # [1029, 1024]
    np.save(f"{args.output}_output.npy", output)
    print(f"Saved {args.output}_*.npy")
    print(f"NOTE: CUDA verify should pass t_raw={t_raw} (not 1000.0) — function multiplies by 1000 internally")

if __name__ == "__main__":
    main()
