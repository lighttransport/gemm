#!/usr/bin/env python3
"""Run Stage 2 shape flow + decoder on CPU with SDPA monkey-patch."""
import os, sys, time
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np, torch.nn.functional as F, json
from safetensors.torch import load_file

# Monkey-patch sparse attention to use SDPA
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
    p.add_argument("--steps", type=int, default=1, help="Euler steps (1 for quick test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="ref_stage2")
    args = p.parse_args()

    device = 'cpu'
    coords = torch.from_numpy(np.load(args.coords)).int()
    cond = torch.from_numpy(np.load(args.features)).float().unsqueeze(0)
    N = coords.shape[0]
    print(f"N={N}, device={device}")

    # Load model
    t0 = time.time()
    sd = load_file(args.stage2)
    sd = {k: v.float() for k, v in sd.items()}
    in_ch = sd['input_layer.weight'].shape[1]
    model = SLatFlowModel(resolution=32, in_channels=in_ch, model_channels=1536,
        cond_channels=1024, out_channels=in_ch, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32', share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True)
    model.load_state_dict(sd, strict=False); del sd
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    # Sample
    torch.manual_seed(args.seed)
    x_feats = torch.randn(N, in_ch)
    neg_cond = torch.zeros_like(cond)
    t_seq = np.linspace(1, 0, args.steps + 1)
    t_seq = 3.0 * t_seq / (1 + 2.0 * t_seq)
    sigma_min = 1e-5

    t0 = time.time()
    for step in range(args.steps):
        tc, tp = t_seq[step], t_seq[step+1]
        tt = torch.tensor([1000*tc], dtype=torch.float32)
        x_st = SparseTensor(feats=x_feats, coords=coords, shape=torch.Size([1,32,32,32]))
        with torch.no_grad():
            if 0.6 <= tc <= 1.0:
                vp = model(x_st, tt, cond)
                vn = model(x_st, tt, neg_cond)
                pred = 7.5*vp.feats + (1-7.5)*vn.feats
                coeff = sigma_min + (1-sigma_min)*tc
                x0p = (1-sigma_min)*x_feats - coeff*vp.feats
                x0c = (1-sigma_min)*x_feats - coeff*pred
                r = x0p.std()/x0c.std()
                x0 = 0.5*(x0c*r) + 0.5*x0c
                pred = ((1-sigma_min)*x_feats - x0)/coeff
            else:
                pred = model(x_st, tt, cond).feats
        x_feats = x_feats - (tc-tp)*pred
        elapsed = time.time()-t0
        print(f"  step {step+1}/{args.steps} t={tc:.4f} std={x_feats.std():.4f} {elapsed:.0f}s")

    # Denormalize
    with open('/tmp/t2cfg/pipeline.json') as f:
        cfg = json.load(f)
    mean = torch.tensor(cfg['args']['shape_slat_normalization']['mean'])[None]
    std_v = torch.tensor(cfg['args']['shape_slat_normalization']['std'])[None]
    slat = x_feats * std_v + mean

    np.save(f"{args.output}_slat.npy", slat.numpy())
    np.save(f"{args.output}_coords.npy", coords.numpy())
    np.save(f"{args.output}_noise.npy", torch.randn(N, in_ch).numpy())  # won't match — seed consumed
    print(f"Saved {args.output}_slat.npy: shape={slat.shape}, std={slat.std():.2f}")
    print(f"Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
