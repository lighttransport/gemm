#!/usr/bin/env python3
"""Debug Stage 2 DiT block 0 intermediate values."""
import os, sys
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np, torch.nn.functional as F
from safetensors.torch import load_file

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
    p.add_argument("--stage2", required=True)
    p.add_argument("--coords", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    coords_full = torch.from_numpy(np.load(args.coords)).int()
    cond = torch.from_numpy(np.load(args.features)).float().unsqueeze(0)
    N = min(args.N, coords_full.shape[0])
    coords = coords_full[:N]

    sd = load_file(args.stage2)
    sd = {k: v.float() for k, v in sd.items()}
    in_ch = sd['input_layer.weight'].shape[1]

    model = SLatFlowModel(resolution=32, in_channels=in_ch, model_channels=1536,
        cond_channels=1024, out_channels=in_ch, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32', share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True)
    model.load_state_dict(sd, strict=False); del sd
    model.eval()

    torch.manual_seed(args.seed)
    noise = torch.randn(N, in_ch)

    # === Manual forward to get intermediates ===
    x_st = SparseTensor(feats=noise, coords=coords, shape=torch.Size([1,32,32,32]))

    # 1. Input embedding
    h_st = model.input_layer(x_st)
    x = h_st.feats
    print(f"input_emb: std={x.std():.4f}, [:4]={x[0,:4].tolist()}")

    # 2. Timestep embedding
    t = torch.tensor([1000.0])
    t_emb = model.t_embedder(t)
    print(f"t_emb: std={t_emb.std():.4f}, [:4]={t_emb[0,:4].tolist()}")

    # 3. Modulation for block 0
    mod_input = F.silu(t_emb)
    mod_all = model.adaLN_modulation(mod_input)  # [1, 6*D]
    mod_block0 = mod_all + model.blocks[0].modulation[None]  # add per-block bias
    print(f"modulation: std={mod_all.std():.4f}, [:8]={mod_all[0,:8].tolist()}")
    print(f"block0_bias: std={model.blocks[0].modulation.std():.4f}, [:8]={model.blocks[0].modulation[:8].tolist()}")
    print(f"mod_w_block0: std={mod_block0.std():.4f}, [:8]={mod_block0[0,:8].tolist()}")

    # 4. Split modulation into 6 chunks
    shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mod_block0.chunk(6, dim=-1)
    print(f"shift_sa: std={shift_sa.std():.4f}, [:4]={shift_sa[0,:4].tolist()}")
    print(f"scale_sa: std={scale_sa.std():.4f}, [:4]={scale_sa[0,:4].tolist()}")
    print(f"gate_sa: std={gate_sa.std():.4f}, [:4]={gate_sa[0,:4].tolist()}")

    # 5. adaLN: LN(x) * (1 + scale) + shift
    x_ln = F.layer_norm(x, (1536,))
    x_adaln = x_ln * (1 + scale_sa) + shift_sa
    print(f"LN(x): std={x_ln.std():.4f}, [:4]={x_ln[0,:4].tolist()}")
    print(f"adaln_out: std={x_adaln.std():.4f}, [:4]={x_adaln[0,:4].tolist()}")

    # 6. QKV
    blk = model.blocks[0]
    qkv = blk.self_attn.to_qkv(x_adaln)
    q, k, v = qkv.chunk(3, dim=-1)
    print(f"Q_raw: std={q.std():.4f}, [:4]={q[0,:4].tolist()}")
    print(f"K_raw: std={k.std():.4f}, [:4]={k[0,:4].tolist()}")
    print(f"V_raw: std={v.std():.4f}, [:4]={v[0,:4].tolist()}")

    # 7. QK RMSNorm
    q = q.view(N, 12, 128)
    k = k.view(N, 12, 128)
    q = blk.self_attn.q_rms_norm(q)
    k = blk.self_attn.k_rms_norm(k)
    print(f"Q_rmsnorm: std={q.std():.4f}, q[0,0,:4]={q[0,0,:4].tolist()}")
    print(f"K_rmsnorm: std={k.std():.4f}, k[0,0,:4]={k[0,0,:4].tolist()}")

if __name__ == "__main__":
    main()
