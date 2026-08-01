#!/usr/bin/env python3
"""Dump TRELLIS.2 DiT block-0 detailed sub-step intermediates matching HIP --dump-b0."""
import sys, os, argparse
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../cpu/trellis2/trellis2_repo'))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def save(outdir, name, tensor):
    arr = tensor.detach().float().cpu().numpy()
    np.save(os.path.join(outdir, f'{name}.npy'), arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--noise',    required=True)
    ap.add_argument('--stage1',   required=True)
    ap.add_argument('--timestep', type=float, default=700.0)
    ap.add_argument('--output-dir', default='.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')

    features = torch.from_numpy(np.load(args.features)).float().unsqueeze(0).to(device)
    noise    = torch.from_numpy(np.load(args.noise)).float().to(device)

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.modules.attention.full_attn import scaled_dot_product_attention
    from trellis2.modules.attention.rope import RotaryPositionEmbedder

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

    with torch.no_grad():
        h = noise.view(1, 8, -1).permute(0, 2, 1).contiguous()  # [1, 4096, 8]
        h = model.input_layer(h)                                # [1, 4096, 1536]

        t = torch.tensor([args.timestep], device=device)
        t_emb_raw = model.t_embedder(t)
        t_emb = model.adaLN_modulation(t_emb_raw)

        block = model.blocks[0]
        mod_total = block.modulation + t_emb  # [1, 6*1536]
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mod_total.chunk(6, dim=1)
        mod_concat = torch.cat([shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp], dim=1).squeeze(0)
        save(args.output_dir, 'ref_b0_mod', mod_concat)

        x = h  # [1, 4096, 1536]
        # ---- self-attn path ----
        h1 = block.norm1(x)
        h1 = h1 * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        save(args.output_dir, 'ref_b0_ln_h_sa', h1.squeeze(0))

        sa = block.self_attn
        B, L, C = h1.shape
        qkv = sa.to_qkv(h1).reshape(B, L, 3, sa.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)
        q = sa.q_rms_norm(q)
        k = sa.k_rms_norm(k)
        phases = model.rope_phases
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
        save(args.output_dir, 'ref_b0_q_post', q.reshape(B, L, -1).squeeze(0))
        save(args.output_dir, 'ref_b0_k_post', k.reshape(B, L, -1).squeeze(0))
        save(args.output_dir, 'ref_b0_v',      v.reshape(B, L, -1).squeeze(0))

        attn = scaled_dot_product_attention(q, k, v).reshape(B, L, -1)
        sa_proj = sa.to_out(attn)
        save(args.output_dir, 'ref_b0_sa_proj', sa_proj.squeeze(0))

        x = x + sa_proj * gate_sa.unsqueeze(1)
        save(args.output_dir, 'ref_b0_h_post_sa', x.squeeze(0))

        # ---- cross-attn path ----
        h2 = block.norm2(x)
        ca = block.cross_attn
        qc = ca.to_q(h2).reshape(B, L, ca.num_heads, -1)
        kv = ca.to_kv(features).reshape(1, features.shape[1], 2, ca.num_heads, -1)
        qc = ca.q_rms_norm(qc)
        kc, vc = kv.unbind(dim=2)
        kc = ca.k_rms_norm(kc)
        ca_attn = scaled_dot_product_attention(qc, kc, vc).reshape(B, L, -1)
        ca_proj = ca.to_out(ca_attn)
        save(args.output_dir, 'ref_b0_ca_proj', ca_proj.squeeze(0))
        x = x + ca_proj
        save(args.output_dir, 'ref_b0_h_post_ca', x.squeeze(0))

        # ---- MLP path ----
        h3 = block.norm3(x)
        h3 = h3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        save(args.output_dir, 'ref_b0_ln_h_mlp', h3.squeeze(0))
        mlp_proj = block.mlp(h3)
        save(args.output_dir, 'ref_b0_mlp_proj', mlp_proj.squeeze(0))
        x = x + mlp_proj * gate_mlp.unsqueeze(1)
        save(args.output_dir, 'ref_b0_h_post_mlp', x.squeeze(0))

    print('done')


if __name__ == '__main__':
    main()
