"""Dump ShapeVAE decoder reference outputs for CUDA verification.

Usage:
    uv run python dump_vae.py \
        --vae-path /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
        [--grid-res 8] [--outdir output]
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--grid-res", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sd = load_file(args.vae_path)
    W, H, HD = 1024, 16, 64

    torch.manual_seed(42)
    latents = torch.randn(1, 4096, 64, dtype=torch.float32)
    np.save(os.path.join(args.outdir, "vae_input_latents.npy"), latents.numpy()[0])

    # Post-KL projection
    x = F.linear(
        latents,
        sd["post_kl.weight"].float(),
        sd["post_kl.bias"].float(),
    )
    np.save(os.path.join(args.outdir, "vae_post_kl.npy"), x.numpy()[0])

    # 16 transformer blocks
    for bi in range(16):
        p = f"transformer.resblocks.{bi}."
        normed = F.layer_norm(
            x, [W],
            sd[p + "ln_1.weight"].float(),
            sd[p + "ln_1.bias"].float(),
            eps=1e-6,
        )
        qkv = F.linear(normed, sd[p + "attn.c_qkv.weight"].float())
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, H, 3, HD)
        q, k, v = qkv.unbind(dim=3)
        q = F.layer_norm(
            q, [HD],
            sd[p + "attn.attention.q_norm.weight"].float(),
            sd[p + "attn.attention.q_norm.bias"].float(),
            eps=1e-6,
        )
        k = F.layer_norm(
            k, [HD],
            sd[p + "attn.attention.k_norm.weight"].float(),
            sd[p + "attn.attention.k_norm.bias"].float(),
            eps=1e-6,
        )
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).reshape(B, N, W)
        attn = F.linear(
            attn,
            sd[p + "attn.c_proj.weight"].float(),
            sd[p + "attn.c_proj.bias"].float(),
        )
        x = x + attn
        n2 = F.layer_norm(
            x, [W],
            sd[p + "ln_2.weight"].float(),
            sd[p + "ln_2.bias"].float(),
            eps=1e-6,
        )
        h = F.gelu(F.linear(
            n2,
            sd[p + "mlp.c_fc.weight"].float(),
            sd[p + "mlp.c_fc.bias"].float(),
        ))
        x = x + F.linear(
            h,
            sd[p + "mlp.c_proj.weight"].float(),
            sd[p + "mlp.c_proj.bias"].float(),
        )
        if bi in (0, 8, 15):
            np.save(os.path.join(args.outdir, f"vae_block_{bi}.npy"), x.numpy()[0])
            print(f"  Block {bi}: mean={x.mean():.6f} std={x.std():.6f}")

    np.save(os.path.join(args.outdir, "vae_decoded_latents.npy"), x.numpy()[0])

    # SDF query through Fourier cross-attn geo decoder
    gr = args.grid_res
    c1d = torch.linspace(-1, 1, gr)
    gx, gy, gz = torch.meshgrid(c1d, c1d, c1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    freqs = 2.0 ** torch.arange(8, dtype=torch.float32)
    embed = coords.unsqueeze(-1) * freqs
    embed = embed.reshape(coords.shape[0], -1)
    qe = torch.cat([coords, torch.sin(embed), torch.cos(embed)], dim=-1)

    cp = "geo_decoder."
    qp = F.linear(
        qe,
        sd[cp + "query_proj.weight"].float(),
        sd[cp + "query_proj.bias"].float(),
    )
    qp = qp.unsqueeze(0)

    cap = cp + "cross_attn_decoder."
    ql = F.layer_norm(
        qp, [W],
        sd[cap + "ln_1.weight"].float(),
        sd[cap + "ln_1.bias"].float(),
        eps=1e-6,
    )
    kl = F.layer_norm(
        x, [W],
        sd[cap + "ln_2.weight"].float(),
        sd[cap + "ln_2.bias"].float(),
        eps=1e-6,
    )
    Q = F.linear(ql, sd[cap + "attn.c_q.weight"].float())
    KV = F.linear(kl, sd[cap + "attn.c_kv.weight"].float())
    Nq, Nkv = qp.shape[1], 4096
    KV = KV.reshape(1, Nkv, H, 2, HD)
    K, V = KV.unbind(dim=3)
    Q = Q.reshape(1, Nq, H, HD)
    Q = F.layer_norm(
        Q, [HD],
        sd[cap + "attn.attention.q_norm.weight"].float(),
        sd[cap + "attn.attention.q_norm.bias"].float(),
        eps=1e-6,
    )
    K = F.layer_norm(
        K, [HD],
        sd[cap + "attn.attention.k_norm.weight"].float(),
        sd[cap + "attn.attention.k_norm.bias"].float(),
        eps=1e-6,
    )
    cs = 8192
    outs = []
    for i in range(0, Nq, cs):
        outs.append(F.scaled_dot_product_attention(
            Q[:, i:i + cs].transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
        ).transpose(1, 2))
    ao = torch.cat(outs, dim=1).reshape(1, Nq, W)
    ao = F.linear(
        ao,
        sd[cap + "attn.c_proj.weight"].float(),
        sd[cap + "attn.c_proj.bias"].float(),
    )
    qo = qp + ao
    m = F.layer_norm(
        qo, [W],
        sd[cap + "ln_3.weight"].float(),
        sd[cap + "ln_3.bias"].float(),
        eps=1e-6,
    )
    m = F.gelu(F.linear(
        m,
        sd[cap + "mlp.c_fc.weight"].float(),
        sd[cap + "mlp.c_fc.bias"].float(),
    ))
    qo = qo + F.linear(
        m,
        sd[cap + "mlp.c_proj.weight"].float(),
        sd[cap + "mlp.c_proj.bias"].float(),
    )
    qo = F.layer_norm(
        qo, [W],
        sd[cp + "ln_post.weight"].float(),
        sd[cp + "ln_post.bias"].float(),
        eps=1e-6,
    )
    sdf = F.linear(
        qo,
        sd[cp + "output_proj.weight"].float(),
        sd[cp + "output_proj.bias"].float(),
    )
    sdf = sdf.squeeze(0).squeeze(-1).numpy().reshape(gr, gr, gr)
    np.save(os.path.join(args.outdir, "vae_sdf_grid.npy"), sdf)
    print(f"  SDF: {sdf.shape}, min={sdf.min():.6f} max={sdf.max():.6f}")
    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
