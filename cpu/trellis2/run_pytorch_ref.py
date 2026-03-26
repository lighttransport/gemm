#!/usr/bin/env python3
"""
run_pytorch_ref.py - Run TRELLIS.2 Stage 1 pipeline with PyTorch reference.

Implements the exact same pipeline as our C/CUDA code:
  DINOv3 features -> DiT flow sampling (12 Euler steps, CFG=7.5) -> Decoder -> Occupancy -> Mesh

Usage:
  python run_pytorch_ref.py --stage1 <ss_flow.st> --decoder <ss_dec.st> \
      --features <features.npy> [--grid 64] [--seed 42] [--output out.obj]
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def load_bf16_safetensors(path):
    """Load safetensors, converting BF16->F32."""
    sd = load_file(path)
    return {k: v.float() if v.dtype == torch.bfloat16 else v.float() for k, v in sd.items()}


# ---- DiT Components ----

def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32, device=t.device) / half * np.log(10000))
    args = t * freqs
    return torch.cat([torch.cos(args), torch.sin(args)])


def rms_norm(x, gamma, eps=1e-6):
    """Per-head RMSNorm. x: [N, dim], gamma: [n_heads, head_dim]."""
    n_heads = gamma.shape[0]
    head_dim = gamma.shape[1]
    x = x.view(x.shape[0], n_heads, head_dim)
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * rms * gamma.unsqueeze(0)
    return x.view(x.shape[0], -1)


def rope_3d(x, cos_table, sin_table, n_freqs, axis_dim):
    """Apply 3D RoPE. x: [N, dim], tables: [N, 3, n_freqs]."""
    N, dim = x.shape
    n_heads = dim // (axis_dim * 3 + (dim - axis_dim * 3))  # approximate
    head_dim = dim // n_heads if n_heads > 0 else dim
    n_heads = dim // head_dim

    x = x.view(N, n_heads, head_dim)
    for axis in range(3):
        base = axis * axis_dim
        c = cos_table[:, axis, :]  # [N, n_freqs]
        s = sin_table[:, axis, :]
        x0 = x[:, :, base:base+n_freqs]
        x1 = x[:, :, base+n_freqs:base+2*n_freqs]
        x[:, :, base:base+n_freqs] = x0 * c.unsqueeze(1) - x1 * s.unsqueeze(1)
        x[:, :, base+n_freqs:base+2*n_freqs] = x0 * s.unsqueeze(1) + x1 * c.unsqueeze(1)
    return x.view(N, dim)


def dit_forward(x_t, t_val, cond, sd, device):
    """Single DiT denoising step."""
    dim = 1536
    heads = 12
    head_dim = 128
    ffn = 8192
    n_blocks = 30
    N = x_t.shape[0]

    # Timestep embedding
    t_emb = timestep_embedding(torch.tensor(t_val, device=device), 256)
    h = F.linear(t_emb, sd['t_embedder.mlp.0.weight'], sd['t_embedder.mlp.0.bias'])
    h = F.silu(h)
    t_emb = F.linear(h, sd['t_embedder.mlp.2.weight'], sd['t_embedder.mlp.2.bias'])

    # Input embedding
    hidden = F.linear(x_t, sd['input_layer.weight'], sd['input_layer.bias'])

    # Precompute RoPE tables
    gs = 16
    n_freqs = head_dim // 6
    axis_dim = 2 * n_freqs
    freqs = 1.0 / (10000.0 ** (torch.arange(n_freqs, dtype=torch.float32, device=device) / n_freqs))
    cos_table = torch.zeros(N, 3, n_freqs, device=device)
    sin_table = torch.zeros(N, 3, n_freqs, device=device)
    for i in range(N):
        z, y, x = i // (gs*gs), (i // gs) % gs, i % gs
        for axis, coord in enumerate([z, y, x]):
            angles = coord * freqs
            cos_table[i, axis] = torch.cos(angles)
            sin_table[i, axis] = torch.sin(angles)

    for bi in range(n_blocks):
        p = f'blocks.{bi}.'

        # Shared modulation
        mod_input = F.silu(t_emb)
        mod = F.linear(mod_input, sd['adaLN_modulation.1.weight'], sd['adaLN_modulation.1.bias'])
        mod = mod + sd[f'{p}modulation']
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6)

        # === Self-attention with adaLN ===
        normed = F.layer_norm(hidden, [dim], eps=1e-6)
        normed = normed * (1 + scale_sa) + shift_sa

        qkv = F.linear(normed, sd[f'{p}self_attn.to_qkv.weight'])
        q, k, v = qkv.chunk(3, dim=-1)

        q = rms_norm(q, sd[f'{p}self_attn.q_rms_norm.gamma'])
        k = rms_norm(k, sd[f'{p}self_attn.k_rms_norm.gamma'])

        q = rope_3d(q, cos_table, sin_table, n_freqs, axis_dim)
        k = rope_3d(k, cos_table, sin_table, n_freqs, axis_dim)

        # Attention
        q = q.view(N, heads, head_dim).transpose(0, 1)  # [H, N, HD]
        k = k.view(N, heads, head_dim).transpose(0, 1)
        v = v.view(N, heads, head_dim).transpose(0, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(0, 1).contiguous().view(N, dim)

        proj = F.linear(attn, sd[f'{p}self_attn.to_out.weight'], sd[f'{p}self_attn.to_out.bias'])
        hidden = hidden + gate_sa * proj

        # === Cross-attention ===
        normed = F.layer_norm(hidden, [dim],
                              weight=sd[f'{p}norm2.weight'], bias=sd[f'{p}norm2.bias'], eps=1e-6)
        cross_q = F.linear(normed, sd[f'{p}cross_attn.to_q.weight'], sd[f'{p}cross_attn.to_q.bias'])
        cross_kv = F.linear(cond, sd[f'{p}cross_attn.to_kv.weight'], sd[f'{p}cross_attn.to_kv.bias'])
        cross_k, cross_v = cross_kv.chunk(2, dim=-1)

        cross_q = rms_norm(cross_q, sd[f'{p}cross_attn.q_rms_norm.gamma'])
        cross_k = rms_norm(cross_k, sd[f'{p}cross_attn.k_rms_norm.gamma'])

        q2 = cross_q.view(N, heads, head_dim).transpose(0, 1)
        k2 = cross_k.view(cond.shape[0], heads, head_dim).transpose(0, 1)
        v2 = cross_v.view(cond.shape[0], heads, head_dim).transpose(0, 1)
        ca = F.scaled_dot_product_attention(q2, k2, v2)
        ca = ca.transpose(0, 1).contiguous().view(N, dim)

        ca_proj = F.linear(ca, sd[f'{p}cross_attn.to_out.weight'], sd[f'{p}cross_attn.to_out.bias'])
        hidden = hidden + ca_proj

        # === MLP with adaLN ===
        normed = F.layer_norm(hidden, [dim], eps=1e-6)
        normed = normed * (1 + scale_mlp) + shift_mlp
        ff = F.linear(normed, sd[f'{p}mlp.mlp.0.weight'], sd[f'{p}mlp.mlp.0.bias'])
        ff = F.gelu(ff)
        ff = F.linear(ff, sd[f'{p}mlp.mlp.2.weight'], sd[f'{p}mlp.mlp.2.bias'])
        hidden = hidden + gate_mlp * ff

    out = F.linear(hidden, sd['out_layer.weight'], sd['out_layer.bias'])
    return out


# ---- Decoder Components ----

def groupnorm_3d(x, w, b, G=32):
    """x: [1, C, D, H, W]"""
    return F.group_norm(x, G, weight=w, bias=b, eps=1e-5)


def resblock_3d(x, sd, prefix, G=32):
    """ResBlock3d: GN->SiLU->Conv3d->GN->SiLU->Conv3d + skip"""
    h = groupnorm_3d(x, sd[f'{prefix}norm1.weight'], sd[f'{prefix}norm1.bias'], G)
    h = F.silu(h)
    h = F.conv3d(h, sd[f'{prefix}conv1.weight'], sd[f'{prefix}conv1.bias'], padding=1)
    h = groupnorm_3d(h, sd[f'{prefix}norm2.weight'], sd[f'{prefix}norm2.bias'], G)
    h = F.silu(h)
    h = F.conv3d(h, sd[f'{prefix}conv2.weight'], sd[f'{prefix}conv2.bias'], padding=1)
    return h + x


def pixel_shuffle_3d(x, factor=2):
    """[B, C*f^3, D, H, W] -> [B, C, D*f, H*f, W*f]"""
    B, C, D, H, W = x.shape
    c_out = C // (factor ** 3)
    x = x.view(B, c_out, factor, factor, factor, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(B, c_out, D*factor, H*factor, W*factor)


def decoder_forward(latent, sd, device):
    """Decode [8,16,16,16] -> [1,64,64,64] occupancy."""
    x = latent.unsqueeze(0).to(device)  # [1, 8, 16, 16, 16]

    # conv_in
    x = F.conv3d(x, sd['input_layer.weight'], sd['input_layer.bias'], padding=1)

    # middle blocks
    for i in range(2):
        x = resblock_3d(x, sd, f'middle_block.{i}.')

    # blocks 0-1 at 16^3
    for i in range(2):
        x = resblock_3d(x, sd, f'blocks.{i}.')

    # upsample 1: conv(512->1024) + pixel_shuffle -> [128, 32^3]
    x = F.conv3d(x, sd['blocks.2.conv.weight'], sd['blocks.2.conv.bias'], padding=1)
    x = pixel_shuffle_3d(x, 2)

    # blocks 3-4 at 32^3
    for i in range(3, 5):
        x = resblock_3d(x, sd, f'blocks.{i}.')

    # upsample 2: conv(128->256) + pixel_shuffle -> [32, 64^3]
    x = F.conv3d(x, sd['blocks.5.conv.weight'], sd['blocks.5.conv.bias'], padding=1)
    x = pixel_shuffle_3d(x, 2)

    # blocks 6-7 at 64^3
    for i in range(6, 8):
        x = resblock_3d(x, sd, f'blocks.{i}.')

    # output: GN -> SiLU -> Conv3d(32->1)
    x = groupnorm_3d(x, sd['out_layer.0.weight'], sd['out_layer.0.bias'], 32)
    x = F.silu(x)
    x = F.conv3d(x, sd['out_layer.2.weight'], sd['out_layer.2.bias'], padding=1)

    return x.squeeze(0).squeeze(0)  # [64, 64, 64]


# ---- Marching Cubes (via skimage or simple export) ----

def export_mesh(occupancy, path, grid_size=64, threshold=0.0):
    """Export occupancy grid to .obj using marching cubes."""
    occ = occupancy.cpu().numpy()

    if grid_size < 64:
        from scipy.ndimage import zoom
        factor = grid_size / 64.0
        occ = zoom(occ, factor, order=1)
        print(f"  Downsampled 64^3 -> {grid_size}^3")

    try:
        from skimage.measure import marching_cubes
        verts, faces, normals, _ = marching_cubes(occ, level=threshold)
        # Normalize to [0, 1]
        verts = verts / (np.array(occ.shape) - 1)
    except ImportError:
        print("  skimage not available, writing raw .npy instead")
        np.save(path.replace('.obj', '.npy'), occ)
        return

    with open(path, 'w') as f:
        f.write(f"# PyTorch reference mesh: {len(verts)} vertices, {len(faces)} triangles\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Wrote {path}: {len(verts)} verts, {len(faces)} faces")


# ---- Sampling ----

def rescale_t(t, rescale_val):
    return t * rescale_val / (1 + (rescale_val - 1) * t)


def sample_stage1(cond, dit_sd, device, n_steps=12, cfg_scale=7.5, rescale_val=5.0, seed=42):
    N, in_ch = 4096, 8
    torch.manual_seed(seed)
    x = torch.randn(N, in_ch, device=device)

    zeros_cond = torch.zeros_like(cond)

    for step in range(n_steps):
        t_start = 1.0 - step / n_steps
        t_end = 1.0 - (step + 1) / n_steps
        t_cur = rescale_t(t_start, rescale_val)
        t_next = rescale_t(t_end, rescale_val)
        dt = t_next - t_cur

        t0 = time.time()
        with torch.no_grad():
            v_cond = dit_forward(x, t_cur, cond, dit_sd, device)
            v_uncond = dit_forward(x, t_cur, zeros_cond, dit_sd, device)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + dt * v
        elapsed = (time.time() - t0) * 1000
        print(f"  step {step+1}/{n_steps}  t={t_start:.4f}  dt={dt:.4f}  {elapsed:.0f} ms")

    return x.view(in_ch, 16, 16, 16)


def main():
    parser = argparse.ArgumentParser(description="PyTorch TRELLIS.2 Stage 1 reference")
    parser.add_argument("--stage1", required=True, help="Stage 1 flow model safetensors")
    parser.add_argument("--decoder", required=True, help="Structure decoder safetensors")
    parser.add_argument("--features", required=True, help="DINOv3 features .npy [1029, 1024]")
    parser.add_argument("--grid", type=int, nargs='+', default=[64, 32], help="MC grid sizes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--output", default="ref", help="Output prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load features
    features = torch.from_numpy(np.load(args.features)).float().to(device)
    print(f"Features: {features.shape}")

    # Load DiT weights
    print("\n=== Loading Stage 1 DiT ===")
    t0 = time.time()
    dit_sd = load_bf16_safetensors(args.stage1)
    dit_sd = {k: v.to(device) for k, v in dit_sd.items()}
    print(f"  Loaded {len(dit_sd)} tensors in {time.time()-t0:.1f}s")

    # Sample
    print(f"\n=== Stage 1: Flow Sampling ({args.steps} steps, cfg={args.cfg}) ===")
    t0 = time.time()
    latent = sample_stage1(features, dit_sd, device, args.steps, args.cfg, seed=args.seed)
    dit_time = time.time() - t0
    print(f"  DiT total: {dit_time:.1f}s")

    # Save latent
    latent_np = latent.cpu().numpy()
    np.save(f"{args.output}_latent.npy", latent_np)
    print(f"  Latent: shape={latent_np.shape}, min={latent_np.min():.2f}, max={latent_np.max():.2f}")

    # Free DiT weights
    del dit_sd
    torch.cuda.empty_cache()

    # Load decoder
    print("\n=== Loading Decoder ===")
    dec_sd = load_file(args.decoder)
    dec_sd = {k: v.float().to(device) for k, v in dec_sd.items()}
    print(f"  Loaded {len(dec_sd)} tensors")

    # Decode
    print("\n=== Decoding ===")
    t0 = time.time()
    with torch.no_grad():
        occupancy = decoder_forward(latent, dec_sd, device)
    dec_time = time.time() - t0
    print(f"  Decoder: {dec_time:.1f}s")

    occ_np = occupancy.cpu().numpy()
    np.save(f"{args.output}_occupancy.npy", occ_np)
    n_occ = (occ_np > 0).sum()
    print(f"  Occupancy: shape={occ_np.shape}, min={occ_np.min():.2f}, max={occ_np.max():.2f}")
    print(f"  Occupied (logit>0): {n_occ}/{occ_np.size} ({100*n_occ/occ_np.size:.1f}%)")

    # Export meshes at different grid sizes
    print("\n=== Mesh Export ===")
    for gs in args.grid:
        out_path = f"{args.output}_{gs}.obj"
        print(f"  Grid {gs}:")
        export_mesh(occupancy, out_path, gs)

    print(f"\n=== Summary ===")
    print(f"  DiT sampling: {dit_time:.1f}s")
    print(f"  Decoder: {dec_time:.1f}s")
    print(f"  Total: {dit_time + dec_time:.1f}s")


if __name__ == "__main__":
    main()
