#!/usr/bin/env python3
"""
Qwen-Image VAE decoder layer-by-layer PyTorch reference.

Dumps intermediate outputs at each stage for comparison with C implementation.

Usage:
    uv run python run_vae_reference.py \
        --vae-path /mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors \
        --latent-h 8 --latent-w 8 --seed 42 --output-dir output/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def save(arr, path):
    """Save tensor/ndarray as .npy."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    np.save(path, arr)
    print(f"  saved {path}  shape={arr.shape}  "
          f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")


def causal_pad_3d(x, kd, kh, kw):
    """Causal padding for 3D conv: replicate-pad temporal, reflect-pad spatial."""
    # Pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
    pw = (kw - 1) // 2
    ph = (kh - 1) // 2
    pd = kd - 1  # causal: all padding on left
    return F.pad(x, (pw, pw, ph, ph, pd, 0), mode='replicate')


def resblock_forward(x, st, prefix):
    """Run a single residual block."""
    # Norm1 -> SiLU -> Conv1
    gamma1 = st.get_tensor(f"{prefix}.residual.0.gamma").float()
    h = F.group_norm(x, 32, weight=gamma1.view(-1))
    h = F.silu(h)
    w1 = st.get_tensor(f"{prefix}.residual.2.weight").float()
    b1 = st.get_tensor(f"{prefix}.residual.2.bias").float()
    h = F.conv3d(causal_pad_3d(h, 3, 3, 3), w1, b1)

    # Norm2 -> SiLU -> Conv2
    gamma2 = st.get_tensor(f"{prefix}.residual.3.gamma").float()
    h = F.group_norm(h, 32, weight=gamma2.view(-1))
    h = F.silu(h)
    w2 = st.get_tensor(f"{prefix}.residual.6.weight").float()
    b2 = st.get_tensor(f"{prefix}.residual.6.bias").float()
    h = F.conv3d(causal_pad_3d(h, 3, 3, 3), w2, b2)

    # Shortcut
    try:
        sc_w = st.get_tensor(f"{prefix}.shortcut.weight").float()
        sc_b = st.get_tensor(f"{prefix}.shortcut.bias").float()
        x = F.conv3d(x, sc_w, sc_b)
    except Exception:
        pass

    return x + h


def mid_attention(x, st):
    """Middle block self-attention."""
    B, C, T, H, W = x.shape
    spatial = T * H * W

    norm_g = st.get_tensor("decoder.middle.1.norm.gamma").float()
    qkv_w = st.get_tensor("decoder.middle.1.to_qkv.weight").float()
    qkv_b = st.get_tensor("decoder.middle.1.to_qkv.bias").float()
    proj_w = st.get_tensor("decoder.middle.1.proj.weight").float()
    proj_b = st.get_tensor("decoder.middle.1.proj.bias").float()

    # GroupNorm
    x_norm = F.group_norm(x, 32, weight=norm_g.view(-1))
    x_flat = x_norm.reshape(B, C, spatial)

    # QKV via 1x1 conv (matmul)
    qkv = torch.einsum('oi,bis->bos', qkv_w.view(3 * C, C), x_flat)
    qkv = qkv + qkv_b.view(1, 3 * C, 1)
    q, k, v = qkv.chunk(3, dim=1)

    # Attention
    scale = C ** -0.5
    attn = torch.bmm(q.permute(0, 2, 1), k) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(v, attn.permute(0, 2, 1))

    # Output projection
    out = torch.einsum('oi,bis->bos', proj_w.view(C, C), out)
    out = out + proj_b.view(1, C, 1)

    return x + out.reshape(B, C, T, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', required=True)
    parser.add_argument('--latent-h', type=int, default=8)
    parser.add_argument('--latent-w', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"Loading VAE from {args.vae_path}")
    st = safe_open(args.vae_path, framework="pt", device="cpu")

    # Generate random latent (same PRNG as C code uses seed=42)
    z = torch.randn(1, 16, 1, args.latent_h, args.latent_w)
    print(f"Latent: {z.shape}")
    save(z[0, :, 0], f"{args.output_dir}/vae_00_input.npy")

    # post_quant_conv (conv2): 1x1x1 conv
    pqc_w = st.get_tensor("conv2.weight").float()
    pqc_b = st.get_tensor("conv2.bias").float()
    x = F.conv3d(z, pqc_w, pqc_b)
    save(x[0, :, 0], f"{args.output_dir}/vae_01_post_quant.npy")

    # decoder.conv1
    w = st.get_tensor("decoder.conv1.weight").float()
    b = st.get_tensor("decoder.conv1.bias").float()
    x = F.conv3d(causal_pad_3d(x, 3, 3, 3), w, b)
    save(x[0, :, 0], f"{args.output_dir}/vae_02_dec_conv1.npy")

    # Middle block
    x = resblock_forward(x, st, "decoder.middle.0")
    save(x[0, :, 0], f"{args.output_dir}/vae_03_mid_res0.npy")

    x = mid_attention(x, st)
    save(x[0, :, 0], f"{args.output_dir}/vae_04_mid_attn.npy")

    x = resblock_forward(x, st, "decoder.middle.2")
    save(x[0, :, 0], f"{args.output_dir}/vae_05_mid_res2.npy")

    # Upsample blocks
    h, w_size = x.shape[3], x.shape[4]
    stage = 6
    for i in range(15):
        prefix = f"decoder.upsamples.{i}"

        # Check if this has a residual block
        try:
            st.get_tensor(f"{prefix}.residual.2.weight")
            x = resblock_forward(x, st, prefix)
            save(x[0, :, 0], f"{args.output_dir}/vae_{stage:02d}_up{i}_res.npy")
            stage += 1
        except Exception:
            pass  # resample-only block

        # Check for resample
        try:
            rs_w = st.get_tensor(f"{prefix}.resample.1.weight").float()
            rs_b = st.get_tensor(f"{prefix}.resample.1.bias").float()

            # NN upsample 2x spatial
            B, C, T, H, W = x.shape
            x_up = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

            # Conv2d on spatial (take T=0 slice, apply 2D conv, add T back)
            x_2d = F.conv2d(x_up[:, :, 0], rs_w, rs_b, padding=1)
            x = x_2d.unsqueeze(2)
            save(x[0, :, 0], f"{args.output_dir}/vae_{stage:02d}_up{i}_resample.npy")
            stage += 1
            print(f"  upsample {i}: resample -> {x.shape}")
        except Exception:
            pass

        # Skip time_conv for image mode

    # Head: GroupNorm -> SiLU -> Conv
    head_g = st.get_tensor("decoder.head.0.gamma").float()
    head_w = st.get_tensor("decoder.head.2.weight").float()
    head_b = st.get_tensor("decoder.head.2.bias").float()

    x = F.group_norm(x, 32, weight=head_g.view(-1))
    x = F.silu(x)
    x = F.conv3d(causal_pad_3d(x, 3, 3, 3), head_w, head_b)

    rgb = x[0, :, 0]  # [3, H, W]
    save(rgb, f"{args.output_dir}/vae_{stage:02d}_output.npy")

    # Save as image
    img = rgb.permute(1, 2, 0).numpy()
    img = np.clip(img * 0.5 + 0.5, 0, 1)
    img = (img * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img).save(f"{args.output_dir}/vae_output.png")
        print(f"  saved {args.output_dir}/vae_output.png")
    except Exception as e:
        print(f"  (no PIL: {e})")

    print("Done.")


if __name__ == "__main__":
    main()
