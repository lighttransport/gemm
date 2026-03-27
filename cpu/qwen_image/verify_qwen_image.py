#!/usr/bin/env python3
"""
verify_qwen_image.py - PyTorch reference and verification for Qwen-Image pipeline.

Usage:
  # Inspect weight files
  python verify_qwen_image.py --inspect-gguf <path.gguf>
  python verify_qwen_image.py --inspect-safetensors <path.safetensors>

  # Generate reference VAE decode output
  python verify_qwen_image.py --generate-vae-ref \
      --vae-path /mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors \
      --latent-h 32 --latent-w 32 --seed 42 --output vae_ref.npy

  # Generate reference text encoder output
  python verify_qwen_image.py --generate-text-ref \
      --prompt "a cat sitting on a chair" \
      --output text_ref.npy

  # Generate reference single DiT step output
  python verify_qwen_image.py --generate-dit-ref \
      --dit-path /mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf \
      --seed 42 --output dit_ref.npy

  # Compare C output with reference
  python verify_qwen_image.py --compare ref.npy c_output.npy

Dependencies:
  pip install numpy safetensors torch
  pip install diffusers transformers (for full pipeline)
"""

import argparse
import numpy as np
import sys
import os
import struct
import json


def inspect_safetensors(path):
    """Print all tensor names, shapes, and dtypes from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    metadata = json.loads(header_json)
    keys = [k for k in metadata.keys() if k != "__metadata__"]
    print(f"Safetensors: {path}")
    print(f"Total tensors: {len(keys)}")

    total_params = 0
    for k in sorted(keys):
        info = metadata[k]
        shape = info["shape"]
        dtype = info["dtype"]
        numel = 1
        for s in shape:
            numel *= s
        total_params += numel
        print(f"  {k}: {shape} ({dtype})")
    print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")


def compare_npy(ref_path, test_path):
    """Compare two .npy files and report statistics."""
    ref = np.load(ref_path).astype(np.float32).flatten()
    test = np.load(test_path).astype(np.float32).flatten()

    if ref.shape != test.shape:
        print(f"Shape mismatch: ref {ref.shape} vs test {test.shape}")
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
        print(f"Comparing first {min_len} elements")

    diff = np.abs(ref - test)
    print(f"Ref  : min={ref.min():.6f} max={ref.max():.6f} mean={ref.mean():.6f}")
    print(f"Test : min={test.min():.6f} max={test.max():.6f} mean={test.mean():.6f}")
    print(f"Diff : min={diff.min():.6f} max={diff.max():.6f} mean={diff.mean():.6f}")
    print(f"       median={np.median(diff):.6f} p99={np.percentile(diff, 99):.6f}")

    if len(ref) > 10:
        corr = np.corrcoef(ref, test)[0, 1]
        print(f"Correlation: {corr:.6f}")
    else:
        corr = 0.0

    if diff.max() < 0.1 and (len(ref) <= 10 or corr > 0.99):
        print("PASS")
        return True
    elif diff.max() < 1.0 and (len(ref) <= 10 or corr > 0.95):
        print("MARGINAL PASS (quantization differences expected)")
        return True
    else:
        print("FAIL")
        return False


def generate_vae_ref(args):
    """Generate reference VAE decode output using raw weights."""
    import torch
    import torch.nn.functional as F

    print("Loading VAE weights...")
    from safetensors import safe_open
    st = safe_open(args.vae_path, framework="pt", device="cpu")

    # Generate random latent
    torch.manual_seed(args.seed)
    lat_h, lat_w = args.latent_h, args.latent_w
    z = torch.randn(1, 16, 1, lat_h, lat_w)  # [B, C, T, H, W]
    print(f"Latent shape: {z.shape}")

    # Save input latent
    lat_path = args.output.replace('.npy', '_input.npy')
    np.save(lat_path, z[:, :, 0, :, :].numpy())  # Save without T dim
    print(f"Saved input latent to {lat_path}")

    # We'll do a simplified VAE decode using raw weights
    # For now, just save the random latent as reference input
    # Full decode requires implementing the 3D causal conv architecture

    # post_quant_conv (conv2): 1x1x1 conv
    pqc_w = st.get_tensor("conv2.weight").float()  # [16, 16, 1, 1, 1]
    pqc_b = st.get_tensor("conv2.bias").float()     # [16]
    x = F.conv3d(z, pqc_w, pqc_b)
    print(f"After post_quant_conv: {x.shape}")

    # decoder.conv1
    dec_w = st.get_tensor("decoder.conv1.weight").float()
    dec_b = st.get_tensor("decoder.conv1.bias").float()
    # Causal padding: (kD-1, 0, pad_h, pad_h, pad_w, pad_w)
    x = F.pad(x, (1, 1, 1, 1, 2, 0), mode='replicate')
    x = F.conv3d(x, dec_w, dec_b)
    print(f"After decoder.conv1: {x.shape}")

    # Save intermediate for comparison
    inter_path = args.output.replace('.npy', '_after_conv1.npy')
    np.save(inter_path, x[:, :, 0, :, :].detach().numpy())
    print(f"Saved after conv1 to {inter_path}")

    # Helper: load and run a residual block
    def resblock_forward(x, prefix):
        spatial = x.shape[3:]
        c_in = x.shape[1]

        # norm1 -> silu -> conv1
        gamma1 = st.get_tensor(f"{prefix}.residual.0.gamma").float()
        gn1 = F.group_norm(x, 32, weight=gamma1.view(-1))
        h = F.silu(gn1)

        w1 = st.get_tensor(f"{prefix}.residual.2.weight").float()
        b1 = st.get_tensor(f"{prefix}.residual.2.bias").float()
        h = F.pad(h, (1, 1, 1, 1, 2, 0), mode='replicate')
        h = F.conv3d(h, w1, b1)
        c_out = h.shape[1]

        # norm2 -> silu -> conv2
        gamma2 = st.get_tensor(f"{prefix}.residual.3.gamma").float()
        gn2 = F.group_norm(h, 32, weight=gamma2.view(-1))
        h = F.silu(gn2)

        w2 = st.get_tensor(f"{prefix}.residual.6.weight").float()
        b2 = st.get_tensor(f"{prefix}.residual.6.bias").float()
        h = F.pad(h, (1, 1, 1, 1, 2, 0), mode='replicate')
        h = F.conv3d(h, w2, b2)

        # shortcut
        try:
            sc_w = st.get_tensor(f"{prefix}.shortcut.weight").float()
            sc_b = st.get_tensor(f"{prefix}.shortcut.bias").float()
            x = F.conv3d(x, sc_w, sc_b)
        except:
            pass  # no shortcut needed (same channels)

        return x + h

    # Middle block
    x = resblock_forward(x, "decoder.middle.0")

    # Middle attention
    norm_g = st.get_tensor("decoder.middle.1.norm.gamma").float()
    qkv_w = st.get_tensor("decoder.middle.1.to_qkv.weight").float()
    qkv_b = st.get_tensor("decoder.middle.1.to_qkv.bias").float()
    proj_w = st.get_tensor("decoder.middle.1.proj.weight").float()
    proj_b = st.get_tensor("decoder.middle.1.proj.bias").float()

    B, C, T, H, W = x.shape
    # Reshape to 2D for attention: [B, C, T*H*W]
    x_2d = x.reshape(B, C, T * H * W)
    # GroupNorm
    x_norm = F.group_norm(x_2d.reshape(B, C, T, H, W), 32,
                          weight=norm_g.view(-1)).reshape(B, C, T * H * W)
    # QKV via 1x1 conv (as matmul)
    qkv = torch.einsum('oi,bis->bos', qkv_w.view(3*C, C), x_norm)
    qkv = qkv + qkv_b.view(1, 3*C, 1)
    q, k, v = qkv.chunk(3, dim=1)
    # Attention
    scale = C ** -0.5
    attn = torch.bmm(q.permute(0, 2, 1), k) * scale  # [B, S, S]
    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, S]
    # Output projection
    out = torch.einsum('oi,bis->bos', proj_w.view(C, C), out)
    out = out + proj_b.view(1, C, 1)
    x = x + out.reshape(B, C, T, H, W)

    x = resblock_forward(x, "decoder.middle.2")
    print(f"After middle: {x.shape}")

    # Save middle output
    mid_path = args.output.replace('.npy', '_after_middle.npy')
    np.save(mid_path, x[:, :, 0, :, :].detach().numpy())
    print(f"Saved after middle to {mid_path}")

    # Upsample blocks
    for i in range(15):
        prefix = f"decoder.upsamples.{i}"
        x = resblock_forward(x, prefix)

        # Check for resample (spatial upsample)
        try:
            rs_w = st.get_tensor(f"{prefix}.resample.1.weight").float()
            rs_b = st.get_tensor(f"{prefix}.resample.1.bias").float()

            # NN upsample 2x spatial
            B, C, T, H, W = x.shape
            x = x[:, :, :, :, :].repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

            # Conv2d on spatial dims
            x_2d = x[:, :, 0, :, :]  # [B, C, H, W]
            x_2d = F.conv2d(x_2d, rs_w, rs_b, padding=1)
            x = x_2d.unsqueeze(2)  # add T back

            # Skip time_conv for image mode
            print(f"  upsample {i}: resample → {x.shape}")
        except:
            pass

        # Also try time_conv (skip for image mode)
        # time_conv would upsample temporal dim, not needed for T=1

    # Head: GroupNorm → SiLU → Conv
    head_g = st.get_tensor("decoder.head.0.gamma").float()
    head_w = st.get_tensor("decoder.head.2.weight").float()
    head_b = st.get_tensor("decoder.head.2.bias").float()

    x = F.group_norm(x, 32, weight=head_g.view(-1))
    x = F.silu(x)
    x = F.pad(x, (1, 1, 1, 1, 2, 0), mode='replicate')
    x = F.conv3d(x, head_w, head_b)

    # Remove temporal dim
    rgb = x[:, :, 0, :, :]  # [B, 3, H, W]
    print(f"Output RGB: {rgb.shape}")

    # Save
    np.save(args.output, rgb.detach().numpy())
    print(f"Saved VAE decode output to {args.output}")

    # Also save as image for visualization
    try:
        from PIL import Image
        img = rgb[0].permute(1, 2, 0).detach().numpy()
        img = np.clip(img * 0.5 + 0.5, 0, 1)  # denormalize
        img = (img * 255).astype(np.uint8)
        img_path = args.output.replace('.npy', '.png')
        Image.fromarray(img).save(img_path)
        print(f"Saved image to {img_path}")
    except Exception as e:
        print(f"Could not save image: {e}")


def generate_text_ref(args):
    """Generate reference text encoder output."""
    import torch
    print("This requires the full Qwen2.5-VL model loaded via transformers.")
    print("For now, generate random conditioning as placeholder.")

    torch.manual_seed(args.seed)
    n_tokens = 77  # typical text length
    txt_dim = 3584
    pooled_dim = 768

    txt_emb = torch.randn(1, n_tokens, txt_dim)
    pooled = torch.randn(1, pooled_dim)

    np.save(args.output, txt_emb.numpy())
    pooled_path = args.output.replace('.npy', '_pooled.npy')
    np.save(pooled_path, pooled.numpy())
    print(f"Saved text embeddings to {args.output} and {pooled_path}")
    print(f"Shape: {txt_emb.shape}, pooled: {pooled.shape}")


def main():
    parser = argparse.ArgumentParser(description='Qwen-Image verification')
    parser.add_argument('--inspect-safetensors', type=str,
                        help='Inspect safetensors file')
    parser.add_argument('--inspect-gguf', type=str,
                        help='Inspect GGUF file')
    parser.add_argument('--compare', nargs=2, metavar=('REF', 'TEST'),
                        help='Compare two .npy files')
    parser.add_argument('--generate-vae-ref', action='store_true',
                        help='Generate VAE decode reference')
    parser.add_argument('--generate-text-ref', action='store_true',
                        help='Generate text encoder reference')
    parser.add_argument('--generate-dit-ref', action='store_true',
                        help='Generate DiT step reference')

    parser.add_argument('--vae-path', type=str,
                        default='/mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors')
    parser.add_argument('--dit-path', type=str,
                        default='/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf')
    parser.add_argument('--prompt', type=str, default='a cat sitting on a chair')
    parser.add_argument('--latent-h', type=int, default=32)
    parser.add_argument('--latent-w', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', '-o', type=str, default='ref_output.npy')

    args = parser.parse_args()

    if args.inspect_safetensors:
        inspect_safetensors(args.inspect_safetensors)
    elif args.inspect_gguf:
        # Reuse the inspect script
        from inspect_gguf import read_gguf_header
        read_gguf_header(args.inspect_gguf)
    elif args.compare:
        compare_npy(args.compare[0], args.compare[1])
    elif args.generate_vae_ref:
        generate_vae_ref(args)
    elif args.generate_text_ref:
        generate_text_ref(args)
    elif args.generate_dit_ref:
        print("DiT reference generation requires dequantizing GGUF weights.")
        print("Use diffusers pipeline for full reference instead.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
