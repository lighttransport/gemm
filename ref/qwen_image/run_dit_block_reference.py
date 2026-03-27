#!/usr/bin/env python3
"""
Qwen-Image DiT block 0 full forward reference.

Dequantizes all block 0 weights and runs the complete dual-stream block:
  adaLN → QKV → QK-norm → joint attention → output proj → gated residual
  adaLN → FFN (GELU) → gated residual

Dumps intermediates at each step.

Usage:
    uv run python run_dit_block_reference.py \
        --dit-path /mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf \
        --output-dir output/
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---- GGUF Q4_0 dequantization (corrected nibble order) ----

def dequant_q4_0(data, n_elements):
    """Dequantize Q4_0 tensor. GGML: low nibbles → pos 0-15, high → 16-31."""
    n_blocks = n_elements // 32
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * 18:(i + 1) * 18]
        d = np.frombuffer(block[:2], dtype=np.float16)[0].astype(np.float32)
        quants = block[2:18]
        for j in range(16):
            byte = int(quants[j])
            v0 = (byte & 0x0F) - 8
            v1 = (byte >> 4) - 8
            result[i * 32 + j] = float(v0) * d
            result[i * 32 + j + 16] = float(v1) * d
    return result


def dequant_q4_1(data, n_elements):
    n_blocks = n_elements // 32
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * 20:(i + 1) * 20]
        d = np.frombuffer(block[:2], dtype=np.float16)[0].astype(np.float32)
        m = np.frombuffer(block[2:4], dtype=np.float16)[0].astype(np.float32)
        quants = block[4:20]
        for j in range(16):
            byte = int(quants[j])
            v0 = byte & 0x0F
            v1 = byte >> 4
            result[i * 32 + j] = float(v0) * d + m
            result[i * 32 + j + 16] = float(v1) * d + m
    return result


# ---- GGUF reader ----

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def skip_value(f, vtype):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype == 8:
        read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_type == 8:
            for _ in range(arr_len): read_string(f)
        else:
            f.read(arr_len * sizes.get(arr_type, 4))
    else:
        f.read(sizes.get(vtype, 4))


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.tensors = {}
        self._parse_header()

    def _parse_header(self):
        with open(self.path, "rb") as f:
            assert f.read(4) == b'GGUF'
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                skip_value(f, vtype)
            for _ in range(n_tensors):
                name = read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                self.tensors[name] = (dims, ttype, offset)
            self.data_start = f.tell()
            alignment = 32
            self.data_start = ((self.data_start + alignment - 1) // alignment) * alignment

    def load(self, name):
        """Load and dequantize a tensor. Returns [n_out, n_in] for 2D weights."""
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found")
        dims, ttype, offset = self.tensors[name]
        n_elements = 1
        for d in dims: n_elements *= d

        with open(self.path, "rb") as f:
            f.seek(self.data_start + offset)
            if ttype == 0:  # F32
                data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
            elif ttype == 30:  # BF16
                raw = np.frombuffer(f.read(n_elements * 2), dtype=np.uint16).copy()
                data = (raw.astype(np.uint32) << 16).view(np.float32)
            elif ttype == 2:  # Q4_0
                data = dequant_q4_0(f.read((n_elements // 32) * 18), n_elements)
            elif ttype == 3:  # Q4_1
                data = dequant_q4_1(f.read((n_elements // 32) * 20), n_elements)
            elif ttype in (31, 32, 33):  # Q4_0_x_x variants
                data = dequant_q4_0(f.read((n_elements // 32) * 18), n_elements)
            else:
                raise ValueError(f"Unsupported type {ttype}")

        if len(dims) == 2:
            data = data.reshape(dims[1], dims[0])  # GGML [cols, rows] → [rows, cols]
        return data


def save(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    np.save(path, arr)
    print(f"  saved {path}  shape={arr.shape}  "
          f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")


def rmsnorm_per_head(x, w, n_heads, head_dim, eps=1e-6):
    """Per-head RMSNorm: x [N, dim], w [head_dim]."""
    N = x.shape[0]
    x = x.reshape(N, n_heads, head_dim)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x = x / rms * torch.tensor(w, dtype=torch.float32).view(1, 1, head_dim)
    return x.reshape(N, n_heads * head_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dit-path', required=True)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gguf = GGUFReader(args.dit_path)

    n_img = 16; n_txt = 10; dim = 3072; n_heads = 24; head_dim = 128
    in_ch = 64; txt_dim = 3584; mlp_hidden = 12288
    eps = 1e-6

    # Load inputs from previous run
    img_in = np.load(f"{args.output_dir}/dit_img_input.npy")
    txt_in = np.load(f"{args.output_dir}/dit_txt_input.npy")
    t_emb = np.load(f"{args.output_dir}/dit_t_emb.npy")

    img = torch.tensor(np.load(f"{args.output_dir}/dit_img_projected.npy")).float()
    txt = torch.tensor(np.load(f"{args.output_dir}/dit_txt_projected.npy")).float()
    t_emb_t = torch.tensor(t_emb).float()

    print("=== Block 0 Forward ===")

    # ---- Modulation ----
    t_silu = F.silu(t_emb_t)

    img_mod_w = gguf.load("transformer_blocks.0.img_mod.1.weight")
    img_mod_b = gguf.load("transformer_blocks.0.img_mod.1.bias")
    img_mod = t_silu @ torch.tensor(img_mod_w).T + torch.tensor(img_mod_b)
    img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = \
        img_mod.reshape(6, dim)

    txt_mod_w = gguf.load("transformer_blocks.0.txt_mod.1.weight")
    txt_mod_b = gguf.load("transformer_blocks.0.txt_mod.1.bias")
    txt_mod = t_silu @ torch.tensor(txt_mod_w).T + torch.tensor(txt_mod_b)
    txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = \
        txt_mod.reshape(6, dim)

    # ---- adaLN + QKV ----
    # Image
    img_mean = img.mean(dim=-1, keepdim=True)
    img_var = img.var(dim=-1, keepdim=True, unbiased=False)
    img_normed = (img - img_mean) / torch.sqrt(img_var + eps)
    img_normed = img_normed * (1.0 + img_scale1) + img_shift1
    save(img_normed, f"{args.output_dir}/dit_blk0_img_adaln.npy")

    # Image QKV
    q_w = gguf.load("transformer_blocks.0.attn.to_q.weight")
    q_b = gguf.load("transformer_blocks.0.attn.to_q.bias")
    k_w = gguf.load("transformer_blocks.0.attn.to_k.weight")
    k_b = gguf.load("transformer_blocks.0.attn.to_k.bias")
    v_w = gguf.load("transformer_blocks.0.attn.to_v.weight")
    v_b = gguf.load("transformer_blocks.0.attn.to_v.bias")

    img_q = img_normed @ torch.tensor(q_w).T + torch.tensor(q_b)
    img_k = img_normed @ torch.tensor(k_w).T + torch.tensor(k_b)
    img_v = img_normed @ torch.tensor(v_w).T + torch.tensor(v_b)

    # Text adaLN + QKV
    txt_mean = txt.mean(dim=-1, keepdim=True)
    txt_var = txt.var(dim=-1, keepdim=True, unbiased=False)
    txt_normed = (txt - txt_mean) / torch.sqrt(txt_var + eps)
    txt_normed = txt_normed * (1.0 + txt_scale1) + txt_shift1

    add_q_w = gguf.load("transformer_blocks.0.attn.add_q_proj.weight")
    add_q_b = gguf.load("transformer_blocks.0.attn.add_q_proj.bias")
    add_k_w = gguf.load("transformer_blocks.0.attn.add_k_proj.weight")
    add_k_b = gguf.load("transformer_blocks.0.attn.add_k_proj.bias")
    add_v_w = gguf.load("transformer_blocks.0.attn.add_v_proj.weight")
    add_v_b = gguf.load("transformer_blocks.0.attn.add_v_proj.bias")

    txt_q = txt_normed @ torch.tensor(add_q_w).T + torch.tensor(add_q_b)
    txt_k = txt_normed @ torch.tensor(add_k_w).T + torch.tensor(add_k_b)
    txt_v = txt_normed @ torch.tensor(add_v_w).T + torch.tensor(add_v_b)

    # ---- QK RMSNorm ----
    norm_q_w = gguf.load("transformer_blocks.0.attn.norm_q.weight")
    norm_k_w = gguf.load("transformer_blocks.0.attn.norm_k.weight")
    norm_aq_w = gguf.load("transformer_blocks.0.attn.norm_added_q.weight")
    norm_ak_w = gguf.load("transformer_blocks.0.attn.norm_added_k.weight")

    img_q = rmsnorm_per_head(img_q, norm_q_w, n_heads, head_dim, eps)
    img_k = rmsnorm_per_head(img_k, norm_k_w, n_heads, head_dim, eps)
    txt_q = rmsnorm_per_head(txt_q, norm_aq_w, n_heads, head_dim, eps)
    txt_k = rmsnorm_per_head(txt_k, norm_ak_w, n_heads, head_dim, eps)

    save(img_q, f"{args.output_dir}/dit_blk0_img_q_normed.npy")

    # ---- RoPE ----
    # 2D RoPE for image (height, width), 1D for text
    # axes_dims = [16, 56, 56]: temporal=16, height=56, width=56
    rope_theta = 10000.0
    axes_dims = [16, 56, 56]
    t_dim, h_dim, w_dim = axes_dims

    # Image RoPE: 2D (height + width)
    hp = int(n_img ** 0.5)  # assume square
    wp = n_img // hp
    for tok in range(n_img):
        ph = tok // wp
        pw = tok % wp
        for h in range(n_heads):
            off = h * head_dim
            # Height RoPE at offset t_dim
            for i in range(h_dim // 2):
                freq = 1.0 / (rope_theta ** (2.0 * i / h_dim))
                angle = ph * freq
                cs, sn = np.cos(angle), np.sin(angle)
                idx = t_dim + 2 * i
                for arr in [img_q, img_k]:
                    v0 = arr[tok, off + idx].item()
                    v1 = arr[tok, off + idx + 1].item()
                    arr[tok, off + idx] = v0 * cs - v1 * sn
                    arr[tok, off + idx + 1] = v0 * sn + v1 * cs
            # Width RoPE at offset t_dim + h_dim
            for i in range(w_dim // 2):
                freq = 1.0 / (rope_theta ** (2.0 * i / w_dim))
                angle = pw * freq
                cs, sn = np.cos(angle), np.sin(angle)
                idx = t_dim + h_dim + 2 * i
                for arr in [img_q, img_k]:
                    v0 = arr[tok, off + idx].item()
                    v1 = arr[tok, off + idx + 1].item()
                    arr[tok, off + idx] = v0 * cs - v1 * sn
                    arr[tok, off + idx + 1] = v0 * sn + v1 * cs

    # Text RoPE: 1D
    for tok in range(n_txt):
        for h in range(n_heads):
            off = h * head_dim
            for i in range(head_dim // 2):
                freq = 1.0 / (rope_theta ** (2.0 * i / head_dim))
                angle = tok * freq
                cs, sn = np.cos(angle), np.sin(angle)
                for arr in [txt_q, txt_k]:
                    v0 = arr[tok, off + 2*i].item()
                    v1 = arr[tok, off + 2*i + 1].item()
                    arr[tok, off + 2*i] = v0 * cs - v1 * sn
                    arr[tok, off + 2*i + 1] = v0 * sn + v1 * cs

    # ---- Joint Attention ----
    q = torch.cat([txt_q, img_q], dim=0)  # [T+S, dim]
    k = torch.cat([txt_k, img_k], dim=0)
    v = torch.cat([txt_v, img_v], dim=0)

    # Reshape for multi-head: [N, H, D]
    N_total = q.shape[0]
    q_mh = q.reshape(N_total, n_heads, head_dim)
    k_mh = k.reshape(N_total, n_heads, head_dim)
    v_mh = v.reshape(N_total, n_heads, head_dim)

    # Attention: [H, N, D] format for bmm
    q_mh = q_mh.permute(1, 0, 2)  # [H, N, D]
    k_mh = k_mh.permute(1, 0, 2)
    v_mh = v_mh.permute(1, 0, 2)

    scale = head_dim ** -0.5
    attn_w = torch.bmm(q_mh, k_mh.transpose(1, 2)) * scale  # [H, N, N]
    attn_w = torch.softmax(attn_w, dim=-1)
    attn_out = torch.bmm(attn_w, v_mh)  # [H, N, D]
    attn_out = attn_out.permute(1, 0, 2).reshape(N_total, dim)  # [N, dim]

    # Split
    txt_attn = attn_out[:n_txt]
    img_attn = attn_out[n_txt:]

    # Output projections
    out_w = gguf.load("transformer_blocks.0.attn.to_out.0.weight")
    out_b = gguf.load("transformer_blocks.0.attn.to_out.0.bias")
    img_attn_proj = img_attn @ torch.tensor(out_w).T + torch.tensor(out_b)

    add_out_w = gguf.load("transformer_blocks.0.attn.to_add_out.weight")
    add_out_b = gguf.load("transformer_blocks.0.attn.to_add_out.bias")
    txt_attn_proj = txt_attn @ torch.tensor(add_out_w).T + torch.tensor(add_out_b)

    # Gated residual
    img = img + img_gate1 * img_attn_proj
    txt = txt + txt_gate1 * txt_attn_proj
    save(img, f"{args.output_dir}/dit_blk0_img_after_attn.npy")
    save(txt, f"{args.output_dir}/dit_blk0_txt_after_attn.npy")

    # ---- MLP with adaLN ----
    # Image MLP
    img_mean2 = img.mean(dim=-1, keepdim=True)
    img_var2 = img.var(dim=-1, keepdim=True, unbiased=False)
    img_n2 = (img - img_mean2) / torch.sqrt(img_var2 + eps)
    img_n2 = img_n2 * (1.0 + img_scale2) + img_shift2

    mlp_up_w = gguf.load("transformer_blocks.0.img_mlp.net.0.proj.weight")
    mlp_up_b = gguf.load("transformer_blocks.0.img_mlp.net.0.proj.bias")
    mlp_dn_w = gguf.load("transformer_blocks.0.img_mlp.net.2.weight")
    mlp_dn_b = gguf.load("transformer_blocks.0.img_mlp.net.2.bias")

    img_ffn = img_n2 @ torch.tensor(mlp_up_w).T + torch.tensor(mlp_up_b)
    img_ffn = F.gelu(img_ffn, approximate='tanh')
    img_ffn = img_ffn @ torch.tensor(mlp_dn_w).T + torch.tensor(mlp_dn_b)
    img = img + img_gate2 * img_ffn

    # Text MLP
    txt_mean2 = txt.mean(dim=-1, keepdim=True)
    txt_var2 = txt.var(dim=-1, keepdim=True, unbiased=False)
    txt_n2 = (txt - txt_mean2) / torch.sqrt(txt_var2 + eps)
    txt_n2 = txt_n2 * (1.0 + txt_scale2) + txt_shift2

    txt_up_w = gguf.load("transformer_blocks.0.txt_mlp.net.0.proj.weight")
    txt_up_b = gguf.load("transformer_blocks.0.txt_mlp.net.0.proj.bias")
    txt_dn_w = gguf.load("transformer_blocks.0.txt_mlp.net.2.weight")
    txt_dn_b = gguf.load("transformer_blocks.0.txt_mlp.net.2.bias")

    txt_ffn = txt_n2 @ torch.tensor(txt_up_w).T + torch.tensor(txt_up_b)
    txt_ffn = F.gelu(txt_ffn, approximate='tanh')
    txt_ffn = txt_ffn @ torch.tensor(txt_dn_w).T + torch.tensor(txt_dn_b)
    txt = txt + txt_gate2 * txt_ffn

    save(img, f"{args.output_dir}/dit_blk0_img_after_mlp.npy")
    save(txt, f"{args.output_dir}/dit_blk0_txt_after_mlp.npy")

    print("Done. Block 0 complete.")


if __name__ == "__main__":
    main()
