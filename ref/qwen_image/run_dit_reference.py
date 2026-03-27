#!/usr/bin/env python3
"""
Qwen-Image DiT single-block PyTorch reference using GGUF weights.

Dequantizes Q4_0 weights from GGUF and runs the first block's operations
step-by-step, dumping intermediates for comparison.

Usage:
    uv run python run_dit_reference.py \
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


# ---- GGUF Q4_0 dequantization ----

def dequant_q4_0_block(data):
    """Dequantize a single Q4_0 block (32 elements from 18 bytes).
    GGML layout: low nibbles → positions 0-15, high nibbles → positions 16-31."""
    d = np.frombuffer(data[:2], dtype=np.float16)[0].astype(np.float32)
    quants = np.frombuffer(data[2:18], dtype=np.uint8)
    result = np.zeros(32, dtype=np.float32)
    for j in range(16):
        byte = int(quants[j])
        v0 = (byte & 0x0F) - 8   # low nibble
        v1 = (byte >> 4) - 8     # high nibble
        result[j]      = float(v0) * d   # low nibble → pos j (0-15)
        result[j + 16] = float(v1) * d   # high nibble → pos j+16 (16-31)
    return result


def dequant_q4_0(data, n_elements):
    """Dequantize Q4_0 tensor."""
    n_blocks = n_elements // 32
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block_data = data[i * 18:(i + 1) * 18]
        result[i * 32:(i + 1) * 32] = dequant_q4_0_block(block_data)
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
            for _ in range(arr_len):
                read_string(f)
        else:
            f.read(arr_len * sizes.get(arr_type, 4))
    else:
        f.read(sizes.get(vtype, 4))


def load_gguf_tensor(path, tensor_name):
    """Load and dequantize a single tensor from GGUF file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b'GGUF'
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Skip KV pairs
        for _ in range(n_kv):
            read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            skip_value(f, vtype)

        # Read tensor infos
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = (dims, ttype, offset)

        if tensor_name not in tensors:
            print(f"Tensor '{tensor_name}' not found!")
            return None, None

        dims, ttype, offset = tensors[tensor_name]

        # Compute data section start (aligned)
        data_start = f.tell()
        alignment = 32
        data_start = ((data_start + alignment - 1) // alignment) * alignment

        # Read tensor data
        f.seek(data_start + offset)

        # Total elements
        n_elements = 1
        for d in dims:
            n_elements *= d

        if ttype == 0:  # F32
            data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
        elif ttype == 1:  # F16
            data = np.frombuffer(f.read(n_elements * 2), dtype=np.float16).astype(np.float32)
        elif ttype == 30:  # BF16
            raw = np.frombuffer(f.read(n_elements * 2), dtype=np.uint16).copy()
            # BF16 to F32: shift left 16 bits
            f32_int = raw.astype(np.uint32) << 16
            data = f32_int.view(np.float32)
        elif ttype == 2:  # Q4_0
            n_bytes = (n_elements // 32) * 18
            raw = f.read(n_bytes)
            data = dequant_q4_0(raw, n_elements)
        elif ttype == 3:  # Q4_1
            block_size = 32
            type_size = 20
            n_blocks = n_elements // block_size
            raw = f.read(n_blocks * type_size)
            # Q4_1: min + d + 16 bytes quants
            data = np.zeros(n_elements, dtype=np.float32)
            for i in range(n_blocks):
                blk = raw[i * type_size:(i + 1) * type_size]
                d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
                m = np.frombuffer(blk[2:4], dtype=np.float16)[0].astype(np.float32)
                quants = np.frombuffer(blk[4:20], dtype=np.uint8)
                for j in range(16):
                    lo = int(quants[j]) & 0x0F
                    hi = int(quants[j]) >> 4
                    data[i * 32 + 2 * j] = float(lo) * d + m
                    data[i * 32 + 2 * j + 1] = float(hi) * d + m
        elif ttype in (31, 32, 33):  # Q4_0_4_4, Q4_0_4_8, Q4_0_8_8
            # Same quantization as Q4_0 but reordered for SIMD
            # For dequantization, treat identically to Q4_0
            n_bytes = (n_elements // 32) * 18
            raw = f.read(n_bytes)
            data = dequant_q4_0(raw, n_elements)
        else:
            print(f"Unsupported type {ttype} for {tensor_name}")
            return None, dims

        # GGML shape: dims[0]=cols, dims[1]=rows
        # Reshape to [rows, cols] for weight matrices
        if len(dims) == 2:
            data = data.reshape(dims[1], dims[0])  # [n_out, n_in]
        elif len(dims) == 1:
            pass  # keep flat

        return data, dims


def save(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    np.save(path, arr)
    print(f"  saved {path}  shape={arr.shape}  "
          f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dit-path', required=True)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gguf = args.dit_path

    # Generate deterministic small test input matching C code
    n_img = 16  # 4x4 patches (from 8x8 latent, ps=2)
    n_txt = 10
    dim = 3072
    in_ch = 64
    txt_dim = 3584

    np.random.seed(42)
    img_tok = (np.random.randn(n_img, in_ch) * 0.1).astype(np.float32)
    txt_tok = (np.random.randn(n_txt, txt_dim) * 0.1).astype(np.float32)
    save(img_tok, f"{args.output_dir}/dit_img_input.npy")
    save(txt_tok, f"{args.output_dir}/dit_txt_input.npy")

    timestep = 500.0

    print("Loading timestep embedder weights...")

    # Timestep embedding
    t_fc1_w, _ = load_gguf_tensor(gguf, "time_text_embed.timestep_embedder.linear_1.weight")
    t_fc1_b, _ = load_gguf_tensor(gguf, "time_text_embed.timestep_embedder.linear_1.bias")
    t_fc2_w, _ = load_gguf_tensor(gguf, "time_text_embed.timestep_embedder.linear_2.weight")
    t_fc2_b, _ = load_gguf_tensor(gguf, "time_text_embed.timestep_embedder.linear_2.bias")

    # Sinusoidal embedding
    half = 128
    freqs = np.exp(-np.arange(half, dtype=np.float32) / half * np.log(10000.0))
    angles = timestep * freqs
    t_sin = np.concatenate([np.sin(angles), np.cos(angles)])
    t_emb = t_sin @ t_fc1_w.T + t_fc1_b.flatten()
    t_emb = t_emb / (1.0 + np.exp(-t_emb))  # SiLU
    t_emb = t_emb @ t_fc2_w.T + t_fc2_b.flatten()
    save(t_emb, f"{args.output_dir}/dit_t_emb.npy")

    # Text input: RMSNorm -> Linear
    print("Loading text input weights...")
    txt_norm_w, _ = load_gguf_tensor(gguf, "txt_norm.weight")
    txt_in_w, _ = load_gguf_tensor(gguf, "txt_in.weight")
    txt_in_b, _ = load_gguf_tensor(gguf, "txt_in.bias")

    txt = txt_tok.copy()
    # RMSNorm
    rms = np.sqrt(np.mean(txt ** 2, axis=-1, keepdims=True) + 1e-6)
    txt = txt / rms * txt_norm_w.flatten()
    # Linear
    txt = txt @ txt_in_w.T + txt_in_b.flatten()
    save(txt, f"{args.output_dir}/dit_txt_projected.npy")

    # Image input: Linear
    print("Loading image input weights...")
    img_in_w, _ = load_gguf_tensor(gguf, "img_in.weight")
    img_in_b, _ = load_gguf_tensor(gguf, "img_in.bias")
    img = img_tok @ img_in_w.T + img_in_b.flatten()
    save(img, f"{args.output_dir}/dit_img_projected.npy")

    # Block 0: modulation
    print("Loading block 0 weights...")
    blk0_img_mod_w, _ = load_gguf_tensor(gguf, "transformer_blocks.0.img_mod.1.weight")
    blk0_img_mod_b, _ = load_gguf_tensor(gguf, "transformer_blocks.0.img_mod.1.bias")

    t_silu = t_emb.copy()
    t_silu = t_silu / (1.0 + np.exp(-t_silu))
    img_mod = t_silu @ blk0_img_mod_w.T + blk0_img_mod_b.flatten()
    save(img_mod, f"{args.output_dir}/dit_blk0_img_mod.npy")

    print("Done. Compare with: uv run python compare.py --ref-dir output/ --ours-dir ../../cpu/qwen_image/")


if __name__ == "__main__":
    main()
