#!/usr/bin/env python3
"""Trace DiT forward pass step-by-step using raw FP8 weights, comparing with C output."""
import struct, math, json
import numpy as np

def fp8_e4m3_to_f32(b):
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0: return 0.0
    if e == 15 and m == 7: return 0.0  # NaN → 0
    val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -val if s else val

def load_st_f32(st_data, st_hdr, name):
    """Load a tensor from safetensors as F32."""
    t = st_hdr[name]
    shape = t['shape']
    offsets = t['data_offsets']
    dtype = t['dtype']
    start = offsets[0]
    n = 1
    for s in shape: n *= s

    raw = st_data[start:start + (offsets[1] - offsets[0])]

    if dtype in ('F8_E4M3', 'F8_E4M3FN'):
        return np.array([fp8_e4m3_to_f32(b) for b in raw], dtype=np.float32).reshape(shape)
    elif dtype == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32).reshape(shape)
    elif dtype == 'F32':
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def our_prng_noise(seed, n):
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        vals[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return vals

def main():
    dit_path = '/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors'

    # Load safetensors
    print("Loading FP8 safetensors...")
    with open(dit_path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_offset = 8 + hdr_len
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        st_data_raw = f.read()
    st_data = st_data_raw[data_offset:]
    # Fix offsets relative to data start
    st_hdr = {k: v for k, v in hdr.items() if k != '__metadata__'}

    # Generate noise [16, 32, 32] and patchify
    noise = our_prng_noise(42, 16 * 32 * 32).reshape(16, 32, 32)
    ps = 2
    hp, wp = 16, 16
    n_img = 256
    tok_dim = 64  # 16 * 2 * 2

    tokens = np.zeros((n_img, tok_dim), dtype=np.float32)
    for py in range(hp):
        for px in range(wp):
            tok = py * wp + px
            idx = 0
            for c in range(16):
                for dy in range(ps):
                    for dx in range(ps):
                        tokens[tok, idx] = noise[c, py*ps+dy, px*ps+dx]
                        idx += 1
    print(f"Patchified tokens: {tokens.shape}, max={np.abs(tokens).max():.4f}")

    # 1. img_in projection: Linear(64 → 3072)
    img_in_w = load_st_f32(st_data, st_hdr, 'img_in.weight')  # [3072, 64]
    img_in_b = load_st_f32(st_data, st_hdr, 'img_in.bias')    # [3072]
    print(f"img_in weight: {img_in_w.shape}, dtype_in_file={st_hdr['img_in.weight']['dtype']}")
    print(f"  weight max={np.abs(img_in_w).max():.4f}, bias max={np.abs(img_in_b).max():.4f}")

    # Y = X @ W^T + b  → [n_img, 3072]
    img_projected = tokens @ img_in_w.T + img_in_b
    print(f"img_projected: max={np.abs(img_projected).max():.4f}")
    print(f"  C code reports: img_input max=20.0128")

    # 2. Timestep embedding for t=1000 (dynamic scheduler)
    t = 1000.0
    half = 128
    t_sin = np.zeros(256, dtype=np.float32)
    for i in range(half):
        freq = math.exp(-i / half * math.log(10000.0))
        angle = t * freq
        t_sin[i] = math.cos(angle)
        t_sin[half + i] = math.sin(angle)

    # t_emb = SiLU(FC1(t_sin)) then FC2
    t_fc1_w = load_st_f32(st_data, st_hdr, 'time_text_embed.timestep_embedder.linear_1.weight')  # [3072, 256]
    t_fc1_b = load_st_f32(st_data, st_hdr, 'time_text_embed.timestep_embedder.linear_1.bias')
    t_fc2_w = load_st_f32(st_data, st_hdr, 'time_text_embed.timestep_embedder.linear_2.weight')  # [3072, 3072]
    t_fc2_b = load_st_f32(st_data, st_hdr, 'time_text_embed.timestep_embedder.linear_2.bias')
    print(f"\nt_fc1_w: {t_fc1_w.shape}, max={np.abs(t_fc1_w).max():.4f}")

    t_emb = t_sin @ t_fc1_w.T + t_fc1_b
    t_emb = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))  # SiLU
    t_emb = t_emb @ t_fc2_w.T + t_fc2_b
    print(f"t_emb: max={np.abs(t_emb).max():.4f}")

    # 3. Block 0 modulation: SiLU(t_emb) → Linear(3072 → 18432)
    t_silu = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))
    mod_w = load_st_f32(st_data, st_hdr, 'transformer_blocks.0.img_mod.1.weight')  # [18432, 3072]
    mod_b = load_st_f32(st_data, st_hdr, 'transformer_blocks.0.img_mod.1.bias')
    mod = t_silu @ mod_w.T + mod_b  # [18432]

    shift1 = mod[0:3072]
    scale1 = mod[3072:6144]
    gate1 = mod[6144:9216]
    print(f"\nBlock 0 modulation:")
    print(f"  shift1 max={np.abs(shift1).max():.4f}  (C: 25.8201)")
    print(f"  scale1 max={np.abs(scale1).max():.4f}  (C: 29.0304)")
    print(f"  gate1  max={np.abs(gate1).max():.4f}  (C: 7.3061)")

    # 4. adaLN: LN(img) * (1+scale) + shift
    # LayerNorm (no affine)
    img = img_projected
    mean = img.mean(axis=1, keepdims=True)
    var = ((img - mean) ** 2).mean(axis=1, keepdims=True)
    img_ln = (img - mean) / np.sqrt(var + 1e-6)
    img_adaln = img_ln * (1.0 + scale1) + shift1
    print(f"  adaln max={np.abs(img_adaln).max():.4f}  (C: 73.2786)")

    print("\n=== Summary ===")
    print(f"img_projected max: Python={np.abs(img_projected).max():.4f}  C=20.0128")
    print(f"mod shift1 max:    Python={np.abs(shift1).max():.4f}  C=25.8201")
    print(f"mod scale1 max:    Python={np.abs(scale1).max():.4f}  C=29.0304")
    print(f"mod gate1 max:     Python={np.abs(gate1).max():.4f}  C=7.3061")
    print(f"adaln max:         Python={np.abs(img_adaln).max():.4f}  C=73.2786")

if __name__ == "__main__":
    main()
