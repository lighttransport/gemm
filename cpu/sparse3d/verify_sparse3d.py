#!/usr/bin/env python3
"""
verify_sparse3d.py - End-to-end verification of sparse3d.h against PyTorch reference

Usage:
  python verify_sparse3d.py --generate --output ref_output/
  ./test_sparse3d --export c_output/ --input ref_output/
  python verify_sparse3d.py --compare ref_output/ c_output/

  # Or all-in-one:
  python verify_sparse3d.py --all
"""

import argparse
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save(directory, name, arr):
    """Save numpy array to .npy file."""
    np.save(os.path.join(directory, name + ".npy"), arr)


def load(directory, name):
    """Load numpy array from .npy file."""
    return np.load(os.path.join(directory, name + ".npy"))


def rand_float(rng, shape, lo=-0.5, hi=0.5):
    """Generate random float32 in [lo, hi]."""
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def rand_coords_unique(rng, N, grid_size, batch_size=1):
    """Generate N unique random voxel coordinates.
    Returns int32 array of shape [N, 4] = (batch, z, y, x).
    """
    coords_set = set()
    coords = np.zeros((N, 4), dtype=np.int32)
    i = 0
    while i < N:
        b = int(rng.integers(0, batch_size))
        z = int(rng.integers(0, grid_size))
        y = int(rng.integers(0, grid_size))
        x = int(rng.integers(0, grid_size))
        key = (b, z, y, x)
        if key not in coords_set:
            coords_set.add(key)
            coords[i] = [b, z, y, x]
            i += 1
    # Sort by batch index for proper batch_starts computation
    order = np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def compute_batch_starts(coords, batch_size):
    """Compute CSR-style batch boundaries from sorted coords."""
    starts = np.zeros(batch_size + 1, dtype=np.int32)
    for i in range(len(coords)):
        b = coords[i, 0]
        starts[b + 1:] = i + 1
    starts[batch_size] = len(coords)
    # Fix: ensure monotonic
    for b in range(batch_size):
        if starts[b + 1] < starts[b]:
            starts[b + 1] = starts[b]
    return starts


# ============================================================
# PyTorch Reference Implementations
# ============================================================

def ref_linear(src, weight, bias):
    """Y = X @ W.T + bias. weight: [out_C, in_C], bias: [out_C] or None."""
    Y = src @ weight.T
    if bias is not None:
        Y = Y + bias
    return Y


def ref_layernorm(src, weight, bias, eps=1e-6):
    """Per-token layer norm."""
    x = torch.from_numpy(src)
    w = torch.from_numpy(weight)
    b = torch.from_numpy(bias)
    C = x.shape[-1]
    out = F.layer_norm(x, [C], w, b, eps=eps)
    return out.numpy()


def ref_gelu(x):
    """GELU with tanh approximation."""
    return F.gelu(torch.from_numpy(x), approximate='tanh').numpy()


def ref_silu(x):
    """SiLU = x * sigmoid(x)."""
    return F.silu(torch.from_numpy(x)).numpy()


def ref_conv3d(coords, feats, weight, bias, in_C, out_C):
    """Sparse submanifold convolution with 3x3x3 kernel.

    weight: [out_C, 27, in_C]
    bias: [out_C] or None
    coords: [N, 4] = (batch, z, y, x)
    feats: [N, in_C]
    """
    N = len(coords)
    # Build hash: (batch, z, y, x) -> index
    coord_hash = {}
    for i in range(N):
        key = tuple(coords[i].tolist())
        coord_hash[key] = i

    # 3x3x3 offsets
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                offsets.append((dz, dy, dx))

    out = np.zeros((N, out_C), dtype=np.float32)
    if bias is not None:
        out[:] = bias[np.newaxis, :]

    for i in range(N):
        b, z, y, x = coords[i]
        for k, (dz, dy, dx) in enumerate(offsets):
            nkey = (int(b), int(z + dz), int(y + dy), int(x + dx))
            j = coord_hash.get(nkey, -1)
            if j < 0:
                continue
            # out[i] += W[k] @ feats[j]
            # W layout: [out_C, 27, in_C] -> W[:, k, :] is [out_C, in_C]
            w_k = weight[:, k, :]  # [out_C, in_C]
            out[i] += w_k @ feats[j]

    return out


def ref_attention(qkv, coords, batch_starts, n_heads, head_dim, batch_size):
    """Per-batch scaled dot-product attention.

    qkv: [N, 3*dim] where dim = n_heads * head_dim
    Returns: [N, dim]
    """
    dim = n_heads * head_dim
    N = len(coords)
    out = np.zeros((N, dim), dtype=np.float32)

    for b in range(batch_size):
        start = batch_starts[b]
        end = batch_starts[b + 1]
        seq_len = end - start
        if seq_len <= 0:
            continue

        batch_qkv = qkv[start:end]  # [seq_len, 3*dim]
        Q = batch_qkv[:, :dim]           # [seq_len, dim]
        K = batch_qkv[:, dim:2*dim]      # [seq_len, dim]
        V = batch_qkv[:, 2*dim:3*dim]    # [seq_len, dim]

        # Reshape to [n_heads, seq_len, head_dim]
        Q_h = Q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        K_h = K.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        V_h = V.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        scale = 1.0 / np.sqrt(head_dim)
        for h in range(n_heads):
            scores = Q_h[h] @ K_h[h].T * scale  # [seq_len, seq_len]
            # Softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            scores = np.exp(scores - scores_max)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            attn_out = scores @ V_h[h]  # [seq_len, head_dim]
            out[start:end, h*head_dim:(h+1)*head_dim] = attn_out

    return out


def ref_rope_3d(qk, coords, n_heads, head_dim, rope_freqs, n_freqs):
    """3D RoPE for sparse coordinates.

    qk: [N, n_heads * head_dim]
    coords: [N, 4] = (batch, z, y, x)
    rope_freqs: [n_freqs]

    Split head_dim into 3 axes (z, y, x), each with axis_dim = 2*n_freqs.
    For each axis, rotate pairs (v[j], v[j+n_freqs]) by theta = coord * freq[j].
    """
    out = qk.copy()
    axis_dim = 2 * n_freqs
    N = len(coords)

    for i in range(N):
        cz = float(coords[i, 1])
        cy = float(coords[i, 2])
        cx = float(coords[i, 3])
        coord_vals = [cz, cy, cx]

        for h in range(n_heads):
            v = out[i, h * head_dim:(h + 1) * head_dim]

            for axis in range(3):
                coord = coord_vals[axis]
                base = axis * axis_dim

                for j in range(n_freqs):
                    theta = coord * rope_freqs[j]
                    cs = np.cos(theta)
                    sn = np.sin(theta)

                    idx0 = base + j
                    idx1 = base + j + n_freqs
                    if idx1 >= head_dim:
                        break

                    v0 = v[idx0]
                    v1 = v[idx1]
                    v[idx0] = v0 * cs - v1 * sn
                    v[idx1] = v0 * sn + v1 * cs

    return out


def ref_downsample(coords, feats, factor):
    """Downsample by floor division, mean pooling.

    Returns (out_coords, out_feats) sorted by unique downsampled coords.
    """
    N, C = feats.shape
    new_coords = coords.copy()
    new_coords[:, 1:] = coords[:, 1:] // factor

    # Find unique coords and their inverse mapping
    # Use structured array for uniqueness
    coord_tuples = [tuple(c) for c in new_coords]
    unique_map = {}
    order = []
    mapping = np.zeros(N, dtype=np.int32)

    for i, ct in enumerate(coord_tuples):
        if ct not in unique_map:
            unique_map[ct] = len(order)
            order.append(ct)
        mapping[i] = unique_map[ct]

    out_N = len(order)
    out_coords = np.array(order, dtype=np.int32)
    out_feats = np.zeros((out_N, C), dtype=np.float32)
    counts = np.zeros(out_N, dtype=np.int32)

    for i in range(N):
        oi = mapping[i]
        out_feats[oi] += feats[i]
        counts[oi] += 1

    for i in range(out_N):
        if counts[i] > 1:
            out_feats[i] /= counts[i]

    return out_coords, out_feats, counts


def ref_upsample(src_coords, src_feats, factor, target_coords):
    """Nearest-neighbor upsample: target_feats[i] = src_feats[lookup(target_coord // factor)]."""
    N_tgt = len(target_coords)
    C = src_feats.shape[1]

    # Build source hash
    src_hash = {}
    for i in range(len(src_coords)):
        key = tuple(src_coords[i].tolist())
        src_hash[key] = i

    out_feats = np.zeros((N_tgt, C), dtype=np.float32)
    for i in range(N_tgt):
        b = int(target_coords[i, 0])
        sz = int(target_coords[i, 1]) // factor
        sy = int(target_coords[i, 2]) // factor
        sx = int(target_coords[i, 3]) // factor
        j = src_hash.get((b, sz, sy, sx), -1)
        if j >= 0:
            out_feats[i] = src_feats[j]

    return out_feats


# ============================================================
# Generate test data + reference outputs
# ============================================================

def generate_all(output_dir):
    """Generate deterministic test inputs and PyTorch reference outputs."""
    if not HAS_TORCH:
        print("ERROR: PyTorch required for --generate. Install with: pip install torch")
        sys.exit(1)

    ensure_dir(output_dir)
    rng = np.random.default_rng(42)

    # --- 1. Linear ---
    N_lin, in_C, out_C = 100, 32, 64
    lin_input = rand_float(rng, (N_lin, in_C))
    lin_weight = rand_float(rng, (out_C, in_C), -0.1, 0.1)
    lin_bias = rand_float(rng, (out_C,), -0.1, 0.1)
    lin_ref = ref_linear(lin_input, lin_weight, lin_bias)
    save(output_dir, "linear_input", lin_input)
    save(output_dir, "linear_weight", lin_weight)
    save(output_dir, "linear_bias", lin_bias)
    save(output_dir, "linear_ref", lin_ref)
    print(f"  linear: input {lin_input.shape}, weight {lin_weight.shape}, output {lin_ref.shape}")

    # --- 2. LayerNorm ---
    N_ln, C_ln = 100, 64
    ln_input = rand_float(rng, (N_ln, C_ln))
    ln_weight = rand_float(rng, (C_ln,), 0.5, 1.5)
    ln_bias = rand_float(rng, (C_ln,), -0.1, 0.1)
    ln_ref = ref_layernorm(ln_input, ln_weight, ln_bias, eps=1e-6)
    save(output_dir, "layernorm_input", ln_input)
    save(output_dir, "layernorm_weight", ln_weight)
    save(output_dir, "layernorm_bias", ln_bias)
    save(output_dir, "layernorm_ref", ln_ref)
    print(f"  layernorm: input {ln_input.shape}, output {ln_ref.shape}")

    # --- 3. GELU / SiLU ---
    act_input = rand_float(rng, (1000,), -3.0, 3.0)
    gelu_ref = ref_gelu(act_input.copy())
    silu_ref = ref_silu(act_input.copy())
    save(output_dir, "activation_input", act_input)
    save(output_dir, "gelu_ref", gelu_ref)
    save(output_dir, "silu_ref", silu_ref)
    print(f"  gelu/silu: input {act_input.shape}")

    # --- 4. Conv3d ---
    N_conv, grid_conv, in_C_conv, out_C_conv = 500, 16, 16, 16
    conv_coords = rand_coords_unique(rng, N_conv, grid_conv, batch_size=1)
    conv_feats = rand_float(rng, (N_conv, in_C_conv))
    conv_weight = rand_float(rng, (out_C_conv, 27, in_C_conv), -0.1, 0.1)
    conv_bias = rand_float(rng, (out_C_conv,), -0.1, 0.1)
    conv_ref = ref_conv3d(conv_coords, conv_feats, conv_weight, conv_bias,
                          in_C_conv, out_C_conv)
    save(output_dir, "conv3d_coords", conv_coords)
    save(output_dir, "conv3d_input", conv_feats)
    save(output_dir, "conv3d_weight", conv_weight)
    save(output_dir, "conv3d_bias", conv_bias)
    save(output_dir, "conv3d_ref", conv_ref)
    print(f"  conv3d: N={N_conv}, in_C={in_C_conv}, out_C={out_C_conv}")

    # --- 5. Attention ---
    # 2 batches: 100 + 150 voxels
    N_attn_b0, N_attn_b1 = 100, 150
    N_attn = N_attn_b0 + N_attn_b1
    n_heads_attn, head_dim_attn = 2, 64
    dim_attn = n_heads_attn * head_dim_attn  # 128

    attn_coords = np.zeros((N_attn, 4), dtype=np.int32)
    # Batch 0: N_attn_b0 unique voxels
    for i in range(N_attn_b0):
        attn_coords[i] = [0, i // 100, (i // 10) % 10, i % 10]
    # Batch 1: N_attn_b1 unique voxels
    for i in range(N_attn_b1):
        attn_coords[N_attn_b0 + i] = [1, i // 100, (i // 10) % 10, i % 10]

    attn_qkv = rand_float(rng, (N_attn, 3 * dim_attn), -0.1, 0.1)
    batch_starts = np.array([0, N_attn_b0, N_attn], dtype=np.int32)
    attn_ref = ref_attention(attn_qkv, attn_coords, batch_starts,
                             n_heads_attn, head_dim_attn, 2)
    save(output_dir, "attention_coords", attn_coords)
    save(output_dir, "attention_qkv", attn_qkv)
    save(output_dir, "attention_ref", attn_ref)
    print(f"  attention: N={N_attn} ({N_attn_b0}+{N_attn_b1}), "
          f"n_heads={n_heads_attn}, head_dim={head_dim_attn}")

    # --- 6. 3D RoPE ---
    N_rope = 50
    n_heads_rope, head_dim_rope = 2, 24  # 8 per axis, n_freqs=4
    n_freqs_rope = head_dim_rope // 6
    dim_rope = n_heads_rope * head_dim_rope

    rope_coords = rand_coords_unique(rng, N_rope, 16, batch_size=1)
    rope_qk = rand_float(rng, (N_rope, dim_rope))
    rope_freqs = np.array([1.0 / (10000.0 ** (2.0 * j / (2.0 * n_freqs_rope)))
                           for j in range(n_freqs_rope)], dtype=np.float32)
    rope_ref = ref_rope_3d(rope_qk.copy(), rope_coords, n_heads_rope,
                            head_dim_rope, rope_freqs, n_freqs_rope)
    save(output_dir, "rope_coords", rope_coords)
    save(output_dir, "rope_input", rope_qk)
    save(output_dir, "rope_freqs", rope_freqs)
    save(output_dir, "rope_ref", rope_ref)
    print(f"  rope_3d: N={N_rope}, n_heads={n_heads_rope}, head_dim={head_dim_rope}, "
          f"n_freqs={n_freqs_rope}")

    # --- 7. Downsample ---
    N_ds, grid_ds, C_ds = 200, 16, 8
    ds_coords = rand_coords_unique(rng, N_ds, grid_ds, batch_size=1)
    ds_feats = rand_float(rng, (N_ds, C_ds))
    ds_out_coords, ds_out_feats, ds_counts = ref_downsample(ds_coords, ds_feats, 2)
    save(output_dir, "downsample_coords", ds_coords)
    save(output_dir, "downsample_input", ds_feats)
    save(output_dir, "downsample_ref_coords", ds_out_coords)
    save(output_dir, "downsample_ref_feats", ds_out_feats)
    save(output_dir, "downsample_ref_counts", ds_counts)
    print(f"  downsample: N={N_ds} -> {len(ds_out_coords)}")

    # --- 8. Upsample ---
    # Use downsample output as source, original coords as target
    up_src_coords = ds_out_coords
    up_src_feats = ds_out_feats
    up_tgt_coords = ds_coords  # original fine coords
    up_ref = ref_upsample(up_src_coords, up_src_feats, 2, up_tgt_coords)
    save(output_dir, "upsample_src_coords", up_src_coords)
    save(output_dir, "upsample_src_feats", up_src_feats)
    save(output_dir, "upsample_tgt_coords", up_tgt_coords)
    save(output_dir, "upsample_ref", up_ref)
    print(f"  upsample: src_N={len(up_src_coords)}, tgt_N={len(up_tgt_coords)}")

    print(f"\nAll reference data saved to {output_dir}/")


# ============================================================
# Compare
# ============================================================

def compare_all(ref_dir, c_dir):
    """Compare Python reference vs C outputs."""
    all_pass = True

    def check_maxdiff(name, ref, c_out, threshold):
        nonlocal all_pass
        diff = np.abs(ref - c_out)
        maxdiff = diff.max()
        meandiff = diff.mean()
        passed = maxdiff < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: max_diff={maxdiff:.2e}, mean_diff={meandiff:.2e}, "
              f"threshold={threshold:.0e} [{status}]")
        if not passed:
            all_pass = False
            # Show worst locations
            flat_idx = np.argmax(diff.ravel())
            idx = np.unravel_index(flat_idx, diff.shape)
            print(f"    worst at {idx}: ref={ref[idx]:.6f}, c={c_out[idx]:.6f}")
        return passed

    def check_correlation(name, ref, c_out, threshold):
        nonlocal all_pass
        # Per-row correlation (treating each voxel as a vector)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
            c_out = c_out.reshape(1, -1)
        corrs = []
        for i in range(len(ref)):
            r = ref[i]
            c = c_out[i]
            if np.std(r) < 1e-10 or np.std(c) < 1e-10:
                corrs.append(1.0 if np.allclose(r, c, atol=1e-6) else 0.0)
            else:
                corrs.append(float(np.corrcoef(r, c)[0, 1]))
        min_corr = min(corrs)
        mean_corr = np.mean(corrs)
        passed = min_corr > threshold
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: min_corr={min_corr:.6f}, mean_corr={mean_corr:.6f}, "
              f"threshold={threshold} [{status}]")
        if not passed:
            all_pass = False
            worst_idx = np.argmin(corrs)
            print(f"    worst row: {worst_idx}, corr={corrs[worst_idx]:.6f}")
        return passed

    print("\n=== Comparing Reference vs C Outputs ===\n")

    # 1. Linear
    try:
        lin_ref = load(ref_dir, "linear_ref")
        lin_c = load(c_dir, "linear_output")
        check_maxdiff("Linear", lin_ref, lin_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  Linear: SKIP ({e})")
        all_pass = False

    # 2. LayerNorm
    try:
        ln_ref = load(ref_dir, "layernorm_ref")
        ln_c = load(c_dir, "layernorm_output")
        check_maxdiff("LayerNorm", ln_ref, ln_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  LayerNorm: SKIP ({e})")
        all_pass = False

    # 3. GELU
    try:
        gelu_ref = load(ref_dir, "gelu_ref")
        gelu_c = load(c_dir, "gelu_output")
        check_maxdiff("GELU", gelu_ref, gelu_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  GELU: SKIP ({e})")
        all_pass = False

    # 4. SiLU
    try:
        silu_ref = load(ref_dir, "silu_ref")
        silu_c = load(c_dir, "silu_output")
        check_maxdiff("SiLU", silu_ref, silu_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  SiLU: SKIP ({e})")
        all_pass = False

    # 5. Conv3d
    try:
        conv_ref = load(ref_dir, "conv3d_ref")
        conv_c = load(c_dir, "conv3d_output")
        check_maxdiff("Conv3d", conv_ref, conv_c, 1e-3)
    except FileNotFoundError as e:
        print(f"  Conv3d: SKIP ({e})")
        all_pass = False

    # 6. Attention
    try:
        attn_ref = load(ref_dir, "attention_ref")
        attn_c = load(c_dir, "attention_output")
        check_maxdiff("Attention (abs)", attn_ref, attn_c, 1e-3)
        check_correlation("Attention (corr)", attn_ref, attn_c, 0.999)
    except FileNotFoundError as e:
        print(f"  Attention: SKIP ({e})")
        all_pass = False

    # 7. 3D RoPE
    try:
        rope_ref = load(ref_dir, "rope_ref")
        rope_c = load(c_dir, "rope_output")
        check_maxdiff("RoPE 3D", rope_ref, rope_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  RoPE 3D: SKIP ({e})")
        all_pass = False

    # 8. Downsample
    try:
        ds_ref_feats = load(ref_dir, "downsample_ref_feats")
        ds_c_feats = load(c_dir, "downsample_feats")
        ds_ref_coords = load(ref_dir, "downsample_ref_coords")
        ds_c_coords = load(c_dir, "downsample_coords_out")

        # Compare N
        ref_N = len(ds_ref_feats)
        c_N = len(ds_c_feats)
        if ref_N != c_N:
            print(f"  Downsample: FAIL (N mismatch: ref={ref_N}, c={c_N})")
            all_pass = False
        else:
            # Sort both by coords for comparison (order may differ)
            ref_order = np.lexsort(ds_ref_coords[:, ::-1].T)
            c_order = np.lexsort(ds_c_coords[:, ::-1].T)
            coords_match = np.array_equal(ds_ref_coords[ref_order], ds_c_coords[c_order])
            if not coords_match:
                print(f"  Downsample coords: FAIL (coords mismatch)")
                all_pass = False
            else:
                print(f"  Downsample coords: PASS (N={ref_N})")
            check_maxdiff("Downsample feats",
                         ds_ref_feats[ref_order], ds_c_feats[c_order], 1e-5)
    except FileNotFoundError as e:
        print(f"  Downsample: SKIP ({e})")
        all_pass = False

    # 9. Upsample
    try:
        up_ref = load(ref_dir, "upsample_ref")
        up_c = load(c_dir, "upsample_output")
        check_maxdiff("Upsample", up_ref, up_c, 1e-5)
    except FileNotFoundError as e:
        print(f"  Upsample: SKIP ({e})")
        all_pass = False

    print()
    if all_pass:
        print("=== ALL CHECKS PASSED ===")
    else:
        print("=== SOME CHECKS FAILED ===")
    return all_pass


# ============================================================
# All-in-one
# ============================================================

def run_all():
    import subprocess

    ref_dir = "ref_output"
    c_dir = "c_output"

    print("=== Step 1: Generate reference data ===\n")
    generate_all(ref_dir)

    print("\n=== Step 2: Build and run C test ===\n")
    # Build
    ret = subprocess.run(["make", "ARCH=native"], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"Build failed:\n{ret.stderr}")
        sys.exit(1)
    print("Build OK")

    # Run C test in export mode
    ret = subprocess.run(["./test_sparse3d", "--export", c_dir, "--input", ref_dir],
                         capture_output=True, text=True)
    print(ret.stderr, end="")
    if ret.returncode != 0:
        print(f"C test failed (exit code {ret.returncode})")
        sys.exit(1)

    print("\n=== Step 3: Compare ===")
    passed = compare_all(ref_dir, c_dir)
    sys.exit(0 if passed else 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify sparse3d.h against PyTorch reference")
    parser.add_argument("--generate", action="store_true",
                        help="Generate reference .npy files")
    parser.add_argument("--output", type=str, default="ref_output",
                        help="Output directory for generated files")
    parser.add_argument("--compare", nargs=2, metavar=("REF_DIR", "C_DIR"),
                        help="Compare reference vs C output directories")
    parser.add_argument("--all", action="store_true",
                        help="Generate + build + run C + compare (all-in-one)")
    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.generate:
        print("Generating reference data...")
        generate_all(args.output)
    elif args.compare:
        compare_all(args.compare[0], args.compare[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
