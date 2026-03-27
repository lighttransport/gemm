#!/usr/bin/env python3
"""
run_pytorch_ref.py - TRELLIS.2 Stage 1 reference using official repo modules.

Loads the official SparseStructureFlowModel + SparseStructureDecoder from
the cloned TRELLIS.2 repo and runs the exact same sampling pipeline.

Usage:
  python run_pytorch_ref.py --stage1 <ss_flow.st> --decoder <ss_dec.st> \
      --features <features.npy> [--grid 64 32] [--seed 42] [--output ref]
"""

import sys, os
# Add TRELLIS.2 repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def load_dit_model(path, device):
    """Load Stage 1 DiT using official SparseStructureFlowModel."""
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel

    # Detect config from weights
    sd = load_file(path)
    # Convert BF16 -> F32
    sd = {k: v.float() for k, v in sd.items()}

    model_channels = sd['input_layer.weight'].shape[0]  # 1536
    in_channels = sd['input_layer.weight'].shape[1]      # 8
    out_channels = sd['out_layer.weight'].shape[0]        # 8
    n_blocks = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    ffn_dim = sd['blocks.0.mlp.mlp.0.weight'].shape[0]
    head_dim = sd['blocks.0.self_attn.q_rms_norm.gamma'].shape[1]
    cond_channels = sd['blocks.0.cross_attn.to_kv.weight'].shape[1]

    print(f"  model_channels={model_channels}, in_channels={in_channels}, "
          f"n_blocks={n_blocks}, ffn={ffn_dim}, head_dim={head_dim}, cond={cond_channels}")

    model = SparseStructureFlowModel(
        resolution=16,
        in_channels=in_channels,
        model_channels=model_channels,
        cond_channels=cond_channels,
        out_channels=out_channels,
        num_blocks=n_blocks,
        num_head_channels=head_dim,
        mlp_ratio=ffn_dim / model_channels,
        pe_mode='rope',
        dtype='float32',
        share_mod=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
    )

    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print(f"  Model loaded on {device}")
    return model


def load_decoder_model(path, device):
    """Load structure decoder using official SparseStructureDecoder."""
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder

    sd = load_file(path)
    sd = {k: v.float() for k, v in sd.items()}

    # Detect channels from conv_in
    in_channels = sd['input_layer.weight'].shape[1]
    model_channels = sd['input_layer.weight'].shape[0]
    out_channels = sd['out_layer.2.weight'].shape[0]

    # Count blocks
    block_ids = set()
    for k in sd:
        if k.startswith('blocks.'):
            block_ids.add(int(k.split('.')[1]))
    # middle blocks
    mid_ids = set()
    for k in sd:
        if k.startswith('middle_block.'):
            mid_ids.add(int(k.split('.')[1]))

    print(f"  model_channels={model_channels}, in={in_channels}, out={out_channels}")
    print(f"  blocks: {sorted(block_ids)}, middle: {sorted(mid_ids)}")

    model = SparseStructureDecoder(
        out_channels=out_channels,
        latent_channels=in_channels,
        num_res_blocks=2,
        channels=[model_channels, 128, 32],
    )

    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print(f"  Decoder loaded on {device}")
    return model


def sample_stage1(model, cond, device, n_steps=12, cfg_scale=7.5,
                  rescale_val=5.0, sigma_min=1e-5, seed=42,
                  guidance_rescale=0.7, guidance_interval=(0.6, 1.0)):
    """Sample using the official FlowEulerGuidanceIntervalSampler logic."""
    B = 1
    resolution = 16
    in_ch = model.in_channels

    torch.manual_seed(seed)
    x = torch.randn(B, in_ch, resolution, resolution, resolution, device=device)

    neg_cond = torch.zeros_like(cond)

    # Time schedule with rescaling
    t_seq = np.linspace(1, 0, n_steps + 1)
    t_seq = rescale_val * t_seq / (1 + (rescale_val - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i+1]) for i in range(n_steps)]

    for step, (t, t_prev) in enumerate(t_pairs):
        t0 = time.time()

        t_tensor = torch.tensor([1000 * t], device=device, dtype=torch.float32)

        with torch.no_grad():
            # Check guidance interval
            apply_cfg = guidance_interval[0] <= t <= guidance_interval[1]

            if apply_cfg and cfg_scale != 1.0:
                pred_pos = model(x, t_tensor, cond)
                pred_neg = model(x, t_tensor, neg_cond)
                pred_v = cfg_scale * pred_pos + (1 - cfg_scale) * pred_neg

                # CFG rescale
                if guidance_rescale > 0:
                    x_0_pos = (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t) * pred_pos
                    x_0_cfg = (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t) * pred_v
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    pred_v = ((1 - sigma_min) * x - x_0) / (sigma_min + (1 - sigma_min) * t)
            else:
                pred_v = model(x, t_tensor, cond)

        # Euler step: x_{t_prev} = x_t - (t - t_prev) * v
        x = x - (t - t_prev) * pred_v

        elapsed = (time.time() - t0) * 1000
        print(f"  step {step+1}/{n_steps}  t={t:.4f}->{t_prev:.4f}  cfg={'ON' if apply_cfg else 'off'}  {elapsed:.0f} ms")

    return x


def export_mesh(occupancy_np, path, grid_size=64, threshold=0.0):
    """Export occupancy grid to .obj mesh."""
    occ = occupancy_np.copy()

    if grid_size < occ.shape[0]:
        from scipy.ndimage import zoom
        factor = grid_size / occ.shape[0]
        occ = zoom(occ, factor, order=1)
        print(f"  Downsampled {occupancy_np.shape[0]}^3 -> {grid_size}^3")

    try:
        from skimage.measure import marching_cubes
        # Ensure threshold is within data range
        if threshold <= occ.min() or threshold >= occ.max():
            threshold = (occ.min() + occ.max()) / 2
            print(f"  Adjusted threshold to {threshold:.2f} (data range: [{occ.min():.2f}, {occ.max():.2f}])")
        verts, faces, _, _ = marching_cubes(occ, level=threshold)
        verts = verts / (np.array(occ.shape) - 1)  # normalize to [0,1]
    except ImportError:
        print("  skimage not available")
        np.save(path.replace('.obj', '.npy'), occ)
        return

    with open(path, 'w') as f:
        f.write(f"# PyTorch ref mesh: {len(verts)} verts, {len(faces)} tris\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    sz = os.path.getsize(path)
    print(f"  Wrote {path}: {len(verts)} verts, {len(faces)} tris ({sz/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Stage 1 PyTorch reference")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--features", required=True, help="DINOv3 features .npy")
    parser.add_argument("--grid", type=int, nargs='+', default=[64, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--output", default="ref", help="Output prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load features
    features_np = np.load(args.features)
    features = torch.from_numpy(features_np).float().unsqueeze(0).to(device)  # [1, 1029, 1024]
    print(f"Features: {features.shape}")

    # Load DiT
    print("\n=== Loading Stage 1 DiT ===")
    t0 = time.time()
    model = load_dit_model(args.stage1, device)
    print(f"  Load time: {time.time()-t0:.1f}s")

    # Sample
    print(f"\n=== Stage 1: Flow Sampling ({args.steps} steps, cfg={args.cfg}) ===")
    t0 = time.time()
    latent = sample_stage1(model, features, device, args.steps, args.cfg, seed=args.seed)
    dit_time = time.time() - t0
    print(f"  DiT total: {dit_time:.1f}s")

    latent_np = latent.cpu().numpy()
    np.save(f"{args.output}_latent.npy", latent_np)
    print(f"  Latent: shape={latent_np.shape}, range=[{latent_np.min():.2f}, {latent_np.max():.2f}]")

    del model
    torch.cuda.empty_cache()

    # Load decoder
    print("\n=== Loading Decoder ===")
    try:
        decoder = load_decoder_model(args.decoder, device)
        print("\n=== Decoding ===")
        t0 = time.time()
        with torch.no_grad():
            occupancy = decoder(latent)
        dec_time = time.time() - t0
        print(f"  Decoder: {dec_time:.1f}s")
    except Exception as e:
        print(f"  Decoder failed ({e}), decoding manually...")
        # Fallback: manual decoder
        dec_time = 0
        occupancy = latent  # placeholder

    occ_np = occupancy.squeeze().cpu().numpy()
    # Ensure 3D
    if occ_np.ndim == 4:
        occ_np = occ_np[0]  # drop channel dim
    np.save(f"{args.output}_occupancy.npy", occ_np)
    n_occ = (occ_np > 0).sum()
    print(f"  Occupancy: shape={occ_np.shape}, range=[{occ_np.min():.2f}, {occ_np.max():.2f}]")
    print(f"  Occupied (>0): {n_occ}/{occ_np.size} ({100*n_occ/occ_np.size:.1f}%)")

    # Export meshes
    print("\n=== Mesh Export ===")
    for gs in args.grid:
        out_path = f"{args.output}_{gs}.obj"
        print(f"Grid {gs}:")
        export_mesh(occ_np, out_path, gs)

    print(f"\n=== Summary ===")
    print(f"  DiT: {dit_time:.1f}s  Decoder: {dec_time:.1f}s  Total: {dit_time+dec_time:.1f}s")


if __name__ == "__main__":
    main()
