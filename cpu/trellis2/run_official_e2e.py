#!/usr/bin/env python3
"""
End-to-end TRELLIS.2 Stage 1 using official repo code.
Converts timm DINOv3 weights to transformers format on the fly.
"""
import sys, os
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.sparse_structure_vae import SparseStructureDecoder


def load_dinov3_from_timm(timm_path, device):
    """Load timm DINOv3 weights into transformers DINOv3ViTModel."""
    from transformers import DINOv3ViTModel, DINOv3ViTConfig

    timm_sd = load_file(timm_path)

    config = DINOv3ViTConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        intermediate_size=4096, patch_size=16, image_size=512,
        num_register_tokens=4, rope_theta=100.0,
    )
    model = DINOv3ViTModel(config)

    # Map timm -> transformers
    new_sd = {}
    new_sd['embeddings.cls_token'] = timm_sd['cls_token']
    new_sd['embeddings.register_tokens'] = timm_sd['reg_token']
    new_sd['embeddings.patch_embeddings.weight'] = timm_sd['patch_embed.proj.weight']
    new_sd['embeddings.patch_embeddings.bias'] = timm_sd['patch_embed.proj.bias']

    for i in range(24):
        tp = f'blocks.{i}.'
        hp = f'layer.{i}.'

        new_sd[f'{hp}norm1.weight'] = timm_sd[f'{tp}norm1.weight']
        new_sd[f'{hp}norm1.bias'] = timm_sd[f'{tp}norm1.bias']
        new_sd[f'{hp}norm2.weight'] = timm_sd[f'{tp}norm2.weight']
        new_sd[f'{hp}norm2.bias'] = timm_sd[f'{tp}norm2.bias']

        # Split fused QKV [3072, 1024] -> Q [1024, 1024], K [1024, 1024], V [1024, 1024]
        qkv_w = timm_sd[f'{tp}attn.qkv.weight']
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        new_sd[f'{hp}attention.q_proj.weight'] = q_w
        new_sd[f'{hp}attention.k_proj.weight'] = k_w
        new_sd[f'{hp}attention.v_proj.weight'] = v_w
        # timm has no QKV bias, create zeros
        new_sd[f'{hp}attention.q_proj.bias'] = torch.zeros(1024)
        new_sd[f'{hp}attention.v_proj.bias'] = torch.zeros(1024)

        new_sd[f'{hp}attention.o_proj.weight'] = timm_sd[f'{tp}attn.proj.weight']
        new_sd[f'{hp}attention.o_proj.bias'] = timm_sd[f'{tp}attn.proj.bias']

        new_sd[f'{hp}layer_scale1.lambda1'] = timm_sd[f'{tp}gamma_1']
        new_sd[f'{hp}layer_scale2.lambda1'] = timm_sd[f'{tp}gamma_2']

        new_sd[f'{hp}mlp.up_proj.weight'] = timm_sd[f'{tp}mlp.fc1.weight']
        new_sd[f'{hp}mlp.up_proj.bias'] = timm_sd[f'{tp}mlp.fc1.bias']
        new_sd[f'{hp}mlp.down_proj.weight'] = timm_sd[f'{tp}mlp.fc2.weight']
        new_sd[f'{hp}mlp.down_proj.bias'] = timm_sd[f'{tp}mlp.fc2.bias']

    # Load converted weights
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    model = model.to(device).eval()
    return model


def extract_dinov3_features(model, image_path, image_size=512):
    """Extract features matching official DinoV3FeatureExtractor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)

    # Run through model layers (matching official extract_features)
    with torch.no_grad():
        hidden_states = model.embeddings(img, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(img)
        for layer_module in model.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)

    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dinov3", required=True, help="Timm DINOv3 safetensors")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--grid", type=int, nargs='+', default=[64, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="official")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # === DINOv3 ===
    print("\n=== DINOv3 (timm->transformers) ===")
    t0 = time.time()
    dino = load_dinov3_from_timm(args.dinov3, device)
    features = extract_dinov3_features(dino, args.image)
    print(f"  Features: {features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")
    print(f"  Time: {time.time()-t0:.1f}s")
    np.save(f"{args.output}_features.npy", features[0].cpu().numpy())
    del dino; torch.cuda.empty_cache()

    # === Stage 1 DiT ===
    print("\n=== Stage 1 DiT ===")
    sd = load_file(args.stage1)
    sd = {k: v.float() for k, v in sd.items()}
    model = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    del sd

    # Sampling
    print(f"\n=== Sampling (12 steps) ===")
    sigma_min = 1e-5
    torch.manual_seed(args.seed)
    x = torch.randn(1, 8, 16, 16, 16, device=device)
    neg_cond = torch.zeros_like(features)

    t_seq = np.linspace(1, 0, 13)
    t_seq = 5.0 * t_seq / (1 + 4.0 * t_seq)

    t0_all = time.time()
    for step in range(12):
        t_cur, t_prev = t_seq[step], t_seq[step+1]
        t_tensor = torch.tensor([1000 * t_cur], device=device, dtype=torch.float32)
        st = time.time()
        with torch.no_grad():
            if 0.6 <= t_cur <= 1.0:
                pred_pos = model(x, t_tensor, features)
                pred_neg = model(x, t_tensor, neg_cond)
                pred_v = 7.5 * pred_pos + (1 - 7.5) * pred_neg
                # CFG rescale 0.7
                x_0_pos = (1-sigma_min)*x - (sigma_min+(1-sigma_min)*t_cur)*pred_pos
                x_0_cfg = (1-sigma_min)*x - (sigma_min+(1-sigma_min)*t_cur)*pred_v
                std_pos = x_0_pos.std(dim=list(range(1,5)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1,5)), keepdim=True)
                x_0 = 0.7*(x_0_cfg*std_pos/std_cfg) + 0.3*x_0_cfg
                pred_v = ((1-sigma_min)*x - x_0)/(sigma_min+(1-sigma_min)*t_cur)
            else:
                pred_v = model(x, t_tensor, features)
        x = x - (t_cur - t_prev) * pred_v
        print(f"  step {step+1}/12 t={t_cur:.4f} {'CFG' if 0.6<=t_cur<=1.0 else 'noG'} {(time.time()-st)*1000:.0f}ms")

    dit_time = time.time() - t0_all
    print(f"  Total: {dit_time:.1f}s, latent range=[{x.min():.3f}, {x.max():.3f}]")
    np.save(f"{args.output}_latent.npy", x.cpu().numpy())
    del model; torch.cuda.empty_cache()

    # === Decoder ===
    print("\n=== Decoder ===")
    dec_sd = load_file(args.decoder)
    dec_sd = {k: v.float() for k, v in dec_sd.items()}
    decoder = SparseStructureDecoder(
        out_channels=1, latent_channels=8, num_res_blocks=2, channels=[512, 128, 32])
    decoder.load_state_dict(dec_sd, strict=False)
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        occupancy = decoder(x)
    occ_np = occupancy.squeeze().cpu().numpy()
    if occ_np.ndim == 4: occ_np = occ_np[0]
    np.save(f"{args.output}_occupancy.npy", occ_np)
    n = (occ_np > 0).sum()
    print(f"  Occupancy: shape={occ_np.shape}, range=[{occ_np.min():.2f}, {occ_np.max():.2f}]")
    print(f"  Occupied: {n}/{occ_np.size} ({100*n/occ_np.size:.1f}%)")

    # === Mesh ===
    print("\n=== Mesh Export ===")
    from skimage.measure import marching_cubes
    from scipy.ndimage import zoom
    for gs in args.grid:
        occ = occ_np.copy()
        if gs < 64: occ = zoom(occ, gs/64.0, order=1)
        thresh = 0.0
        if thresh <= occ.min() or thresh >= occ.max():
            thresh = (occ.min()+occ.max())/2
        verts, faces, _, _ = marching_cubes(occ, level=thresh)
        verts = verts / (np.array(occ.shape)-1)
        path = f"{args.output}_{gs}.obj"
        with open(path, 'w') as f:
            f.write(f"# {len(verts)} verts, {len(faces)} tris\n")
            for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for fc in faces: f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
        print(f"  {path}: {len(verts)} verts, {len(faces)} tris ({os.path.getsize(path)/1e6:.1f}MB)")

    print(f"\nDone (DiT: {dit_time:.1f}s)")

if __name__ == "__main__":
    main()
