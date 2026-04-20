#!/usr/bin/env python3
"""
Generate TRELLIS.2 Stage 1 PyTorch reference outputs for HIP/ROCm verification.

Runs the full Stage 1 pipeline (DINOv3 → DiT 12-step → Decoder) on GPU
(ROCm or CUDA — both use torch.device("cuda")) and saves .npy files for
layer-by-layer comparison with the HIP implementation.

Usage:
  python gen_stage1_ref.py \
    --image teapot.png \
    --dinov3 /path/to/dinov3-vitl16/model.safetensors \
    --stage1 /path/to/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
    --decoder /path/to/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
    --seed 42 --output-dir /tmp/t2ref
"""
import sys, os, time, argparse
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../cpu/trellis2/trellis2_repo'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


def load_dinov3_from_timm(timm_path, device):
    """Load timm DINOv3 ViT-L/16 weights into transformers DINOv3ViTModel."""
    from transformers import DINOv3ViTModel, DINOv3ViTConfig

    timm_sd = load_file(timm_path)
    config = DINOv3ViTConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        intermediate_size=4096, patch_size=16, image_size=512,
        num_register_tokens=4, rope_theta=100.0,
    )
    model = DINOv3ViTModel(config)

    new_sd = {}
    new_sd['embeddings.cls_token'] = timm_sd['cls_token']
    new_sd['embeddings.register_tokens'] = timm_sd['reg_token']
    new_sd['embeddings.patch_embeddings.weight'] = timm_sd['patch_embed.proj.weight']
    new_sd['embeddings.patch_embeddings.bias'] = timm_sd['patch_embed.proj.bias']

    for i in range(24):
        tp = f'blocks.{i}.'
        hp = f'model.layer.{i}.'
        new_sd[f'{hp}norm1.weight'] = timm_sd[f'{tp}norm1.weight']
        new_sd[f'{hp}norm1.bias']   = timm_sd[f'{tp}norm1.bias']
        new_sd[f'{hp}norm2.weight'] = timm_sd[f'{tp}norm2.weight']
        new_sd[f'{hp}norm2.bias']   = timm_sd[f'{tp}norm2.bias']
        qkv_w = timm_sd[f'{tp}attn.qkv.weight']
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        new_sd[f'{hp}attention.q_proj.weight'] = q_w
        new_sd[f'{hp}attention.k_proj.weight'] = k_w
        new_sd[f'{hp}attention.v_proj.weight'] = v_w
        new_sd[f'{hp}attention.q_proj.bias'] = torch.zeros(1024)
        new_sd[f'{hp}attention.v_proj.bias'] = torch.zeros(1024)
        new_sd[f'{hp}attention.o_proj.weight'] = timm_sd[f'{tp}attn.proj.weight']
        new_sd[f'{hp}attention.o_proj.bias']   = timm_sd[f'{tp}attn.proj.bias']
        new_sd[f'{hp}layer_scale1.lambda1'] = timm_sd[f'{tp}gamma_1']
        new_sd[f'{hp}layer_scale2.lambda1'] = timm_sd[f'{tp}gamma_2']
        new_sd[f'{hp}mlp.up_proj.weight']   = timm_sd[f'{tp}mlp.fc1.weight']
        new_sd[f'{hp}mlp.up_proj.bias']     = timm_sd[f'{tp}mlp.fc1.bias']
        new_sd[f'{hp}mlp.down_proj.weight'] = timm_sd[f'{tp}mlp.fc2.weight']
        new_sd[f'{hp}mlp.down_proj.bias']   = timm_sd[f'{tp}mlp.fc2.bias']

    model.load_state_dict(new_sd, strict=False)
    return model.to(device).eval()


def preprocess_image(input_img):
    """Port of Trellis2ImageTo3DPipeline.preprocess_image (pipelines/trellis2_image_to_3d.py:127).

    If RGBA with real alpha: use alpha directly; else rembg to produce alpha.
    Then bbox-crop around alpha>0.8, multiply rgb*alpha (black background).
    """
    has_alpha = False
    if input_img.mode == 'RGBA':
        a = np.array(input_img)[:, :, 3]
        if not np.all(a == 255):
            has_alpha = True
    max_size = max(input_img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_img = input_img.resize((int(input_img.width * scale), int(input_img.height * scale)),
                                     Image.Resampling.LANCZOS)
    if has_alpha:
        output = input_img
    else:
        from rembg import remove, new_session  # lazy import
        output = remove(input_img.convert('RGB'), session=new_session('u2net'))
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    ys, xs = np.where(alpha > 0.8 * 255)
    if len(xs) == 0:
        raise RuntimeError('preprocess_image: no foreground pixels after rembg/alpha')
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    bbox = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    output = output.crop(bbox)
    arr = np.array(output).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4]   # black background
    return Image.fromarray((arr * 255).astype(np.uint8))


def extract_dinov3_features(model, image_path, image_size=512):
    """Extract DINOv3 features [1, 1029, 1024] matching official pipeline."""
    img = Image.open(image_path)
    img = preprocess_image(img)
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(next(model.parameters()).device)
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    with torch.no_grad():
        hidden = model.embeddings(x, bool_masked_pos=None)
        pos_emb = model.rope_embeddings(x)
        for layer in model.model.layer:
            hidden = layer(hidden, position_embeddings=pos_emb)
    return F.layer_norm(hidden, hidden.shape[-1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',    required=True)
    parser.add_argument('--dinov3',   required=True, help='timm DINOv3 safetensors')
    parser.add_argument('--stage1',   required=True, help='Stage 1 DiT safetensors (BF16)')
    parser.add_argument('--decoder',  required=True, help='Stage 1 decoder safetensors')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--steps',    type=int, default=12)
    parser.add_argument('--cfg',      type=float, default=7.5)
    parser.add_argument('--rescale',  type=float, default=0.7)
    parser.add_argument('--sigma-min', type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print(f'CPU mode (torch {torch.__version__})')

    # ── DINOv3 ──────────────────────────────────────────────────────────────
    print('\n=== DINOv3 (timm→transformers) ===')
    t0 = time.time()
    dino = load_dinov3_from_timm(args.dinov3, device)
    features = extract_dinov3_features(dino, args.image)  # [1, 1029, 1024]
    print(f'  features: {features.shape}  range=[{features.min():.3f}, {features.max():.3f}]')
    print(f'  time: {time.time()-t0:.1f}s')
    feat_np = features[0].cpu().float().numpy()
    np.save(os.path.join(args.output_dir, 'ref_features.npy'), feat_np)
    print(f'  saved ref_features.npy  {feat_np.shape}')
    del dino; torch.cuda.empty_cache()

    # ── Stage 1 DiT ─────────────────────────────────────────────────────────
    print('\n=== Stage 1 DiT ===')
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.modules.utils import manual_cast

    sd = {k: v.float() for k, v in load_file(args.stage1).items()}
    model = SparseStructureFlowModel(
        resolution=16, in_channels=8, model_channels=1536, cond_channels=1024,
        out_channels=8, num_blocks=30, num_head_channels=128,
        mlp_ratio=8192/1536, pe_mode='rope', dtype='float32',
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    del sd

    torch.manual_seed(args.seed)
    x = torch.randn(1, 8, 16, 16, 16, device=device)
    noise_init = x.cpu().float().numpy()
    np.save(os.path.join(args.output_dir, 'ref_noise_init.npy'), noise_init)
    print(f'  saved ref_noise_init.npy  seed={args.seed}')

    neg_cond = torch.zeros_like(features)
    sigma_min = args.sigma_min

    # Rescale t sequence
    t_seq = np.linspace(1, 0, args.steps + 1)
    t_seq = 5.0 * t_seq / (1 + 4.0 * t_seq)

    t0_all = time.time()
    for step in range(args.steps):
        t_cur, t_prev = t_seq[step], t_seq[step + 1]
        t_tensor = torch.tensor([1000.0 * t_cur], device=device)
        st = time.time()
        with torch.no_grad():
            if 0.6 <= t_cur <= 1.0:
                pred_pos = model(x, t_tensor, features)
                pred_neg = model(x, t_tensor, neg_cond)
                pred_v = args.cfg * pred_pos + (1 - args.cfg) * pred_neg
                x0_pos = (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t_cur) * pred_pos
                x0_cfg = (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t_cur) * pred_v
                std_pos = x0_pos.std(dim=list(range(1, 5)), keepdim=True)
                std_cfg = x0_cfg.std(dim=list(range(1, 5)), keepdim=True)
                x0 = args.rescale * (x0_cfg * std_pos / std_cfg) + (1 - args.rescale) * x0_cfg
                pred_v = ((1 - sigma_min) * x - x0) / (sigma_min + (1 - sigma_min) * t_cur)
            else:
                pred_v = model(x, t_tensor, features)
        x = x - (t_cur - t_prev) * pred_v
        step_np = x.cpu().float().numpy()
        np.save(os.path.join(args.output_dir, f'ref_latent_step{step}.npy'), step_np)
        tag = 'CFG' if 0.6 <= t_cur <= 1.0 else 'noG'
        print(f'  step {step+1:2d}/{args.steps}  t={t_cur:.4f}  {tag}  {(time.time()-st)*1000:.0f}ms')

    dit_time = time.time() - t0_all
    latent_np = x.cpu().float().numpy()
    np.save(os.path.join(args.output_dir, 'ref_latent.npy'), latent_np)
    print(f'  DiT total: {dit_time:.1f}s')
    print(f'  saved ref_latent.npy  range=[{latent_np.min():.3f}, {latent_np.max():.3f}]')
    del model; torch.cuda.empty_cache()

    # ── Decoder ──────────────────────────────────────────────────────────────
    print('\n=== Stage 1 Decoder ===')
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder

    dec_sd = {k: v.float() for k, v in load_file(args.decoder).items()}
    decoder = SparseStructureDecoder(
        out_channels=1, latent_channels=8, num_res_blocks=2, channels=[512, 128, 32])
    decoder.load_state_dict(dec_sd, strict=False)
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        occ = decoder(x)
    occ_np = occ.squeeze().cpu().float().numpy()
    if occ_np.ndim == 4:
        occ_np = occ_np[0]
    np.save(os.path.join(args.output_dir, 'ref_occupancy.npy'), occ_np)
    n_occ = (occ_np > 0).sum()
    print(f'  occupancy: {occ_np.shape}  {n_occ}/{occ_np.size} ({100*n_occ/occ_np.size:.1f}%)')
    print(f'  saved ref_occupancy.npy')

    print(f'\nDone. All outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()
