#!/usr/bin/env python3
"""
DA3 PyTorch reference inference.

Runs the Depth-Anything-3 model using bare PyTorch (no external DA3 package)
to produce reference depth/confidence outputs for comparison with our C/CUDA
implementation.

Usage:
    uv run python run_reference.py \
        --model-dir /path/to/da3-small \
        --image input.jpg \
        --output-dir output/ \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Model components (matching safetensors key structure exactly)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Standard 2-layer MLP with GELU (matching DINOv2)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads, use_qknorm=False, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.use_qknorm = use_qknorm
        self.use_rope = use_rope
        if use_qknorm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, rope_cos_y=None, rope_sin_y=None,
                rope_cos_x=None, rope_sin_x=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        if self.use_qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and rope_cos_y is not None:
            # Apply RoPE to spatial tokens only (skip CLS at position 0)
            q_cls, q_sp = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_sp = k[:, :, :1, :], k[:, :, 1:, :]
            q_sp = apply_rope(q_sp, rope_cos_y, rope_sin_y,
                              rope_cos_x, rope_sin_x)
            k_sp = apply_rope(k_sp, rope_cos_y, rope_sin_y,
                              rope_cos_x, rope_sin_x)
            q = torch.cat([q_cls, q_sp], dim=2)
            k = torch.cat([k_cls, k_sp], dim=2)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_qknorm=False,
                 use_rope=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, use_qknorm, use_rope)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope_cos_y=None, rope_sin_y=None,
                rope_cos_x=None, rope_sin_x=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope_cos_y, rope_sin_y,
                                    rope_cos_x, rope_sin_x))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def apply_rope(x, cos_y, sin_y, cos_x, sin_x):
    """Apply 2D rotary position embedding. x: (B, heads, N, head_dim)

    The C code rotates pairs WITHIN each half of head_dim:
      Y rotation: (v[j], v[j+quarter]) for j=0..quarter-1  (first half)
      X rotation: (v[half+j], v[half+j+quarter])            (second half)
    """
    quarter = x.shape[-1] // 4
    # Split into 4 quarters
    x_y1 = x[..., :quarter]             # indices 0..15
    x_y2 = x[..., quarter:2*quarter]    # indices 16..31
    x_x1 = x[..., 2*quarter:3*quarter]  # indices 32..47
    x_x2 = x[..., 3*quarter:]           # indices 48..63

    # Y rotation on first half
    y1_rot = x_y1 * cos_y - x_y2 * sin_y
    y2_rot = x_y1 * sin_y + x_y2 * cos_y

    # X rotation on second half
    x1_rot = x_x1 * cos_x - x_x2 * sin_x
    x2_rot = x_x1 * sin_x + x_x2 * cos_x

    return torch.cat([y1_rot, y2_rot, x1_rot, x2_rot], dim=-1)


def build_rope_cache(grid_size, head_dim, device, theta=10000.0):
    """Build 2D RoPE cos/sin cache for grid_size x grid_size patches."""
    half = head_dim // 2     # 32
    quarter = half // 2      # 16
    # freq[j] = 1 / (theta ^ (2*j / half))
    freq = 1.0 / (theta ** (torch.arange(0, quarter, dtype=torch.float32,
                                          device=device) * 2.0 / half))
    gy = torch.arange(grid_size, dtype=torch.float32, device=device)
    gx = torch.arange(grid_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    angle_y = yy.unsqueeze(1) * freq.unsqueeze(0)  # (n_patches, quarter)
    angle_x = xx.unsqueeze(1) * freq.unsqueeze(0)
    cos_y = angle_y.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, n_patches, quarter)
    sin_y = angle_y.sin().unsqueeze(0).unsqueeze(0)
    cos_x = angle_x.cos().unsqueeze(0).unsqueeze(0)
    sin_x = angle_x.sin().unsqueeze(0).unsqueeze(0)
    return cos_y, sin_y, cos_x, sin_x


class DinoV2ViT(nn.Module):
    """ViT-S/14 backbone matching DA3 DinoV2 config."""

    def __init__(self, dim=384, num_heads=6, depth=12, patch_size=14,
                 img_size=518, out_layers=(5, 7, 9, 11), rope_start=4,
                 qknorm_start=4, cat_token=True, alt_start=None):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 37
        self.out_layers = list(out_layers)
        self.rope_start = rope_start
        self.qknorm_start = qknorm_start
        self.cat_token = cat_token
        self.num_heads = num_heads
        # alt_start: layer at which camera_token replaces CLS token
        self.alt_start = alt_start if alt_start is not None else rope_start

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.grid_size ** 2, dim))
        # Learned camera token: (1, 2, dim) — ref_token[0] used for single-view
        self.camera_token = nn.Parameter(torch.zeros(1, 2, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_rope = (i >= rope_start)
            use_qknorm = (i >= qknorm_start)
            self.blocks.append(TransformerBlock(
                dim, num_heads, use_qknorm=use_qknorm, use_rope=use_rope))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        rope_cos_y, rope_sin_y, rope_cos_x, rope_sin_x = build_rope_cache(
            self.grid_size, self.dim // self.num_heads, x.device)

        features = []
        for i, block in enumerate(self.blocks):
            # At alt_start, inject camera token into CLS position
            if i == self.alt_start and self.camera_token is not None:
                # For single-view: use ref_token (index 0)
                ref_token = self.camera_token[:, 0, :].expand(B, -1)
                x = x.clone()
                x[:, 0, :] = ref_token

            if i >= self.rope_start:
                x = block(x, rope_cos_y, rope_sin_y, rope_cos_x, rope_sin_x)
            else:
                x = block(x)
            if i in self.out_layers:
                # Save raw hidden states (no backbone norm) — matches C code
                # CLS token concat + norm happens in the DPT head
                features.append(x.clone())

        return features


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class FeatureFusionBlock(nn.Module):
    """RefineNet-style feature fusion block matching DPT."""

    def __init__(self, features, has_skip=True):
        super().__init__()
        # out_conv is a 1x1 Conv2d (not a ResidualConvUnit)
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)
        if has_skip:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.has_skip = has_skip
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x, skip=None):
        if skip is not None and self.has_skip:
            # Match C code convention: output = skip + RCU1(upsample(deeper))
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=True)
            x = skip + self.resConfUnit1(x)
        x = self.resConfUnit2(x)
        x = self.out_conv(x)
        return x


class DualDPTHead(nn.Module):
    """DualDPT head: reassemble + fuse + output convolutions."""

    def __init__(self, dim_in=768, features=64, out_channels=(48, 96, 192, 384),
                 output_dim=2):
        super().__init__()
        self.output_dim = output_dim

        # Layer norm applied to features before projection
        self.norm = nn.LayerNorm(dim_in)

        # Reassemble: project from dim_in to out_channels[i]
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, bias=True)
            for oc in out_channels
        ])

        # Resize convolutions (upsample path)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        # Scratch: fusion layers
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1,
                                   bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1,
                                   bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1,
                                   bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1,
                                   bias=False)

        # refinenet4 has no skip input (no resConfUnit1)
        self.refinenet4 = FeatureFusionBlock(features, has_skip=False)
        self.refinenet3 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet2 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet1 = FeatureFusionBlock(features, has_skip=True)

        # Output conv: Conv2d(64, 32, 3, pad=1) + ReLU
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        # Output conv2: Sequential(Conv2d(32, 32, 3, pad=1), ReLU, Conv2d(32, out_dim, 1))
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, output_dim, 1),
        )

    def forward(self, features, grid_size, patch_size=14):
        """features: list of 4 tensors, each (B, 1+n_patches, dim) raw hidden."""
        B = features[0].shape[0]

        # Reassemble: concat [patch, CLS], norm, reshape, project
        outs = []
        for i, (feat, proj, resize) in enumerate(zip(
                features, self.projects, self.resize_layers)):
            # feat is raw hidden: (B, 1+n_patches, dim) with CLS at position 0
            # Concat: [patch_token, CLS_token] for each patch → (B, n_patches, 2*dim)
            dim = feat.shape[-1]
            patch_tokens = feat[:, 1:, :]  # (B, n_patches, dim)
            cls_expanded = feat[:, :1, :].expand(-1, patch_tokens.shape[1], -1)
            cat = torch.cat([patch_tokens, cls_expanded], dim=-1)  # (B, n_patches, 2*dim)
            # Apply head norm (dim_in = 2*dim)
            cat = self.norm(cat)
            x = cat.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
            x = proj(x)
            x = resize(x)
            outs.append(x)

        # Fusion path (bottom-up)
        l4 = self.layer4_rn(outs[3])
        l3 = self.layer3_rn(outs[2])
        l2 = self.layer2_rn(outs[1])
        l1 = self.layer1_rn(outs[0])

        path4 = self.refinenet4(l4)
        path3 = self.refinenet3(path4, l3)
        path2 = self.refinenet2(path3, l2)
        path1 = self.refinenet1(path2, l1)

        # Output head — match C code: all convs at fusion resolution,
        # then caller upsamples to final resolution
        out = F.relu(self.output_conv1(path1))  # at fusion res (148x148)
        out = self.output_conv2(out)             # at fusion res (148x148)
        return out  # (B, output_dim, fusion_h, fusion_w)


class DepthAnything3(nn.Module):
    """Full DA3 model: backbone + DualDPT head."""

    def __init__(self, config):
        super().__init__()
        net_cfg = config["net"]
        head_cfg = config["head"]

        name = net_cfg.get("name", "vits")
        if name == "vits":
            dim, heads, depth = 384, 6, 12
        elif name == "vitb":
            dim, heads, depth = 768, 12, 12
        elif name == "vitl":
            dim, heads, depth = 1024, 16, 24
        else:
            raise ValueError(f"Unknown backbone: {name}")

        self.backbone = DinoV2ViT(
            dim=dim, num_heads=heads, depth=depth,
            out_layers=net_cfg.get("out_layers", [5, 7, 9, 11]),
            rope_start=net_cfg.get("rope_start", 4),
            qknorm_start=net_cfg.get("qknorm_start", 4),
            cat_token=net_cfg.get("cat_token", True),
            alt_start=net_cfg.get("alt_start", net_cfg.get("rope_start", 4)),
        )

        self.head = DualDPTHead(
            dim_in=head_cfg.get("dim_in", 768),
            features=head_cfg.get("features", 64),
            out_channels=tuple(head_cfg.get("out_channels", [48, 96, 192, 384])),
            output_dim=head_cfg.get("output_dim", 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features, self.backbone.grid_size,
                        self.backbone.patch_size)
        return out


# ---------------------------------------------------------------------------
# Weight loading from safetensors
# ---------------------------------------------------------------------------

def load_weights(model, st_path):
    """Load safetensors weights into our model, handling key name mapping."""
    state = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    mapped = {}
    for st_key, tensor in state.items():
        key = st_key

        # Strip "model." prefix
        if key.startswith("model."):
            key = key[6:]

        # Map backbone: "backbone.pretrained." -> "backbone."
        key = key.replace("backbone.pretrained.", "backbone.")

        # Map: backbone.patch_embed.proj -> backbone.patch_embed (Conv2d)
        key = key.replace("backbone.patch_embed.proj.", "backbone.patch_embed.")

        # Head scratch layer_rn -> head.layer*_rn
        key = key.replace("head.scratch.layer1_rn.", "head.layer1_rn.")
        key = key.replace("head.scratch.layer2_rn.", "head.layer2_rn.")
        key = key.replace("head.scratch.layer3_rn.", "head.layer3_rn.")
        key = key.replace("head.scratch.layer4_rn.", "head.layer4_rn.")

        # Head scratch refinenet -> head.refinenet
        key = key.replace("head.scratch.refinenet", "head.refinenet")
        # Skip aux refinenets
        if "_aux" in key:
            continue

        # Head scratch output_conv -> head.output_conv
        key = key.replace("head.scratch.output_conv1.", "head.output_conv1.")
        key = key.replace("head.scratch.output_conv2.", "head.output_conv2.")

        # Map backbone.camera_token → backbone.camera_token (keep it)

        # Skip cam_enc, cam_dec, aux heads, gsdpt, mask_token
        if any(key.startswith(p) for p in [
            "cam_enc", "cam_dec", "aux_", "gsdpt", "backbone.mask_token"
        ]):
            continue

        mapped[key] = tensor

    # Diagnostics
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(mapped.keys())

    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        print(f"WARNING: {len(missing)} missing keys:")
        for k in sorted(missing)[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if unexpected:
        print(f"INFO: {len(unexpected)} skipped/unmapped keys:")
        for k in sorted(unexpected)[:10]:
            print(f"  {k}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected) - 10} more")

    model.load_state_dict(mapped, strict=False)
    loaded = model_keys & loaded_keys
    print(f"Loaded {len(loaded)}/{len(model_keys)} parameters")
    return len(loaded), len(model_keys)


# ---------------------------------------------------------------------------
# Preprocessing (matches our C implementation exactly)
# ---------------------------------------------------------------------------

def preprocess(img_np, img_size=518):
    """
    Preprocess image for DA3 inference.
    img_np: (H, W, 3) uint8 numpy array
    Returns: (1, 3, img_size, img_size) float32 tensor
    """
    img = torch.from_numpy(img_np).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear",
                        align_corners=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DA3 PyTorch reference inference")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing model.safetensors and config.json")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--no-cam-token", action="store_true",
                        help="Disable camera_token injection (match C code behavior)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)
    with open(config_path) as f:
        full_config = json.load(f)
    config = full_config.get("config", full_config)
    print(f"Model config loaded: {config_path}")

    # Load image
    pil_img = Image.open(args.image).convert("RGB")
    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]
    print(f"Input image: {args.image} ({orig_w}x{orig_h})")

    # Build model
    device = torch.device(args.device)
    model = DepthAnything3(config)
    print(f"Model built: {sum(p.numel() for p in model.parameters())} params")

    # Load weights
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        candidates = list(model_dir.glob("*.safetensors"))
        if candidates:
            st_path = candidates[0]
        else:
            print(f"ERROR: No .safetensors file in {model_dir}")
            sys.exit(1)
    n_loaded, n_total = load_weights(model, str(st_path))
    if n_loaded < n_total:
        print(f"WARNING: Only {n_loaded}/{n_total} params loaded — "
              "results will be unreliable!")

    # Optionally disable camera_token to match C code (for depth comparison)
    if args.no_cam_token:
        model.backbone.camera_token = None
        print("Camera token injection DISABLED (matching C code)")

    model = model.to(device).eval()

    # Preprocess
    inp = preprocess(img_np).to(device)
    print(f"Preprocessed input: {inp.shape} on {device}")

    # Run inference
    with torch.no_grad():
        out = model(inp)  # (1, 2, H_out, W_out)

    print(f"Raw output: {out.shape}")

    # Apply activation: depth = exp(logit[0]), confidence = exp(logit[1]) + 1
    logit_depth = out[0, 0]
    logit_conf = out[0, 1]
    depth_small = torch.exp(logit_depth)
    conf_small = torch.exp(logit_conf) + 1.0

    print(f"DPT output size: {depth_small.shape}")
    print(f"  depth:      min={depth_small.min():.4f}  "
          f"max={depth_small.max():.4f}  mean={depth_small.mean():.4f}")
    print(f"  confidence: min={conf_small.min():.4f}  "
          f"max={conf_small.max():.4f}  mean={conf_small.mean():.4f}")

    # Upsample to original resolution
    depth_up = F.interpolate(
        depth_small.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy().astype(np.float32)

    conf_up = F.interpolate(
        conf_small.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy().astype(np.float32)

    print(f"Upsampled to: {depth_up.shape}")
    print(f"  depth:      min={depth_up.min():.4f}  "
          f"max={depth_up.max():.4f}  mean={depth_up.mean():.4f}")

    # Save .npy
    depth_path = output_dir / "depth_ref.npy"
    conf_path = output_dir / "confidence_ref.npy"
    np.save(str(depth_path), depth_up)
    np.save(str(conf_path), conf_up)
    print(f"Saved: {depth_path} ({depth_up.dtype}, {depth_up.shape})")
    print(f"Saved: {conf_path}")

    # Save normalized 16-bit PGM for visual inspection
    pgm_path = output_dir / "depth_ref.pgm"
    d_min, d_max = depth_up.min(), depth_up.max()
    d_range = d_max - d_min
    if d_range < 1e-6:
        d_range = 1.0
    normalized = ((depth_up - d_min) / d_range * 65535.0).astype(np.uint16)
    normalized_be = normalized.byteswap()
    with open(pgm_path, "wb") as f:
        header = f"P5\n{orig_w} {orig_h}\n65535\n".encode()
        f.write(header)
        f.write(normalized_be.tobytes())
    print(f"Saved: {pgm_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
