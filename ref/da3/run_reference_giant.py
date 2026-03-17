#!/usr/bin/env python3
"""
DA3 Giant PyTorch reference inference — all output modalities.

Produces reference outputs for depth, confidence, pose, rays, and gaussians
to compare against our C/CUDA implementation.

Usage:
    uv run python run_reference_giant.py \
        --model-dir /mnt/nvme02/models/gemm/da3-giant \
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

# Import shared components from run_reference.py
from run_reference import (
    Mlp, LayerScale, Attention,
    apply_rope, build_rope_cache,
    ResidualConvUnit, FeatureFusionBlock,
    preprocess,
)


# ---------------------------------------------------------------------------
# SwiGLU MLP for ViT-Giant backbone
# ---------------------------------------------------------------------------

class SwiGLUMlp(nn.Module):
    """SwiGLU MLP: w12 = gate_up [2*hidden, dim], w3 = down [dim, hidden]."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        gu = self.w12(x)  # (B, N, 2*hidden)
        hidden = gu.shape[-1] // 2
        gate = F.silu(gu[..., :hidden])
        up = gu[..., hidden:]
        return self.w3(gate * up)


class TransformerBlockGiant(nn.Module):
    """Transformer block with configurable MLP (GELU or SwiGLU)."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_qknorm=False,
                 use_rope=False, use_swiglu=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, use_qknorm, use_rope)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            # SwiGLU with 2/3 hidden ratio (but giant uses full ratio)
            # Giant: dim=1536, hidden=4096 (from w3 shape)
            self.mlp = SwiGLUMlp(dim, hidden_dim)
        else:
            self.mlp = Mlp(dim, hidden_dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope_cos_y=None, rope_sin_y=None,
                rope_cos_x=None, rope_sin_x=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope_cos_y, rope_sin_y,
                                    rope_cos_x, rope_sin_x))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2ViTGiant(nn.Module):
    """ViT backbone with SwiGLU support for Giant model."""

    def __init__(self, dim=1536, num_heads=24, depth=40, patch_size=14,
                 img_size=518, out_layers=(19, 27, 33, 39), rope_start=13,
                 qknorm_start=13, cat_token=True, use_swiglu=True,
                 ffn_hidden=4096):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.out_layers = list(out_layers)
        self.rope_start = rope_start
        self.qknorm_start = qknorm_start
        self.cat_token = cat_token
        self.num_heads = num_heads

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.grid_size ** 2, dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_rope = (i >= rope_start)
            use_qknorm = (i >= qknorm_start)
            # For SwiGLU, use ffn_hidden directly (not mlp_ratio)
            mlp_ratio = ffn_hidden / dim if use_swiglu else 4.0
            self.blocks.append(TransformerBlockGiant(
                dim, num_heads, mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm, use_rope=use_rope,
                use_swiglu=use_swiglu))

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
            if i >= self.rope_start:
                x = block(x, rope_cos_y, rope_sin_y, rope_cos_x, rope_sin_x)
            else:
                x = block(x)
            if i in self.out_layers:
                features.append(x.clone())

        return features


# ---------------------------------------------------------------------------
# DPT Head for Giant model (correct out_mid + reassemble/fuse helpers)
# ---------------------------------------------------------------------------

class DPTHead(nn.Module):
    """DPT head with correct intermediate dimensions for all model sizes."""

    def __init__(self, dim_in=3072, features=256,
                 out_channels=(256, 512, 1024, 1024), output_dim=2,
                 out_mid=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, bias=True)
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1,
                                   bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1,
                                   bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1,
                                   bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1,
                                   bias=False)
        self.refinenet4 = FeatureFusionBlock(features, has_skip=False)
        self.refinenet3 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet2 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet1 = FeatureFusionBlock(features, has_skip=True)

        neck_co = features // 2  # 128 for giant
        self.output_conv1 = nn.Conv2d(features, neck_co, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(neck_co, out_mid, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_mid, output_dim, 1),
        )

    def reassemble(self, features, grid_size):
        """Concat [patch, CLS], norm, project, resize, adapt."""
        B = features[0].shape[0]
        adapted = []
        for i, (feat, proj, resize) in enumerate(zip(
                features, self.projects, self.resize_layers)):
            patch = feat[:, 1:, :]
            cls_exp = feat[:, :1, :].expand(-1, patch.shape[1], -1)
            cat = self.norm(torch.cat([patch, cls_exp], dim=-1))
            x = cat.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
            x = resize(proj(x))
            x = getattr(self, f"layer{i+1}_rn")(x)
            adapted.append(x)
        return adapted

    def fuse(self, adapted):
        p4 = self.refinenet4(adapted[3])
        p3 = self.refinenet3(p4, adapted[2])
        p2 = self.refinenet2(p3, adapted[1])
        return self.refinenet1(p2, adapted[0])

    def forward(self, features, grid_size, patch_size=14):
        adapted = self.reassemble(features, grid_size)
        path1 = self.fuse(adapted)
        out = F.relu(self.output_conv1(path1))
        out = self.output_conv2(out)
        return out, adapted


# ---------------------------------------------------------------------------
# CameraDec: backbone_norm(CLS) → MLP(GELU) → 3 heads → pose[9]
# ---------------------------------------------------------------------------

class CameraDec(nn.Module):
    def __init__(self, dim_in, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim_in
        self.mlp_dim = mlp_dim
        # backbone.0 / backbone.2 — Sequential(Linear+GELU, Linear+GELU)
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
        )
        self.fc_t = nn.Linear(mlp_dim, 3)
        self.fc_qvec = nn.Linear(mlp_dim, 4)
        self.fc_fov = nn.Linear(mlp_dim, 2)

    def forward(self, cls_normed):
        """cls_normed: (B, dim_in) — backbone-normed CLS token (possibly doubled)."""
        h = self.backbone(cls_normed)
        t = self.fc_t(h)
        q = self.fc_qvec(h)
        fov = self.fc_fov(h)
        # pose = [t(3), qvec(4), fov(2)]
        return torch.cat([t, q, fov], dim=-1)  # (B, 9)


# ---------------------------------------------------------------------------
# Aux DPT: separate refinenet fusion + multi-layer output convs → rays
# ---------------------------------------------------------------------------

class AuxOutputConv2(nn.Module):
    """output_conv2_aux per level: Conv3x3 → [GroupNorm] → ReLU → Conv1x1→7.
    GroupNorm is optional (only level 0 has it in the giant model)."""
    def __init__(self, in_ch, mid_ch=32, out_ch=7, has_gn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.has_gn = has_gn
        if has_gn:
            self.gn = nn.GroupNorm(1, mid_ch)
        self.out = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        if self.has_gn:
            x = self.gn(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class AuxDPT(nn.Module):
    """Aux DPT branch for rays: shares adapted features with main DPT,
    but has its own refinenet fusion and output convolutions.
    oc1/oc2 layers are built dynamically from safetensors weight shapes."""

    def __init__(self, features=256, oc1_shapes=None):
        """oc1_shapes: list of 4 lists of (ci, co) tuples for oc1 Conv2d layers."""
        super().__init__()
        self.features = features

        # Aux refinenet fusion blocks
        self.refinenet4_aux = FeatureFusionBlock(features, has_skip=False)
        self.refinenet3_aux = FeatureFusionBlock(features, has_skip=True)
        self.refinenet2_aux = FeatureFusionBlock(features, has_skip=True)
        self.refinenet1_aux = FeatureFusionBlock(features, has_skip=True)

        # oc1: built from safetensors weight shapes
        if oc1_shapes is None:
            # Default for giant: 5 layers 256→128→256→128→256→128
            oc1_shapes = [[(256, 128), (128, 256), (256, 128), (128, 256),
                           (256, 128)]] * 4
        self.oc1 = nn.ModuleList()
        last_co = features
        for lv_shapes in oc1_shapes:
            layers = nn.ModuleList()
            for ci, co in lv_shapes:
                layers.append(nn.Conv2d(ci, co, 3, padding=1))
                last_co = co
            self.oc1.append(layers)

        # oc2: input channels = last oc1 output
        # Only level 0 has GroupNorm; levels 1-3 skip GN
        self.oc2 = nn.ModuleList()
        for lv in range(4):
            self.oc2.append(AuxOutputConv2(last_co, 32, 7, has_gn=(lv == 0)))

    def forward(self, adapted):
        """adapted: list of 4 adapted feature maps."""
        path4 = self.refinenet4_aux(adapted[3])
        path3 = self.refinenet3_aux(path4, adapted[2])
        path2 = self.refinenet2_aux(path3, adapted[1])
        path1 = self.refinenet1_aux(path2, adapted[0])

        # Apply oc1 chains + oc2 at each level
        paths = [path1, path2, path3, path4]
        all_outs = []
        for lv in range(4):
            x = paths[lv]
            for layer in self.oc1[lv]:
                x = layer(x)
            x = self.oc2[lv](x)
            all_outs.append(x)

        # Use level 0 (finest)
        out = all_outs[0]  # (B, 7, H, W)
        rays = out[:, :6, :, :]
        ray_conf = torch.exp(out[:, 6:7, :, :]) + 1.0
        return rays, ray_conf


# ---------------------------------------------------------------------------
# GSDPT: separate DPT head + image merger → 38-ch gaussians
# ---------------------------------------------------------------------------

class GSDPT(nn.Module):
    """Gaussian Splatting DPT head — independent DPT + RGB merger injection."""

    def __init__(self, dim_in=3072, features=256,
                 out_channels=(256, 512, 1024, 1024), output_dim=38):
        super().__init__()
        self.output_dim = output_dim
        self.features = features

        # Readout norm (optional — some models don't have it)
        self.has_norm = False  # will be set True if weights found
        self.norm = nn.LayerNorm(dim_in)

        # Projections
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, bias=True)
            for oc in out_channels
        ])

        # Resize layers (same structure as main DPT)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        # Adapters (layer_rn)
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1,
                                   bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1,
                                   bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1,
                                   bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1,
                                   bias=False)

        # RefineNet fusion
        self.refinenet4 = FeatureFusionBlock(features, has_skip=False)
        self.refinenet3 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet2 = FeatureFusionBlock(features, has_skip=True)
        self.refinenet1 = FeatureFusionBlock(features, has_skip=True)

        # Output convs: neck (NO ReLU for GSDPT), out_0 (ReLU), out_2
        neck_co = features // 2  # 128
        out_mid = 32
        self.output_conv1 = nn.Conv2d(features, neck_co, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(neck_co, out_mid, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_mid, output_dim, 1),
        )

        # Images merger: 3 stride-2 Conv2d layers on RGB
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            # No activation on last layer
        )

    def forward(self, features, grid_size, img_norm):
        """
        features: list of 4 raw hidden tensors (B, 1+np, dim)
        grid_size: spatial grid size
        img_norm: preprocessed image (B, 3, target_h, target_w)
        """
        B = features[0].shape[0]

        # Readout: concat [patch, CLS], norm, project, resize
        outs = []
        for i, (feat, proj, resize) in enumerate(zip(
                features, self.projects, self.resize_layers)):
            dim = feat.shape[-1]
            patch_tokens = feat[:, 1:, :]
            cls_expanded = feat[:, :1, :].expand(-1, patch_tokens.shape[1], -1)
            cat = torch.cat([patch_tokens, cls_expanded], dim=-1)
            if self.has_norm:
                cat = self.norm(cat)
            x = cat.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
            x = proj(x)
            x = resize(x)
            outs.append(x)

        # Adapter
        l4 = self.layer4_rn(outs[3])
        l3 = self.layer3_rn(outs[2])
        l2 = self.layer2_rn(outs[1])
        l1 = self.layer1_rn(outs[0])

        # Fusion
        path4 = self.refinenet4(l4)
        path3 = self.refinenet3(path4, l3)
        path2 = self.refinenet2(path3, l2)
        path1 = self.refinenet1(path2, l1)

        # Output: neck (NO ReLU for GSDPT)
        neck_out = self.output_conv1(path1)

        # Merger: process RGB image
        merger_out = self.images_merger(img_norm)
        # Upsample merger to match neck spatial size
        if merger_out.shape[2:] != neck_out.shape[2:]:
            merger_out = F.interpolate(merger_out, size=neck_out.shape[2:],
                                       mode="bilinear", align_corners=True)
        # Inject (element-wise add) — only merger_out channels overlap
        neck_out[:, :merger_out.shape[1]] += merger_out

        # output_conv2: Conv3x3 + ReLU + Conv1x1 → 38ch
        out = self.output_conv2(neck_out)
        return out  # (B, 38, H, W)


# ---------------------------------------------------------------------------
# Full DA3 Giant model
# ---------------------------------------------------------------------------

class DepthAnything3Giant(nn.Module):
    """DA3 Giant model: backbone + DualDPT + CameraDec + AuxDPT + GSDPT."""

    def __init__(self, config, oc1_shapes=None):
        super().__init__()
        net_cfg = config["net"]
        head_cfg = config["head"]

        name = net_cfg.get("name", "vitg")
        if name == "vitg":
            dim, heads, depth, ffn, swiglu = 1536, 24, 40, 4096, True
        elif name == "vitl":
            dim, heads, depth, ffn, swiglu = 1024, 16, 24, 4096, False
        elif name == "vitb":
            dim, heads, depth, ffn, swiglu = 768, 12, 12, 3072, False
        elif name == "vits":
            dim, heads, depth, ffn, swiglu = 384, 6, 12, 1536, False
        else:
            raise ValueError(f"Unknown backbone: {name}")

        self.dim = dim

        self.backbone = DinoV2ViTGiant(
            dim=dim, num_heads=heads, depth=depth,
            out_layers=net_cfg.get("out_layers", [19, 27, 33, 39]),
            rope_start=net_cfg.get("rope_start", 13),
            qknorm_start=net_cfg.get("qknorm_start", 13),
            cat_token=net_cfg.get("cat_token", True),
            use_swiglu=swiglu,
            ffn_hidden=ffn,
        )

        # Main DPT head (depth + confidence)
        self.head = DPTHead(
            dim_in=head_cfg.get("dim_in", 3072),
            features=head_cfg.get("features", 256),
            out_channels=tuple(head_cfg.get("out_channels",
                                            [256, 512, 1024, 1024])),
            output_dim=head_cfg.get("output_dim", 2),
        )

        # CameraDec (pose estimation)
        cam_dec_cfg = config.get("cam_dec", {})
        cam_dec_dim_in = cam_dec_cfg.get("dim_in", dim * 2)
        self.cam_dec = CameraDec(dim_in=cam_dec_dim_in, mlp_dim=cam_dec_dim_in)

        # Aux DPT (rays)
        self.aux_dpt = AuxDPT(features=head_cfg.get("features", 256),
                               oc1_shapes=oc1_shapes)

        # GSDPT (gaussians)
        gs_cfg = config.get("gs_head", {})
        self.gsdpt = GSDPT(
            dim_in=gs_cfg.get("dim_in", dim * 2),
            features=gs_cfg.get("features", 256),
            out_channels=tuple(gs_cfg.get("out_channels",
                                          [256, 512, 1024, 1024])),
            output_dim=gs_cfg.get("output_dim", 38),
        )

    def forward(self, x, img_norm=None):
        features = self.backbone(x)
        gs = self.backbone.grid_size
        ps = self.backbone.patch_size

        # Depth + confidence (also returns adapted features for aux reuse)
        dpt_out, adapted = self.head(features, gs, ps)

        # Pose: backbone_norm(CLS) → CameraDec
        last_feat = features[-1]
        cls_normed = self.backbone.norm(last_feat[:, 0, :])
        cls_input = torch.cat([cls_normed, cls_normed], dim=-1)
        pose = self.cam_dec(cls_input)

        # Aux DPT (rays) — reuses adapted features from main DPT
        rays, ray_conf = self.aux_dpt(adapted)

        # GSDPT (gaussians)
        gs_out = self.gsdpt(features, gs, img_norm)

        return dpt_out, pose, rays, ray_conf, gs_out


# ---------------------------------------------------------------------------
# Weight loading for giant model
# ---------------------------------------------------------------------------

def scan_aux_oc1_shapes(st_path):
    """Scan safetensors for aux oc1 layer shapes → list of 4 lists of (ci, co)."""
    shapes = [[] for _ in range(4)]
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            if "output_conv1_aux" in key and "weight" in key:
                # model.head.scratch.output_conv1_aux.{level}.{layer}.weight
                parts = key.split(".")
                # Find level and layer indices
                idx = parts.index("output_conv1_aux")
                level = int(parts[idx + 1])
                layer = int(parts[idx + 2])
                w = f.get_tensor(key)
                co, ci = w.shape[0], w.shape[1]
                while len(shapes[level]) <= layer:
                    shapes[level].append(None)
                shapes[level][layer] = (ci, co)
    # Clean up None entries
    for lv in range(4):
        shapes[lv] = [s for s in shapes[lv] if s is not None]
    return shapes


def load_giant_weights(model, st_path):
    """Load safetensors weights for the full giant model."""
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
        key = key.replace("backbone.patch_embed.proj.", "backbone.patch_embed.")

        # --- Main DPT head ---
        key = key.replace("head.scratch.layer1_rn.", "head.layer1_rn.")
        key = key.replace("head.scratch.layer2_rn.", "head.layer2_rn.")
        key = key.replace("head.scratch.layer3_rn.", "head.layer3_rn.")
        key = key.replace("head.scratch.layer4_rn.", "head.layer4_rn.")

        # Main refinenet (skip aux)
        if "head.scratch.refinenet" in key and "_aux" not in key:
            key = key.replace("head.scratch.refinenet", "head.refinenet")
        # Aux refinenet
        elif "head.scratch.refinenet" in key and "_aux" in key:
            # refinenet1_aux → aux_dpt.refinenet1_aux
            key = key.replace("head.scratch.", "aux_dpt.")

        # Main output convs (skip aux)
        if "head.scratch.output_conv1." in key:
            key = key.replace("head.scratch.output_conv1.", "head.output_conv1.")
        elif "head.scratch.output_conv2." in key and "aux" not in key:
            key = key.replace("head.scratch.output_conv2.", "head.output_conv2.")

        # Aux output conv1 chains: output_conv1_aux.{level}.{layer}.weight
        if "head.scratch.output_conv1_aux." in key:
            key = key.replace("head.scratch.output_conv1_aux.", "aux_dpt.oc1.")

        # Aux output conv2: output_conv2_aux.{level}.{si}.weight
        if "head.scratch.output_conv2_aux." in key:
            key = key.replace("head.scratch.output_conv2_aux.", "aux_dpt.oc2_raw.")

        # --- CameraDec ---
        if key.startswith("cam_dec."):
            # cam_dec.backbone.0.weight → cam_dec.backbone.0.weight (matches nn.Sequential)
            # cam_dec.backbone.2.weight → cam_dec.backbone.2.weight
            # cam_dec.fc_t.weight, cam_dec.fc_qvec.weight
            # cam_dec.fc_fov.0.weight → cam_dec.fc_fov.weight
            key = key.replace("cam_dec.fc_fov.0.", "cam_dec.fc_fov.")
            pass  # keys already match

        # --- GSDPT ---
        if key.startswith("gs_head."):
            key = key.replace("gs_head.", "gsdpt.")
            # Norm: readout_projects.0.0.* or norm.* → gsdpt.norm.*
            if "readout_projects.0.0." in key:
                key = key.replace("readout_projects.0.0.", "norm.")
            # gsdpt.norm.weight/bias (when key is already correct)
            # Handle explicit norm mapping for keys like "gsdpt.norm.weight"
            # Projects, resize_layers — already correct
            # Scratch
            key = key.replace("gsdpt.scratch.layer1_rn.", "gsdpt.layer1_rn.")
            key = key.replace("gsdpt.scratch.layer2_rn.", "gsdpt.layer2_rn.")
            key = key.replace("gsdpt.scratch.layer3_rn.", "gsdpt.layer3_rn.")
            key = key.replace("gsdpt.scratch.layer4_rn.", "gsdpt.layer4_rn.")
            key = key.replace("gsdpt.scratch.refinenet", "gsdpt.refinenet")
            key = key.replace("gsdpt.scratch.output_conv1.", "gsdpt.output_conv1.")
            key = key.replace("gsdpt.scratch.output_conv2.", "gsdpt.output_conv2.")
            # Images merger
            key = key.replace("gsdpt.images_merger.", "gsdpt.images_merger.")
            # Skip aux within gs_head
            if "_aux" in key:
                continue

        # Skip cam_enc, mask_token
        if any(key.startswith(p) for p in ["cam_enc", "backbone.mask_token"]):
            continue

        # SwiGLU MLP key mapping: w12 stays as w12, w3 stays as w3
        # (our SwiGLUMlp uses self.w12 and self.w3 which match the safetensors keys)
        # Standard MLP: fc1/fc2 also match directly

        mapped[key] = tensor

    # --- Handle aux output conv2 key remapping ---
    oc2_keys = sorted([k for k in mapped if k.startswith("aux_dpt.oc2_raw.")])
    for k in oc2_keys:
        # aux_dpt.oc2_raw.{level}.{si}.{weight|bias}
        parts = k.split(".")
        level = int(parts[2])
        si = int(parts[3])
        wb = parts[4]
        tensor = mapped.pop(k)
        # si=0 → conv, si=2 → gn, si=5 → out
        if si == 0:
            target_key = f"aux_dpt.oc2.{level}.conv.{wb}"
        elif si == 2:
            target_key = f"aux_dpt.oc2.{level}.gn.{wb}"
        elif si == 5:
            target_key = f"aux_dpt.oc2.{level}.out.{wb}"
        else:
            continue
        mapped[target_key] = tensor

    # Load state dict
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
        for k in sorted(unexpected)[:20]:
            print(f"  {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")

    # Remove keys for optional components that don't exist in weights
    # GSDPT norm: remove from expected if not in weights
    if "gsdpt.norm.weight" not in loaded_keys:
        mapped.pop("gsdpt.norm.weight", None)
        mapped.pop("gsdpt.norm.bias", None)
        # Remove from model's expected keys by marking norm as unused
        model.gsdpt.has_norm = False

    model.load_state_dict(mapped, strict=False)
    loaded = model_keys & set(mapped.keys())
    # Adjust for removed optional keys
    n_expected = len(model_keys) - len(missing - set(mapped.keys()))
    print(f"Loaded {len(loaded)}/{len(model_keys)} parameters")
    return len(loaded), len(model_keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DA3 Giant PyTorch reference — all outputs")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--prefix", default="giant",
                        help="Output filename prefix")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        full_config = json.load(f)
    config = full_config.get("config", full_config)
    print(f"Config: {config_path}")

    # Load image
    pil_img = Image.open(args.image).convert("RGB")
    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    # Find safetensors file
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        candidates = list(model_dir.glob("*.safetensors"))
        st_path = candidates[0] if candidates else None
        if not st_path:
            print(f"ERROR: No .safetensors in {model_dir}")
            sys.exit(1)

    # Pre-scan aux oc1 shapes before building model
    oc1_shapes = scan_aux_oc1_shapes(str(st_path))
    print(f"Aux oc1: {[len(s) for s in oc1_shapes]} layers per level")

    # Build model
    device = torch.device(args.device)
    model = DepthAnything3Giant(config, oc1_shapes=oc1_shapes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Load weights
    n_loaded, n_total = load_giant_weights(model, str(st_path))

    # Use bfloat16 on CUDA to fit in GPU memory (giant model is ~5GB in fp32)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()

    # Preprocess
    inp = preprocess(img_np).to(device=device, dtype=dtype)
    img_norm = inp.clone()
    print(f"Input: {inp.shape} ({inp.dtype}) on {device}")

    # Run inference
    with torch.no_grad():
        dpt_out, pose, rays, ray_conf, gs_out = model(inp, img_norm)

    prefix = args.prefix

    # --- Depth + Confidence ---
    depth_small = torch.exp(dpt_out[0, 0])
    conf_small = torch.exp(dpt_out[0, 1]) + 1.0
    print(f"\nDepth:      min={depth_small.min():.4f}  max={depth_small.max():.4f}  "
          f"mean={depth_small.mean():.4f}")
    print(f"Confidence: min={conf_small.min():.4f}  max={conf_small.max():.4f}  "
          f"mean={conf_small.mean():.4f}")

    depth_up = F.interpolate(
        depth_small.float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy().astype(np.float32)
    conf_up = F.interpolate(
        conf_small.float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy().astype(np.float32)

    np.save(str(output_dir / f"depth_{prefix}_ref.npy"), depth_up)
    np.save(str(output_dir / f"confidence_{prefix}_ref.npy"), conf_up)
    print(f"Saved depth/confidence: {depth_up.shape}")

    # --- Pose ---
    pose_np = pose[0].float().cpu().numpy().astype(np.float32)
    np.save(str(output_dir / f"pose_{prefix}_ref.npy"), pose_np)
    print(f"\nPose: t=[{pose_np[0]:.6f}, {pose_np[1]:.6f}, {pose_np[2]:.6f}]")
    print(f"      q=[{pose_np[3]:.6f}, {pose_np[4]:.6f}, {pose_np[5]:.6f}, {pose_np[6]:.6f}]")
    print(f"    fov=[{pose_np[7]:.6f}, {pose_np[8]:.6f}]")

    # --- Rays ---
    rays_up = F.interpolate(
        rays.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze(0).cpu().numpy().astype(np.float32)
    ray_conf_up = F.interpolate(
        ray_conf.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy().astype(np.float32)

    np.save(str(output_dir / f"rays_{prefix}_ref.npy"), rays_up)
    np.save(str(output_dir / f"ray_conf_{prefix}_ref.npy"), ray_conf_up)
    print(f"\nRays: {rays_up.shape}, conf: {ray_conf_up.shape}")
    for c in range(6):
        print(f"  ch{c}: min={rays_up[c].min():.4f}  max={rays_up[c].max():.4f}")
    print(f"  conf: min={ray_conf_up.min():.4f}  max={ray_conf_up.max():.4f}")

    # --- Gaussians ---
    gs_up = F.interpolate(
        gs_out.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=True
    ).squeeze(0).cpu().numpy().astype(np.float32)

    np.save(str(output_dir / f"gaussians_{prefix}_ref.npy"), gs_up)
    print(f"\nGaussians: {gs_up.shape}")
    for c in range(min(9, gs_up.shape[0])):
        print(f"  ch{c}: min={gs_up[c].min():.4f}  max={gs_up[c].max():.4f}")
    print(f"  ... ({gs_up.shape[0]} total channels)")

    print("\nDone.")


if __name__ == "__main__":
    main()
