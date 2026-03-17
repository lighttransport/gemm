#!/usr/bin/env python3
"""
DA3 Nested-Giant-Large PyTorch reference — ALL output modalities.

Anyview (ViT-G): depth, confidence, pose, rays (cam_token), gaussians
Metric (ViT-L):  metric depth, sky segmentation

Usage:
    uv run python run_reference_nested.py \
        --model-dir /mnt/nvme02/models/gemm/da3-nested-giant-1.1 \
        --image output/street_scene.jpg --output-dir output/ --device cuda
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from run_reference import (
    Mlp, LayerScale, Attention, TransformerBlock,
    apply_rope, build_rope_cache,
    ResidualConvUnit, FeatureFusionBlock, preprocess,
)
from run_reference_giant import (
    SwiGLUMlp, TransformerBlockGiant, DinoV2ViTGiant,
    DPTHead, CameraDec, AuxDPT, AuxOutputConv2, GSDPT,
    scan_aux_oc1_shapes,
)


# ---------------------------------------------------------------------------
# Metric backbone: ViT-L (no RoPE, no QKNorm, no camera_token, GELU MLP)
# ---------------------------------------------------------------------------

class DinoV2ViTMetric(nn.Module):
    """ViT-L backbone for metric depth — simpler than the anyview backbone."""

    def __init__(self, dim=1024, num_heads=16, depth=24, patch_size=14,
                 img_size=518, out_layers=(4, 11, 17, 23), cat_token=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.out_layers = list(out_layers)
        self.cat_token = cat_token
        self.num_heads = num_heads

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.grid_size ** 2, dim))

        # All blocks: standard GELU MLP, no RoPE, no QKNorm
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_qknorm=False, use_rope=False)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_layers:
                if self.cat_token:
                    normed = self.norm(x)
                    cls_exp = normed[:, :1, :].expand(-1, normed.shape[1]-1, -1)
                    feat = torch.cat([normed[:, 1:, :], cls_exp], dim=-1)
                else:
                    # cat_token=False: just patch tokens, no CLS concat, no norm
                    feat = x[:, 1:, :]  # (B, n_patches, dim)
                features.append(feat)
        return features


# ---------------------------------------------------------------------------
# Metric DPT Head (output_dim=1, cat_token=false → dim_in=1024)
# ---------------------------------------------------------------------------

class MetricDPTHead(nn.Module):
    """Standard DPT head for metric depth + sky segmentation."""

    def __init__(self, dim_in=1024, features=256,
                 out_channels=(256, 512, 1024, 1024)):
        super().__init__()
        # No norm (cat_token=false, dim_in=raw dim)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, 1, bias=True) for oc in out_channels
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

        neck_co = features // 2  # 128
        out_mid = 32
        # Main output: metric depth (1 channel)
        self.output_conv1 = nn.Conv2d(features, neck_co, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(neck_co, out_mid, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_mid, 1, 1),
        )
        # Sky segmentation branch (shares neck output)
        self.sky_output_conv2 = nn.Sequential(
            nn.Conv2d(neck_co, out_mid, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_mid, 1, 1),
        )

    def forward(self, features, grid_size):
        B = features[0].shape[0]
        outs = []
        for i, (feat, proj, resize) in enumerate(zip(
                features, self.projects, self.resize_layers)):
            # cat_token=False: features are (B, n_patches, dim), no CLS concat
            x = feat.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
            x = resize(proj(x))
            x = getattr(self, f"layer{i+1}_rn")(x)
            outs.append(x)

        p4 = self.refinenet4(outs[3])
        p3 = self.refinenet3(p4, outs[2])
        p2 = self.refinenet2(p3, outs[1])
        p1 = self.refinenet1(p2, outs[0])

        neck = F.relu(self.output_conv1(p1))
        metric_depth = self.output_conv2(neck)    # (B, 1, H, W)
        sky_seg = self.sky_output_conv2(neck)     # (B, 1, H, W)
        return metric_depth, sky_seg


# ---------------------------------------------------------------------------
# Full Nested Model
# ---------------------------------------------------------------------------

class NestedDA3(nn.Module):
    def __init__(self, config, oc1_shapes=None):
        super().__init__()
        av_cfg = config["anyview"]
        met_cfg = config["metric"]
        av_net = av_cfg["net"]
        av_head = av_cfg["head"]
        met_net = met_cfg["net"]
        met_head = met_cfg["head"]

        # --- Anyview (ViT-G) ---
        self.av_backbone = DinoV2ViTGiant(
            dim=1536, num_heads=24, depth=40,
            out_layers=av_net.get("out_layers", [19, 27, 33, 39]),
            rope_start=av_net.get("rope_start", 13),
            qknorm_start=av_net.get("qknorm_start", 13),
            cat_token=True, use_swiglu=True, ffn_hidden=4096,
        )
        self.av_head = DPTHead(
            dim_in=av_head.get("dim_in", 3072),
            features=av_head.get("features", 256),
            out_channels=tuple(av_head.get("out_channels",
                                           [256, 512, 1024, 1024])),
            output_dim=2, out_mid=32,
        )
        cam_dec_cfg = av_cfg.get("cam_dec", {})
        self.av_cam_dec = CameraDec(
            dim_in=cam_dec_cfg.get("dim_in", 3072), mlp_dim=3072)
        self.av_aux_dpt = AuxDPT(
            features=av_head.get("features", 256), oc1_shapes=oc1_shapes)
        gs_cfg = av_cfg.get("gs_head", {})
        self.av_gsdpt = GSDPT(
            dim_in=gs_cfg.get("dim_in", 3072),
            features=gs_cfg.get("features", 256),
            out_channels=tuple(gs_cfg.get("out_channels",
                                          [256, 512, 1024, 1024])),
            output_dim=gs_cfg.get("output_dim", 38),
        )

        # --- Metric (ViT-L) ---
        self.met_backbone = DinoV2ViTMetric(
            dim=1024, num_heads=16, depth=24,
            out_layers=met_net.get("out_layers", [4, 11, 17, 23]),
            cat_token=met_net.get("cat_token", False),
        )
        self.met_head = MetricDPTHead(
            dim_in=met_head.get("dim_in", 1024),
            features=met_head.get("features", 256),
            out_channels=tuple(met_head.get("out_channels",
                                            [256, 512, 1024, 1024])),
        )

    def forward(self, x, img_norm=None):
        # === Anyview ===
        av_feats = self.av_backbone(x)
        gs = self.av_backbone.grid_size
        ps = self.av_backbone.patch_size

        dpt_out, adapted = self.av_head(av_feats, gs, ps)

        # Pose
        last = av_feats[-1]
        cls_normed = self.av_backbone.norm(last[:, 0, :])
        pose = self.av_cam_dec(torch.cat([cls_normed, cls_normed], dim=-1))

        # Rays
        rays, ray_conf = self.av_aux_dpt(adapted)

        # Gaussians
        gs_out = self.av_gsdpt(av_feats, gs, img_norm)

        # === Metric ===
        met_feats = self.met_backbone(x)
        metric_depth, sky_seg = self.met_head(met_feats,
                                               self.met_backbone.grid_size)

        return dpt_out, pose, rays, ray_conf, gs_out, metric_depth, sky_seg


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_nested_weights(model, st_path):
    state = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    mapped = {}
    for st_key, tensor in state.items():
        key = st_key
        if key.startswith("model."):
            key = key[6:]

        # === Metric model (da3_metric.*) ===
        if key.startswith("da3_metric."):
            key = key[len("da3_metric."):]
            key = key.replace("backbone.pretrained.", "backbone.")
            key = key.replace("backbone.patch_embed.proj.", "backbone.patch_embed.")
            key = key.replace("head.scratch.layer1_rn.", "head.layer1_rn.")
            key = key.replace("head.scratch.layer2_rn.", "head.layer2_rn.")
            key = key.replace("head.scratch.layer3_rn.", "head.layer3_rn.")
            key = key.replace("head.scratch.layer4_rn.", "head.layer4_rn.")
            key = key.replace("head.scratch.refinenet", "head.refinenet")
            key = key.replace("head.scratch.output_conv1.", "head.output_conv1.")
            key = key.replace("head.scratch.output_conv2.", "head.output_conv2.")
            key = key.replace("head.scratch.sky_output_conv2.",
                              "head.sky_output_conv2.")
            key = "met_" + key  # prefix with met_
            mapped[key] = tensor
            continue

        # === Anyview model ===
        # Nested model uses "da3." prefix: model.da3.backbone -> backbone
        if key.startswith("da3."):
            key = key[4:]
        key = key.replace("backbone.pretrained.", "backbone.")
        key = key.replace("backbone.patch_embed.proj.", "backbone.patch_embed.")

        # Head
        key = key.replace("head.scratch.layer1_rn.", "head.layer1_rn.")
        key = key.replace("head.scratch.layer2_rn.", "head.layer2_rn.")
        key = key.replace("head.scratch.layer3_rn.", "head.layer3_rn.")
        key = key.replace("head.scratch.layer4_rn.", "head.layer4_rn.")

        if "head.scratch.refinenet" in key and "_aux" not in key:
            key = key.replace("head.scratch.refinenet", "head.refinenet")
        elif "head.scratch.refinenet" in key and "_aux" in key:
            key = key.replace("head.scratch.", "aux_dpt.")

        if "head.scratch.output_conv1." in key:
            key = key.replace("head.scratch.output_conv1.", "head.output_conv1.")
        elif "head.scratch.output_conv2." in key and "aux" not in key:
            key = key.replace("head.scratch.output_conv2.", "head.output_conv2.")

        if "head.scratch.output_conv1_aux." in key:
            key = key.replace("head.scratch.output_conv1_aux.", "aux_dpt.oc1.")
        if "head.scratch.output_conv2_aux." in key:
            key = key.replace("head.scratch.output_conv2_aux.", "aux_dpt.oc2_raw.")

        # CameraDec
        if key.startswith("cam_dec."):
            key = key.replace("cam_dec.fc_fov.0.", "cam_dec.fc_fov.")

        # GSDPT
        if key.startswith("gs_head."):
            key = key.replace("gs_head.", "gsdpt.")
            if "readout_projects.0.0." in key:
                key = key.replace("readout_projects.0.0.", "norm.")
            key = key.replace("gsdpt.scratch.layer1_rn.", "gsdpt.layer1_rn.")
            key = key.replace("gsdpt.scratch.layer2_rn.", "gsdpt.layer2_rn.")
            key = key.replace("gsdpt.scratch.layer3_rn.", "gsdpt.layer3_rn.")
            key = key.replace("gsdpt.scratch.layer4_rn.", "gsdpt.layer4_rn.")
            key = key.replace("gsdpt.scratch.refinenet", "gsdpt.refinenet")
            key = key.replace("gsdpt.scratch.output_conv1.", "gsdpt.output_conv1.")
            key = key.replace("gsdpt.scratch.output_conv2.", "gsdpt.output_conv2.")
            if "_aux" in key:
                continue

        if any(key.startswith(p) for p in ["cam_enc", "backbone.mask_token"]):
            continue

        # Prefix anyview keys with av_
        key = "av_" + key
        mapped[key] = tensor

    # Handle aux oc2 remapping
    oc2_keys = sorted([k for k in mapped if "oc2_raw." in k])
    for k in oc2_keys:
        parts = k.split(".")
        raw_idx = parts.index("oc2_raw")
        level = int(parts[raw_idx + 1])
        si = int(parts[raw_idx + 2])
        wb = parts[raw_idx + 3]
        tensor = mapped.pop(k)
        prefix = ".".join(parts[:raw_idx])
        if si == 0:
            mapped[f"{prefix}.oc2.{level}.conv.{wb}"] = tensor
        elif si == 2:
            mapped[f"{prefix}.oc2.{level}.gn.{wb}"] = tensor
        elif si == 5:
            mapped[f"{prefix}.oc2.{level}.out.{wb}"] = tensor

    # GSDPT norm handling (may not exist)
    if "av_gsdpt.norm.weight" not in mapped:
        model.av_gsdpt.has_norm = False

    # Load
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(mapped.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        print(f"WARNING: {len(missing)} missing keys:")
        for k in sorted(missing)[:15]:
            print(f"  {k}")
        if len(missing) > 15:
            print(f"  ... and {len(missing)-15} more")
    if unexpected:
        print(f"INFO: {len(unexpected)} skipped keys:")
        for k in sorted(unexpected)[:10]:
            print(f"  {k}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected)-10} more")

    model.load_state_dict(mapped, strict=False)
    loaded = model_keys & loaded_keys
    print(f"Loaded {len(loaded)}/{len(model_keys)} params")
    return len(loaded), len(model_keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", default="nested")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "config.json") as f:
        config = json.load(f)["config"]

    pil_img = Image.open(args.image).convert("RGB")
    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    st_path = model_dir / "model.safetensors"
    oc1_shapes = scan_aux_oc1_shapes(str(st_path))
    print(f"Aux oc1: {[len(s) for s in oc1_shapes]} layers/level")

    device = torch.device(args.device)
    model = NestedDA3(config, oc1_shapes=oc1_shapes)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    load_nested_weights(model, str(st_path))

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()

    inp = preprocess(img_np).to(device=device, dtype=dtype)
    img_norm = inp.clone()
    print(f"Input: {inp.shape} ({inp.dtype})")

    with torch.no_grad():
        dpt_out, pose, rays, ray_conf, gs_out, met_depth, sky_seg = \
            model(inp, img_norm)

    pfx = args.prefix

    # --- Depth ---
    d = torch.exp(dpt_out[0, 0]).float()
    c = (torch.exp(dpt_out[0, 1]) + 1).float()
    d_up = F.interpolate(d.unsqueeze(0).unsqueeze(0),
                         (orig_h, orig_w), mode="bilinear",
                         align_corners=True).squeeze().cpu().numpy()
    np.save(str(output_dir / f"depth_{pfx}_ref.npy"), d_up.astype(np.float32))
    print(f"\nDepth: [{d.min():.4f}, {d.max():.4f}]")

    # --- Pose ---
    p = pose[0].float().cpu().numpy()
    np.save(str(output_dir / f"pose_{pfx}_ref.npy"), p.astype(np.float32))
    print(f"Pose: t={p[:3]}  q={p[3:7]}  fov={p[7:]}")

    # --- Rays ---
    r = rays.float()
    r_up = F.interpolate(r, (orig_h, orig_w), mode="bilinear",
                         align_corners=True).squeeze(0).cpu().numpy()
    rc = ray_conf.float()
    rc_up = F.interpolate(rc, (orig_h, orig_w), mode="bilinear",
                          align_corners=True).squeeze().cpu().numpy()
    np.save(str(output_dir / f"rays_{pfx}_ref.npy"), r_up.astype(np.float32))
    np.save(str(output_dir / f"ray_conf_{pfx}_ref.npy"),
            rc_up.astype(np.float32))
    print(f"Rays: [{r_up.min():.4f}, {r_up.max():.4f}], "
          f"all_zero={np.allclose(r_up, 0)}")
    print(f"Ray conf: [{rc_up.min():.4f}, {rc_up.max():.4f}]")

    # --- Gaussians ---
    gs = gs_out.float()
    gs_up = F.interpolate(gs, (orig_h, orig_w), mode="bilinear",
                          align_corners=True).squeeze(0).cpu().numpy()
    np.save(str(output_dir / f"gaussians_{pfx}_ref.npy"),
            gs_up.astype(np.float32))
    print(f"Gaussians: {gs_up.shape}, ch0=[{gs_up[0].min():.1f},{gs_up[0].max():.1f}]")

    # --- Metric Depth ---
    md = met_depth[0, 0].float()
    md_up = F.interpolate(md.unsqueeze(0).unsqueeze(0),
                          (orig_h, orig_w), mode="bilinear",
                          align_corners=True).squeeze().cpu().numpy()
    np.save(str(output_dir / f"metric_depth_{pfx}_ref.npy"),
            md_up.astype(np.float32))
    print(f"Metric depth: [{md_up.min():.4f}, {md_up.max():.4f}]")

    # --- Sky Segmentation ---
    ss = sky_seg[0, 0].float()
    ss_up = F.interpolate(ss.unsqueeze(0).unsqueeze(0),
                          (orig_h, orig_w), mode="bilinear",
                          align_corners=True).squeeze().cpu().numpy()
    np.save(str(output_dir / f"sky_seg_{pfx}_ref.npy"),
            ss_up.astype(np.float32))
    print(f"Sky seg: [{ss_up.min():.4f}, {ss_up.max():.4f}]")

    # Save PGMs for visual inspection
    for name, data in [("depth", d_up), ("metric_depth", md_up)]:
        dmin, dmax = data.min(), data.max()
        rng = dmax - dmin if dmax - dmin > 1e-6 else 1.0
        norm16 = ((data - dmin) / rng * 65535).astype(np.uint16).byteswap()
        pgm = output_dir / f"{name}_{pfx}_ref.pgm"
        with open(pgm, "wb") as fp:
            fp.write(f"P5\n{orig_w} {orig_h}\n65535\n".encode())
            fp.write(norm16.tobytes())
        print(f"Saved: {pgm}")

    print("\nDone.")


if __name__ == "__main__":
    main()
