"""Dump Hunyuan3D-2.1 paint UNet reference activations with ONLY the MA
(multiview cross-view) attention path enabled.

Phase 4.3 validation oracle for cuda/hy3d_paint. Two sub-modes:
    --rope 0   plain MA self-attn over (b*n_pbr, n_gen*L, C) tokens [no RoPE]
    --rope 1   MA with PoseRoPE (3-axis rotary) on Q/K, voxel indices from
               position maps.

Inputs (shared with the other dump_paint_unet_*.py — same seed):
    in_sample.npy                [1, 2, 2, 4, 64, 64]  f32
    in_embeds_normal.npy         [1, 2, 4, 64, 64]     f32
    in_embeds_position.npy       [1, 2, 4, 64, 64]     f32
    in_encoder_hidden_states.npy [1, 2, 77, 1024]      f32
    in_timestep.npy              [1]                   i64

Additional MA inputs:
    in_position_maps.npy         [1, 2, 3, 512, 512]   f32   (only --rope 1)

Outputs:
    out_ma.npy                   [4, 4, 64, 64]        f32   full forward
    voxel_indices_<L>.npy        [1, n_gen*L, 3]       i64   (only --rope 1)

Usage:
  uv run --with torch --with diffusers --with safetensors --with einops \\
      --with transformers --with packaging --with numpy \\
      python ref/hy3d/dump_paint_unet_ma.py \\
      --unet /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet \\
      --outdir /tmp/hy3d_paint_unet_ma_ref --rope 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

HY3D = "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dpaint"
sys.path.insert(0, HY3D)
import importlib.util as _ilu  # noqa: E402
import types as _types  # noqa: E402

_pkg_root = _types.ModuleType("hpb_stub")
_pkg_root.__path__ = []
_pkg_unet = _types.ModuleType("hpb_stub.unet")
_pkg_unet.__path__ = [os.path.join(HY3D, "hunyuanpaintpbr/unet")]
sys.modules["hpb_stub"] = _pkg_root
sys.modules["hpb_stub.unet"] = _pkg_unet

_spec_ap = _ilu.spec_from_file_location(
    "hpb_stub.unet.attn_processor",
    os.path.join(HY3D, "hunyuanpaintpbr/unet/attn_processor.py"))
_ap = _ilu.module_from_spec(_spec_ap)
sys.modules["hpb_stub.unet.attn_processor"] = _ap
_spec_ap.loader.exec_module(_ap)

_spec = _ilu.spec_from_file_location(
    "hpb_stub.unet.modules",
    os.path.join(HY3D, "hunyuanpaintpbr/unet/modules.py"))
_mod = _ilu.module_from_spec(_spec)
sys.modules["hpb_stub.unet.modules"] = _mod
_spec.loader.exec_module(_mod)
UNet2p5DConditionModel = _mod.UNet2p5DConditionModel
calc_multires_voxel_idxs = _mod.calc_multires_voxel_idxs


class MaOnlyWrapper(UNet2p5DConditionModel):
    """Wrapper that builds Basic2p5DTransformerBlocks with use_ma=True only."""

    def __init__(self, unet, use_position_rope):
        nn.Module.__init__(self)
        self.unet = unet
        self.train_sched = None
        self.val_sched = None
        self.use_ma = True
        self.use_ra = False
        self.use_mda = False
        self.use_dino = False
        self.use_position_rope = use_position_rope
        self.use_learned_text_clip = True
        self.use_dual_stream = False
        self.pbr_setting = ["albedo", "mr"]
        self.pbr_token_channels = 77

        self.init_attention(
            self.unet,
            use_ma=True, use_ra=False, use_mda=False, use_dino=False,
            pbr_setting=self.pbr_setting,
        )
        self.init_condition(use_dino=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unet_ma_ref")
    ap.add_argument("--rope", type=int, default=0, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    config_path = os.path.join(args.unet, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    inner = UNet2DConditionModel(**cfg)
    model = MaOnlyWrapper(inner, use_position_rope=bool(args.rope))
    model.unet.conv_in = torch.nn.Conv2d(
        12, inner.conv_in.out_channels,
        kernel_size=inner.conv_in.kernel_size,
        stride=inner.conv_in.stride,
        padding=inner.conv_in.padding,
        dilation=inner.conv_in.dilation,
        groups=inner.conv_in.groups,
        bias=inner.conv_in.bias is not None,
    )

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                      map_location="cpu", weights_only=True)
    SKIP_PREFIXES = ("unet_dual.",)
    SKIP_TOKENS = ("attn_refview", "attn_dino", ".processor.", "image_proj_model_dino")
    state = {}
    for k, v in ckpt.items():
        if any(k.startswith(p) for p in SKIP_PREFIXES):
            continue
        if any(t in k for t in SKIP_TOKENS):
            continue
        state[k] = v.to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} keys; missing={len(missing)}, unexpected={len(unexpected)}",
          file=sys.stderr)
    if missing:
        print(f"  first missing: {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}", file=sys.stderr)

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    torch.manual_seed(args.seed)
    B, N_pbr, N_gen = 1, 2, 2
    H = W = 64
    sample = torch.randn(B, N_pbr, N_gen, 4, H, W, device=device)
    embeds_normal = torch.randn(B, N_gen, 4, H, W, device=device)
    embeds_position = torch.randn(B, N_gen, 4, H, W, device=device)
    timestep = torch.tensor([500], dtype=torch.int64, device=device)
    encoder_hidden_states = torch.randn(B, N_pbr, 77, 1024, device=device)

    def save(name, t, dtype=np.float32):
        path = os.path.join(args.outdir, f"in_{name}.npy")
        arr = t.detach().cpu().numpy()
        arr = arr.astype(np.int64) if dtype == np.int64 else arr.astype(np.float32)
        np.save(path, arr)

    save("sample", sample)
    save("embeds_normal", embeds_normal)
    save("embeds_position", embeds_position)
    save("encoder_hidden_states", encoder_hidden_states)
    save("timestep", timestep, dtype=np.int64)

    cache = {}
    extra = {}
    if args.rope:
        # Generate dummy position maps. Use the latent resolution H=64 directly so
        # that voxel_grid_resolutions=[64,32,16,8] match the UNet's spatial levels
        # (calc_multires_voxel_idxs uses [H, H//2, H//4, H//8]).
        position_maps = torch.rand(B, N_gen, 3, H, W, device=device)
        save("position_maps", position_maps)
        extra["position_maps"] = position_maps
        # Pre-compute voxel indices and dump for the C-side check.
        v = calc_multires_voxel_idxs(
            position_maps,
            grid_resolutions=[H, H // 2, H // 4, H // 8],
            voxel_resolutions=[H * 8, H * 4, H * 2, H])
        for L_key, info in v.items():
            np.save(os.path.join(args.outdir, f"voxel_indices_{L_key}.npy"),
                    info["voxel_indices"].detach().cpu().numpy().astype(np.int64))
            print(f"  voxel L={L_key} res={info['voxel_resolution']}", file=sys.stderr)

    with torch.no_grad():
        out = model(
            sample, timestep, encoder_hidden_states,
            embeds_normal=embeds_normal,
            embeds_position=embeds_position,
            cache=cache,
            **extra,
        )
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "sample"):
        out = out.sample

    np.save(os.path.join(args.outdir, "out_ma.npy"),
            out.detach().cpu().numpy().astype(np.float32))
    print(f"out_ma {tuple(out.shape)} range=[{out.min():+.3f},{out.max():+.3f}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
