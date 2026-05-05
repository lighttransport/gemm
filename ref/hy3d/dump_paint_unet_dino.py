"""Dump Hunyuan3D-2.1 paint UNet reference activations with ONLY the DINO
cross-attn path enabled (use_dino=True, use_mda=use_ma=use_ra=False).

Phase 4.2 validation oracle for cuda/hy3d_paint. Tighter than the full
"all-paths" wrapper dump because we isolate the DINO residual: any diff
vs the stock-UNet path comes from `image_proj_model_dino` + per-block
`attn_dino` only.

Inputs (shared with dump_paint_unet_wrapper.py — use the same seed):
    in_sample.npy                [1, 2, 2, 4, 64, 64]  f32
    in_embeds_normal.npy         [1, 2, 4, 64, 64]     f32
    in_embeds_position.npy       [1, 2, 4, 64, 64]     f32
    in_encoder_hidden_states.npy [1, 2, 77, 1024]      f32
    in_dino_hidden_states.npy    [1, 257, 1536]        f32
    in_timestep.npy              [1]                   i64

Outputs:
    out_dino.npy                 [4, 4, 64, 64]        f32  full forward
    dino_proj.npy                [1, 4, 1024]          f32  image_proj_model_dino output

Usage:
  uv run --with torch --with diffusers --with safetensors --with einops \\
      python ref/hy3d/dump_paint_unet_dino.py \\
      --unet /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet \\
      --outdir /tmp/hy3d_paint_unet_dino_ref
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


class DinoOnlyWrapper(UNet2p5DConditionModel):
    """Wrapper that builds Basic2p5DTransformerBlocks with use_dino=True only."""

    def __init__(self, unet):
        nn.Module.__init__(self)
        self.unet = unet
        self.train_sched = None
        self.val_sched = None
        self.use_ma = False
        self.use_ra = False
        self.use_mda = False
        self.use_dino = True
        self.use_position_rope = False
        self.use_learned_text_clip = True
        self.use_dual_stream = False
        self.pbr_setting = ["albedo", "mr"]
        self.pbr_token_channels = 77

        # Skip unet_dual entirely (use_ra=False).
        self.init_attention(
            self.unet,
            use_ma=False,
            use_ra=False,
            use_mda=False,
            use_dino=True,
            pbr_setting=self.pbr_setting,
        )
        self.init_condition(use_dino=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unet_dino_ref")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config_path = os.path.join(args.unet, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    inner = UNet2DConditionModel(**cfg)
    model = DinoOnlyWrapper(inner)
    # Patch conv_in 4->12 (matches from_pretrained).
    model.unet.conv_in = torch.nn.Conv2d(
        12,
        inner.conv_in.out_channels,
        kernel_size=inner.conv_in.kernel_size,
        stride=inner.conv_in.stride,
        padding=inner.conv_in.padding,
        dilation=inner.conv_in.dilation,
        groups=inner.conv_in.groups,
        bias=inner.conv_in.bias is not None,
    )

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                      map_location="cpu", weights_only=True)
    # Strip wrapper-only keys we never built (use_ma/use_ra/use_mda all False
    # and unet_dual disabled).
    SKIP_PREFIXES = ("unet_dual.",)
    SKIP_TOKENS = ("attn_multiview", "attn_refview", "attn1.processor")
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
    B, N_pbr, N_gen, N_ref = 1, 2, 2, 1
    H = W = 64
    sample = torch.randn(B, N_pbr, N_gen, 4, H, W, device=device)
    embeds_normal = torch.randn(B, N_gen, 4, H, W, device=device)
    embeds_position = torch.randn(B, N_gen, 4, H, W, device=device)
    timestep = torch.tensor([500], dtype=torch.int64, device=device)
    encoder_hidden_states = torch.randn(B, N_pbr, 77, 1024, device=device)
    ref_latents = torch.randn(B, N_ref, 4, H, W, device=device)
    dino_hidden_states = torch.randn(B, 257, 1536, device=device)

    def save(name, t, dtype=np.float32):
        path = os.path.join(args.outdir, f"in_{name}.npy")
        arr = t.detach().cpu().numpy()
        arr = arr.astype(np.int64) if dtype == np.int64 else arr.astype(np.float32)
        np.save(path, arr)

    save("sample", sample)
    save("embeds_normal", embeds_normal)
    save("embeds_position", embeds_position)
    save("encoder_hidden_states", encoder_hidden_states)
    save("ref_latents", ref_latents)
    save("dino_hidden_states", dino_hidden_states)
    save("timestep", timestep, dtype=np.int64)

    # Cache the projected DINO output for tighter validation.
    cache = {}
    with torch.no_grad():
        dino_proj = model.image_proj_model_dino(dino_hidden_states)
    np.save(os.path.join(args.outdir, "dino_proj.npy"),
            dino_proj.detach().cpu().numpy().astype(np.float32))
    cache["dino_hidden_states_proj"] = dino_proj

    with torch.no_grad():
        out = model(
            sample, timestep, encoder_hidden_states,
            embeds_normal=embeds_normal,
            embeds_position=embeds_position,
            dino_hidden_states=dino_hidden_states,
            cache=cache,
        )
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "sample"):
        out = out.sample

    np.save(os.path.join(args.outdir, "out_dino.npy"),
            out.detach().cpu().numpy().astype(np.float32))
    print(f"out_dino {tuple(out.shape)} range=[{out.min():+.3f},{out.max():+.3f}]",
          file=sys.stderr)
    print(f"dino_proj {tuple(dino_proj.shape)}", file=sys.stderr)


if __name__ == "__main__":
    main()
