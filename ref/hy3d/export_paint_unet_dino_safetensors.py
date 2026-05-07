"""Convert Hunyuan3D-2.1 paint UNet `.bin` checkpoint to a stock-SD-2.1
diffusers-naming safetensors that ALSO retains the per-block `attn_dino.*`
sibling weights and the top-level `image_proj_model_dino.*` projector.

Phase 4.2 of cuda/hy3d_paint loads this file (instead of paint_unet_stock)
to validate the DINO cross-attn path. Other custom paths
(attn_multiview / attn_refview / attn1.processor / unet_dual /
learned_text_clip_*) are still skipped.

Usage:
  uv run --with torch --with safetensors python ref/hy3d/export_paint_unet_dino_safetensors.py \\
      [--unet /mnt/disk01/.../paintpbr-v2-1/unet] \\
      [--out  /mnt/disk01/.../paintpbr-v2-1/unet/paint_unet_dino.safetensors]
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--out",  default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.unet, "paint_unet_dino.safetensors")

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                       map_location="cpu", weights_only=True)

    # Skip all Phase-4-other custom paths but keep attn_dino + image_proj_model_dino.
    SKIP = ("attn_multiview", "attn_refview", ".processor.")
    state = {}
    for k, v in ckpt.items():
        if k.startswith("unet_dual."):
            continue
        if not k.startswith("unet."):
            continue
        kk = k[len("unet."):]
        if any(t in kk for t in SKIP):
            continue
        # learned_text_clip_* belongs to RA / MDA paths, not DINO.
        if kk.startswith("learned_text_clip"):
            continue
        # Keep image_proj_model_dino.*, drop transformer nesting only on
        # inner-UNet keys (image_proj is top-level, attn_dino is sibling
        # of `.transformer.` so neither contains that token).
        kk = kk.replace(".transformer.", ".")
        state[kk] = v.contiguous().to(torch.float32)

    save_file(state, out)
    sz = os.path.getsize(out)
    n_dino_attn = sum(1 for k in state if "attn_dino" in k)
    n_dino_proj = sum(1 for k in state if k.startswith("image_proj_model_dino"))
    print(f"wrote {len(state)} tensors -> {out}  ({sz/1e9:.2f} GB)",
          f"[attn_dino={n_dino_attn}, image_proj_model_dino={n_dino_proj}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
