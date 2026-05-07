"""Convert Hunyuan3D-2.1 paint UNet `.bin` checkpoint into a stock-SD-2.1
safetensors file with diffusers naming. Drops dual-stream + custom-attn
parameters; only the inner stock UNet (with 12-ch conv_in) is exported.

Phase 3 only needs the stock attn path. Phase 4 will add a separate dump
for the custom processors.

Usage:
  uv run --with torch --with safetensors python ref/hy3d/export_paint_unet_safetensors.py \\
      [--unet /mnt/disk01/.../paintpbr-v2-1/unet] \\
      [--out  /mnt/disk01/.../paintpbr-v2-1/unet/paint_unet_stock.safetensors]
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
    out = args.out or os.path.join(args.unet, "paint_unet_stock.safetensors")

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                       map_location="cpu", weights_only=True)

    SKIP = ("attn_multiview", "attn_refview", "attn_dino", ".processor.")
    state = {}
    for k, v in ckpt.items():
        if not k.startswith("unet."):
            continue
        kk = k[len("unet."):]
        if any(t in kk for t in SKIP):
            continue
        kk = kk.replace(".transformer.", ".")
        # learned_text_clip_* / image_proj_model_dino live under unet.* but
        # are wrapper-only (Phase 4).
        if kk.startswith("learned_text_clip") or kk.startswith("image_proj_model"):
            continue
        state[kk] = v.contiguous().to(torch.float32)

    save_file(state, out)
    sz = os.path.getsize(out)
    print(f"wrote {len(state)} tensors -> {out}  ({sz/1e9:.2f} GB)", file=sys.stderr)


if __name__ == "__main__":
    main()
