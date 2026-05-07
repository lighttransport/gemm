"""Convert Hunyuan3D-2.1 paint UNet `.bin` checkpoint to a safetensors that
keeps stock SD-2.1 keys + per-block `attn_multiview.*` (Phase 4.3) +
per-block per-material `attn1.processor.{q,k,v,out}_mr.*` (Phase 4.4 MDA path).

Skips: unet_dual.*, attn_refview, attn_dino, image_proj_model_dino,
       learned_text_clip_*. Keeps `.processor.` (needed for `_mr` weights).

Usage:
  uv run --with torch --with safetensors --with packaging --with numpy \\
      python ref/hy3d/export_paint_unet_mda_safetensors.py
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
    out = args.out or os.path.join(args.unet, "paint_unet_mda.safetensors")

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                      map_location="cpu", weights_only=True)

    SKIP = ("attn_refview", "attn_dino")
    state = {}
    for k, v in ckpt.items():
        if k.startswith("unet_dual."):
            continue
        if not k.startswith("unet."):
            continue
        kk = k[len("unet."):]
        if any(t in kk for t in SKIP):
            continue
        if kk.startswith("learned_text_clip"):
            continue
        if kk.startswith("image_proj_model_dino"):
            continue
        # The wrapper buries diffusers attention modules under `.transformer.` —
        # strip it so keys match the stock safetensors layout.
        kk = kk.replace(".transformer.", ".")
        state[kk] = v.contiguous().to(torch.float32)

    save_file(state, out)
    sz = os.path.getsize(out)
    n_ma = sum(1 for k in state if "attn_multiview" in k)
    n_mr = sum(1 for k in state if "attn1.processor." in k)
    print(f"wrote {len(state)} tensors -> {out}  ({sz/1e9:.2f} GB)"
          f"  [attn_multiview={n_ma} attn1.processor={n_mr}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
