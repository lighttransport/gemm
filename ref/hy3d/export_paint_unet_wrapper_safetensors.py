"""Convert Hunyuan3D-2.1 paint UNet `.bin` checkpoint to a single safetensors
file containing the FULL wrapper state — stock UNet + custom attention
weights (attn_multiview / attn_refview / attn_dino + per-material `_mr`
sub-modules) + image_proj_model_dino + learned_text_clip_* + unet_dual
(reference stream).

Used by Phase 4 of cuda/hy3d_paint to load the wrapper-aware C runner.
For Phase 3 (stock UNet only) keep using export_paint_unet_safetensors.py.

Usage:
  uv run --with torch --with safetensors python ref/hy3d/export_paint_unet_wrapper_safetensors.py \\
      [--unet /mnt/disk01/.../paintpbr-v2-1/unet] \\
      [--out  .../paintpbr-v2-1/unet/paint_unet_wrapper.safetensors]
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
    ap.add_argument("--skip-dual", action="store_true",
                    help="Skip unet_dual.* keys (Phase 4.2-4.4 don't need them).")
    args = ap.parse_args()
    out = args.out or os.path.join(args.unet, "paint_unet_wrapper.safetensors")

    bin_path = os.path.join(args.unet, "diffusion_pytorch_model.bin")
    print(f"loading {bin_path} ...", file=sys.stderr)
    ckpt = torch.load(bin_path, map_location="cpu", weights_only=True)

    state = {}
    n_skipped_dual = 0
    for k, v in ckpt.items():
        if args.skip_dual and k.startswith("unet_dual."):
            n_skipped_dual += 1
            continue
        state[k] = v.contiguous().to(torch.float32)

    save_file(state, out)
    sz = os.path.getsize(out)
    print(f"wrote {len(state)} tensors -> {out}  ({sz/1e9:.2f} GB)"
          + (f"   (skipped {n_skipped_dual} unet_dual.* keys)" if args.skip_dual else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
