"""Convert Hunyuan3D-2.1 paint UNet `.bin` checkpoint to a safetensors that
keeps stock SD-2.1 keys + per-block `attn_refview.*` (+ `_mr` siblings) +
the full `unet_dual.*` second copy of the UNet (Phase 4.5 RA path).

Skips: attn_dino, attn_multiview, image_proj_model_dino,
       attn1.processor (MDA-only weights),
       learned_text_clip_albedo / _mr (kept: learned_text_clip_ref).

Both unet_dual and the main UNet's `attn_refview` keep `.transformer.`
collapsed to nothing so paint_unet_ra.safetensors looks like two siblings:
  unet_dual.<sd21 keys>...  (no attn_refview, no _mr — dual stream is RA-write only)
  <sd21 keys>...
  down_blocks.<i>.attentions.<j>.transformer_blocks.<k>.attn_refview.{to_q,to_k,to_v,to_out.0}.{weight,bias}
  ....attn_refview.processor.to_v_mr.weight
  ....attn_refview.processor.to_out_mr.0.{weight,bias}

Usage:
  uv run --with torch --with safetensors --with packaging --with numpy \\
      python ref/hy3d/export_paint_unet_ra_safetensors.py
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
    out = args.out or os.path.join(args.unet, "paint_unet_ra.safetensors")

    ckpt = torch.load(os.path.join(args.unet, "diffusion_pytorch_model.bin"),
                      map_location="cpu", weights_only=True)

    # Tokens whose presence anywhere in a key removes it (irrelevant attn paths).
    SKIP_TOKENS_MAIN = ("attn_dino", "attn_multiview", "image_proj_model_dino")
    # MDA-only weights live under `attn1.processor.` (to_q_mr/etc.).
    # RA `_mr` weights live under `attn_refview.processor.` — keep those.
    state = {}
    for k, v in ckpt.items():
        if k.startswith("unet_dual."):
            kk = k[len("unet_dual."):]
            # Inside unet_dual we only need stock SD-2.1 weights (no attn_refview,
            # no specialized siblings — dual stream just runs vanilla UNet to
            # populate norm_hidden_states caches per-block).
            if any(t in kk for t in ("attn_dino", "attn_multiview",
                                       "attn_refview", "image_proj_model_dino")):
                continue
            if "attn1.processor." in kk:
                continue
            if kk.startswith("learned_text_clip"):
                continue
            kk = kk.replace(".transformer.", ".")
            state["unet_dual." + kk] = v.contiguous().to(torch.float32)
            continue
        if not k.startswith("unet."):
            continue
        kk = k[len("unet."):]
        if any(t in kk for t in SKIP_TOKENS_MAIN):
            continue
        if "attn1.processor." in kk:
            continue
        # Only keep learned_text_clip_ref (used to drive unet_dual text input).
        if kk.startswith("learned_text_clip_albedo") or kk.startswith("learned_text_clip_mr"):
            continue
        kk = kk.replace(".transformer.", ".")
        state[kk] = v.contiguous().to(torch.float32)

    save_file(state, out)
    sz = os.path.getsize(out)
    n_dual = sum(1 for k in state if k.startswith("unet_dual."))
    n_ref  = sum(1 for k in state if "attn_refview" in k and not k.startswith("unet_dual."))
    print(f"wrote {len(state)} tensors -> {out}  ({sz/1e9:.2f} GB)"
          f"  [unet_dual={n_dual} attn_refview={n_ref}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
