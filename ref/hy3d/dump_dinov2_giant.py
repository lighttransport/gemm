"""Dump DINOv2-giant encoder reference outputs for CUDA-side validation.

Matches what hy3dpaint's Dino_v2 wrapper (hy3dpaint/hunyuanpaintpbr/unet/modules.py)
does for the multiview paint pipeline: loads facebook/dinov2-giant from a
local directory, preprocesses a condition image via BitImageProcessor
(resize shortest edge -> center crop 224 -> ImageNet normalize), and
returns the last_hidden_state.

Produces .npy tensors for:
  dinov2g_input.npy       [1, 3, 224, 224]  preprocessed image the CUDA
                                              runner should consume verbatim
  dinov2g_patch_embed.npy [1, 257, 1536]    post patch + CLS + pos_embed
                                              (first hidden state)
  dinov2g_hidden_<i>.npy  [1, 257, 1536]    selected intermediate hidden
                                              states (layers 0, 10, 20, 30, 39)
  dinov2g_output.npy      [1, 257, 1536]    last_hidden_state after the
                                              final LayerNorm

Usage:
  uv run python dump_dinov2_giant.py \
      --model /mnt/disk01/models/dinov2-giant \
      --image /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png \
      --outdir output_dinov2g/
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/disk01/models/dinov2-giant")
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", default="output_dinov2g")
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    print(f"Loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).eval()

    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    x = inputs["pixel_values"].to(dtype)
    print(f"input shape {tuple(x.shape)} dtype {x.dtype}")
    np.save(os.path.join(args.outdir, "dinov2g_input.npy"),
            x.float().cpu().numpy())

    # Hook each Dinov2Layer + the embedding module to capture intermediates.
    cap = {}

    def make_hook(name):
        def _h(mod, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            cap[name] = out.detach().float().cpu().numpy()
        return _h

    handles = []
    handles.append(model.embeddings.register_forward_hook(make_hook("embed")))
    tap_layers = (0, 10, 20, 30, 39)
    for i, layer in enumerate(model.encoder.layer):
        if i in tap_layers:
            handles.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

    with torch.inference_mode():
        out = model(x)
    last = out.last_hidden_state.detach().float().cpu().numpy()
    for h in handles:
        h.remove()

    print(f"output shape {last.shape} mean={last.mean():+.4f} std={last.std():.4f}")
    np.save(os.path.join(args.outdir, "dinov2g_patch_embed.npy"), cap["embed"])
    for i in tap_layers:
        np.save(os.path.join(args.outdir, f"dinov2g_hidden_{i}.npy"),
                cap[f"layer_{i}"])
    np.save(os.path.join(args.outdir, "dinov2g_output.npy"), last)
    print(f"Wrote inputs/outputs to {args.outdir}")


if __name__ == "__main__":
    main()
