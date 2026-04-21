#!/usr/bin/env python3
"""Dump per-block Hiera outputs + FPN features for SAM2 CUDA verification.

Outputs NPY files into <outdir>:
  patch_embed.npy         (1, H/4, W/4, embed_dim)
  block{N}_input.npy      input to block N (pre-residual)
  block{N}_output.npy     output of block N (post-residual + MLP)
  intermediate_{k}.npy    Hiera intermediate_hidden_states (FPN inputs, k=0..3)
  fpn_{k}.npy             FPN output feature k (k=0..3)

For sam2.1-hiera-tiny:
  stages=[1,2,7,2]  total_blocks=12
  global_att @ blocks 5,7,9
  embed_dim_per_stage=[96,192,384,768], num_heads=[1,2,4,8]
  window_size_per_stage=[8,4,14,7], query_stride=(2,2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.utils import logging as hf_logging


def save(p: Path, x: torch.Tensor) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x.detach().cpu().float().numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    hf_logging.set_verbosity_error()

    # TF32 matmuls on Ampere+ introduce ~1e-3 drift vs strict F32 — disable for
    # reference tracing.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model).to(device).eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]  # (1, 3, 1024, 1024)

    backbone = model.vision_encoder.backbone  # Sam2HieraDetModel
    neck = model.vision_encoder.neck          # FPN

    captures: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        # Patch embed
        pe = backbone.patch_embed(pixel_values)  # (1, H/4, W/4, 96)
        save(outdir / "patch_embed.npy", pe)

        hs = pe
        # Add absolute pos embed if available
        if getattr(backbone, "pos_embed", None) is not None:
            # HF: hidden_states + _get_pos_embed
            hs = hs + backbone._get_pos_embed(hs.shape[1:3])
            save(outdir / "patch_embed_pos.npy", hs)

        for i, blk in enumerate(backbone.blocks):
            save(outdir / f"block{i}_input.npy", hs)
            hs = blk(hs)
            save(outdir / f"block{i}_output.npy", hs)

        save(outdir / "backbone_last.npy", hs)

        # Run through whole backbone forward for intermediates (HF path)
        out = backbone(pixel_values=pixel_values, output_hidden_states=True)
        for k, t in enumerate(out.intermediate_hidden_states or ()):
            save(outdir / f"intermediate_{k}.npy", t)

        # FPN
        # neck expects list of intermediate feature maps (B, C, H, W) format -
        # HF stores BHWC then neck permutes. Just call vision_encoder.
        ve_out = model.vision_encoder(pixel_values=pixel_values)
        for k, t in enumerate(ve_out.fpn_hidden_states or ()):
            save(outdir / f"fpn_{k}.npy", t)

    # Config dump (best-effort — attribute names vary by transformers version)
    cfg = backbone.config
    keys = [
        "embed_dim", "hidden_size", "num_attention_heads",
        "blocks_per_stage", "embed_dim_per_stage",
        "num_attention_heads_per_stage", "window_size_per_stage",
        "global_attention_blocks", "query_stride",
        "num_query_pool_stages", "mlp_ratio",
        "window_abs_pos_embed_size", "layer_norm_eps",
    ]
    lines = []
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            try:
                v = list(v)
            except TypeError:
                pass
            lines.append(f"{k}={v}")
    (outdir / "config.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote trace to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
