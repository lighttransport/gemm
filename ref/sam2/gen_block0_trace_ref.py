#!/usr/bin/env python3
"""Dump SAM2 block-0 trace tensors for CUDA debugging.

Outputs in outdir:
- block0_input.npy
- block0_ln1.npy
- block0_attn.npy
- block0_res1.npy
- block0_ln2.npy
- block0_mlp.npy
- block0_output.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.models.sam2.modeling_sam2 import window_partition, window_unpartition
from transformers.utils import logging as hf_logging


def save(path: Path, t: torch.Tensor) -> None:
    np.save(path, t.detach().cpu().numpy().astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/disk01/sam2/sam2.1-hiera-tiny")
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    hf_logging.set_verbosity_error()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    processor = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model).eval()
    block = model.vision_encoder.backbone.blocks[0]

    image = Image.open(args.image).convert("RGB")
    inputs = processor(
        images=image,
        input_points=[[[[256.0, 256.0]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )

    pv = inputs["pixel_values"]
    bb = model.vision_encoder.backbone

    with torch.no_grad():
        x = bb.patch_embed(pv)
        x = x + bb._get_pos_embed(x.shape[1:3])

        ln1 = block.layer_norm1(x)
        w, pad_hw = window_partition(ln1, block.window_size)
        wa = block.attn(w)
        attn = window_unpartition(wa, block.window_size, pad_hw, (x.shape[1], x.shape[2]))

        res1 = x + attn
        ln2 = block.layer_norm2(res1)
        mlp = block.mlp(ln2)
        out = res1 + mlp

    save(outdir / "block0_input.npy", x[0])
    save(outdir / "block0_ln1.npy", ln1[0])
    save(outdir / "block0_attn.npy", attn[0])
    save(outdir / "block0_res1.npy", res1[0])
    save(outdir / "block0_ln2.npy", ln2[0])
    save(outdir / "block0_mlp.npy", mlp[0])
    save(outdir / "block0_output.npy", out[0])
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
