#!/usr/bin/env python3
"""Dump SAM2 block-0 reference tensors.

Outputs:
- block0_input.npy  : float32 (256,256,96)
- block0_output.npy : float32 (256,256,96)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.utils import logging as hf_logging


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

    image = Image.open(args.image).convert("RGB")
    inputs = processor(
        images=image,
        input_points=[[[[256.0, 256.0]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )

    pv = inputs["pixel_values"]  # (1,3,1024,1024)
    bb = model.vision_encoder.backbone

    with torch.no_grad():
        x = bb.patch_embed(pv)
        x = x + bb._get_pos_embed(x.shape[1:3])
        y = bb.blocks[0](x)

    np.save(outdir / "block0_input.npy", x[0].cpu().numpy().astype(np.float32))
    np.save(outdir / "block0_output.npy", y[0].cpu().numpy().astype(np.float32))
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
