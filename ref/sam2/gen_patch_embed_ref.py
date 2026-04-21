#!/usr/bin/env python3
"""Dump SAM2 patch-embed reference tensors.

Outputs:
- input_pixel_values.npy : float32 (3,1024,1024)
- patch_conv.npy         : float32 (96,256,256)
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
    ap.add_argument("--model", default="sam2.1-hiera-tiny")
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
    with torch.no_grad():
        conv = model.vision_encoder.backbone.patch_embed.projection(pv)  # (1,96,256,256)

    np.save(outdir / "input_pixel_values.npy", pv[0].cpu().numpy().astype(np.float32))
    np.save(outdir / "patch_conv.npy", conv[0].cpu().numpy().astype(np.float32))
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
