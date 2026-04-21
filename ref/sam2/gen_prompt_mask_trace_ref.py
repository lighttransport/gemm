#!/usr/bin/env python3
"""Dump prompt-encoder + mask-decoder reference tensors for SAM2 CUDA verification.

Outputs into <outdir>:
  prompt_inputs.txt           points/labels used
  prompt_sparse.npy           (B, N, 256)  sparse_prompt_embeddings
  prompt_dense.npy            (B, 256, 64, 64) dense_prompt_embeddings
  md_image_embed.npy          (B, 256, 64, 64) image_embeddings (fpn_2)
  md_image_pe.npy             (B, 256, 64, 64) image_positional_embeddings
  md_high_res_0.npy           (B, 32, 256, 256)  high-res feature 0
  md_high_res_1.npy           (B, 64, 128, 128)  high-res feature 1
  md_low_res_masks.npy        (B, N_masks, 256, 256)
  md_iou_scores.npy           (B, N_masks)
  md_high_res_masks.npy       (B, N_masks, H_img, W_img)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.utils import logging as hf_logging


def save(p: Path, x) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    np.save(p, x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x", type=float, default=256.0)
    ap.add_argument("--y", type=float, default=256.0)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    hf_logging.set_verbosity_error()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model).to(device).eval()

    image = Image.open(args.image).convert("RGB")
    points = [[[[args.x, args.y]]]]
    labels = [[[1]]]
    inputs = processor(images=image, input_points=points, input_labels=labels,
                        return_tensors="pt").to(device)

    with torch.no_grad():
        # Run vision encoder → get FPN outputs + pos encs
        ve = model.vision_encoder(pixel_values=inputs["pixel_values"])
        fpn = ve.fpn_hidden_states  # tuple — ordered [fpn_0 (hi-res) .. fpn_N (lo-res)]
        pe  = ve.fpn_position_encoding

        # Per HF Sam2Model.forward:
        #   image_embeddings = last fpn + no_memory_embedding (broadcast per-channel)
        #   image_pe         = get_image_wide_positional_embeddings() (shared_image_embedding)
        #   high_res[0/1]    = conv_s0/conv_s1 projections of fpn[0/1]
        no_mem = model.no_memory_embedding.view(1, -1, 1, 1)  # (1,1,256) -> (1,256,1,1)
        image_embeddings = fpn[-1] + no_mem
        image_pe = model.get_image_wide_positional_embeddings()
        high_res = [
            model.mask_decoder.conv_s0(fpn[0]),
            model.mask_decoder.conv_s1(fpn[1]),
        ]

        save(outdir / "md_image_embed.npy", image_embeddings)
        save(outdir / "md_image_pe.npy", image_pe)
        for i, t in enumerate(high_res):
            save(outdir / f"md_high_res_{i}.npy", t)

        # Prompt encoder
        input_points = inputs.get("input_points")
        input_labels = inputs.get("input_labels")
        input_boxes = inputs.get("input_boxes")
        input_masks = inputs.get("input_masks")
        sparse, dense = model.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        save(outdir / "prompt_sparse.npy", sparse)
        save(outdir / "prompt_dense.npy", dense)

        # Full forward (mask decoder)
        out = model(**inputs, multimask_output=True)
        save(outdir / "md_iou_scores.npy", out.iou_scores)
        save(outdir / "md_low_res_masks.npy", out.pred_masks)

    (outdir / "prompt_inputs.txt").write_text(
        f"points={points}\nlabels={labels}\nimage={args.image}\n"
        f"orig_size={image.size}\n"
    )
    print(f"wrote prompt/mask trace to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
