#!/usr/bin/env python3
"""Generate minimal SAM2 reference tensors for CUDA bring-up.

Requires: transformers>=4.57, torch, pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.utils import logging as hf_logging


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def parse_csv_floats(s: str) -> list[float]:
    vals: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def parse_csv_ints(s: str) -> list[int]:
    vals: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny",
                    help="HF repo id or local model directory")
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--points", default="", help="comma list: x1,y1,x2,y2,...")
    ap.add_argument("--labels", default="", help="comma list: l1,l2,...")
    ap.add_argument("--box", default="", help="comma list: x0,y0,x1,y1")
    ap.add_argument("--x", type=float, default=256.0)
    ap.add_argument("--y", type=float, default=256.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_logging.set_verbosity_error()

    processor = Sam2Processor.from_pretrained(args.model)
    model = Sam2Model.from_pretrained(args.model).to(device).eval()

    image = Image.open(args.image).convert("RGB")
    pvals = parse_csv_floats(args.points) if args.points else []
    lvals = parse_csv_ints(args.labels) if args.labels else []
    box_vals = parse_csv_floats(args.box) if args.box else []

    if pvals and (len(pvals) % 2) != 0:
        raise ValueError("--points must contain x,y pairs")
    n_points = len(pvals) // 2
    if n_points > 0 and lvals and len(lvals) != n_points:
        raise ValueError("--labels count must match number of points")

    if n_points == 0:
        points = [[[args.x, args.y]]]
        labels = [[1]]
    else:
        points = [[[pvals[i * 2 + 0], pvals[i * 2 + 1]] for i in range(n_points)]]
        labels = [lvals if lvals else [1] * n_points]

    proc_kwargs = {
        "images": image,
        "input_points": [points],
        "input_labels": [labels],
        "return_tensors": "pt",
    }
    if box_vals:
        if len(box_vals) != 4:
            raise ValueError("--box requires 4 values: x0,y0,x1,y1")
        proc_kwargs["input_boxes"] = [[[box_vals[0], box_vals[1], box_vals[2], box_vals[3]]]]

    inputs = processor(**proc_kwargs).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    save_npy(outdir / "pred_masks.npy", outputs.pred_masks.detach().cpu().numpy())
    save_npy(outdir / "iou_scores.npy", outputs.iou_scores.detach().cpu().numpy())
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
