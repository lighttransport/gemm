#!/usr/bin/env python3
"""
moge_pointmap.py - generate a SAM3D-compatible MoGe pointmap.

Writes a float32 NumPy array with shape (H, W, 3).  The CUDA runner can
consume the output with --pointmap, or generate it on demand with --moge.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def default_model_path():
    local = "/mnt/disk01/models/moge-vitl/model.pt"
    return local if os.path.exists(local) else "Ruicheng/moge-vitl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input RGB image")
    ap.add_argument("--out", required=True, help="output .npy pointmap path")
    ap.add_argument("--model", default=default_model_path(),
                    help="MoGe HF repo id or local model.pt path")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--hf-cache", default="/mnt/disk01/models/hf-cache",
                    help="HF cache directory; ignored if it does not exist")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--use-fp16", action="store_true",
                    help="request MoGe fp16 inference (default is fp32)")
    args = ap.parse_args()

    if args.hf_cache and os.path.isdir(args.hf_cache):
        os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)

    try:
        import torch
        from moge.model.v1 import MoGeModel
    except Exception as exc:
        print(f"[moge_pointmap] cannot import torch/moge: {exc}", file=sys.stderr)
        return 1

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[moge_pointmap] cuda requested but unavailable; using cpu", file=sys.stderr)
        device = "cpu"

    img = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8, copy=True)
    print(f"[moge_pointmap] image {img.shape} uint8", file=sys.stderr, flush=True)

    try:
        model = MoGeModel.from_pretrained(args.model).to(device).eval()
    except Exception as exc:
        print(f"[moge_pointmap] model load failed ({args.model}): {exc}", file=sys.stderr)
        return 2
    print(f"[moge_pointmap] model loaded from {args.model} on {device}", file=sys.stderr,
          flush=True)

    tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    infer_kwargs = {
        "resolution_level": args.resolution_level,
        "use_fp16": bool(args.use_fp16 and device.startswith("cuda")),
    }
    if args.num_tokens is not None:
        infer_kwargs["num_tokens"] = args.num_tokens

    try:
        with torch.inference_mode():
            out = model.infer(tensor, **infer_kwargs)
    except Exception as exc:
        print(f"[moge_pointmap] inference failed: {exc}", file=sys.stderr)
        return 3

    if "points" not in out:
        print("[moge_pointmap] MoGe output has no 'points' tensor", file=sys.stderr)
        return 4

    pmap = out["points"].detach().float().cpu().numpy()
    if pmap.ndim != 3 or pmap.shape[2] != 3:
        print(f"[moge_pointmap] bad pointmap shape: {pmap.shape}", file=sys.stderr)
        return 4

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, np.ascontiguousarray(pmap.astype(np.float32)))
    finite = np.isfinite(pmap).mean() * 100.0
    print(f"[moge_pointmap] wrote {args.out} shape={pmap.shape} "
          f"dtype=float32 finite={finite:.2f}%", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
