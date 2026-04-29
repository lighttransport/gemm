#!/usr/bin/env python3
"""
Generate a self-contained reference for sam3d_body_preprocess_image
and the TopdownAffine helpers.

Outputs (under --outdir, default /tmp/sam3d_body_ref_pp):
  raw_image.npy            (H, W, 3)   u8     raw RGB image
  bbox_xyxy.npy            (4,)        f32    [x1, y1, x2, y2]
  bbox_center_expect.npy   (2,)        f32    GetBBoxCenterScale center
  bbox_scale_expect.npy    (2,)        f32    GetBBoxCenterScale scale (after fix_ar)
  warp_mat_expect.npy      (2, 3)      f32    get_warp_matrix(center, scale, 512, 512)
  preprocess_expect.npy    (3, 512, 512) f32  normalized input to dinov3

All computed via the upstream implementation (cv2 + numpy only; no
sam_3d_body model required, so this runs without CUDA / HF download).
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image


def fix_aspect_ratio(scale, aspect_ratio):
    sw, sh = float(scale[0]), float(scale[1])
    if sw > sh * aspect_ratio:
        return np.array([sw, sw / aspect_ratio], dtype=np.float32)
    else:
        return np.array([sh * aspect_ratio, sh], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * pt[0] - s * pt[1], s * pt[0] + c * pt[1]], dtype=np.float32)


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def get_warp_matrix(center, scale, rot, output_size):
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    rot_rad = np.deg2rad(rot)
    src_w = scale[0]
    dst_w, dst_h = output_size
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    help="x1 y1 x2 y2; defaults to full image")
    ap.add_argument("--out-w", type=int, default=512)
    ap.add_argument("--out-h", type=int, default=512)
    ap.add_argument("--outdir", default="/tmp/sam3d_body_ref_pp")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    if args.bbox is None:
        bbox = np.array([0.0, 0.0, float(W), float(H)], dtype=np.float32)
    else:
        bbox = np.asarray(args.bbox, dtype=np.float32)

    # GetBBoxCenterScale(padding=1.25)
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale_raw = np.array([(x2 - x1) * 1.25, (y2 - y1) * 1.25], dtype=np.float32)
    # TopdownAffine: aspect_ratio=0.75 → then out_w/out_h=1.0
    s1 = fix_aspect_ratio(scale_raw, 0.75)
    scale = fix_aspect_ratio(s1, args.out_w / args.out_h)
    warp = get_warp_matrix(center, scale, 0.0, (args.out_w, args.out_h)).astype(np.float32)

    # cv2.warpAffine + ToTensor + ImageNet norm
    warped = cv2.warpAffine(img, warp, (args.out_w, args.out_h), flags=cv2.INTER_LINEAR)
    chw = warped.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
    pre = (chw - mean) / std

    np.save(os.path.join(args.outdir, "raw_image.npy"), np.ascontiguousarray(img))
    np.save(os.path.join(args.outdir, "bbox_xyxy.npy"), bbox.astype(np.float32))
    np.save(os.path.join(args.outdir, "bbox_center_expect.npy"), center.astype(np.float32))
    np.save(os.path.join(args.outdir, "bbox_scale_expect.npy"), scale.astype(np.float32))
    np.save(os.path.join(args.outdir, "warp_mat_expect.npy"), warp)
    np.save(os.path.join(args.outdir, "preprocess_expect.npy"),
            np.ascontiguousarray(pre))
    print(f"[gen_preprocess_ref] wrote to {args.outdir}", file=sys.stderr)
    print(f"[gen_preprocess_ref] bbox={bbox.tolist()} center={center.tolist()} "
          f"scale={scale.tolist()}", file=sys.stderr)
    print(f"[gen_preprocess_ref] warp=\n{warp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
