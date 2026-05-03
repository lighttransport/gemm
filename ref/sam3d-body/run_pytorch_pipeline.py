#!/usr/bin/env python3
"""Slim PyTorch sam-3d-body runner.

Two ways to use this module:

1. CLI (back-compat):
     python run_pytorch_pipeline.py --image person.jpg [--bbox x0 y0 x1 y1] \\
         [--local-ckpt-dir /mnt/disk01/models/sam3d-body/dinov3] \\
         [--device cuda] --out body.obj

2. As an importable library (server/sam3d/pytorch_runner.py uses this):
     from run_pytorch_pipeline import load_model, run
     model, estimator = load_model(local_ckpt_dir=..., device="cuda")
     out = run(estimator, img_rgb_uint8, bbox_xyxy=[x0,y0,x1,y1])
     # out -> {"verts","faces","kp3d","kp2d","cam_t","mhr_params","bbox"}

Writes a Wavefront OBJ matching what test_sam3d_body emits (same vertex
layout, 1-based face indexing).
"""
import argparse
import json
import os
import sys

import numpy as np


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def _ensure_sam3d_body_on_path():
    for cand in ("/tmp/sam-3d-body", os.environ.get("SAM_3D_BODY_DIR", "")):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)


def _import_sam3d_body():
    try:
        from sam_3d_body import (load_sam_3d_body, load_sam_3d_body_hf,
                                 SAM3DBodyEstimator)
        return load_sam_3d_body, load_sam_3d_body_hf, SAM3DBodyEstimator
    except Exception as e:
        raise RuntimeError(
            "cannot import sam_3d_body. Set SAM_3D_BODY_DIR or restore "
            "/tmp/sam-3d-body with: git clone --depth 1 "
            "https://github.com/facebookresearch/sam-3d-body "
            "/tmp/sam-3d-body") from e


def load_model(local_ckpt_dir=None,
               hf_repo_id="facebook/sam-3d-body-dinov3",
               device="cuda"):
    """Load sam-3d-body model + estimator. Returns (model, estimator).

    estimator.process_one_image(img_bgr, bboxes=...) is the main entry.
    """
    import torch
    torch.set_float32_matmul_precision("highest")
    _ensure_sam3d_body_on_path()
    load_sam_3d_body, load_sam_3d_body_hf, SAM3DBodyEstimator = (
        _import_sam3d_body())

    if local_ckpt_dir:
        ckpt_path = os.path.join(local_ckpt_dir, "model.ckpt")
        mhr_path = os.path.join(local_ckpt_dir, "assets", "mhr_model.pt")
        model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_path,
                                            device=device,
                                            mhr_path=mhr_path)
    else:
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=model_cfg,
        human_detector=None, human_segmentor=None, fov_estimator=None)
    return model, estimator


def _to_numpy(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def run(estimator, img_rgb, bbox_xyxy=None):
    """Run the estimator on a single image.

    img_rgb: HxWx3 uint8 RGB
    bbox_xyxy: optional [x0,y0,x1,y1] in image coordinates (no detector
               available without detectron2, so this is recommended).
    """
    import cv2
    import torch
    img_bgr = cv2.cvtColor(np.asarray(img_rgb, dtype=np.uint8),
                           cv2.COLOR_RGB2BGR)
    bboxes = None
    if bbox_xyxy is not None:
        bboxes = np.asarray([bbox_xyxy], dtype=np.float32)

    with torch.inference_mode():
        outputs = estimator.process_one_image(img_bgr, bboxes=bboxes)

    if not outputs:
        raise RuntimeError("no detections (no bbox provided and no detector)")

    person = outputs[0]
    verts = _to_numpy(person.get("pred_vertices"))
    if verts is None:
        raise RuntimeError("missing pred_vertices in estimator output")
    verts = verts.reshape(-1, 3)

    faces = getattr(estimator, "faces", None)
    if faces is None:
        raise RuntimeError("missing estimator.faces")
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    return {
        "verts":       verts,
        "faces":       faces,
        "kp3d":        _to_numpy(person.get("pred_keypoints_3d")),
        "kp2d":        _to_numpy(person.get("pred_keypoints_2d")),
        "cam_t":       _to_numpy(person.get("pred_cam_t_world",
                                            person.get("pred_cam_t"))),
        "mhr_params":  _to_numpy(person.get("pred_mhr_params",
                                            person.get("mhr_params"))),
        "bbox":        list(bbox_xyxy) if bbox_xyxy is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("x0", "y0", "x1", "y1"))
    ap.add_argument("--hf-repo-id", default="facebook/sam-3d-body-dinov3")
    ap.add_argument("--local-ckpt-dir", default=None,
                    help="local snapshot dir (overrides --hf-repo-id)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="OBJ output path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from PIL import Image
    img_rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, estimator = load_model(local_ckpt_dir=args.local_ckpt_dir,
                              hf_repo_id=args.hf_repo_id,
                              device=args.device)
    try:
        out = run(estimator, img_rgb, bbox_xyxy=args.bbox)
    except RuntimeError as e:
        print(f"[run_pytorch_pipeline] {e}", file=sys.stderr)
        return 2

    write_obj(args.out, out["verts"], out["faces"])
    side = args.out + ".json"
    with open(side, "w") as f:
        json.dump({
            "bbox":       out["bbox"],
            "image_w":    int(img_rgb.shape[1]),
            "image_h":    int(img_rgb.shape[0]),
            "cam_t":      _list_or_none(out["cam_t"]),
            "mhr_params": _list_or_none(out["mhr_params"]),
            "kp3d":       _list_or_none(out["kp3d"]),
            "kp2d":       _list_or_none(out["kp2d"]),
        }, f)
    print(f"[run_pytorch_pipeline] V={out['verts'].shape[0]} "
          f"F={out['faces'].shape[0]} -> {args.out}", file=sys.stderr)
    return 0


def _list_or_none(a):
    if a is None:
        return None
    return np.asarray(a).reshape(-1).tolist() if a.ndim <= 1 \
        else np.asarray(a).tolist()


if __name__ == "__main__":
    sys.exit(main() or 0)
