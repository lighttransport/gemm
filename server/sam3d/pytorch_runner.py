"""In-process PyTorch sam-3d-body backend.

Loads the model once at startup and serves requests directly (no
subprocess). Used by server/sam3d/app.py.

Public API:
    load(local_ckpt_dir=None, hf_repo_id=None, device="cuda")
    infer(img_rgb_uint8, bbox_xyxy=None) -> {
        "obj_text":   str,                 # Wavefront OBJ
        "n_verts":    int,
        "n_faces":    int,
        "json":       dict,                # sidecar metadata
    }
"""
import os
import sys
import threading
import time

import numpy as np


_LOCK = threading.Lock()
_STATE = {"model": None, "estimator": None}


def _ensure_ref_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.abspath(os.path.join(here, "..", "..", "ref",
                                            "sam3d-body"))
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)


def load(local_ckpt_dir=None,
         hf_repo_id="facebook/sam-3d-body-dinov3",
         device="cuda"):
    """Load model + estimator into module state (idempotent)."""
    with _LOCK:
        if _STATE["estimator"] is not None:
            return
        _ensure_ref_on_path()
        from run_pytorch_pipeline import load_model
        t0 = time.time()
        model, estimator = load_model(local_ckpt_dir=local_ckpt_dir,
                                      hf_repo_id=hf_repo_id, device=device)
        print(f"[pytorch_runner] loaded in {time.time()-t0:.1f}s "
              f"(device={device}, "
              f"src={'local:'+local_ckpt_dir if local_ckpt_dir else 'hf:'+hf_repo_id})",
              file=sys.stderr, flush=True)
        _STATE["model"] = model
        _STATE["estimator"] = estimator


def available():
    return _STATE["estimator"] is not None


def _verts_to_obj(verts, faces):
    out = []
    for v in verts:
        out.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for tri in faces:
        out.append(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
    return "".join(out)


def _list_or_none(a):
    if a is None:
        return None
    a = np.asarray(a)
    return a.reshape(-1).tolist() if a.ndim <= 1 else a.tolist()


def infer(img_rgb, bbox_xyxy=None):
    """Run inference; returns obj_text + sidecar JSON dict."""
    if _STATE["estimator"] is None:
        raise RuntimeError("pytorch backend not loaded "
                           "(call load() at server startup)")
    _ensure_ref_on_path()
    from run_pytorch_pipeline import run
    with _LOCK:
        out = run(_STATE["estimator"], img_rgb, bbox_xyxy=bbox_xyxy)
    verts, faces = out["verts"], out["faces"]
    obj = _verts_to_obj(verts, faces)
    side = {
        "bbox":       out["bbox"],
        "image":      {"width":  int(img_rgb.shape[1]),
                       "height": int(img_rgb.shape[0])},
        "cam_t":      _list_or_none(out["cam_t"]),
        "mhr_params": _list_or_none(out["mhr_params"]),
        "keypoints_3d": _list_or_none(out["kp3d"]),
        "keypoints_2d": _list_or_none(out["kp2d"]),
    }
    return {
        "obj_text": obj,
        "n_verts":  int(verts.shape[0]),
        "n_faces":  int(faces.shape[0]),
        "json":     side,
    }
