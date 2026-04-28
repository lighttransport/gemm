#!/usr/bin/env python3
"""
Dump the MHR jit module's weights to a single safetensors + JSON sidecar
for the C port. See ../../cpu/sam3d_body/MHR_PORT.md for the op set this
covers.

Inputs:
  mhr_model.pt — torch.jit ScriptModule shipped with sam-3d-body-dinov3,
                 typically at /mnt/disk01/models/sam3d-body/dinov3/assets/

Outputs:
  <out_dir>/sam3d_body_mhr_jit.safetensors — raw weights/buffers
  <out_dir>/sam3d_body_mhr_jit.json        — scalar constants + index

Env:
  PYTHONPATH=/tmp/sam-3d-body (or wherever sam_3d_body is checked out;
                               not strictly needed for this script since
                               we only use torch.jit)

Run with the project's ref venv:
  /home/syoyo/work/gemm/main/ref/sam3d-body/.venv/bin/python dump_mhr_assets.py
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


# Keys we pull out of the jit module. Names on the left are what we
# export; paths on the right are module attributes on the jit object.
EXPORT_MAP = [
    # Identity/face shape blends -----------------------------------------
    ("blend_shape.shape_vectors",
     "character_torch.blend_shape.shape_vectors"),             # (45,18439,3) f32
    ("blend_shape.base_shape",
     "character_torch.blend_shape.base_shape"),                # (18439,3)   f32
    ("face_expressions.shape_vectors",
     "face_expressions_model.shape_vectors"),                  # (72,18439,3) f32

    # Parameter transform + skeleton -------------------------------------
    ("parameter_transform",
     "character_torch.parameter_transform.parameter_transform"),  # (889,249) f32
    ("skeleton.joint_translation_offsets",
     "character_torch.skeleton.joint_translation_offsets"),    # (127,3) f32
    ("skeleton.joint_prerotations",
     "character_torch.skeleton.joint_prerotations"),           # (127,4) f32 xyzw
    ("skeleton.pmi",
     "character_torch.skeleton.pmi"),                          # (2,266) i64
    ("skeleton.joint_parents",
     "character_torch.skeleton.joint_parents"),                # (127,)  i32

    # LBS ----------------------------------------------------------------
    ("lbs.inverse_bind_pose",
     "character_torch.linear_blend_skinning.inverse_bind_pose"),  # (127,8) f32
    ("lbs.skin_indices_flattened",
     "character_torch.linear_blend_skinning.skin_indices_flattened"),  # (51337,) i32
    ("lbs.skin_weights_flattened",
     "character_torch.linear_blend_skinning.skin_weights_flattened"),  # (51337,) f32
    ("lbs.vert_indices_flattened",
     "character_torch.linear_blend_skinning.vert_indices_flattened"),  # (51337,) i64

    # Pose correctives ---------------------------------------------------
    ("pose_correctives.sparse_indices",
     "pose_correctives_model.pose_dirs_predictor.0.sparse_indices"),  # (2,53136) i64
    ("pose_correctives.sparse_weight",
     "pose_correctives_model.pose_dirs_predictor.0.sparse_weight"),   # (53136,) f32
    ("pose_correctives.linear_weight",
     "pose_correctives_model.pose_dirs_predictor.2.weight"),          # (55317,3000) f32
]

# Scalar / list constants (not tensors). Dumped to the JSON sidecar so
# the loader can initialise hardcoded sizes without re-opening the .pt.
SCALAR_EXPORTS = [
    ("pmi_buffer_sizes",
     "character_torch.skeleton._pmi_buffer_sizes"),            # [65,56,62,83]
]


def _resolve(obj, dotted):
    parts = dotted.split(".")
    for p in parts:
        obj = getattr(obj, p)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhr-model",
                    default="/mnt/disk01/models/sam3d-body/dinov3/assets/mhr_model.pt")
    ap.add_argument("--out-dir",
                    default="/mnt/disk01/models/sam3d-body/safetensors")
    ap.add_argument("--name", default="sam3d_body_mhr_jit",
                    help="base name for the output .safetensors / .json pair")
    args = ap.parse_args()

    if not os.path.exists(args.mhr_model):
        print(f"[dump_mhr_assets] missing {args.mhr_model}", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    m = torch.jit.load(args.mhr_model, map_location="cpu")

    tensors = {}
    manifest = []
    for out_name, path in EXPORT_MAP:
        t = _resolve(m, path)
        if not torch.is_tensor(t):
            print(f"[dump_mhr_assets] {path}: not a tensor", file=sys.stderr)
            return 3
        # safetensors wants contiguous tensors. Clone + detach to be safe.
        t = t.detach().contiguous().clone()
        tensors[out_name] = t
        manifest.append({
            "name": out_name,
            "jit_path": path,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "nbytes": t.element_size() * t.numel(),
        })
        print(f"  {out_name:44s} {tuple(t.shape)} {str(t.dtype).replace('torch.','')}  "
              f"{t.element_size() * t.numel() / 1024 / 1024:.1f} MiB",
              file=sys.stderr)

    scalars = {}
    for out_name, path in SCALAR_EXPORTS:
        v = _resolve(m, path)
        # JIT attrs may come back as torch ints; coerce to native.
        if isinstance(v, (list, tuple)):
            v = [int(x) for x in v]
        elif isinstance(v, int):
            v = int(v)
        else:
            try:
                v = v.tolist()
            except Exception:
                v = str(v)
        scalars[out_name] = v
        print(f"  {out_name:44s} = {v}  (scalar)", file=sys.stderr)

    sft_path  = os.path.join(args.out_dir, args.name + ".safetensors")
    json_path = os.path.join(args.out_dir, args.name + ".json")

    save_file(tensors, sft_path)
    total_mb = sum(m["nbytes"] for m in manifest) / 1024 / 1024
    print(f"[dump_mhr_assets] wrote {sft_path}  ({total_mb:.1f} MiB, "
          f"{len(tensors)} tensors)", file=sys.stderr)

    with open(json_path, "w") as f:
        json.dump({
            "mhr_model": os.path.abspath(args.mhr_model),
            "safetensors": os.path.basename(sft_path),
            "scalars": scalars,
            "manifest": manifest,
        }, f, indent=2)
    print(f"[dump_mhr_assets] wrote {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
