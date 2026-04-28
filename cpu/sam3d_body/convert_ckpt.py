#!/usr/bin/env python3
"""
convert_ckpt.py — slice the sam-3d-body HF checkpoint into per-module
safetensors files consumed by the C runner.

Produces (per `--backbone-name`):
    <out>/sam3d_body_<variant>.safetensors          — encoder backbone
    <out>/sam3d_body_<variant>_decoder.safetensors  — promptable decoder
    <out>/sam3d_body_<variant>_mhr_head.safetensors — MHR head + camera

For backwards compatibility, when `--backbone-name dinov3`
the script ALSO writes the legacy un-tagged filenames
(`sam3d_body_decoder.safetensors`, `sam3d_body_mhr_head.safetensors`)
so existing CPU/CUDA runners that pre-date the variant flag keep
working unchanged. The dinov3 backbone file is already named with
the variant tag (`sam3d_body_dinov3.safetensors`) and matches the
legacy convention.

CRITICAL: the two ckpts ship SEPARATELY trained weights. Decoder
and head tensor SHAPES match across variants but VALUES differ by
~0.1–0.9 max-abs (sanity-checked 2026-04-26). Per-variant tagging
of decoder + MHR head outputs is mandatory to avoid silently mixing
DINOv3 decoder values with ViT-H backbone outputs.

The split heuristic strips the upstream wrapper prefix (`module.` /
`model.` / `backbone.` / etc.) then routes each tensor into one of
the buckets based on its top-level prefix. A manifest is printed to
stderr for verification.

Usage:
    # DINOv3-H+ variant (default backbone-name=dinov3):
    python cpu/sam3d_body/convert_ckpt.py \\
        /mnt/disk01/models/sam3d-body/dinov3 \\
        -o /mnt/disk01/models/sam3d-body/safetensors

    # ViT-H variant — pass --backbone-name vith so the slice file
    # is named sam3d_body_vith.safetensors:
    python cpu/sam3d_body/convert_ckpt.py \\
        /mnt/disk01/models/sam3d-body/vith \\
        --backbone-name vith \\
        -o /mnt/disk01/models/sam3d-body/safetensors
"""

import argparse
import os
import sys


def build_routes(backbone_name):
    return [
        # (destination bucket, tuple-of-key-prefixes)
        # Encoder backbone — file name carries the variant tag so the
        # CPU/CUDA runners can select via `--backbone {dinov3,vith}`.
        (f"sam3d_body_{backbone_name}",          ("backbone.",)),

        # Promptable transformer decoder + prompt encoder + token-fusion
        # MLPs + keypoint/ray conditioning heads. Both body and hand
        # decoders live here; `_hand` variants run the same modules on a
        # cropped hand region.
        (f"sam3d_body_{backbone_name}_decoder",
                                ("decoder.", "decoder_hand.",
                                 "prompt_encoder.", "prompt_to_token.",
                                 "init_to_token_mhr.", "init_to_token_mhr_hand.",
                                 "prev_to_token_mhr.", "prev_to_token_mhr_hand.",
                                 "hand_pe_layer.", "hand_box_embedding.",
                                 "hand_cls_embed.",
                                 "keypoint_feat_linear.",
                                 "keypoint_feat_linear_hand.",
                                 "keypoint_embedding.",
                                 "keypoint_embedding_hand.",
                                 "keypoint3d_embedding.",
                                 "keypoint3d_embedding_hand.",
                                 "keypoint_posemb_linear.",
                                 "keypoint_posemb_linear_hand.",
                                 "keypoint3d_posemb_linear.",
                                 "keypoint3d_posemb_linear_hand.",
                                 "ray_cond_emb.", "ray_cond_emb_hand.")),

        # MHR pose/camera regression head. Operates on the decoder's
        # pooled tokens to produce body pose, camera, and bbox regressors.
        # The MHR _skinning_ assets (kintree + LBS weights) ship in a
        # separate mhr_model.pt and are converted by convert_mhr_assets.py.
        (f"sam3d_body_{backbone_name}_mhr_head",
                                ("head_pose.", "head_pose_hand.",
                                 "head_camera.", "head_camera_hand.",
                                 "bbox_embed.",
                                 "init_camera.", "init_camera_hand.",
                                 "init_pose.", "init_pose_hand.")),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir",
                    help="directory containing model.ckpt (and assets/)")
    ap.add_argument("-o", "--outdir", required=True,
                    help="output directory for per-module safetensors")
    ap.add_argument("--ckpt", default="model.ckpt",
                    help="filename within ckpt_dir (default model.ckpt)")
    ap.add_argument("--backbone-name", default="dinov3",
                    choices=("dinov3", "vith"),
                    help="backbone variant tag — names the output "
                         "safetensors slice (sam3d_body_<tag>.safetensors). "
                         "Decoder + MHR head slices are bit-identical "
                         "across variants so they overwrite each other "
                         "when both runs target the same outdir.")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, args.ckpt)
    print(f"[convert] loading {path}", file=sys.stderr)
    obj = torch.load(path, map_location="cpu", weights_only=False)

    sd = obj
    for key in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]

    if not isinstance(sd, dict):
        print(f"[convert] unexpected payload type: {type(sd)}", file=sys.stderr)
        return 1

    routes = build_routes(args.backbone_name)
    buckets = {name: {} for name, _ in routes}
    unmatched = {}
    for k, v in sd.items():
        kk = k
        for pfx in ("module.", "model."):
            if kk.startswith(pfx):
                kk = kk[len(pfx):]
        if not hasattr(v, "to"):
            unmatched[kk] = v
            continue
        # clone() so tensors that share storage in the ckpt (e.g. MHR
        # kintree assets replicated across body + hand heads) get
        # independent blobs in the safetensors file. safetensors
        # refuses shared storage.
        t = v.detach().cpu().contiguous().clone()

        # The encoder ships bf16. Our CPU matmul path has a threaded
        # fp16 kernel but only a scalar bf16 fallback. To make inference
        # fast while keeping precision tight, selectively cast:
        #   - Weight matrices (2-D; the big matmul kernels) → fp16
        #   - Biases, gains, norms, embeddings (small; precision-sensitive) → fp32
        # Rationale: small tensors contribute <0.1% of memory but every
        # layer consumes their full precision twice; keeping them fp32
        # is free. Large matmul weights dominate bandwidth and benefit
        # from the threaded fp16 gemm; f16 mantissa (10b) >> bf16 (7b)
        # so no precision is lost vs the upstream bf16 compute.
        if t.dtype == torch.bfloat16:
            is_big_matrix = (t.dim() == 2 and min(t.shape) >= 64)
            t = t.to(torch.float16 if is_big_matrix else torch.float32)
        placed = False
        for bucket, prefixes in routes:
            if any(kk.startswith(p) for p in prefixes):
                buckets[bucket][kk] = t
                placed = True
                break
        if not placed:
            unmatched[kk] = t

    # Legacy un-tagged aliases for the dinov3 variant — preserved so
    # existing CPU/CUDA runners + verify binaries that pre-date the
    # variant flag (and hard-code "sam3d_body_decoder.safetensors" /
    # "sam3d_body_mhr_head.safetensors") keep working unchanged.
    legacy_aliases = {}
    if args.backbone_name == "dinov3":
        legacy_aliases = {
            f"sam3d_body_{args.backbone_name}_decoder":   "sam3d_body_decoder",
            f"sam3d_body_{args.backbone_name}_mhr_head":  "sam3d_body_mhr_head",
        }

    for bucket, tensors in buckets.items():
        if not tensors:
            print(f"[convert] WARN: bucket {bucket} is empty "
                  "(no matching prefixes)", file=sys.stderr)
            continue
        out = os.path.join(args.outdir, f"{bucket}.safetensors")
        save_file(tensors, out)
        print(f"[convert] wrote {out}  ({len(tensors)} tensors)",
              file=sys.stderr)
        alias = legacy_aliases.get(bucket)
        if alias:
            alias_path = os.path.join(args.outdir, f"{alias}.safetensors")
            save_file(tensors, alias_path)
            print(f"[convert] wrote {alias_path}  (legacy alias)",
                  file=sys.stderr)

    if unmatched:
        print(f"[convert] WARN: {len(unmatched)} unmatched keys; first 20:",
              file=sys.stderr)
        for k in list(unmatched)[:20]:
            print(f"    {k}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
