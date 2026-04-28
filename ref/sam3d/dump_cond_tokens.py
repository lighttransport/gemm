#!/usr/bin/env python3
"""
dump_cond_tokens.py — targeted dump of DINOv2 + PointPatchEmbed + fuser
outputs for verify_dinov2.c and verify_cond_fuser.c.

This side-steps the full InferencePipelinePointMap (which requires
pytorch3d/kaolin/gsplat/MoGe). We directly instantiate
`ss_generator.module.condition_embedder.backbone` (an EmbedderFuser
wrapping two DINOv2-L/14+reg branches and a PointPatchEmbed), load the
slice of ss_generator.ckpt matching those submodules, and run it on
the given image + mask + optional pointmap.

Writes under --outdir:
    dinov2_tokens.npy     (2, 1+n_patches, 1024) f32 — img + mask branches,
                                                      register tokens dropped
    ppe_tokens.npy        (1+ppe_N, 512)          f32 — if --pointmap given
    cond_tokens.npy       (N, 1024)               f32 — fused output

Usage:
    python ref/sam3d/dump_cond_tokens.py \\
        --image fujisan.jpg --mask fujisan_mask.png \\
        --ss-yaml $MODELS/sam3d/checkpoints/ss_generator.yaml \\
        --ss-ckpt $MODELS/sam3d/checkpoints/ss_generator.ckpt \\
        --outdir /tmp/sam3d_ref --seed 42 \\
        [--pointmap p.npy]
"""

import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
# DINOv2 on CPU: force the pure-PyTorch attention/FFN paths; xformers
# has no CPU kernels and would raise NotImplementedError.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse
import sys
import numpy as np


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[dump_cond_tokens] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def pad_to_square(img_np):
    # Replicate sam3d_objects.data.dataset.tdfy.img_processing.pad_to_square_centered
    # (RGB image, centered pad with zeros).
    h, w = img_np.shape[:2]
    s = max(h, w)
    if img_np.ndim == 3:
        out = np.zeros((s, s, img_np.shape[2]), dtype=img_np.dtype)
        out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img_np
    else:
        out = np.zeros((s, s), dtype=img_np.dtype)
        out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img_np
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--ss-yaml", required=True)
    ap.add_argument("--ss-ckpt", required=True)
    ap.add_argument("--pointmap", default=None)
    ap.add_argument("--outdir", default="/tmp/sam3d_ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=518,
                    help="resize target (matches preprocessor)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from PIL import Image
    import sam3d_objects  # noqa: F401

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    # Force CPU — the torch+cu121 build ships kernels that may not
    # match the local GPU arch; we only need fp32 reference tokens.
    device = "cpu"
    dtype = torch.float32

    # 1. Load + preprocess inputs the same way the PreProcessor does:
    #    pad_to_square_centered → Resize(518) → normalize in DINOv2.
    img_u8 = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    msk_u8 = np.asarray(Image.open(args.mask).convert("L"),   dtype=np.uint8)
    save(args.outdir, "input_image.npy", img_u8)
    save(args.outdir, "input_mask.npy",  msk_u8)

    img_sq = pad_to_square(img_u8)
    msk_sq = pad_to_square(msk_u8)

    img_pt = torch.from_numpy(img_sq).permute(2, 0, 1).float()[None] / 255.0
    # Binary mask from user-supplied single-channel mask, matches the
    # pipeline's `rgb_image_mask = (alpha > 0).float()` path.
    msk_pt = (torch.from_numpy(msk_sq).float() > 0).float()[None, None]

    # Bilinear (RGB) / nearest (mask) resize to (size, size); matches
    # torchvision.transforms.Resize with interpolation=0 for the mask.
    img_pt = torch.nn.functional.interpolate(img_pt, size=(args.size, args.size),
                                             mode="bilinear", align_corners=False)
    msk_pt = torch.nn.functional.interpolate(msk_pt, size=(args.size, args.size),
                                             mode="nearest")

    rgb_image = img_pt
    # 1-channel binary mask — DINOv2 replicates to 3 channels internally
    # (see Dino._preprocess_input: `if x.shape[1] == 1: x.repeat(1,3,1,1)`).
    rgb_image_mask = msk_pt

    # 2. Instantiate the embedder-fuser from ss_generator.yaml.
    conf = OmegaConf.load(args.ss_yaml)
    embedder_conf = conf.module.condition_embedder.backbone
    fuser = instantiate(embedder_conf).to(device).to(dtype).eval()

    # 3. Load matching weights from ss_generator.ckpt. The ckpt carries
    #    a deeply-nested dict; the condition_embedder's weights live under
    #    `_base_models.condition_embedder.` (after unwrapping).
    print(f"[dump_cond_tokens] loading ckpt {args.ss_ckpt}...", file=sys.stderr)
    blob = torch.load(args.ss_ckpt, map_location="cpu", weights_only=False)

    # Drill into the state dict root.
    sd = blob
    while isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    prefix = "_base_models.condition_embedder."
    emb_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            emb_sd[k[len(prefix):]] = v
    missing, unexpected = fuser.load_state_dict(emb_sd, strict=False)
    print(f"[dump_cond_tokens] loaded {len(emb_sd)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})",
          file=sys.stderr)
    if missing:
        print(f"[dump_cond_tokens] first 5 missing: {missing[:5]}", file=sys.stderr)

    # 4. Hook per-branch DINOv2 + PPE outputs before fuser projection.
    caps = {}

    def cap(name):
        def _hook(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            caps[name] = x.detach().float().cpu().numpy()
        return _hook

    # embedder_list[i] = (module, kwargs_info). DINOv2 img = 0, DINOv2 msk = 1, PPE = 2.
    dino_img = fuser.embedder_list[0][0]
    dino_msk = fuser.embedder_list[1][0]
    ppe = fuser.embedder_list[2][0]
    dino_img.register_forward_hook(cap("dinov2_img_tokens"))
    dino_msk.register_forward_hook(cap("dinov2_msk_tokens"))
    ppe.register_forward_hook(cap("ppe_tokens"))

    # Also capture the input tensor fed to Dino.forward — that's what
    # the PreProcessor output (or our manually-prepared tensor) looks
    # like just before `_preprocess_input` does resize + normalize.
    def cap_preproc(name):
        def _hook(_m, i):
            x = i[0].detach().float().cpu().numpy()
            caps[name] = x
        return _hook
    dino_img.register_forward_pre_hook(cap_preproc("dino_img_in"))
    dino_msk.register_forward_pre_hook(cap_preproc("dino_msk_in"))

    # 5. Build kwargs matching EmbedderFuser's expected names.
    #    For pos_group == "full", the fuser expects `rgb_image`,
    #    `rgb_image_mask`, `rgb_pointmap` keys.
    kwargs = {"rgb_image": rgb_image.to(device).to(dtype),
              "rgb_image_mask": rgb_image_mask.to(device).to(dtype)}

    if args.pointmap is not None:
        pmap = np.load(args.pointmap).astype(np.float32)
        save(args.outdir, "pointmap.npy", pmap)
        pmap_pt = torch.from_numpy(pmap).to(device).to(dtype)
        # PointPatchEmbed expects (B, 3, H, W).
        if pmap_pt.ndim == 3:  # (H, W, 3)
            pmap_pt = pmap_pt.permute(2, 0, 1)[None]
        kwargs["rgb_pointmap"] = pmap_pt
    else:
        # Skip PPE branch; fuser will warn about missing key.
        pass

    # 6. Strip the `cropped` variant kwargs from each embedder_list entry
    #    — our C runner only produces the `full` pos_group; matching that
    #    scope means the fuser should project & position-embed only the
    #    full-resolution modality per branch.
    new_embedder_list = []
    for (embedder, kwargs_info) in fuser.embedder_list:
        full_only = [(name, pos) for (name, pos) in kwargs_info if pos == "full"]
        new_embedder_list.append((embedder, full_only))
    fuser.embedder_list = new_embedder_list

    if args.pointmap is None:
        fuser.embedder_list = fuser.embedder_list[:2]

    with torch.inference_mode():
        cond = fuser(**kwargs)

    # 7. Write DINOv2 outputs (register tokens are NOT in Dino.forward's
    #    output — it returns cat([CLS, patch_tokens]) only, so no drop needed).
    if "dinov2_img_tokens" in caps and "dinov2_msk_tokens" in caps:
        dino_stack = np.stack([caps["dinov2_img_tokens"][0],
                               caps["dinov2_msk_tokens"][0]], axis=0)
        save(args.outdir, "dinov2_tokens.npy", dino_stack)
    if "dino_img_in" in caps:
        save(args.outdir, "dino_img_in.npy", caps["dino_img_in"][0])
    if "dino_msk_in" in caps:
        save(args.outdir, "dino_msk_in.npy", caps["dino_msk_in"][0])

    if "ppe_tokens" in caps:
        save(args.outdir, "ppe_tokens.npy", caps["ppe_tokens"][0])

    save(args.outdir, "cond_tokens.npy", cond[0].detach().float().cpu().numpy())
    print(f"[dump_cond_tokens] cond shape={cond.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
