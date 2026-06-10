#!/usr/bin/env python3
"""
gen_image_ref.py — run FAIR's sam-3d-objects pipeline with per-stage
torch hooks, dump float32 .npy tensors to REFDIR for the C runner's
verify_*.c binaries to diff against.

Stages dumped (written only if that stage executed):
    input_image.npy         (H, W, 3)   u8  — loaded RGB
    input_mask.npy          (H, W)      u8  — loaded mask
    pointmap.npy            (H, W, 3)   f32 — MoGe output or user input
    dinov2_tokens.npy       (T, 1024)   f32 — DINOv2-L/14+reg (after reg-drop)
    cond_tokens.npy         (N, 1024)   f32 — CondEmbedderFuser out
    ss_latent.npy           (8, 16, 16, 16) f32
    occupancy.npy           (64, 64, 64)    f32
    slat_feats.npy          (M, C)      f32
    slat_coords.npy         (M, 4)      i32
    gaussians.npy           (G, 17)     f32 — PLY-order

Determinism:
  * torch.manual_seed(args.seed) + numpy seed.
  * torch.set_float32_matmul_precision("highest").

Usage:
  python ref/sam3d/gen_image_ref.py \\
      --image fujisan.jpg --mask fujisan_mask.png \\
      --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml \\
      --outdir /tmp/sam3d_ref --seed 42

Requires the sam3d_objects package (installed into ref/sam3d/.venv)
and the HF checkpoint for `facebook/sam-3d-objects`.
"""

# Must be set BEFORE importing sam3d_objects — its __init__ runs a
# Meta-internal bootstrap submodule that isn't in the OSS release.
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import argparse
import sys
import numpy as np


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[gen_image_ref] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--pipeline-yaml", required=True,
                    help="pipeline.yaml from $MODELS/sam3d/checkpoints")
    ap.add_argument("--pointmap", default=None,
                    help="optional pre-computed pointmap .npy (H,W,3) f32; "
                         "bypasses MoGe (useful when moge isn't installed)")
    ap.add_argument("--outdir", default="/tmp/sam3d_ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-steps", type=int, default=None,
                    help="override ss_inference_steps")
    ap.add_argument("--stage2-steps", type=int, default=None,
                    help="override slat_inference_steps")
    ap.add_argument("--stage1-only", action="store_true",
                    help="stop after SS stage (skips SLAT + GS decoder)")
    ap.add_argument("--skip-run", action="store_true",
                    help="Only dump preprocessed image/mask; skip the model.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Preprocess image + mask so step-1 scaffold has at least one dump
    # even before the model code is installed.
    from PIL import Image
    image_pil = Image.open(args.image).convert("RGB")
    mask_pil = Image.open(args.mask).convert("L")
    img = np.asarray(image_pil, dtype=np.uint8)
    msk = np.asarray(mask_pil,   dtype=np.uint8)
    save(args.outdir, "input_image.npy", img)
    save(args.outdir, "input_mask.npy",  msk)

    if args.skip_run:
        print("[gen_image_ref] --skip-run: dumped preproc only", file=sys.stderr)
        return 0

    try:
        import torch
    except Exception as e:
        print(f"[gen_image_ref] torch not importable: {e}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import sam3d_objects  # noqa: F401 — triggers skip-init path
    except Exception as e:
        print(f"[gen_image_ref] cannot import sam3d_objects / hydra: {e}",
              file=sys.stderr)
        return 1

    # pipeline.yaml uses hydra `_target_` instantiation; the ckpt paths
    # inside are relative to the yaml's own directory, so set
    # `workspace_dir` accordingly.
    yaml_dir = os.path.dirname(os.path.abspath(args.pipeline_yaml))
    conf = OmegaConf.load(args.pipeline_yaml)
    OmegaConf.set_struct(conf, False)
    conf._set_flag("allow_objects", True)
    conf.workspace_dir = yaml_dir
    if os.environ.get("SAM3D_REF_DEVICE"):
        conf.device = os.environ["SAM3D_REF_DEVICE"]

    if args.pointmap is not None:
        # Avoid instantiating/downloading MoGe. compute_pointmap() does
        # not use depth_model when a pointmap is supplied, but current
        # upstream still requires the constructor argument to exist.
        conf.depth_model = None
        # Upstream compile warmup calls compute_pointmap() without our
        # pointmap override, so it would dereference the placeholder.
        conf.compile_model = False

    # The native SAM3D verifiers consume the Gaussian path. Upstream
    # instantiates the mesh decoder unconditionally, and that decoder has
    # an internal CUDA-default grid allocation even for CPU references.
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    InferencePipeline.init_slat_decoder_mesh = (
        lambda self, *a, **kw: torch.nn.Identity()
    )

    if str(conf.device) == "cpu":
        try:
            import xformers.ops

            def sdpa_memory_efficient_attention(q, k, v, attn_bias=None,
                                                p=0.0, scale=None, **_kw):
                if q.ndim == 4:
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 1, 3)
                    v = v.permute(0, 2, 1, 3)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale
                    )
                    return out.permute(0, 2, 1, 3)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale
                )

            xformers.ops.memory_efficient_attention = sdpa_memory_efficient_attention
        except Exception as e:
            print(f"[gen_image_ref] xformers CPU fallback unavailable: {e}",
                  file=sys.stderr)

    pipe = instantiate(conf, _recursive_=True)
    pipe.eval() if hasattr(pipe, "eval") else None

    # Register forward hooks to capture per-stage outputs.
    caps = {}

    def cap(name, post=None):
        def _hook(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            if hasattr(x, "detach"):
                x = x.detach().float().cpu().numpy()
            if post is not None:
                x = post(x)
            caps[name] = x
        return _hook

    # Submodules live under pipe.models (generators/decoders) and
    # pipe.condition_embedders (DINOv2 + fuser wrapper).
    ce = pipe.condition_embedders.get("ss_condition_embedder", None)
    if ce is not None:
        # The wrapper owns the DINOv2 backbone (image + mask branches
        # share weights) and the embedder-fuser. Hook the wrapper's
        # forward to capture the fused cond tokens; hook the DINOv2
        # module directly for per-branch tokens.
        ce.register_forward_hook(cap("cond_tokens"))
        dino = getattr(ce, "image_cond_model", None) or getattr(ce, "dinov2", None)
        if dino is not None:
            dino.register_forward_hook(cap("dinov2_tokens"))

    ssg = pipe.models["ss_generator"] if "ss_generator" in pipe.models else None
    if ssg is not None:
        ssg.register_forward_hook(cap("ss_latent"))

    ssd = pipe.models["ss_decoder"] if "ss_decoder" in pipe.models else None
    if ssd is not None:
        ssd.register_forward_hook(cap("occupancy"))

    slg = pipe.models["slat_generator"] if "slat_generator" in pipe.models else None
    if slg is not None:
        slg.register_forward_hook(cap("slat_feats"))

    sgd = pipe.models["slat_decoder_gs"] if "slat_decoder_gs" in pipe.models else None
    if sgd is not None:
        sgd.register_forward_hook(cap("gaussians"))

    # Load optional pre-computed pointmap
    pmap_tensor = None
    if args.pointmap is not None:
        pmap_np = np.load(args.pointmap)
        pmap_tensor = torch.from_numpy(pmap_np.astype(np.float32))
        save(args.outdir, "pointmap.npy", pmap_np.astype(np.float32))

    run_kwargs = dict(
        image=image_pil, mask=mask_pil, seed=args.seed,
        stage1_only=args.stage1_only,
        decode_formats=["gaussian"],
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
    )
    if pmap_tensor is not None:
        run_kwargs["pointmap"] = pmap_tensor
    if args.stage1_steps is not None:
        run_kwargs["stage1_inference_steps"] = args.stage1_steps
    if args.stage2_steps is not None:
        run_kwargs["stage2_inference_steps"] = args.stage2_steps

    with torch.inference_mode():
        out = pipe.run(**run_kwargs)

    for name, arr in caps.items():
        save(args.outdir, name + ".npy", arr)

    # Canonical end-to-end dump — pipe.run() returns a dict of tensors
    # (coords, scale, rotation, gaussian, ...). Write every torch/ndarray
    # entry for reference.
    if isinstance(out, dict):
        for k, v in out.items():
            try:
                if hasattr(v, "detach"):
                    save(args.outdir, f"out_{k}.npy",
                         v.detach().float().cpu().numpy())
                elif isinstance(v, np.ndarray):
                    save(args.outdir, f"out_{k}.npy", v)
            except Exception as e:
                print(f"[gen_image_ref] skip out_{k}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
