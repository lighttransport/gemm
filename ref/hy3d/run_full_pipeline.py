"""End-to-end PyTorch reference for Hunyuan3D-2.1 shape generation.

Thin wrapper around hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline that:
  - Resolves weights from a local directory (no HuggingFace download) by
    setting HY3DGEN_MODELS for smart_load_model.
  - Defaults to low-VRAM mode (model CPU offload, one stage on GPU at a
    time, sequencing conditioner -> model -> vae).
  - Saves the output mesh as .glb and .obj.

Usage (from ref/hy3d/ with the uv-managed venv):

    uv run python run_full_pipeline.py \
        --image /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png \
        --out  /tmp/hy3d_ref.glb \
        --steps 30 --guidance 5.0 --octree 256 --seed 42

Environment (the script sets these automatically if unset):
    HY3D_REPO       = path that contains the `hy3dshape` python package dir.
                      Default: /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape
    HY3DGEN_MODELS  = parent dir of `Hunyuan3D-2.1/` (the weight tree).
                      Default: /mnt/disk01/models
    HY3D_MODEL_NAME = model folder under HY3DGEN_MODELS.
                      Default: Hunyuan3D-2.1
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image


def _setup_sys_path():
    repo = os.environ.get("HY3D_REPO",
                          "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape")
    if not os.path.isdir(os.path.join(repo, "hy3dshape")):
        sys.exit(f"ERROR: hy3dshape package not found under {repo}. "
                 "Set HY3D_REPO to the repo's hy3dshape/ dir.")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # Point smart_load_model at local weights: it joins
    #   {HY3DGEN_MODELS}/{model_path}/{subfolder}
    os.environ.setdefault("HY3DGEN_MODELS", "/mnt/disk01/models")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, help="input image (PNG/JPG)")
    parser.add_argument("--out", default="hy3d_ref.glb",
                        help="output mesh; '.glb' or '.obj' (an '.obj' sibling is also written)")
    parser.add_argument("--model-name", default=os.environ.get("HY3D_MODEL_NAME", "Hunyuan3D-2.1"),
                        help="model folder under HY3DGEN_MODELS")
    parser.add_argument("--steps", type=int, default=30,
                        help="DiT inference steps")
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="classifier-free-guidance scale")
    parser.add_argument("--octree", type=int, default=256,
                        help="marching-cubes grid resolution")
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--box-v", type=float, default=1.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--low-vram", dest="low_vram", action="store_true",
                        default=True,
                        help="enable model CPU offload (default)")
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false",
                        help="keep all models resident on GPU")
    parser.add_argument("--sequential-offload", action="store_true",
                        help="use accelerate sequential CPU offload (lowest VRAM, slowest)")
    parser.add_argument("--save-latents", default=None,
                        help="if set, dump the final DiT latents to this .npy path")
    args = parser.parse_args()

    _setup_sys_path()

    # Imports happen AFTER sys.path setup
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Loading pipeline: HY3DGEN_MODELS={os.environ['HY3DGEN_MODELS']} "
          f"model={args.model_name} dtype={args.dtype}")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_name, device=args.device, dtype=dtype,
        use_safetensors=False, variant="fp16")
    print(f"  loaded in {time.time() - t0:.1f}s")

    if args.sequential_offload:
        print("Enabling sequential CPU offload (accelerate)...")
        pipe.enable_sequential_cpu_offload()
    elif args.low_vram:
        print("Enabling model CPU offload (one stage on GPU at a time)...")
        pipe.enable_model_cpu_offload()
    else:
        print("All models resident on GPU (no offload).")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGBA")
    if image.mode == "RGB":
        from hy3dshape.rembg import BackgroundRemover
        image = BackgroundRemover()(image)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"Sampling: steps={args.steps} guidance={args.guidance} "
          f"octree={args.octree} seed={args.seed}")
    t0 = time.time()
    outputs = pipe(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        octree_resolution=args.octree,
        mc_level=args.mc_level,
        num_chunks=args.num_chunks,
        box_v=args.box_v,
        generator=generator,
        output_type="trimesh",
    )
    elapsed = time.time() - t0
    mesh = outputs[0]
    print(f"  sampled in {elapsed:.1f}s: "
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mesh.export(out_path)
    print(f"  wrote {out_path}")

    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".obj":
        obj_path = base + ".obj"
        mesh.export(obj_path)
        print(f"  wrote {obj_path}")


if __name__ == "__main__":
    main()
