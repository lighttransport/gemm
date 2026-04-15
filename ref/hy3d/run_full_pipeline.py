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
                        default=False,
                        help="try accelerate.cpu_offload on each sub-module. "
                             "WARNING: upstream's enable_model_cpu_offload is "
                             "half-ported from diffusers and the straight "
                             "accelerate.cpu_offload path triggers a "
                             "meta-tensor bug in torchvision Normalize inside "
                             "the conditioner. Full-resident is recommended "
                             "if your GPU has >=10 GB VRAM.")
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false",
                        help="keep all models resident on GPU (default)")
    parser.add_argument("--sequential-offload", action="store_true",
                        help="use accelerate sequential CPU offload (lowest VRAM, slowest, broken)")
    parser.add_argument("--save-latents", default=None,
                        help="if set, dump the final DiT latents to this .npy path")
    parser.add_argument("--trace-dir", default=None,
                        help="dump per-stage .npy tensors for layer-by-layer "
                             "comparison vs the CUDA runner")
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

    # Upstream Hunyuan3DDiTPipeline inherits enable_model_cpu_offload from
    # diffusers.DiffusionPipeline but never sets up .components / .device /
    # .to, and the pipeline's __call__ mixes CPU latents with CUDA model
    # outputs after offload. Instead, apply accelerate.cpu_offload directly
    # to each top-level sub-module — it attaches forward hooks that move
    # just the model weights onto execution_device during forward and back
    # to CPU after, leaving all other tensors (latents, contexts) on CUDA.
    if args.sequential_offload or args.low_vram:
        from accelerate import cpu_offload
        tag = "sequential CPU offload" if args.sequential_offload \
                else "model CPU offload"
        print(f"Enabling {tag} (accelerate.cpu_offload on each sub-module)...")
        for name in ("conditioner", "model", "vae"):
            mod = getattr(pipe, name)
            if mod is not None:
                cpu_offload(mod, execution_device=args.device)
    else:
        print("All models resident on GPU (no offload).")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGBA")
    if image.mode == "RGB":
        from hy3dshape.rembg import BackgroundRemover
        image = BackgroundRemover()(image)

    # With accelerate.cpu_offload each sub-module still reports execution_device
    # as cuda, so the pipeline's latents/prepare_latents path stays on cuda.
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # ---- Tracing hooks (layer-by-layer debug vs CUDA runner) ----
    hook_handles = []
    if args.trace_dir:
        import numpy as np
        os.makedirs(args.trace_dir, exist_ok=True)

        def _save(name, tensor):
            if hasattr(tensor, "detach"):
                tensor = tensor.detach()
            if hasattr(tensor, "float"):
                tensor = tensor.float().cpu().numpy()
            path = os.path.join(args.trace_dir, name + ".npy")
            np.save(path, tensor)
            print(f"    trace: {name} shape={tensor.shape} "
                  f"min={tensor.min():+.4f} max={tensor.max():+.4f} "
                  f"mean={tensor.mean():+.4f} std={tensor.std():.4f}")

        # Hook the Dinov2 HF model inside the ImageEncoder
        dino = pipe.conditioner.main_image_encoder.model

        def _dino_pre(module, args_):
            _save("01_dino_input", args_[0])

        def _dino_post(module, args_, output):
            # transformers returns a BaseModelOutput[WithPooling]
            lhs = getattr(output, "last_hidden_state", None)
            if lhs is None and isinstance(output, tuple):
                lhs = output[0]
            _save("02_dino_output", lhs)

        hook_handles.append(dino.register_forward_pre_hook(_dino_pre))
        hook_handles.append(dino.register_forward_hook(_dino_post))

        # Hook the DiT model itself: inputs (x, t, contexts) + output
        _dit_state = {"step": 0}

        def _dit_pre(module, args_):
            x, t, contexts = args_[0], args_[1], args_[2]
            step = _dit_state["step"]
            main_ctx = contexts["main"] if isinstance(contexts, dict) else contexts
            if step == 0:
                _save("03_dit_context_cfg", main_ctx)  # first step carries the CFG-doubled context
                _save("04_dit_latents_step0", x)
            _save(f"05_dit_input_x_{step:03d}", x)
            _save(f"05_dit_input_t_{step:03d}",
                  torch.as_tensor(t).reshape(-1))

        def _dit_post(module, args_, output):
            step = _dit_state["step"]
            _save(f"06_dit_output_{step:03d}", output)
            _dit_state["step"] += 1

        hook_handles.append(pipe.model.register_forward_pre_hook(_dit_pre))
        hook_handles.append(pipe.model.register_forward_hook(_dit_post))

        # Hook the ShapeVAE forward (post-KL + decoder). It's called once in _export.
        def _vae_pre(module, args_):
            _save("07_vae_input_latents", args_[0])

        def _vae_post(module, args_, output):
            _save("08_vae_decoded", output)

        hook_handles.append(pipe.vae.register_forward_pre_hook(_vae_pre))
        hook_handles.append(pipe.vae.register_forward_hook(_vae_post))

        print(f"Tracing enabled: {args.trace_dir}")

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

    for h in hook_handles:
        h.remove()

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
