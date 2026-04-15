"""PyTorch reference runner for Hunyuan3D-2.1 texture generation (hy3dpaint).

Loads the multiview diffusion paint pipeline from local weights and textures
an input mesh with a style/condition image. Intended to be run against the
shape-gen output (e.g. /tmp/hy3d_ref.glb from run_full_pipeline.py) so the
two stages can be debugged independently.

All weight paths default to the local mirror:
    /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1
    /mnt/disk01/models/dinov2-giant
    /mnt/disk01/models/realesrgan/RealESRGAN_x4plus.pth
Override via --paint-path / --dino-path / --realesrgan-ckpt.

Usage:
    uv run python run_texturegen.py \
        --mesh /tmp/hy3d_ref.glb \
        --image /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png \
        --out /tmp/hy3d_textured.obj \
        [--trace-dir /tmp/hy3d_paint_trace]

--trace-dir installs forward hooks on the multiview UNet, VAE, and DINOv2
encoder to dump per-stage tensors (01_dino_input, 02_dino_output,
03_unet_sample_NNN, 04_unet_output_NNN, 05_vae_decoded, ...) so the shapes
and activations can be compared against a future CUDA port layer-by-layer.
"""
import argparse
import importlib
import os
import sys
import time
from typing import Optional


def _setup_env(hy3d_repo: str, paint_path: str):
    # hy3dpaint imports are relative (`from DifferentiableRenderer import ...`,
    # `from utils.multiview_utils import ...`); the repo's demo.py prepends
    # the hy3dpaint dir to sys.path and cds into it. We mirror that.
    if not os.path.isdir(hy3d_repo):
        sys.exit(f"ERROR: HY3D_REPO not a directory: {hy3d_repo}")
    hy3dpaint_dir = os.path.join(hy3d_repo, "..", "hy3dpaint")
    hy3dpaint_dir = os.path.abspath(hy3dpaint_dir)
    if not os.path.isdir(hy3dpaint_dir):
        sys.exit(f"ERROR: hy3dpaint dir not found: {hy3dpaint_dir}")
    # hy3dshape (for shape decoding types)
    sys.path.insert(0, hy3d_repo)
    # hy3dpaint top-level (so `from DifferentiableRenderer...` and `from utils...`)
    sys.path.insert(0, hy3dpaint_dir)
    # hunyuanpaintpbr is next to hy3dpaint (referenced as relative import inside paint)
    os.chdir(hy3dpaint_dir)

    # Monkey-patch huggingface_hub.snapshot_download so that multiviewDiffusionNet
    # resolves tencent/Hunyuan3D-2.1 to our local mirror WITHOUT hitting the hub.
    # multiview_utils.py does `os.path.join(snapshot, "hunyuan3d-paintpbr-v2-1")`
    # so we return paint_path (the dir that CONTAINS that subfolder), not its parent.
    local_root = paint_path  # e.g. /mnt/disk01/models/Hunyuan3D-2.1
    import huggingface_hub
    _orig_snapshot = huggingface_hub.snapshot_download

    def _patched_snapshot(repo_id=None, *args, **kwargs):
        if repo_id and "Hunyuan3D-2.1" in str(repo_id):
            return local_root
        return _orig_snapshot(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = _patched_snapshot
    # multiview_utils imports huggingface_hub then does huggingface_hub.snapshot_download,
    # so patching the module attribute is enough.

    return hy3dpaint_dir


def _install_trace_hooks(pipeline, paint_pipe, trace_dir: str):
    import numpy as np
    import torch

    os.makedirs(trace_dir, exist_ok=True)
    handles = []

    def _save(name, t):
        # Recursively unwrap tuples/lists/dicts to a single tensor
        if isinstance(t, (list, tuple)):
            for i, item in enumerate(t):
                _save(f"{name}_{i}", item)
            return
        if isinstance(t, dict):
            for k, v in t.items():
                _save(f"{name}_{k}", v)
            return
        if hasattr(t, "sample") and not isinstance(t, torch.Tensor):
            _save(name, t.sample)
            return
        if isinstance(t, torch.Tensor):
            arr = t.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            try:
                arr = np.asarray(t)
            except Exception as e:
                print(f"    trace SKIP {name}: type={type(t).__name__} err={e}")
                return
        np.save(os.path.join(trace_dir, name + ".npy"), arr)
        print(f"    trace: {name} shape={arr.shape} "
              f"min={arr.min():+.4f} max={arr.max():+.4f} "
              f"mean={arr.mean():+.4f} std={arr.std():.4f}")

    # 1. DINO encoder (if used) — hook the underlying HF model forward
    dino_v2 = getattr(pipeline, "dino_v2", None)
    if dino_v2 is not None:
        hf_dino = getattr(dino_v2, "dino_v2", None)  # Dino_v2 wraps HF AutoModel
        if hf_dino is not None:
            def _dino_pre(module, args_):
                if args_:
                    _save("01_dino_input", args_[0])

            def _dino_post(module, args_, output):
                lhs = getattr(output, "last_hidden_state", None)
                if lhs is None and isinstance(output, tuple):
                    lhs = output[0]
                if lhs is not None:
                    _save("02_dino_output", lhs)

            handles.append(hf_dino.register_forward_pre_hook(_dino_pre))
            handles.append(hf_dino.register_forward_hook(_dino_post))

    # 2. Multiview UNet — hook forward to capture sample / encoder_hidden_states / output per step
    unet = paint_pipe.unet
    state = {"step": 0}

    def _unet_pre(module, args_, kwargs_):
        step = state["step"]
        # UNet2DConditionModel-style signature: (sample, timestep, encoder_hidden_states, ...)
        sample = args_[0] if args_ else kwargs_.get("sample")
        t = args_[1] if len(args_) > 1 else kwargs_.get("timestep")
        ehs = args_[2] if len(args_) > 2 else kwargs_.get("encoder_hidden_states")
        if sample is not None:
            _save(f"03_unet_sample_{step:03d}", sample)
        if t is not None:
            try:
                import torch
                _save(f"03_unet_t_{step:03d}", torch.as_tensor(t).reshape(-1))
            except Exception:
                pass
        if ehs is not None and step == 0:
            _save("03_unet_encoder_hs", ehs)

    def _unet_post(module, args_, kwargs_, output):
        step = state["step"]
        out = getattr(output, "sample", output)
        if out is not None:
            _save(f"04_unet_output_{step:03d}", out)
        state["step"] += 1

    handles.append(unet.register_forward_pre_hook(_unet_pre, with_kwargs=True))
    handles.append(unet.register_forward_hook(_unet_post, with_kwargs=True))

    # 3. VAE — hook decoder forward to capture latents -> images
    vae = paint_pipe.vae
    def _vae_dec_pre(module, args_):
        if args_:
            _save("07_vae_decode_input", args_[0])

    def _vae_dec_post(module, args_, output):
        samp = getattr(output, "sample", output)
        if samp is not None:
            _save("08_vae_decode_output", samp)

    handles.append(vae.decode.__self__.register_forward_pre_hook(_vae_dec_pre))
    # decode is a method, not an nn.Module, so also hook the decoder module directly
    if hasattr(vae, "decoder"):
        handles.append(vae.decoder.register_forward_pre_hook(
            lambda m, a: _save("07_vae_decoder_input", a[0]) if a else None))
        handles.append(vae.decoder.register_forward_hook(
            lambda m, a, o: _save("08_vae_decoder_output", o)))
    return handles


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    ap.add_argument("--mesh", required=True, help="input mesh (.obj or .glb)")
    ap.add_argument("--image", required=True,
                    help="style / condition image (PNG/JPG, ideally with alpha)")
    ap.add_argument("--out", default="/tmp/hy3d_textured.obj",
                    help="output textured mesh path (.obj; a sibling .glb is also written)")
    ap.add_argument("--hy3d-repo", default=os.environ.get(
        "HY3D_REPO", "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape"))
    ap.add_argument("--paint-path", default="/mnt/disk01/models/Hunyuan3D-2.1",
                    help="directory containing hunyuan3d-paintpbr-v2-1/ subfolder")
    ap.add_argument("--dino-path", default="/mnt/disk01/models/dinov2-giant",
                    help="local facebook/dinov2-giant mirror")
    ap.add_argument("--realesrgan-ckpt",
                    default="/mnt/disk01/models/realesrgan/RealESRGAN_x4plus.pth")
    ap.add_argument("--max-view", type=int, default=6,
                    help="number of views to sample (6..9)")
    ap.add_argument("--resolution", type=int, default=512,
                    help="per-view resolution (512 or 768)")
    ap.add_argument("--no-remesh", action="store_true",
                    help="skip the pymeshlab pre-remeshing step")
    ap.add_argument("--trace-dir", default=None,
                    help="dump per-stage .npy tensors from forward hooks")
    args = ap.parse_args()

    hy3dpaint_dir = _setup_env(args.hy3d_repo, args.paint_path)
    print(f"hy3dpaint: {hy3dpaint_dir}")

    # bpy is only used for OBJ -> GLB conversion in DifferentiableRenderer
    # mesh_utils.convert_obj_to_glb. Installing Blender is overkill; stub it
    # and monkey-patch convert_obj_to_glb to use trimesh instead.
    import types
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = types.ModuleType("bpy")

    # Apply the upstream torchvision_fix before importing anything that touches
    # torchvision.transforms.functional_tensor (removed in torchvision 0.17+).
    try:
        tfx = importlib.import_module("utils.torchvision_fix")
        if hasattr(tfx, "apply_fix"):
            tfx.apply_fix()
            print("Applied torchvision_fix.apply_fix()")
    except Exception as e:
        print(f"Warning: torchvision_fix not applied: {e}")

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    # Replace the bpy-based GLB conversion with a trimesh fallback.
    from DifferentiableRenderer import mesh_utils as _mu

    def _trimesh_obj_to_glb(obj_path, glb_path):
        import trimesh
        m = trimesh.load(obj_path, process=False)
        m.export(glb_path)

    _mu.convert_obj_to_glb = _trimesh_obj_to_glb
    import textureGenPipeline as _tgp
    _tgp.convert_obj_to_glb = _trimesh_obj_to_glb

    conf = Hunyuan3DPaintConfig(args.max_view, args.resolution)
    conf.multiview_cfg_path = os.path.join(hy3dpaint_dir, "cfgs", "hunyuan-paint-pbr.yaml")
    conf.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"  # monkey-patched snapshot
    conf.dino_ckpt_path = args.dino_path
    conf.realesrgan_ckpt_path = args.realesrgan_ckpt

    print(f"Loading paint pipeline (this can take ~1 min)...")
    t0 = time.time()
    pipeline = Hunyuan3DPaintPipeline(conf)
    print(f"  loaded in {time.time() - t0:.1f}s")

    paint_pipe = pipeline.models["multiview_model"].pipeline

    hook_handles = []
    if args.trace_dir:
        hook_handles = _install_trace_hooks(
            pipeline.models["multiview_model"], paint_pipe, args.trace_dir)
        print(f"Tracing enabled: {args.trace_dir}")

    print(f"Texturing mesh: {args.mesh}")
    print(f"Style image:    {args.image}")
    t0 = time.time()
    out_path = pipeline(
        mesh_path=args.mesh,
        image_path=args.image,
        output_mesh_path=args.out,
        use_remesh=not args.no_remesh,
        save_glb=True,
    )
    print(f"  textured in {time.time() - t0:.1f}s")
    print(f"  wrote {out_path}")
    glb = out_path.replace(".obj", ".glb")
    if os.path.exists(glb):
        print(f"  wrote {glb}")

    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
