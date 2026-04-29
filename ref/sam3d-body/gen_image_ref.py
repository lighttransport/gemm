#!/usr/bin/env python3
"""
gen_image_ref.py — run FAIR's sam-3d-body pipeline with per-stage torch
hooks, dump float32 .npy tensors to REFDIR for the C runner's
verify_*.c binaries to diff against.

Stages dumped (written only if that stage executed):
    input_image.npy         (H, W, 3)  u8
    image_processed.npy     (Hc, Wc, 3) f32  — crop+resize+normalize
    dinov3_tokens.npy       (N, D)      f32  — DINOv3 encoder
    mhr_params.npy          (519,)      f32  — MHR head regressor
    cam_params.npy          (4,)        f32  — [cam_t_x, cam_t_y, cam_t_z, focal]
    out_vertices.npy        (V, 3)      f32
    out_keypoints_3d.npy    (K, 3)      f32
    out_keypoints_2d.npy    (K, 2)      f32
    out_faces.npy           (F, 3)      i32

Determinism:
  * torch.manual_seed + numpy seed
  * torch.set_float32_matmul_precision("highest")

Usage:
  python ref/sam3d-body/gen_image_ref.py \\
      --image person.jpg \\
      --hf-repo-id facebook/sam-3d-body-dinov3 \\
      --outdir /tmp/sam3d_body_ref --seed 42

Requires the `sam_3d_body` package installed into ref/sam3d-body/.venv
(see README.md) and HF-authenticated access to the gated weights.
"""

import argparse
import os
import sys

import numpy as np


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[gen_image_ref] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB image (.jpg/.png)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("x0", "y0", "x1", "y1"),
                    help="optional bbox in image coords; else use full image")
    ap.add_argument("--hf-repo-id", default="facebook/sam-3d-body-dinov3",
                    help="HuggingFace model repo (gated)")
    ap.add_argument("--local-ckpt-dir", default=None,
                    help="local snapshot dir (skip HF download): contains "
                         "model.ckpt, model_config.yaml, assets/mhr_model.pt")
    ap.add_argument("--device", default="cuda",
                    help='"cuda" or "cpu"')
    ap.add_argument("--outdir", default="/tmp/sam3d_body_ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-run", action="store_true",
                    help="only dump the preprocessed image; skip the model")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Preprocess so step-0 scaffold has at least one dump even before
    # the model installs cleanly.
    from PIL import Image
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    save(args.outdir, "input_image.npy", img)

    if args.skip_run:
        print("[gen_image_ref] --skip-run: dumped raw image only",
              file=sys.stderr)
        return 0

    try:
        import torch
    except Exception as e:
        print(f"[gen_image_ref] torch not importable: {e}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    # Upstream repo ships no setup.py — add the clone directory to sys.path
    # so `import sam_3d_body` works without requiring `pip install -e`.
    for cand in ("/tmp/sam-3d-body", os.environ.get("SAM_3D_BODY_DIR", "")):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)

    try:
        from sam_3d_body import (load_sam_3d_body, load_sam_3d_body_hf,
                                 SAM3DBodyEstimator)
    except Exception as e:
        print(f"[gen_image_ref] cannot import sam_3d_body: {e}",
              file=sys.stderr)
        return 1

    if args.local_ckpt_dir:
        ckpt_path = os.path.join(args.local_ckpt_dir, "model.ckpt")
        mhr_path = os.path.join(args.local_ckpt_dir, "assets", "mhr_model.pt")
        print(f"[gen_image_ref] local ckpt: {ckpt_path}", file=sys.stderr)
        model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_path,
                                            device=args.device,
                                            mhr_path=mhr_path)
    else:
        model, model_cfg = load_sam_3d_body_hf(args.hf_repo_id,
                                               device=args.device)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                                   human_detector=None, human_segmentor=None,
                                   fov_estimator=None)

    # Register forward hooks on the submodules we care about. We can't
    # always statically resolve the exact attribute paths (upstream
    # refactors move things around), so we search by class name /
    # attribute path.
    caps = {}

    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return x

    def cap(name, idx=None):
        """Capture forward output. If output is tuple/list, `idx` picks an
        element; otherwise the whole thing. Dict outputs are flattened."""
        def _hook(_m, _i, o):
            x = o
            if isinstance(o, (tuple, list)):
                x = o[idx] if idx is not None else o[0]
            if isinstance(x, dict):
                out = {k: _to_np(v) for k, v in x.items() if hasattr(v, "detach")}
                caps[name] = out
            else:
                caps[name] = _to_np(x)
        return _hook

    def cap_tuple(name):
        """Capture both elements of a (tokens, image_embeddings) tuple
        output as `{name}__tokens.npy` and `{name}__context.npy`."""
        def _hook(_m, _i, o):
            if isinstance(o, (tuple, list)) and len(o) >= 2:
                caps[name] = {
                    "tokens":  _to_np(o[0]),
                    "context": _to_np(o[1]),
                }
            else:
                caps[name] = _to_np(o)
        return _hook

    def cap_pre(name, arg_idx=0):
        def _pre(_m, args, _kwargs=None):
            if args and arg_idx < len(args):
                caps[name] = _to_np(args[arg_idx])
        return _pre

    def cap_pre_tuple(name, names):
        """Capture several positional args as {name}__{k}.npy for k in `names`."""
        def _pre(_m, args, _kwargs=None):
            out = {}
            for i, k in enumerate(names):
                if i < len(args) and hasattr(args[i], "detach"):
                    out[k] = _to_np(args[i])
            caps[name] = out
        return _pre

    registered = []

    def mods_by_name(suffix_or_eq):
        """Yield (name, module) matching either an exact dotted name or a
        suffix after a dot. We prefer exact matches to avoid grabbing
        submodules of a namespaced attribute."""
        exact, suffix = [], []
        for n, m in model.named_modules():
            if n == suffix_or_eq:
                exact.append((n, m))
            elif n.endswith("." + suffix_or_eq):
                suffix.append((n, m))
        return exact or suffix

    # --- Backbone + heads (kept from prior version). ---
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if cls == "Dinov3Backbone":
            m.register_forward_pre_hook(cap_pre("dinov3_input"))
            m.register_forward_hook(cap("dinov3_tokens"))
            registered.append(("dinov3_tokens", name, cls))
        elif cls == "ViT":
            # ViT-H/16 (vit_hmr_512_384) variant — outputs (B, 1280, 32, 24).
            # Input is the post-W-axis-crop (1, 3, 512, 384) tensor.
            m.register_forward_pre_hook(cap_pre("vith_input"))
            m.register_forward_hook(cap("vith_tokens"))
            registered.append(("vith_tokens", name, cls))
        elif cls == "MHRHead":
            m.register_forward_hook(cap("mhr_params"))
            registered.append(("mhr_params", name, cls))
        elif cls in ("CameraHead", "FovHead"):
            m.register_forward_hook(cap("cam_params"))
            registered.append(("cam_params", name, cls))

    # --- Ray-cond projection: captures batch["ray_cond"] (as 2nd arg)
    #     and the projected image_embeddings_after_ray. Also monkey-patch
    #     the forward to cache the downsampled rays (post-interpolate, with
    #     the trailing z=1 appended) and the fourier-embedded rays — this
    #     lets the C port sidestep reproducing torch's antialias-bilinear
    #     downsample bit-exactly; we verify fourier+conv+LN2d against the
    #     cached intermediate.
    for n, m in mods_by_name("ray_cond_emb"):
        m.register_forward_pre_hook(cap_pre_tuple(
            "ray_cond", ["image_embeddings_pre_ray", "ray_cond"]))
        m.register_forward_hook(cap("image_embeddings_after_ray"))

        _orig_fwd = m.forward
        _ray_mod = m
        def _patched(img_embeddings, rays, _orig=_orig_fwd, _mod=_ray_mod):
            import torch.nn.functional as _F, einops as _ein
            B, D, _h, _w = img_embeddings.shape
            scale = 1.0 / _mod.patch_size
            rays_ds = _F.interpolate(
                rays, scale_factor=(scale, scale),
                mode="bilinear", align_corners=False, antialias=True
            )
            rays_ds = rays_ds.permute(0, 2, 3, 1).contiguous()
            rays_ds = torch.cat([rays_ds, torch.ones_like(rays_ds[..., :1])], dim=-1)
            caps["ray_cond_ds_xyz"] = _to_np(rays_ds)  # (B, _h, _w, 3)
            rays_emb = _mod.camera(pos=rays_ds.reshape(B, -1, 3))  # (B, N, 99)
            caps["ray_cond_fourier"] = _to_np(rays_emb)
            rays_emb = _ein.rearrange(rays_emb, "b (h w) c -> b c h w", h=_h, w=_w).contiguous()
            z = torch.cat([img_embeddings, rays_emb], dim=1)
            caps["ray_cond_preconv"] = _to_np(z)  # (B, D+99, h, w)
            zc = _mod.conv(z)
            caps["ray_cond_postconv"] = _to_np(zc)
            return _mod.norm(zc)
        m.forward = _patched
        registered.append(("image_embeddings_after_ray", n, m.__class__.__name__))
        break  # body decoder only

    # --- Token construction linears. ---
    for n, m in mods_by_name("init_to_token_mhr"):
        m.register_forward_pre_hook(cap_pre("init_to_token_in"))
        m.register_forward_hook(cap("init_token_raw"))
        registered.append(("init_token_raw", n, m.__class__.__name__))
        break
    for n, m in mods_by_name("prev_to_token_mhr"):
        m.register_forward_pre_hook(cap_pre("prev_to_token_in"))
        m.register_forward_hook(cap("prev_token_raw"))
        registered.append(("prev_token_raw", n, m.__class__.__name__))
        break
    for n, m in mods_by_name("prompt_to_token"):
        m.register_forward_pre_hook(cap_pre("prompt_to_token_in"))
        m.register_forward_hook(cap("prompt_token_raw"))
        registered.append(("prompt_token_raw", n, m.__class__.__name__))
        break
    for n, m in mods_by_name("keypoint_posemb_linear"):
        m.register_forward_hook(cap("kp2d_posemb_init"))
        registered.append(("kp2d_posemb_init", n, m.__class__.__name__))
        break
    for n, m in mods_by_name("keypoint3d_posemb_linear"):
        m.register_forward_hook(cap("kp3d_posemb_init"))
        registered.append(("kp3d_posemb_init", n, m.__class__.__name__))
        break

    # --- Per-decoder-layer in/out. Body decoder lives at
    #     model.decoder.layers[i] (PromptableDecoder → ModuleList).
    for n, m in model.named_modules():
        if not n.startswith("decoder.layers."):
            continue
        # n is "decoder.layers.{i}" exactly (not its sub-sub-modules)
        tail = n[len("decoder.layers."):]
        if "." in tail:
            continue
        try:
            i = int(tail)
        except ValueError:
            continue
        m.register_forward_pre_hook(cap_pre_tuple(
            f"decoder_layer{i}_in",
            ["x", "context", "x_pe", "context_pe", "x_mask"]))
        m.register_forward_hook(cap_tuple(f"decoder_layer{i}_out"))
        registered.append((f"decoder_layer{i}", n, m.__class__.__name__))

    # --- Decoder final norm: pre-hook gives raw tokens, post gives norm'd.
    for n, m in mods_by_name("decoder.norm_final"):
        m.register_forward_pre_hook(cap_pre("decoder_out_prenorm"))
        m.register_forward_hook(cap("decoder_out_norm_final"))
        registered.append(("decoder_out_norm_final", n, m.__class__.__name__))
        break

    # --- MHR head internals: raw 519 params BEFORE + AFTER init_estimate add
    #     are useful for isolating head vs init drift. Hook the inner FFN
    #     `proj` which yields the raw 519.
    for n, m in mods_by_name("head_pose.proj"):
        m.register_forward_pre_hook(cap_pre("head_pose_proj_input"))
        m.register_forward_hook(cap("head_pose_proj_raw"))
        registered.append(("head_pose_proj_raw", n, m.__class__.__name__))
        break
    for n, m in mods_by_name("head_camera.proj"):
        m.register_forward_pre_hook(cap_pre("head_camera_proj_input"))
        m.register_forward_hook(cap("head_camera_proj_raw"))
        registered.append(("head_camera_proj_raw", n, m.__class__.__name__))
        break

    # --- MHR jit sub-modules: ScriptModules forbid both pre- and
    #     post-hooks, so we capture intermediates by re-calling the
    #     sub-modules directly after the main pipeline runs (see below,
    #     after process_one_image). We just stash a handle here.
    mhr_root = None
    for n, m in model.named_modules():
        if n.endswith(".mhr") and m.__class__.__name__ in (
                "RecursiveScriptModule", "MHR"):
            mhr_root = m
            registered.append(("(post-run)mhr_sub", n, m.__class__.__name__))
            break

    # --- Grab `batch` (affine_trans / bbox_center / bbox_scale /
    #     ori_img_size / img_size / cam_int) as soon as the decoder forward
    #     starts. `forward_decoder(image_embeddings, init_estimate=None,
    #     keypoints=..., prev_estimate=..., condition_info=..., batch=...,
    #     full_output=None)` — batch is a kw-arg for us.
    # Capture only the FIRST forward_decoder call (body branch, dummy
    # 1-keypoint prompt → 145 tokens). The full inference flow may invoke
    # forward_decoder a second time via run_keypoint_prompt with 4 hand-tip
    # keypoints + prev_estimate (148 tokens); v1 scope is the first pass.
    fd_call_counter = {"n": 0}

    def _cap_batch_pre(_m, _args, kwargs):
        if fd_call_counter["n"] != 0:
            return
        b = kwargs.get("batch", None)
        if b is None:
            return
        out = {}
        for k in ("bbox_center", "bbox_scale", "ori_img_size",
                  "img_size", "cam_int", "affine_trans"):
            if k in b and hasattr(b[k], "detach"):
                out[k] = _to_np(b[k])
        if out:
            caps["decoder_batch"] = out

    for n, m in model.named_modules():
        if n.endswith(".decoder") and m.__class__.__name__ == "PromptableDecoder":
            # Registering on the meta-arch forward_decoder isn't a Module hook,
            # but PromptableDecoder itself isn't called with `batch`. Instead
            # wrap the meta-arch's forward_decoder method once.
            break
    try:
        _orig_forward_decoder = model.forward_decoder
        def _wrapped_fd(*args, **kwargs):
            _cap_batch_pre(model, args, kwargs)
            r = _orig_forward_decoder(*args, **kwargs)
            fd_call_counter["n"] += 1
            return r
        model.forward_decoder = _wrapped_fd
        registered.append(("decoder_batch", "forward_decoder",
                           "meta_arch.wrap"))
    except Exception as e:
        print(f"[gen_image_ref] could not wrap forward_decoder: {e}",
              file=sys.stderr)

    # --- Per-intermediate-layer pose_output capture. Hook the decoder's
    #     internal `token_to_pose_output_fn` closure via a method wrap on
    #     PromptableDecoder.forward. The closure is passed in as the
    #     `token_to_pose_output_fn` kwarg; we swap it for a wrapper that
    #     captures the dict each invocation.
    try:
        dec = None
        for n, m in model.named_modules():
            if n == "decoder" and m.__class__.__name__ == "PromptableDecoder":
                dec = m
                break
        if dec is not None:
            _orig_dec_fwd = dec.forward
            def _dec_wrap(token_embedding, image_embedding, *a, **kw):
                # Skip second forward_decoder call (148-token iterative refine).
                if fd_call_counter["n"] != 0:
                    return _orig_dec_fwd(token_embedding, image_embedding,
                                         *a, **kw)
                orig_pf = kw.get("token_to_pose_output_fn", None)
                orig_kf = kw.get("keypoint_token_update_fn", None)
                layer_poses = []
                layer_in = []
                layer_out_kp = []
                if orig_pf is not None:
                    def _pf_wrap(tokens, prev_pose_output, layer_idx):
                        out = orig_pf(tokens, prev_pose_output, layer_idx)
                        sel = {}
                        for k in ("pred_keypoints_2d_cropped",
                                  "pred_keypoints_2d_depth",
                                  "pred_keypoints_3d",
                                  "pred_keypoints_2d",
                                  "pred_cam", "pred_cam_t"):
                            if k in out and hasattr(out[k], "detach"):
                                sel[k] = _to_np(out[k])
                        layer_poses.append((layer_idx, sel))
                        return out
                    kw["token_to_pose_output_fn"] = _pf_wrap
                if orig_kf is not None:
                    def _kf_wrap(token_embedding, token_augment,
                                 curr_pose_output, layer_idx):
                        layer_in.append((layer_idx, {
                            "tokens_pre_kp": _to_np(token_embedding),
                            "augment_pre_kp": _to_np(token_augment),
                        }))
                        out = orig_kf(token_embedding, token_augment,
                                      curr_pose_output, layer_idx)
                        (te_post, ta_post, _, _) = out
                        layer_out_kp.append((layer_idx, {
                            "tokens_post_kp": _to_np(te_post),
                            "augment_post_kp": _to_np(ta_post),
                        }))
                        return out
                    kw["keypoint_token_update_fn"] = _kf_wrap
                r = _orig_dec_fwd(token_embedding, image_embedding, *a, **kw)
                for li, d in layer_poses:
                    for k, v in d.items():
                        caps[f"decoder_pose_layer{li}__{k}"] = v
                for li, d in layer_in:
                    for k, v in d.items():
                        caps[f"decoder_kp_layer{li}__{k}"] = v
                for li, d in layer_out_kp:
                    for k, v in d.items():
                        caps[f"decoder_kp_layer{li}__{k}"] = v
                return r
            dec.forward = _dec_wrap
            registered.append(("decoder_pose/kp_layer*", "decoder",
                               "PromptableDecoder.wrap"))
    except Exception as e:
        print(f"[gen_image_ref] could not wrap decoder.forward: {e}",
              file=sys.stderr)

    for r in registered:
        print(f"[gen_image_ref] hook: name='{r[1]}' cls={r[2]} -> {r[0]}",
              file=sys.stderr)

    # Run the pipeline. Upstream estimator accepts either a path or an
    # ndarray; pass the ndarray in BGR (cv2 convention) to match
    # process_one_image.
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bboxes = None
    if args.bbox is not None:
        bboxes = np.asarray([args.bbox], dtype=np.float32)

    with torch.inference_mode():
        outputs = estimator.process_one_image(img_bgr, bboxes=bboxes)

    # Re-invoke MHR sub-modules directly to harvest per-stage
    # intermediates (ScriptModules don't accept forward hooks).
    # Inputs were already captured via MHRHead forward hook as
    # `mhr_params__{shape,mhr_model_params,face}`.
    if mhr_root is not None and "mhr_params" in caps and isinstance(
            caps["mhr_params"], dict):
        try:
            mp_d = caps["mhr_params"]
            shape_np = mp_d.get("shape")
            modelp_np = mp_d.get("mhr_model_params")
            face_np = mp_d.get("face")
            if (shape_np is not None and modelp_np is not None
                    and face_np is not None):
                dev = next((p.device for p in mhr_root.parameters()),
                           torch.device("cpu"))
                shape_t = torch.from_numpy(shape_np).to(dev).float()
                modelp_t = torch.from_numpy(modelp_np).to(dev).float()
                face_t = torch.from_numpy(face_np).to(dev).float()
                with torch.inference_mode():
                    bs = modelp_t.shape[0]
                    # Stage 10A: identity blend shape rest pose
                    rest_id = mhr_root.character_torch.blend_shape(shape_t)
                    caps["mhr_blend_shape_out"] = _to_np(rest_id)
                    # Stage 10B: face expressions (zero-in → zero-out expected)
                    face_out = mhr_root.face_expressions_model(face_t)
                    caps["mhr_face_expressions_out"] = _to_np(face_out)
                    # Stage 6: parameter_transform(cat(model_params(204), zeros(45))) → (B,889)
                    pad = torch.zeros(bs, 45, device=dev, dtype=modelp_t.dtype)
                    pt_in = torch.cat([modelp_t, pad], dim=1)
                    joint_params = mhr_root.character_torch.parameter_transform(pt_in)
                    caps["mhr_joint_parameters"] = _to_np(joint_params)
                    # Stage 10C: pose correctives fed from joint parameters
                    pc_out = mhr_root.pose_correctives_model(joint_params)
                    caps["mhr_pose_correctives_out"] = _to_np(pc_out)
                    # Top-level: verts + skel_state for stages 8+11
                    verts, skel = mhr_root(shape_t, modelp_t, face_t)
                    caps["mhr_output"] = {
                        "verts": _to_np(verts),
                        "skel_state": _to_np(skel),
                    }
        except Exception as e:
            print(f"[gen_image_ref] skip mhr sub-module re-invoke: {e}",
                  file=sys.stderr)

    for name, arr in caps.items():
        if isinstance(arr, dict):
            for k, v in arr.items():
                if hasattr(v, "shape"):
                    save(args.outdir, f"{name}__{k}.npy", v)
        elif hasattr(arr, "shape"):
            save(args.outdir, name + ".npy", arr)

    if outputs:
        person = outputs[0]
        for key, tag in [
            ("pred_vertices",     "out_vertices"),
            ("pred_keypoints_3d", "out_keypoints_3d"),
            ("pred_keypoints_2d", "out_keypoints_2d"),
        ]:
            v = person.get(key)
            if v is None:
                continue
            if hasattr(v, "detach"):
                v = v.detach().float().cpu().numpy()
            save(args.outdir, tag + ".npy", np.asarray(v))
        # camera info
        cam_t = person.get("pred_cam_t")
        focal = person.get("focal_length")
        if cam_t is not None and focal is not None:
            if hasattr(cam_t, "detach"):
                cam_t = cam_t.detach().float().cpu().numpy()
            arr = np.concatenate([np.asarray(cam_t).reshape(-1),
                                  np.asarray(focal).reshape(-1)]).astype(np.float32)
            save(args.outdir, "cam_params.npy", arr)

    # MHR faces (static; needed for mesh export parity)
    try:
        faces = getattr(estimator, "faces", None)
        if faces is not None:
            save(args.outdir, "out_faces.npy", np.asarray(faces, dtype=np.int32))
    except Exception as e:
        print(f"[gen_image_ref] skip out_faces: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
