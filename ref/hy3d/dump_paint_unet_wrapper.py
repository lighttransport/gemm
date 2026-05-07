"""Dump Hunyuan3D-2.1 UNet2p5DConditionModel reference activations with each
custom attention path (DINO / MA / MDA / RA) toggled individually.

Used as the validation oracle for Phase 4 of the cuda/hy3d_paint port. Inputs
are shared across combos (same seeded tensors); only the per-path flags
differ. PoseRoPE is disabled here (separate dump for the RoPE variant).

Inputs (one set):
    in_sample.npy                [1, 2, 2, 4, 64, 64] f32
    in_embeds_normal.npy         [1, 2, 4, 64, 64]    f32
    in_embeds_position.npy       [1, 2, 4, 64, 64]    f32
    in_encoder_hidden_states.npy [1, 2, 77, 1024]     f32
    in_ref_latents.npy           [1, 1, 4, 64, 64]    f32
    in_dino_hidden_states.npy    [1, 257, 1536]       f32
    in_timestep.npy              [1]                  i64

Outputs:
    out_<combo>.npy              [4, 4, 64, 64]       f32   (B*N_pbr*N_gen=4)

Combos: none / dino / ma / mda / ra / all  (`--paths` to filter).

Usage (uses ~24 GB host RAM during dual-stream load; CUDA optional):
  uv run --with torch --with diffusers --with safetensors --with einops \\
      python ref/hy3d/dump_paint_unet_wrapper.py \\
      --unet /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet \\
      --outdir /tmp/hy3d_paint_unet_wrap_ref
"""
import argparse
import os
import sys

import numpy as np
import torch

HY3D = "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dpaint"
sys.path.insert(0, HY3D)
# The hunyuanpaintpbr/__init__.py pulls in model.py (pytorch-lightning,
# torchvision) which we don't need. Build a synthetic package so that
# `from .attn_processor import ...` inside modules.py resolves without
# triggering the full __init__.py.
import importlib.util as _ilu  # noqa: E402
import types as _types  # noqa: E402

_pkg_root = _types.ModuleType("hpb_stub")
_pkg_root.__path__ = []
_pkg_unet = _types.ModuleType("hpb_stub.unet")
_pkg_unet.__path__ = [os.path.join(HY3D, "hunyuanpaintpbr/unet")]
sys.modules["hpb_stub"] = _pkg_root
sys.modules["hpb_stub.unet"] = _pkg_unet

_spec_ap = _ilu.spec_from_file_location(
    "hpb_stub.unet.attn_processor",
    os.path.join(HY3D, "hunyuanpaintpbr/unet/attn_processor.py"))
_ap = _ilu.module_from_spec(_spec_ap)
sys.modules["hpb_stub.unet.attn_processor"] = _ap
_spec_ap.loader.exec_module(_ap)

_spec = _ilu.spec_from_file_location(
    "hpb_stub.unet.modules",
    os.path.join(HY3D, "hunyuanpaintpbr/unet/modules.py"))
_mod = _ilu.module_from_spec(_spec)
sys.modules["hpb_stub.unet.modules"] = _mod
_spec.loader.exec_module(_mod)
UNet2p5DConditionModel = _mod.UNet2p5DConditionModel
calc_multires_voxel_idxs = _mod.calc_multires_voxel_idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unet_wrap_ref")
    ap.add_argument("--seed", type=int, default=42)
    # Only the "all-paths-on" combo runs cleanly: turning use_mda off at
    # runtime is fragile because attn1.processor was permanently replaced
    # by SelfAttnProcessor2_0 at init (it requires 5D input). Per-path
    # validation in Phase 4.2-4.5 uses dedicated module-level dumpers.
    ap.add_argument("--paths", default="all")
    ap.add_argument("--rope", type=int, default=0, choices=[0, 1],
                    help="If 1, enable PoseRoPE on the MA path; dumps "
                         "in_position_maps.npy + voxel_indices_<Np>.npy and "
                         "writes out_<combo>_rope.npy.")
    ap.add_argument("--steps", type=int, default=0,
                    help="If >0, also drive a UniPC denoising loop with the "
                         "all-paths-on wrapper UNet (single-batch, no CFG, "
                         "same conditioning each step). Dumps loop_x0.npy, "
                         "loop_timesteps.npy, loop_model_out_<i>.npy, "
                         "loop_x_after_<i>.npy.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("loading wrapper (this loads ~2x UNet weights)...", file=sys.stderr)
    model = UNet2p5DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    print(f"device={device}", file=sys.stderr)

    model.use_position_rope = bool(args.rope)

    torch.manual_seed(args.seed)
    B, N_pbr, N_gen, N_ref = 1, 2, 2, 1
    H = W = 64
    sample = torch.randn(B, N_pbr, N_gen, 4, H, W, device=device)
    embeds_normal = torch.randn(B, N_gen, 4, H, W, device=device)
    embeds_position = torch.randn(B, N_gen, 4, H, W, device=device)
    timestep = torch.tensor([500], dtype=torch.int64, device=device)
    encoder_hidden_states = torch.randn(B, N_pbr, 77, 1024, device=device)
    ref_latents = torch.randn(B, N_ref, 4, H, W, device=device)
    # DINOv2-G outputs 257 tokens (1 cls + 16x16 patches @ 224 input) of dim 1536
    dino_hidden_states = torch.randn(B, 257, 1536, device=device)

    def save(name, t, dtype=np.float32):
        path = os.path.join(args.outdir, f"in_{name}.npy")
        arr = t.detach().cpu().numpy()
        if dtype == np.int64:
            arr = arr.astype(np.int64)
        else:
            arr = arr.astype(np.float32)
        np.save(path, arr)

    save("sample", sample)
    save("embeds_normal", embeds_normal)
    save("embeds_position", embeds_position)
    save("encoder_hidden_states", encoder_hidden_states)
    save("ref_latents", ref_latents)
    save("dino_hidden_states", dino_hidden_states)
    save("timestep", timestep, dtype=np.int64)

    position_maps = None
    if args.rope:
        position_maps = torch.rand(B, N_gen, 3, H, W, device=device)
        save("position_maps", position_maps)
        v = calc_multires_voxel_idxs(
            position_maps,
            grid_resolutions=[H, H // 2, H // 4, H // 8],
            voxel_resolutions=[H * 8, H * 4, H * 2, H])
        for L_key, info in v.items():
            np.save(os.path.join(args.outdir, f"voxel_indices_{L_key}.npy"),
                    info["voxel_indices"].detach().cpu().numpy().astype(np.int64))
            print(f"  voxel L={L_key} res={info['voxel_resolution']}", file=sys.stderr)

    print(f"shapes: sample={tuple(sample.shape)} text={tuple(encoder_hidden_states.shape)} "
          f"ref={tuple(ref_latents.shape)} dino={tuple(dino_hidden_states.shape)}",
          file=sys.stderr)

    combos = {
        "none": dict(use_dino=False, use_ma=False, use_mda=False, use_ra=False),
        "dino": dict(use_dino=True,  use_ma=False, use_mda=False, use_ra=False),
        "ma":   dict(use_dino=False, use_ma=True,  use_mda=False, use_ra=False),
        "mda":  dict(use_dino=False, use_ma=False, use_mda=True,  use_ra=False),
        "ra":   dict(use_dino=False, use_ma=False, use_mda=False, use_ra=True),
        "all":  dict(use_dino=True,  use_ma=True,  use_mda=True,  use_ra=True),
    }
    # Per-Basic2p5DTransformerBlock flags are baked at init_attention time;
    # propagate runtime overrides into every wrapped block.
    Basic2p5D = _mod.Basic2p5DTransformerBlock

    def propagate_flags(unet, flags):
        for m in unet.modules():
            if isinstance(m, Basic2p5D):
                for k, v in flags.items():
                    setattr(m, k, v)

    requested = [p.strip() for p in args.paths.split(",") if p.strip()]
    for combo_name in requested:
        if combo_name not in combos:
            print(f"  unknown combo '{combo_name}', skipping", file=sys.stderr)
            continue
        flags = combos[combo_name]
        for k, v in flags.items():
            setattr(model, k, v)
        propagate_flags(model.unet, flags)
        # NOTE: unet_dual was init_attention'd with all custom flags False
        # (its attn1.processor is the stock AttnProcessor2_0). We must NOT
        # toggle its per-block flags or the wrapper's forward will reshape
        # to 5D and feed it to a 3D-only processor.

        kwargs = {
            "embeds_normal": embeds_normal,
            "embeds_position": embeds_position,
            "cache": {},  # forces dual-stream + dino_proj re-run if enabled
        }
        if flags["use_ra"]:
            kwargs["ref_latents"] = ref_latents
        if flags["use_dino"]:
            kwargs["dino_hidden_states"] = dino_hidden_states
        if args.rope and position_maps is not None:
            kwargs["position_maps"] = position_maps

        with torch.no_grad():
            out = model(sample, timestep, encoder_hidden_states, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "sample"):
            out = out.sample
        out_name = f"{combo_name}_rope" if args.rope else combo_name
        path = os.path.join(args.outdir, f"out_{out_name}.npy")
        np.save(path, out.detach().cpu().numpy().astype(np.float32))
        print(f"  {combo_name:6s} -> {tuple(out.shape)}  range=[{out.min():+.3f},{out.max():+.3f}]",
              file=sys.stderr)

    # Optional Phase 4.11b: scheduler↔UNet integration loop.
    if args.steps > 0:
        from diffusers import UniPCMultistepScheduler
        sch_cfg = dict(
            num_train_timesteps=1000,
            beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
            solver_order=2, solver_type="bh2",
            predict_x0=True, lower_order_final=True,
            final_sigmas_type="zero", steps_offset=1,
        )
        sch = UniPCMultistepScheduler(**sch_cfg)
        sch.set_timesteps(args.steps)
        ts = sch.timesteps  # on cpu

        # all-paths-on
        flags = combos["all"]
        for k, v in flags.items():
            setattr(model, k, v)
        propagate_flags(model.unet, flags)

        # Initial latent: shape [B, N_pbr, N_gen, 4, H, W] in "sample" form.
        # We reuse the same conditioning each step.
        torch.manual_seed(args.seed + 7)
        x = torch.randn(B, N_pbr, N_gen, 4, H, W, device=device)
        # Save Beff-flat initial latent for the C side: [Beff, 4, H, W]
        x_flat0 = x.reshape(B * N_pbr * N_gen, 4, H, W).detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(args.outdir, "loop_x0.npy"), x_flat0)
        np.save(os.path.join(args.outdir, "loop_timesteps.npy"),
                ts.cpu().numpy().astype(np.int64))

        for i, t in enumerate(ts):
            t_dev = t.to(device).reshape(1)
            kwargs_loop = {
                "embeds_normal": embeds_normal,
                "embeds_position": embeds_position,
                "cache": {},
                "ref_latents": ref_latents,
                "dino_hidden_states": dino_hidden_states,
            }
            with torch.no_grad():
                out = model(x, t_dev, encoder_hidden_states, **kwargs_loop)
            if isinstance(out, tuple):
                out = out[0]
            if hasattr(out, "sample"):
                out = out.sample
            mo = out.detach().cpu()  # [Beff, 4, H, W]
            np.save(os.path.join(args.outdir, f"loop_model_out_{i}.npy"),
                    mo.numpy().astype(np.float32))

            # Step in sample-shape: scheduler is per-element so layout is fine.
            x_flat = x.reshape(B * N_pbr * N_gen, 4, H, W).cpu()
            stp = sch.step(mo, t, x_flat, return_dict=True)
            x_flat_new = stp.prev_sample
            np.save(os.path.join(args.outdir, f"loop_x_after_{i}.npy"),
                    x_flat_new.numpy().astype(np.float32))
            x = x_flat_new.reshape(B, N_pbr, N_gen, 4, H, W).to(device)
            print(f"  loop step {i:2d}: t={int(t):4d}  mo range=[{float(mo.min()):+.3f},{float(mo.max()):+.3f}]  "
                  f"x range=[{float(x.min()):+.3f},{float(x.max()):+.3f}]",
                  file=sys.stderr)

        import json
        with open(os.path.join(args.outdir, "loop_meta.json"), "w") as f:
            json.dump(dict(steps=args.steps, cfg=sch_cfg,
                           B=B, N_pbr=N_pbr, N_gen=N_gen, H=H, W=W,
                           Beff=B*N_pbr*N_gen), f, indent=2)
        print(f"wrote {args.steps}-step loop -> {args.outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
