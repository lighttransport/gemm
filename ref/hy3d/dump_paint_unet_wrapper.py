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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("loading wrapper (this loads ~2x UNet weights)...", file=sys.stderr)
    model = UNet2p5DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    print(f"device={device}", file=sys.stderr)

    # PoseRoPE off for this dump (separate variant covers it).
    model.use_position_rope = False

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

        with torch.no_grad():
            out = model(sample, timestep, encoder_hidden_states, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "sample"):
            out = out.sample
        path = os.path.join(args.outdir, f"out_{combo_name}.npy")
        np.save(path, out.detach().cpu().numpy().astype(np.float32))
        print(f"  {combo_name:6s} -> {tuple(out.shape)}  range=[{out.min():+.3f},{out.max():+.3f}]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
