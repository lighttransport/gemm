"""Dump Hunyuan3D-2.1 paint UNet reference activations with ONLY the RA
(reference-attn + dual-stream) path enabled — no MA / MDA / DINO.

Phase 4.5 validation oracle for cuda/hy3d_paint. Same shared inputs as the
4.2/4.3/4.4 dumpers, plus `in_ref_latents.npy [B, N_ref, 4, H, W]`.

Output: out_ra.npy [B*N_pbr*N_gen, 4, H, W] (here [4, 4, 64, 64]) for B=1
N_pbr=2 N_gen=2 N_ref=1.

Usage:
  uv run --with torch --with diffusers --with safetensors --with einops \\
      --with transformers --with packaging --with numpy \\
      python ref/hy3d/dump_paint_unet_ra.py \\
      --unet /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet \\
      --outdir /tmp/hy3d_paint_unet_ra_ref
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

HY3D = "/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dpaint"
sys.path.insert(0, HY3D)
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


class RaOnlyWrapper(UNet2p5DConditionModel):
    """Wrapper that builds Basic2p5DTransformerBlocks with use_ra=True only,
    plus the dual-stream second UNet (so RA's condition_embed_dict is built
    via unet_dual rather than reusing the main unet)."""

    def __init__(self, unet):
        nn.Module.__init__(self)
        self.unet = unet
        self.train_sched = None
        self.val_sched = None
        self.use_ma = False
        self.use_ra = True
        self.use_mda = False
        self.use_dino = False
        self.use_position_rope = False
        self.use_learned_text_clip = True
        self.use_dual_stream = True
        self.pbr_setting = ["albedo", "mr"]
        self.pbr_token_channels = 77

        # Match modules.py order: deepcopy unet BEFORE installing RA on main,
        # so unet_dual stays vanilla. init_attention on dual still installs
        # layer_name on each block (what RA's "w" mode keys by).
        self.unet_dual = copy.deepcopy(unet)
        self.init_attention(self.unet_dual)
        # main UNet attn paths: only RA on the main unet (no MA/MDA/DINO).
        self.init_attention(
            self.unet,
            use_ma=False, use_ra=True, use_mda=False, use_dino=False,
            pbr_setting=self.pbr_setting,
        )
        self.init_condition(use_dino=False)


def _load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    SKIP_TOKENS = ("attn_multiview", "attn_dino", "image_proj_model_dino")
    state = {}
    for k, v in ckpt.items():
        if any(t in k for t in SKIP_TOKENS):
            continue
        if "attn1.processor." in k:
            continue
        state[k] = v.to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} keys; missing={len(missing)}, unexpected={len(unexpected)}",
          file=sys.stderr)
    if missing:
        print(f"  first missing: {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_unet_ra_ref")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    config_path = os.path.join(args.unet, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    inner = UNet2DConditionModel(**cfg)
    model = RaOnlyWrapper(inner)
    # Replace conv_in for the 12-channel multiview input on the MAIN unet.
    model.unet.conv_in = torch.nn.Conv2d(
        12, inner.conv_in.out_channels,
        kernel_size=inner.conv_in.kernel_size,
        stride=inner.conv_in.stride,
        padding=inner.conv_in.padding,
        dilation=inner.conv_in.dilation,
        groups=inner.conv_in.groups,
        bias=inner.conv_in.bias is not None,
    )
    # unet_dual was deep-copied BEFORE conv_in was widened; its conv_in stays at
    # 4 channels (matches ref_latents shape).
    _load_state(model, os.path.join(args.unet, "diffusion_pytorch_model.bin"))

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    torch.manual_seed(args.seed)
    B, N_pbr, N_gen, N_ref = 1, 2, 2, 1
    H = W = 64
    sample = torch.randn(B, N_pbr, N_gen, 4, H, W, device=device)
    embeds_normal = torch.randn(B, N_gen, 4, H, W, device=device)
    embeds_position = torch.randn(B, N_gen, 4, H, W, device=device)
    timestep = torch.tensor([500], dtype=torch.int64, device=device)
    encoder_hidden_states = torch.randn(B, N_pbr, 77, 1024, device=device)
    ref_latents = torch.randn(B, N_ref, 4, H, W, device=device)

    def save(name, t, dtype=np.float32):
        path = os.path.join(args.outdir, f"in_{name}.npy")
        arr = t.detach().cpu().numpy()
        arr = arr.astype(np.int64) if dtype == np.int64 else arr.astype(np.float32)
        np.save(path, arr)

    save("sample", sample)
    save("embeds_normal", embeds_normal)
    save("embeds_position", embeds_position)
    save("encoder_hidden_states", encoder_hidden_states)
    save("ref_latents", ref_latents)
    save("timestep", timestep, dtype=np.int64)

    # Hook RefAttnProcessor2_0 to dump first-call inputs+output.
    _ra_log = {"n": 0}
    _RAP = _ap.RefAttnProcessor2_0
    _orig_call = _RAP.__call__
    def _ra_hook(self, attn, hidden_states, encoder_hidden_states=None, **kw):
        if _ra_log["n"] == 0:
            # Capture intermediates by running attn.to_q/to_k/to_v plus processor.to_v_mr.
            with torch.no_grad():
                Qw = attn.to_q(hidden_states)
                Kw = attn.to_k(encoder_hidden_states)
                Vw_a = attn.to_v(encoder_hidden_states)
                Vw_m = self.to_v_mr(encoder_hidden_states)
                np.save(os.path.join(args.outdir, "ra0_Q.npy"), Qw.detach().cpu().numpy().astype(np.float32))
                np.save(os.path.join(args.outdir, "ra0_K.npy"), Kw.detach().cpu().numpy().astype(np.float32))
                np.save(os.path.join(args.outdir, "ra0_Va.npy"), Vw_a.detach().cpu().numpy().astype(np.float32))
                np.save(os.path.join(args.outdir, "ra0_Vm.npy"), Vw_m.detach().cpu().numpy().astype(np.float32))
                # Replicate the SDPA-with-cat-V step.
                import torch.nn.functional as F_
                Vc = torch.cat([Vw_a, Vw_m], dim=-1)
                B_, N_, _ = Qw.shape
                M_ = Kw.shape[1]
                H_ = attn.heads
                Dq = Kw.shape[-1] // H_
                Dv = Vc.shape[-1] // H_
                Qr = Qw.view(B_, N_, H_, Dq).transpose(1,2)
                Kr = Kw.view(B_, M_, H_, Dq).transpose(1,2)
                Vr = Vc.view(B_, M_, H_, Dv).transpose(1,2)
                Hs = F_.scaled_dot_product_attention(Qr, Kr, Vr)
                np.save(os.path.join(args.outdir, "ra0_Hs.npy"), Hs.detach().cpu().numpy().astype(np.float32))
                print(f"  ra0_Hs {tuple(Hs.shape)} (split chunks of {Dq})", file=sys.stderr)
        out = _orig_call(self, attn, hidden_states, encoder_hidden_states=encoder_hidden_states, **kw)
        if _ra_log["n"] == 0:
            np.save(os.path.join(args.outdir, "ra0_q_in.npy"),
                    hidden_states.detach().cpu().numpy().astype(np.float32))
            np.save(os.path.join(args.outdir, "ra0_kv_in.npy"),
                    encoder_hidden_states.detach().cpu().numpy().astype(np.float32))
            np.save(os.path.join(args.outdir, "ra0_out.npy"),
                    out.detach().cpu().numpy().astype(np.float32))
            print(f"ra0_q_in {tuple(hidden_states.shape)} ra0_kv_in {tuple(encoder_hidden_states.shape)} ra0_out {tuple(out.shape)}",
                  file=sys.stderr)
        _ra_log["n"] += 1
    _RAP.__call__ = _ra_hook

    cache = {}
    with torch.no_grad():
        out = model(
            sample, timestep, encoder_hidden_states,
            embeds_normal=embeds_normal,
            embeds_position=embeds_position,
            ref_latents=ref_latents,
            cache=cache,
        )

    # Dump the per-block condition_embed_dict cache (post norm1 of dual UNet,
    # rearranged "(b n) l c -> b (n l) c" with n=N_ref=1 -> shape [B, L, C]).
    ced = cache.get("condition_embed_dict") or cache.get("cache", {}).get("condition_embed_dict")
    if ced is None:
        print("WARN: no condition_embed_dict in cache", file=sys.stderr)
    else:
        for k, v in ced.items():
            np.save(os.path.join(args.outdir, f"cache_{k}.npy"),
                    v.detach().cpu().numpy().astype(np.float32))
            print(f"cache[{k}] {tuple(v.shape)} mean={v.mean():.6f} std={v.std():.6f}",
                  file=sys.stderr)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "sample"):
        out = out.sample

    np.save(os.path.join(args.outdir, "out_ra.npy"),
            out.detach().cpu().numpy().astype(np.float32))
    print(f"out_ra {tuple(out.shape)} range=[{out.min():+.3f},{out.max():+.3f}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
