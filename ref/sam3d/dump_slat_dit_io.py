#!/usr/bin/env python3
"""
dump_slat_dit_io.py — dump one forward call of the SLAT Flow DiT
(stage-2) so verify_slat_dit.c can diff against it.

Instantiates SLatFlowModelTdfyWrapper directly from
slat_generator.yaml (only the generator.backbone subtree, sidestepping
the full EmbedderFuser which needs DINOv2 + pointmap). Runs one step
of the transformer with a synthetic sparse tensor, fixed t, and
caller-provided cond tokens.

Writes under --outdir:
    slat_dit_in_coords.npy  [N, 4]          i32
    slat_dit_in_feats.npy   [N, 8]          f32
    slat_dit_t.npy          ()              f32
    slat_dit_cond.npy       [1, Ltok, 1024] f32
    slat_dit_out_feats.npy  [N, 8]          f32
"""
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
# spconv's default "auto" falls back to MaskImplicitGemm (CUDA-only) on
# SubMConv3d. Force native so the ref dump runs on CPU on hosts where the
# installed torch lacks a kernel image for the local GPU arch.
os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
import sys
import types
import numpy as np

for mod_name in ("kaolin", "kaolin.utils", "kaolin.utils.testing",
                 "pytorch3d", "pytorch3d.renderer", "gsplat"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["kaolin.utils.testing"].check_tensor = lambda *a, **kw: None
sys.modules["pytorch3d.renderer"].look_at_view_transform = lambda *a, **kw: None


def _install_spconv_cpu_bias_patch():
    """spconv 2.3.8 asserts `!features.is_cpu()` if the SubMConv3d has a bias
    parameter attached. Unregister the bias around the spconv call and add it
    back manually — result is numerically identical and CPU-safe."""
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_spconv import (
        SparseConv3d, SparseInverseConv3d,
    )

    def _wrap(cls):
        orig_forward = cls.forward

        def patched(self, x):
            saved = self.conv.bias
            if saved is not None:
                self.conv._parameters['bias'] = None
            try:
                out = orig_forward(self, x)
            finally:
                if saved is not None:
                    self.conv._parameters['bias'] = saved
            if saved is not None:
                # spconv's SparseConvTensor forbids direct .features assignment.
                out.data = out.data.replace_feature(out.data.features + saved)
            return out

        cls.forward = patched

    _wrap(SparseConv3d)
    _wrap(SparseInverseConv3d)


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[dump_slat_dit_io] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def _strip_to_base_model(sd, model_keys):
    """Pick the prefix from the shipped ckpt that lines up with our
    bare SLatFlowModel state dict. ckpt keys look like
    `reverse_fn.backbone._base_models.generator.{input_layer|blocks|...}`."""
    best = None
    for prefix in (
        "_base_models.generator.reverse_fn.backbone.",
        "generator.reverse_fn.backbone.",
        "reverse_fn.backbone.",
        "",
    ):
        matched = {k[len(prefix):]: v for k, v in sd.items()
                   if k.startswith(prefix)
                   and k[len(prefix):] in model_keys}
        if best is None or len(matched) > len(best[1]):
            best = (prefix, matched)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat-yaml", required=True,
                    help="$MODELS/sam3d/checkpoints/slat_generator.yaml")
    ap.add_argument("--slat-ckpt", required=True,
                    help="$MODELS/sam3d/checkpoints/slat_generator.ckpt")
    ap.add_argument("--coords-npy", default=None,
                    help="optional: sparse coords [N,4] i32")
    ap.add_argument("--feats-npy", default=None,
                    help="optional: sparse feats [N,8] f32")
    ap.add_argument("--cond-npy", default=None,
                    help="optional: cond tokens [1,L,1024]; default random")
    ap.add_argument("--n-voxels", type=int, default=1024)
    ap.add_argument("--cond-tokens", type=int, default=1374,
                    help="DINOv2 img (687) + mask (687) = 1374; any L works")
    ap.add_argument("--t", type=float, default=1.0,
                    help="time value fed to t_embedder (raw, pre time_scale)")
    ap.add_argument("--outdir", default="/tmp/sam3d_ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-trace", action="store_true",
                    help="dump per-stage activations for C-vs-pytorch debugging")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cpu")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import sam3d_objects  # noqa: F401
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    _install_spconv_cpu_bias_patch()

    full = OmegaConf.load(args.slat_yaml)
    backbone_cfg = full.module.generator.backbone.reverse_fn.backbone
    OmegaConf.set_struct(backbone_cfg, False)
    backbone_cfg["use_fp16"] = False
    backbone_cfg["condition_embedder"] = None
    # Keep the Tdfy wrapper but cut the embedder — we feed cond directly.
    model = instantiate(backbone_cfg, _recursive_=True).eval().to(device)

    print(f"[dump_slat_dit_io] loading ckpt {args.slat_ckpt}...", file=sys.stderr)
    blob = torch.load(args.slat_ckpt, map_location="cpu", weights_only=False)
    sd = blob
    while isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    prefix, matched = _strip_to_base_model(sd, set(model.state_dict().keys()))
    assert len(matched) > 0, "no matching weights"
    print(f"[dump_slat_dit_io] prefix='{prefix}' matched {len(matched)} tensors",
          file=sys.stderr)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"[dump_slat_dit_io]   missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    if args.coords_npy and args.feats_npy \
            and os.path.exists(args.coords_npy) \
            and os.path.exists(args.feats_npy):
        coords = np.load(args.coords_npy).astype(np.int32)
        feats  = np.load(args.feats_npy).astype(np.float32)
    else:
        n = args.n_voxels
        cells = rng.choice(64 ** 3, size=n, replace=False)
        z = cells // (64 * 64)
        y = (cells // 64) % 64
        x = cells % 64
        coords = np.stack([np.zeros_like(z), z, y, x], axis=1).astype(np.int32)
        feats = rng.standard_normal((n, 8)).astype(np.float32)

    if args.cond_npy and os.path.exists(args.cond_npy):
        cond = np.load(args.cond_npy).astype(np.float32)
    else:
        cond = rng.standard_normal((1, args.cond_tokens, 1024)).astype(np.float32)

    t = np.float32(args.t)

    save(args.outdir, "slat_dit_in_coords.npy", coords)
    save(args.outdir, "slat_dit_in_feats.npy",  feats)
    save(args.outdir, "slat_dit_cond.npy",      cond)
    save(args.outdir, "slat_dit_t.npy",         np.array(t, dtype=np.float32))

    coords_t = torch.from_numpy(coords).to(device)
    feats_t  = torch.from_numpy(feats).to(device)
    cond_t   = torch.from_numpy(cond).to(device)
    t_t      = torch.tensor([t], dtype=torch.float32, device=device)

    x = sp.SparseTensor(feats=feats_t, coords=coords_t)
    with torch.inference_mode():
        # Base SLatFlowModel.forward — skip the Tdfy wrapper's cond embedder.
        h = model.input_layer(x).type(model.dtype)
        if args.dump_trace:
            save(args.outdir, "py_h_after_input_layer.npy",
                 h.feats.detach().float().cpu().numpy())
        t_emb = model.t_embedder(t_t)
        if model.is_shortcut_model:
            d_t = torch.tensor([1.0], dtype=torch.float32, device=device)
            t_emb = t_emb + model.d_embedder(d_t)
        t_emb = t_emb.type(model.dtype)
        if args.dump_trace:
            save(args.outdir, "py_t_emb.npy",
                 t_emb.detach().float().cpu().numpy().squeeze(0))
        cond_m = cond_t.type(model.dtype)

        skips = []
        for i, blk in enumerate(model.input_blocks):
            if args.dump_trace and i == 1:
                # manually expand input_block_1 to dump intermediates
                import torch.nn.functional as F
                emb_out = blk.emb_layers(t_emb).type(h.dtype)
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                save(args.outdir, "py_ib1_emb_scale.npy",
                     scale.detach().float().cpu().numpy())
                save(args.outdir, "py_ib1_emb_shift.npy",
                     shift.detach().float().cpu().numpy())
                h_down = blk._updown(h)
                save(args.outdir, "py_ib1_xu_feats.npy",
                     h_down.feats.detach().float().cpu().numpy())
                save(args.outdir, "py_ib1_xu_coords.npy",
                     h_down.coords.detach().cpu().numpy().astype(np.int32))
                hh = h_down.replace(blk.norm1(h_down.feats))
                save(args.outdir, "py_ib1_after_norm1.npy",
                     hh.feats.detach().float().cpu().numpy())
                hh = hh.replace(F.silu(hh.feats))
                save(args.outdir, "py_ib1_after_silu1.npy",
                     hh.feats.detach().float().cpu().numpy())
                hh = blk.conv1(hh)
                save(args.outdir, "py_ib1_after_conv1.npy",
                     hh.feats.detach().float().cpu().numpy())
                hh = hh.replace(blk.norm2(hh.feats)) * (1 + scale) + shift
                save(args.outdir, "py_ib1_after_norm2mod.npy",
                     hh.feats.detach().float().cpu().numpy())
                hh = hh.replace(F.silu(hh.feats))
                save(args.outdir, "py_ib1_after_silu2.npy",
                     hh.feats.detach().float().cpu().numpy())
                hh = blk.conv2(hh)
                save(args.outdir, "py_ib1_after_conv2.npy",
                     hh.feats.detach().float().cpu().numpy())
                skip_out = blk.skip_connection(h_down)
                save(args.outdir, "py_ib1_skip_out.npy",
                     skip_out.feats.detach().float().cpu().numpy())
                h = hh + skip_out
            else:
                h = blk(h, t_emb)
            skips.append(h.feats)
            if args.dump_trace:
                save(args.outdir, f"py_h_after_input_block_{i}.npy",
                     h.feats.detach().float().cpu().numpy())
                save(args.outdir, f"py_coords_after_input_block_{i}.npy",
                     h.coords.detach().cpu().numpy().astype(np.int32))
        if model.pe_mode == "ape":
            h = h + model.pos_embedder(h.coords[:, 1:]).type(model.dtype)
        if args.dump_trace:
            save(args.outdir, "py_h_after_ape.npy",
                 h.feats.detach().float().cpu().numpy())
        for i, blk in enumerate(model.blocks):
            h = blk(h, t_emb, cond_m)
            if args.dump_trace and (i == 0 or i == len(model.blocks) - 1):
                save(args.outdir, f"py_h_after_block_{i}.npy",
                     h.feats.detach().float().cpu().numpy())
        for i, (blk, skip) in enumerate(zip(model.out_blocks, reversed(skips))):
            if model.use_skip_connection:
                h = blk(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = blk(h, t_emb)
            if args.dump_trace:
                save(args.outdir, f"py_h_after_out_block_{i}.npy",
                     h.feats.detach().float().cpu().numpy())
        import torch.nn.functional as F
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = model.out_layer(h.type(x.dtype))

    save(args.outdir, "slat_dit_out_feats.npy",
         h.feats.detach().float().cpu().numpy())
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
