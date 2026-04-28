#!/usr/bin/env python3
"""
dump_slat_gs_io.py — dump one forward call of the SLAT → 3D Gaussian
decoder so verify_slat_gs.c can diff against it.

Instantiates SLatGaussianDecoderTdfyWrapper directly from
slat_decoder_gs.yaml (sidesteps the full pipeline, which pulls in
pytorch3d/kaolin/gsplat). Feeds the decoder either a pre-generated
SLAT sparse tensor (coords+feats from step 7 ref) or a deterministic
random sparse tensor.

Writes under --outdir:
    slat_gs_in_coords.npy    [N, 4]   i32   (batch,z,y,x)
    slat_gs_in_feats.npy     [N, 8]   f32
    slat_gs_out_feats.npy    [N, 448] f32   (post LN + out_layer)
    slat_gs_offset_pert.npy  [32, 3]  f32   (the registered buffer)
    slat_gs_rep_xyz.npy      [N*32, 3] f32
    slat_gs_rep_dc.npy       [N*32, 1, 3] f32
    slat_gs_rep_scaling.npy  [N*32, 3] f32
    slat_gs_rep_rotation.npy [N*32, 4] f32
    slat_gs_rep_opacity.npy  [N*32, 1] f32
"""
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

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


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[dump_slat_gs_io] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-yaml", required=True,
                    help="$MODELS/sam3d/checkpoints/slat_decoder_gs.yaml")
    ap.add_argument("--gs-ckpt", required=True,
                    help="$MODELS/sam3d/checkpoints/slat_decoder_gs.ckpt")
    ap.add_argument("--coords-npy", default=None,
                    help="optional: sparse coords [N,4] i32 (batch,z,y,x)")
    ap.add_argument("--feats-npy", default=None,
                    help="optional: sparse feats [N,8] f32 matching coords")
    ap.add_argument("--n-voxels", type=int, default=1024,
                    help="count for synthetic input when --coords-npy absent")
    ap.add_argument("--outdir", default="/tmp/sam3d_ref")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import sam3d_objects  # noqa: F401
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    conf = OmegaConf.load(args.gs_yaml)
    # Force fp32 so the ref is a clean baseline for bf16 C-side drift budget.
    OmegaConf.set_struct(conf, False)
    conf["use_fp16"] = False
    model = instantiate(conf, _recursive_=True).eval()

    print(f"[dump_slat_gs_io] loading ckpt {args.gs_ckpt}...", file=sys.stderr)
    blob = torch.load(args.gs_ckpt, map_location="cpu", weights_only=False)
    sd = blob
    while isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_keys = set(model.state_dict().keys())
    best = None
    for candidate_prefix in ("_base_models.decoder.", "decoder.", ""):
        matched = {k[len(candidate_prefix):]: v for k, v in sd.items()
                   if k.startswith(candidate_prefix)
                   and k[len(candidate_prefix):] in model_keys}
        if best is None or len(matched) > len(best[1]):
            best = (candidate_prefix, matched)
    assert best is not None and len(best[1]) > 0, "no matching weights"
    prefix, matched = best
    print(f"[dump_slat_gs_io] prefix='{prefix}' matched {len(matched)} tensors",
          file=sys.stderr)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"[dump_slat_gs_io]   missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)

    if args.coords_npy and args.feats_npy \
            and os.path.exists(args.coords_npy) \
            and os.path.exists(args.feats_npy):
        coords = np.load(args.coords_npy).astype(np.int32)
        feats  = np.load(args.feats_npy).astype(np.float32)
        assert coords.shape[0] == feats.shape[0]
        assert coords.shape[1] == 4 and feats.shape[1] == 8
    else:
        rng = np.random.default_rng(args.seed)
        n = args.n_voxels
        # deterministic unique coords in a 64^3 grid; batch=0
        cells = rng.choice(64 ** 3, size=n, replace=False)
        z = cells // (64 * 64)
        y = (cells // 64) % 64
        x = cells % 64
        coords = np.stack([np.zeros_like(z), z, y, x], axis=1).astype(np.int32)
        feats = rng.standard_normal((n, 8)).astype(np.float32)

    save(args.outdir, "slat_gs_in_coords.npy", coords)
    save(args.outdir, "slat_gs_in_feats.npy", feats)

    coords_t = torch.from_numpy(coords)
    feats_t  = torch.from_numpy(feats)
    x = sp.SparseTensor(feats=feats_t, coords=coords_t)

    with torch.inference_mode():
        # Run the base transformer + LN + out_layer manually so we get the
        # pre-to_representation 448-channel sparse feats too.
        h = model.input_layer(x)
        if model.pe_mode == "ape":
            h = h + model.pos_embedder(x.coords[:, 1:])
        h = h.type(model.dtype)
        for blk in model.blocks:
            h = blk(h)
        h = h.type(x.dtype)
        import torch.nn.functional as F
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = model.out_layer(h)

    save(args.outdir, "slat_gs_out_feats.npy", h.feats.detach().float().cpu().numpy())
    save(args.outdir, "slat_gs_offset_pert.npy",
         model.offset_perturbation.detach().float().cpu().numpy())

    # to_representation (mirror of decoder_gs.SLatGaussianDecoder.to_representation
    # but without building the Gaussian wrapper — just dump channel-split outputs).
    with torch.inference_mode():
        feats_all = h.feats.detach().float().cpu()
        coords_all = x.coords.detach().cpu()
        cfg = model.rep_config
        G = cfg["num_gaussians"]
        # h.layout[i] is a Python slice selecting voxels owned by batch i.
        sel = h.layout[0]
        xyz_vox = (coords_all[sel][:, 1:].float() + 0.5) / model.resolution
        rng_xyz = model.layout["_xyz"]["range"]
        offset = feats_all[sel][:, rng_xyz[0]:rng_xyz[1]].reshape(-1, G, 3)
        offset = offset * cfg["lr"]["_xyz"]
        if cfg["perturb_offset"]:
            offset = offset + model.offset_perturbation.detach().float().cpu()
        offset = torch.tanh(offset) / model.resolution * 0.5 * cfg["voxel_size"]
        rep_xyz = (xyz_vox.unsqueeze(1) + offset).flatten(0, 1)

        def chan(name, shape_tail, lr_key):
            r = model.layout[name]["range"]
            arr = feats_all[sel][:, r[0]:r[1]].reshape(-1, *shape_tail).flatten(0, 1)
            return (arr * cfg["lr"][lr_key]).detach().float().cpu().numpy()

        save(args.outdir, "slat_gs_rep_xyz.npy",
             rep_xyz.detach().float().cpu().numpy())
        save(args.outdir, "slat_gs_rep_dc.npy",       chan("_features_dc", (G, 1, 3), "_features_dc"))
        save(args.outdir, "slat_gs_rep_scaling.npy",  chan("_scaling",     (G, 3),    "_scaling"))
        save(args.outdir, "slat_gs_rep_rotation.npy", chan("_rotation",    (G, 4),    "_rotation"))
        save(args.outdir, "slat_gs_rep_opacity.npy",  chan("_opacity",     (G, 1),    "_opacity"))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
