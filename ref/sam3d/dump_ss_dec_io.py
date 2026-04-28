#!/usr/bin/env python3
"""
dump_ss_dec_io.py — dump one forward call of the sparse-structure VAE
decoder so verify_ss_decoder.c can diff against it.

Instantiates SparseStructureDecoderTdfyWrapper directly from
ss_decoder.yaml (sidestepping the full pipeline, which pulls in
pytorch3d/kaolin/gsplat). Feeds the decoder either a random-gaussian
latent or the SS-DiT output for the shape stream, if available.

Writes under --outdir:
    ss_dec_in.npy    [8, 16, 16, 16]  f32
    ss_dec_out.npy   [1, 64, 64, 64]  f32
"""
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse
import sys
import types
import numpy as np

# Stub out submodules that tdfy_dit/models/__init__ tries to import
# eagerly but we don't need for the VAE decoder.
for mod_name in ("kaolin", "kaolin.utils", "kaolin.utils.testing",
                 "pytorch3d", "pytorch3d.renderer", "gsplat"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["kaolin.utils.testing"].check_tensor = lambda *a, **kw: None
sys.modules["pytorch3d.renderer"].look_at_view_transform = lambda *a, **kw: None


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[dump_ss_dec_io] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss-yaml", required=True,
                    help="$MODELS/sam3d/checkpoints/ss_decoder.yaml")
    ap.add_argument("--ss-ckpt", required=True,
                    help="$MODELS/sam3d/checkpoints/ss_decoder.ckpt")
    ap.add_argument("--latent-npy", default=None,
                    help="optional: pre-existing shape latent (e.g. "
                         "ss_dit_out_shape.npy from step 4). If absent "
                         "uses a fixed-seed random-gaussian latent.")
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

    conf = OmegaConf.load(args.ss_yaml)
    model = instantiate(conf, _recursive_=True).eval()

    print(f"[dump_ss_dec_io] loading ckpt {args.ss_ckpt}...", file=sys.stderr)
    blob = torch.load(args.ss_ckpt, map_location="cpu", weights_only=False)
    sd = blob
    while isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_keys = set(model.state_dict().keys())
    best = None
    for candidate_prefix in [
        "_base_models.decoder.",
        "decoder.",
        "",
    ]:
        matched = {k[len(candidate_prefix):]: v for k, v in sd.items()
                   if k.startswith(candidate_prefix)
                   and k[len(candidate_prefix):] in model_keys}
        if len(matched) > len(best[1] if best else {}):
            best = (candidate_prefix, matched)
    assert best is not None and len(best[1]) > 0, "no matching weights found"
    prefix, matched = best
    print(f"[dump_ss_dec_io] prefix='{prefix}' matched {len(matched)} tensors",
          file=sys.stderr)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"[dump_ss_dec_io]   missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)

    # Latent input: (1, 8, 16, 16, 16).
    if args.latent_npy and os.path.exists(args.latent_npy):
        arr = np.load(args.latent_npy).astype(np.float32)
        if arr.ndim == 2 and arr.shape == (4096, 8):
            # Reshape (N=4096, C=8) → (C=8, D=16, H=16, W=16).
            arr = arr.reshape(16, 16, 16, 8).transpose(3, 0, 1, 2)
        elif arr.ndim == 4:
            if arr.shape[-1] == 8:
                arr = arr.transpose(3, 0, 1, 2)
        else:
            raise ValueError(f"unexpected latent shape {arr.shape}")
        x = torch.from_numpy(np.ascontiguousarray(arr))[None]
    else:
        x = torch.randn((1, 8, 16, 16, 16), dtype=torch.float32)

    save(args.outdir, "ss_dec_in.npy", x[0].numpy())

    with torch.inference_mode():
        y = model(x)

    # y might be dict or tensor. Strip batch dim.
    if isinstance(y, dict):
        y = next(iter(y.values()))
    arr = y.detach().float().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]  # (C, D, H, W)
    save(args.outdir, "ss_dec_out.npy", arr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
