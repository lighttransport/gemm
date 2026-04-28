#!/usr/bin/env python3
"""
dump_ss_dit_io.py — dump one forward call into the SS Flow DiT for
verify_ss_dit.c to diff against.

Side-steps the full InferencePipelinePointMap (which drags in pytorch3d,
kaolin, gsplat, MoGe) and directly instantiates just the
SparseStructureFlowTdfyWrapper from ss_generator.yaml, loads matching
weights from ss_generator.ckpt, and runs one forward pass with:
  * random-gaussian latents (torch.manual_seed(args.seed)) — shape (16³×8
    + 1×6 + 1×3 + 1×3 + 1×1)
  * cond tokens loaded from --cond-npy (defaults to
    /tmp/sam3d_ref/cond_tokens.npy produced by dump_cond_tokens.py)
  * fixed t, d values (defaults match step-0 of the shortcut sampler)

Writes under --outdir:
    ss_dit_in_<name>.npy     (token_len, in_channels) f32
    ss_dit_out_<name>.npy    (token_len, in_channels) f32
    ss_dit_cond.npy          (N_cond, 1024) f32    — copy of input cond
    ss_dit_t.npy             () f32
    ss_dit_d.npy             () f32
"""
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse
import sys
import types
import numpy as np

# tdfy_dit/models/__init__ eagerly imports SLatGaussianDecoder + the mesh
# representation chain, which needs kaolin / pytorch3d / flexicubes. We
# only use SparseStructureFlowTdfyWrapper here, so stub the missing deps.
for mod_name in ("kaolin", "kaolin.utils", "kaolin.utils.testing",
                 "pytorch3d", "pytorch3d.renderer", "gsplat"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["kaolin.utils.testing"].check_tensor = lambda *a, **kw: None
sys.modules["pytorch3d.renderer"].look_at_view_transform = lambda *a, **kw: None


def save(outdir, name, arr):
    p = os.path.join(outdir, name)
    np.save(p, np.ascontiguousarray(arr))
    print(f"[dump_ss_dit_io] wrote {p} shape={arr.shape} dtype={arr.dtype}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss-yaml", required=True,
                    help="$MODELS/sam3d/checkpoints/ss_generator.yaml")
    ap.add_argument("--ss-ckpt", required=True,
                    help="$MODELS/sam3d/checkpoints/ss_generator.ckpt")
    ap.add_argument("--cond-npy", default="/tmp/sam3d_ref/cond_tokens.npy")
    ap.add_argument("--outdir", default="/tmp/sam3d_ref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t", type=float, default=1.0,
                    help="timestep pre-time_scale (upstream uses t ∈ [0, 1])")
    ap.add_argument("--d", type=float, default=1.0,
                    help="shortcut jump size; 1.0 = full jump to t=0")
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

    # Descend to the SparseStructureFlowTdfyWrapper config.
    bb_conf = conf.module.generator.backbone.reverse_fn.backbone
    # The SS wrapper needs condition_embedder=None at instantiate time
    # and doesn't use cfg at inference here.
    bb_conf.condition_embedder = None

    model = instantiate(bb_conf, _recursive_=True).eval()

    # Load weights from ckpt. Try prefixes until one matches.
    print(f"[dump_ss_dit_io] loading ckpt {args.ss_ckpt}...", file=sys.stderr)
    blob = torch.load(args.ss_ckpt, map_location="cpu", weights_only=False)
    sd = blob
    while isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_keys = set(model.state_dict().keys())
    best = None
    for candidate_prefix in [
        "_base_models.generator.reverse_fn.backbone.",
        "_base_models.generator.backbone.reverse_fn.backbone.",
        "generator.reverse_fn.backbone.",
        "",
    ]:
        matched = {k[len(candidate_prefix):]: v for k, v in sd.items()
                   if k.startswith(candidate_prefix)
                   and k[len(candidate_prefix):] in model_keys}
        if len(matched) > len(best[1] if best else {}):
            best = (candidate_prefix, matched)
    assert best is not None and len(best[1]) > 0, "no matching weights found"
    prefix, matched = best
    print(f"[dump_ss_dit_io] prefix='{prefix}' matched {len(matched)} tensors",
          file=sys.stderr)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"[dump_ss_dit_io]   missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    if missing:
        print(f"[dump_ss_dit_io]   first missing: {missing[:5]}", file=sys.stderr)

    # Random latents per the 5 modalities.
    spec = [
        ("shape",                 (4096, 8)),
        ("6drotation_normalized", (1,    6)),
        ("translation",           (1,    3)),
        ("scale",                 (1,    3)),
        ("translation_scale",     (1,    1)),
    ]
    latents = {}
    for name, shape in spec:
        x = torch.randn((1,) + shape, dtype=torch.float32)
        latents[name] = x
        save(args.outdir, f"ss_dit_in_{name}.npy", x[0].numpy())

    # cond: load dumped tensor.
    cond_np = np.load(args.cond_npy).astype(np.float32)
    # Expect (N, 1024) or (1, N, 1024)
    if cond_np.ndim == 3:
        cond_np = cond_np[0]
    cond = torch.from_numpy(cond_np)[None]
    save(args.outdir, "ss_dit_cond.npy", cond_np)

    t = torch.tensor([args.t], dtype=torch.float32)
    d = torch.tensor([args.d], dtype=torch.float32)
    save(args.outdir, "ss_dit_t.npy", np.asarray(args.t, np.float32))
    save(args.outdir, "ss_dit_d.npy", np.asarray(args.d, np.float32))

    with torch.inference_mode():
        out = model(latents, t, cond, d=d)

    for name, _ in spec:
        arr = out[name].detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        save(args.outdir, f"ss_dit_out_{name}.npy", arr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
