"""Extract submodule state-dicts from the multi-component sam-3d-
objects `.ckpt` files and emit them as per-module `.safetensors`
files that our single-file loader (common/safetensors.h) can consume.

The shipped checkpoints are flat pickle state-dicts with deeply
nested keys. For v1 we only need these slices:

  ss_generator.ckpt:
    _base_models.condition_embedder.module_list.0.backbone.*  → sam3d_dinov2.safetensors
    _base_models.condition_embedder.module_list.1.backbone.*  → (same weights, second instance; skipped)
    _base_models.condition_embedder.module_list.2.*           → sam3d_point_patch_embed.safetensors
    _base_models.condition_embedder.projection_net.*          → sam3d_cond_fuser.safetensors
    _base_models.generator.reverse_fn.backbone.*              → sam3d_ss_dit.safetensors

  slat_generator.ckpt:
    _base_models.generator.reverse_fn.backbone.*              → sam3d_slat_dit.safetensors

  ss_decoder.ckpt        (already flat)                        → sam3d_ss_decoder.safetensors
  slat_decoder_gs.ckpt   (already flat)                        → sam3d_slat_gs_decoder.safetensors

Usage:
    python convert_ckpt.py <hf-ckpt-dir> -o <outdir>
    # e.g. <hf-ckpt-dir> = /mnt/disk01/models/sam3d/checkpoints
    #      <outdir>      = /mnt/disk01/models/sam3d/safetensors
"""
import argparse
import os
import sys

# Per-file extraction map: src_ckpt → [(key_prefix(es), out_safetensors)].
# key_prefix may be a str or a tuple of str; tuple entries merge into
# the same output file (used so the fuser's sibling `idx_emb` tensor
# ships alongside its `projection_nets.*` block).
EXTRACT_MAP = [
    ("ss_generator.ckpt", [
        ("_base_models.condition_embedder.module_list.0.backbone.",
         "sam3d_dinov2.safetensors"),
        ("_base_models.condition_embedder.module_list.2.",
         "sam3d_point_patch_embed.safetensors"),
        (("_base_models.condition_embedder.projection_nets.",
          "_base_models.condition_embedder."),
         "sam3d_cond_fuser.safetensors"),
        ("_base_models.generator.reverse_fn.backbone.",
         "sam3d_ss_dit.safetensors"),
    ]),
    ("slat_generator.ckpt", [
        ("_base_models.generator.reverse_fn.backbone.",
         "sam3d_slat_dit.safetensors"),
    ]),
    ("ss_decoder.ckpt", [
        ("", "sam3d_ss_decoder.safetensors"),
    ]),
    ("slat_decoder_gs.ckpt", [
        ("", "sam3d_slat_gs_decoder.safetensors"),
    ]),
]

def load_ckpt(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for wrap in ("state_dict", "model", "module", "weights"):
            if wrap in obj and isinstance(obj[wrap], dict):
                return obj[wrap]
        return obj
    raise RuntimeError(f"{path}: not a dict")

def save_safetensors(outpath, tensors):
    import torch
    from safetensors.torch import save_file
    # Move everything to CPU + contiguous float tensors; safetensors
    # rejects non-tensor entries.
    clean = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().cpu().contiguous()
        else:
            # Sub-dicts / misc objects — skip. Warn only for
            # suspiciously-named keys so we notice missing tensors.
            print(f"  skipping non-tensor {k} ({type(v).__name__})",
                  file=sys.stderr)
    save_file(clean, outpath)
    total = sum(t.numel() * t.element_size() for t in clean.values())
    print(f"  wrote {outpath}  ({len(clean)} tensors, {total/1e6:.1f} MB)",
          file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--only", help="comma-separated module tags to extract "
                                    "(dinov2, point_patch_embed, cond_fuser, "
                                    "ss_dit, slat_dit, ss_decoder, slat_gs_decoder)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(args.ckpt_dir.rstrip('/')),
                                         "safetensors")
    os.makedirs(outdir, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    for src_name, slices in EXTRACT_MAP:
        src = os.path.join(args.ckpt_dir, src_name)
        if not os.path.exists(src):
            print(f"  missing {src}, skipping", file=sys.stderr)
            continue
        want_slices = [(pref, dst) for pref, dst in slices
                       if only is None or
                          any(dst.endswith(f"sam3d_{t}.safetensors") for t in only)]
        if not want_slices:
            continue
        print(f"[convert_ckpt] loading {src}", file=sys.stderr)
        sd = load_ckpt(src)
        for prefix, dst_name in want_slices:
            dst = os.path.join(outdir, dst_name)
            slice_ = {}
            prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
            for k, v in sd.items():
                # pick the most specific matching prefix so broad
                # fallbacks (e.g. the bare condition_embedder. root)
                # only catch direct children, not nested paths already
                # claimed by a longer prefix.
                best = max((p for p in prefixes if k.startswith(p)),
                           key=len, default=None)
                if best is None:
                    continue
                tail = k[len(best):]
                # For the broad fallback prefix, require it to be a
                # direct child (no further dot) to avoid double-capture.
                if best != max(prefixes, key=len) and "." in tail:
                    continue
                slice_[tail] = v
            if not slice_:
                print(f"  no keys matched {prefixes!r}", file=sys.stderr)
                continue
            save_safetensors(dst, slice_)

if __name__ == "__main__":
    sys.exit(main() or 0)
