"""Convert Hunyuan3D-2.1 ShapeVAE .ckpt to .safetensors.

Upstream ships `hunyuan3d-vae-v2-1/model.fp16.ckpt` (PyTorch pickle).
The C runner wants a .safetensors file. This script does the flat conversion.

Usage:
    uv run python export_vae_safetensors.py \
        --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/model.fp16.ckpt \
        --outdir /mnt/disk01/models/Hunyuan3D-2.1
"""
import argparse
import os
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--name", type=str, default="vae.safetensors")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Coerce any non-tensor leaves out and contiguous-ify
    clean = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous()
    print(f"  {len(clean)} tensors")

    out = os.path.join(args.outdir, args.name)
    save_file(clean, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
