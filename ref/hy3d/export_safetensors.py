"""Export DiT + conditioner components from combined .ckpt to .safetensors.

Usage:
    uv run python export_safetensors.py \
        --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
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
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    cond_path = os.path.join(args.outdir, "conditioner.safetensors")
    if not os.path.exists(cond_path):
        sd = {
            (k[len("conditioner."):] if k.startswith("conditioner.") else k): v
            for k, v in ckpt["conditioner"].items()
        }
        print(f"  Saving conditioner ({len(sd)} tensors) -> {cond_path}")
        save_file(sd, cond_path)
    else:
        print(f"  Conditioner exists: {cond_path}")

    model_path = os.path.join(args.outdir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"  Saving DiT ({len(ckpt['model'])} tensors) -> {model_path}")
        save_file(ckpt["model"], model_path)
    else:
        print(f"  DiT exists: {model_path}")

    print("Done.")


if __name__ == "__main__":
    main()
