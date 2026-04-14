"""Dump DINOv2 encoder reference outputs for CUDA verification.

Usage:
    uv run python dump_dinov2.py \
        --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
        [--image path.png] [--outdir output]
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image


def preprocess_image(img_path, size=518):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return ((img - mean) / std).transpose(2, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cond_sd = {
        (k[len("conditioner."):] if k.startswith("conditioner.") else k): v
        for k, v in ckpt["conditioner"].items()
    }

    from transformers import Dinov2Model, Dinov2Config
    config = Dinov2Config(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        patch_size=14,
        image_size=518,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        layerscale_value=1.0,
    )
    model = Dinov2Model(config)
    prefix = "main_image_encoder.model."
    new_sd = {k[len(prefix):]: v for k, v in cond_sd.items() if k.startswith(prefix)}
    model.load_state_dict(new_sd, strict=False)
    model = model.float().eval()

    if args.image:
        img = preprocess_image(args.image)
    else:
        np.random.seed(42)
        img = (np.random.randn(3, 518, 518).astype(np.float32) * 0.1)
    np.save(os.path.join(args.outdir, "dinov2_input.npy"), img)

    with torch.no_grad():
        out = model(
            torch.from_numpy(img).unsqueeze(0).float(),
            output_hidden_states=True,
        )
    result = out.last_hidden_state.numpy()[0]
    np.save(os.path.join(args.outdir, "dinov2_output.npy"), result)
    print(f"  Output: {result.shape}, mean={result.mean():.6f}, std={result.std():.6f}")

    if out.hidden_states:
        for i, hs in enumerate(out.hidden_states):
            if i in (0, 12, 23, 24):
                np.save(os.path.join(args.outdir, f"dinov2_hidden_{i}.npy"), hs.numpy()[0])

    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
