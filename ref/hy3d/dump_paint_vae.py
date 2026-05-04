"""Dump SD-2.1 paint VAE encoder + decoder reference outputs.

Used to validate the native CUDA port of the Hunyuan3D-2.1 paint VAE
(stock diffusers AutoencoderKL, ch=[128,256,512,512], latent_ch=4).

Produces:
    {prefix}_input.npy        [3, H, W]   float32   normalized RGB in [-1, 1]
    {prefix}_latent_mean.npy  [4, H/8, W/8] float32 mean of encoder posterior
    {prefix}_latent_logvar.npy[4, H/8, W/8] float32 logvar
    {prefix}_latent.npy       [4, H/8, W/8] float32 mean (deterministic encode)
    {prefix}_recon.npy        [3, H, W]   float32   decoded RGB in [-1, 1]

Usage:
  uv run --with torch --with diffusers --with safetensors --with pillow \\
      python dump_paint_vae.py \\
      --image /tmp/some_view.png \\
      --vae /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/vae \\
      --outdir /tmp/hy3d_paint_vae_ref \\
      [--resolution 512]
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL


def load_image(path: str, res: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = arr * 2.0 - 1.0                           # [-1, 1]
    return arr.transpose(2, 0, 1)                   # CHW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--vae", default="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/vae")
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_vae_ref")
    ap.add_argument("--prefix", default="ref")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype).eval()
    vae.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loaded vae (latent_ch={vae.config.latent_channels}, "
          f"block_out={list(vae.config.block_out_channels)})")

    rgb = load_image(args.image, args.resolution)
    x = torch.from_numpy(rgb)[None].to(vae.device, dtype)

    with torch.no_grad():
        post = vae.encode(x).latent_dist
        mean = post.mean
        logvar = post.logvar
        z = mean  # deterministic
        recon = vae.decode(z).sample

    os.makedirs(args.outdir, exist_ok=True)
    def save(name, t):
        path = os.path.join(args.outdir, f"{args.prefix}_{name}.npy")
        np.save(path, t.detach().cpu().float().numpy()[0])
        return path

    save("input", x)
    save("latent_mean", mean)
    save("latent_logvar", logvar)
    save("latent", z)
    save("recon", recon)

    print(f"input  range=[{x.min().item():+.3f}, {x.max().item():+.3f}] shape={tuple(x.shape)}")
    print(f"latent range=[{z.min().item():+.3f}, {z.max().item():+.3f}] shape={tuple(z.shape)}")
    print(f"recon  range=[{recon.min().item():+.3f}, {recon.max().item():+.3f}] shape={tuple(recon.shape)}")
    print(f"wrote 5 .npy files to {args.outdir}")


if __name__ == "__main__":
    main()
