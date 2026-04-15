#!/usr/bin/env python3
"""Run diffusers VAE decode on the pinned final_latent_packed.bin
(with BN de-norm and unpatchify) and save the RAW F32 RGB tensor for diffing."""
import numpy as np
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                           torch_dtype=torch.bfloat16)
vae = pipe.vae.to("cuda")

# Load packed final latent: [n_patches, 128] F32
packed = np.fromfile("final_latent_packed.bin", dtype=np.float32).reshape(256, 128)
# For 256×256 output at vae_scale_factor=8, latent is 32x32, packed is 16x16 patches
ph, pw = 16, 16

# Reshape packed [ph*pw, 128] -> [1, 128, ph, pw]  (diffusers' _unpack_latents)
t = torch.from_numpy(packed).reshape(ph, pw, 128).permute(2, 0, 1).unsqueeze(0)
t = t.to("cuda", dtype=torch.bfloat16)

# BN denorm in 128-channel space
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(t.device, t.dtype)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(t.device, t.dtype)
t = t * bn_std + bn_mean

# Unpatchify 128 -> 32 ch, 16x16 -> 32x32 spatial
# diffusers uses Flux2KleinPipeline._unpatchify_latents
t = Flux2KleinPipeline._unpatchify_latents(t)
print("latent for vae.decode:", tuple(t.shape), t.dtype, "mean", t.float().mean().item(), "std", t.float().std().item())

with torch.no_grad():
    img = vae.decode(t, return_dict=False)[0]
print("vae raw out:", tuple(img.shape), img.dtype, "mean", img.float().mean().item(),
      "min", img.float().min().item(), "max", img.float().max().item())

img_f32 = img[0].detach().to(torch.float32).cpu().numpy()  # [3, H, W]
img_f32.tofile("vae_out_ref.bin")
print(f"saved vae_out_ref.bin ({img_f32.shape} F32)")

# Also save postprocessed uint8 image for sanity
img_norm = ((img_f32 * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
from PIL import Image
Image.fromarray(img_norm).save("vae_out_ref.png")
print("saved vae_out_ref.png")
