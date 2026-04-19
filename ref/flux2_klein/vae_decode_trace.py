#!/usr/bin/env python3
"""Run diffusers VAE decode in F32 and dump per-block intermediates.

Hooks into vae.decoder to capture:
  - x after conv_in
  - x after mid_block
  - x after each up_block
  - x after norm_out+SiLU
  - x final (conv_out)
Saves each as raw F32 CHW .bin for diffing against HIP per-stage outputs.
"""
import numpy as np
import torch
from diffusers import Flux2KleinPipeline

# F32 decode
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                           torch_dtype=torch.float32)
vae = pipe.vae.to("cuda").eval()
for p in vae.parameters():
    p.requires_grad_(False)

# Load packed final latent
packed = np.fromfile("final_latent_packed.bin", dtype=np.float32).reshape(256, 128)
ph, pw = 16, 16
t = torch.from_numpy(packed).reshape(ph, pw, 128).permute(2, 0, 1).unsqueeze(0).to("cuda")

# BN + unpatch (match pipeline)
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(t.device, t.dtype)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(t.device, t.dtype)
t = t * bn_std + bn_mean
t = Flux2KleinPipeline._unpatchify_latents(t)
print(f"vae input  : {tuple(t.shape)} mean {t.mean().item():.4f} std {t.std().item():.4f}")

# Dump VAE input
t[0].detach().cpu().numpy().astype(np.float32).tofile("vae_in_f32.bin")

# Monkey-patch the decoder to log intermediates
dec = vae.decoder

def save(name, x):
    arr = x[0].detach().to(torch.float32).cpu().numpy()
    arr.tofile(f"vae_trace_{name}.bin")
    print(f"  {name:15s} shape {arr.shape} mean {arr.mean():.4f} std {arr.std():.4f} min {arr.min():.4f} max {arr.max():.4f}")

with torch.no_grad():
    # post_quant_conv
    x = vae.post_quant_conv(t) if vae.post_quant_conv is not None else t
    save("post_q", x)
    # conv_in
    x = dec.conv_in(x)
    save("conv_in", x)
    # mid_block
    x = dec.mid_block(x, None)
    save("mid", x)
    # up_blocks
    for i, up in enumerate(dec.up_blocks):
        x = up(x, None)
        save(f"up{i}", x)
    # norm_out + act
    x = dec.conv_norm_out(x)
    save("norm_out", x)
    x = dec.conv_act(x)
    save("act", x)
    x = dec.conv_out(x)
    save("conv_out", x)

print(f"final: mean {x.mean().item():.4f} std {x.std().item():.4f} min {x.min().item():.4f} max {x.max().item():.4f}")

from PIL import Image
img = x[0].detach().float().cpu().numpy()
u8 = ((img * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
Image.fromarray(u8).save("vae_out_f32.png")
img.tofile("vae_out_f32.bin")
print("saved vae_out_f32.png / vae_out_f32.bin")
