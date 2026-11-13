#!/usr/bin/env python3
"""Cosmos3-Edge-Policy-DROID vae/ (diffusers AutoencoderKLWan, bf16, Wan2.2-TI2V-5B) -> the layout cuda/praxis/cuda_wan22_vae.h loads
(original Wan2.2_VAE.pth key names, fp32, see ref/praxis/convert_vae.py), then verifies the weights against the Wan2.2 VAE already on disk.

  ~/.venv/bin/python -I ref/cosmos3/convert_vae.py [--src .../cosmos3-edge-droid/vae/diffusion_pytorch_model.safetensors]
        [--out /mnt/disk1/praxis/cosmos3-vae-wan] [--orig /mnt/disk1/praxis/wan2.2-vae/model.safetensors]
Verification: every converted tensor must equal bf16_round(orig_fp32) bit for bit (the checkpoint stores the VAE in bf16)."""
import argparse, os, re, torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/mnt/disk1/praxis/cosmos3-edge-droid/vae/diffusion_pytorch_model.safetensors")
ap.add_argument("--out", default="/mnt/disk1/praxis/cosmos3-vae-wan")
ap.add_argument("--orig", default="/mnt/disk1/praxis/wan2.2-vae/model.safetensors")
a = ap.parse_args()

RES = [("norm1.gamma", "residual.0.gamma"), ("conv1", "residual.2"), ("norm2.gamma", "residual.3.gamma"), ("conv2", "residual.6"), ("conv_shortcut", "shortcut")]
def rename(k):
    m = {"quant_conv": "conv1", "post_quant_conv": "conv2"}
    for s, d in m.items():
        if k.startswith(s + "."): return d + k[len(s):]
    for part in ("encoder", "decoder"):
        if not k.startswith(part + "."): continue
        r = k[len(part) + 1:]
        top = {"conv_in": "conv1", "conv_out": "head.2", "norm_out.gamma": "head.0.gamma"}
        for s, d in top.items():
            if r == s or r.startswith(s + "."): return f"{part}.{d}" + r[len(s):]
        mm = re.match(r"mid_block\.resnets\.(\d)\.(.*)", r)
        if mm:
            body = mm.group(2)
            for s, d in RES:
                if body.startswith(s): return f"{part}.middle.{0 if mm.group(1) == '0' else 2}.{d}{body[len(s):]}"
        mm = re.match(r"mid_block\.attentions\.0\.(.*)", r)
        if mm: return f"{part}.middle.1.{mm.group(1)}"
        blk = "down_blocks" if part == "encoder" else "up_blocks"; dst = "downsamples" if part == "encoder" else "upsamples"
        mm = re.match(blk + r"\.(\d+)\.resnets\.(\d+)\.(.*)", r)
        if mm:
            for s, d in RES:
                if mm.group(3).startswith(s): return f"{part}.{dst}.{mm.group(1)}.{dst}.{mm.group(2)}.{d}{mm.group(3)[len(s):]}"
        mm = re.match(blk + r"\.(\d+)\.(downsampler|upsampler)\.(.*)", r)
        if mm: return f"{part}.{dst}.{mm.group(1)}.{dst}.{2 if part == 'encoder' else 3}.{mm.group(3)}"
    raise KeyError(k)

sd, names = {}, {}
with safe_open(a.src, "pt") as f:
    for k in f.keys():
        n = rename(k)
        assert n not in sd, (k, n)
        sd[n] = f.get_tensor(k).float().contiguous(); names[n] = k
os.makedirs(a.out, exist_ok=True)
save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
print(f"converted {len(sd)} tensors -> {a.out}/model.safetensors")

bad = 0; extra = 0
with safe_open(a.orig, "pt") as g:
    ok_keys = set(g.keys())
    for n, v in sd.items():
        if n not in ok_keys: print("NOT IN ORIG:", n, names[n]); extra += 1; continue
        o = g.get_tensor(n)
        if o.shape != v.shape: print("SHAPE", n, o.shape, v.shape); bad += 1; continue
        same = torch.equal(o.bfloat16().float(), v)
        if not same:
            bad += 1; print("DIFF", n, "max|d|=%.3e" % (o.bfloat16().float() - v).abs().max().item())
    missing = ok_keys - set(sd)
print(f"verification vs {a.orig}: tensors {len(sd)}, not-in-orig {extra}, missing-from-converted {len(missing)}, differing after bf16 rounding of the orig: {bad}")
raise SystemExit(1 if bad or extra or missing else 0)
