#!/usr/bin/env python3
"""Wan2.2_VAE.pth (fp32 pickle) -> model.safetensors (fp32, names kept)."""
import argparse, os, torch
from safetensors.torch import save_file
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=PRAXIS_ROOT + "/wan2.2-ti2v/Wan2.2_VAE.pth")
ap.add_argument("--out", default=PRAXIS_ROOT + "/wan2.2-vae")
a = ap.parse_args()
sd = torch.load(a.src, map_location="cpu", weights_only=True)
sd = {k: v.detach().float().contiguous() for k, v in sd.items()}
os.makedirs(a.out, exist_ok=True)
save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
print("saved", len(sd), "tensors,", sum(v.numel() for v in sd.values()) / 1e6, "M params ->", a.out)
