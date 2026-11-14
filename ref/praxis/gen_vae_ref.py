#!/usr/bin/env python3
"""Dump Wan2.2 VAE reference using praxis's own vae.py (the oracle).

Input video: fixed-seed smooth random clip (B=1,3,T=1+4k,H,W), in [-1,1]
(video.npy is the [-1,1] clip; encode_eager is fed (v+1)*127.5 because it
expects [0,255] frames).
Output: video.npy, latent.npy (policy-normalised), recon.npy (decode(encode),
uint8-valued as vae.decode returns), recon_unit.npy (decode_unit, float [-1,1],
shape [1,3,T',H,W]) and, with --inter, intermediate tensors (channel-first
[1,C,T,H,W], chunks concatenated along T):
  enc_conv1 enc_down0..3 enc_mid enc_head dec_conv2 dec_conv1 dec_mid
  dec_up0..3 dec_head
--fp32 : plain fp32 reference (raw fp32 weights, no autocast), same loops as
         encode_eager / decode_unit. Use with WAN_VAE_RAW_WEIGHTS=1 in the C runner.
"""
import argparse, os, sys, numpy as np, torch
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))
ap = argparse.ArgumentParser()
ap.add_argument("--praxis-src", default=PRAXIS_ROOT + "/praxis-1-oss")
ap.add_argument("--weights", default=PRAXIS_ROOT + "/wan2.2-ti2v/Wan2.2_VAE.pth")
ap.add_argument("--out", default="tmp/praxis_ref/vae")
ap.add_argument("--frames", type=int, default=5)
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--device", default="cuda")
ap.add_argument("--inter", action="store_true", help="dump intermediate tensors")
ap.add_argument("--fp32", action="store_true", help="fp32 weights, no autocast")
a = ap.parse_args()
sys.path.insert(0, a.praxis_src)
import praxis.vae as V
from praxis.vae import Autoencoder
os.makedirs(a.out, exist_ok=True)

if a.fp32:
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    vae = Autoencoder(decoder_impl="reference")
    vae.load_state_dict(state, strict=True)
    vae = vae.to(a.device).eval().requires_grad_(False)
else:
    vae = Autoencoder.from_pretrained(a.weights, device=a.device, decoder_impl="reference")

taps = {}
def hook(name):
    def f(m, i, o):
        taps.setdefault(name, []).append(o.detach().float().cpu())
    return f
if a.inter:
    e, d = vae.encoder, vae.decoder
    e.conv1.register_forward_hook(hook("enc_conv1"))
    for i, b in enumerate(e.downsamples): b.register_forward_hook(hook(f"enc_down{i}"))
    e.middle.register_forward_hook(hook("enc_mid")) if False else None
    e.middle[2].register_forward_hook(hook("enc_mid"))
    e.head[2].register_forward_hook(hook("enc_head"))
    d.conv1.register_forward_hook(hook("dec_conv1"))
    d.middle[2].register_forward_hook(hook("dec_mid"))
    for i, b in enumerate(d.upsamples): b.register_forward_hook(hook(f"dec_up{i}"))
    d.head[2].register_forward_hook(hook("dec_head"))
    vae.conv2.register_forward_hook(hook("dec_conv2"))

g = torch.Generator().manual_seed(0)
v = torch.randn(1, 3, a.frames, a.size // 8, a.size // 8, generator=g)
v = torch.nn.functional.interpolate(v.flatten(1, 2), size=(a.size, a.size), mode="bilinear").view(1, 3, a.frames, a.size, a.size)
v = v.clamp(-1, 1).to(a.device)

@torch.no_grad()
def encode_fp32(x_in):
    x = V.patchify(x_in)
    caches = V.Caches(V.count_causal_convs(vae.encoder))
    n = 1 + (x.shape[2] - 1) // 4
    mean, inv_std = vae.scale
    outs = []
    for i in range(n):
        caches.index = 0
        sl = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1): 1 + 4 * i]
        mu = vae.conv1(vae.encoder(sl, caches)).chunk(2, dim=1)[0]
        outs.append((mu - mean.view(1, 48, 1, 1, 1)) * inv_std.view(1, 48, 1, 1, 1))
    return V.to_policy_latent(torch.cat(outs, dim=2))

@torch.no_grad()
def decode_fp32(latent):
    z = V.from_policy_latent(latent)
    mean, inv_std = vae.scale
    z = z / inv_std.view(1, 48, 1, 1, 1) + mean.view(1, 48, 1, 1, 1)
    x = vae.conv2(z)
    caches = V.Caches(V.count_causal_convs(vae.decoder))
    outs = []
    for i in range(z.shape[2]):
        caches.index = 0
        outs.append(vae.decoder(x[:, :, i:i + 1], caches, first_chunk=i == 0))
    return V.unpatchify(torch.cat(outs, dim=2)).float().clamp_(-1, 1)

with torch.no_grad():
    if a.fp32:
        z = encode_fp32(v)
        ru = decode_fp32(z)
    else:
        # encode_eager expects frames in [0,255] and maps them to [-1,1] itself
        z = vae.encode_eager((v + 1.0) * 127.5)
        ru = vae.decode_unit(z)
    vv = ru[0].permute(1, 0, 2, 3).float().mul(0.5).add(0.5).clamp(0, 1)
    r = (vv * 255).to(torch.uint8)
print("video", tuple(v.shape), "latent", tuple(z.shape), "recon", tuple(r.shape))
for n, t in (("video", v), ("latent", z), ("recon", r), ("recon_unit", ru)):
    np.save(f"{a.out}/{n}.npy", t.float().cpu().numpy())
for n, lst in taps.items():
    np.save(f"{a.out}/{n}.npy", torch.cat(lst, dim=2).numpy())
    print("dumped", n, tuple(torch.cat(lst, dim=2).shape))
