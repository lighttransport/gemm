#!/usr/bin/env python3
"""PyTorch ROCm tex_dec cold rep 0 vs warm rep 1+ — mirrors HIP T2_TEX_REPS=N."""
import os, sys, time, argparse
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runner')))
from shim_bootstrap import install_all  # noqa: E402
install_all()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np, torch
from safetensors.torch import load_file
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
import trellis2.modules.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument('--weights', default='/mnt/disk1/hf-cache/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors')
ap.add_argument('--dump', default='/mnt/disk1/tmp/tex_knight_r512_fresh')
ap.add_argument('--reps', type=int, default=6)
args = ap.parse_args()

dec = SparseUnetVaeDecoder(out_channels=6, model_channels=[1024,512,256,128,64],
    latent_channels=32, num_blocks=[4,16,8,4,0],
    block_type=['SparseConvNeXtBlock3d']*5,
    up_block_type=['SparseResBlockC2S3d']*4,
    block_args=[{'use_checkpoint': False}]*5, use_fp16=True, pred_subdiv=False)
dec.load_state_dict(load_file(args.weights), strict=False)
dec = dec.cuda().eval()

feats = torch.from_numpy(np.load(f'{args.dump}/tex_slat_feats.npy')).cuda().float()
coords = torch.from_numpy(np.load(f'{args.dump}/tex_slat_coords.npy')).cuda().int()
if coords.shape[-1] == 3:
    b = torch.zeros(coords.shape[0], 1, dtype=torch.int32, device='cuda')
    coords = torch.cat([b, coords], dim=-1)
print(f"feats {tuple(feats.shape)} coords {tuple(coords.shape)}", flush=True)

# Pre-load c2s caches dumped by gen_stage2_ref. Keyed by `channel2spatial_2` per stage scale.
caches = {}
for scale in (2, 4, 8, 16):
    p = f'{args.dump}/cache_scale{scale}_c2s'
    caches[scale] = (
        torch.from_numpy(np.load(f'{p}_x_coords.npy')).cuda().int(),
        torch.from_numpy(np.load(f'{p}_idx.npy')).cuda().long(),
        torch.from_numpy(np.load(f'{p}_subidx.npy')).cuda().long(),
    )

from fractions import Fraction
def make():
    t = sp.SparseTensor(feats=feats, coords=coords)
    # Cache key is `str(_scale)`. Decoder starts at scale (1,1,1) with downsampled
    # coords; first c2s_2 looks up at scale (1,1,1) producing scale (1/2,...) etc.
    # Dump scales 16,8,4,2 correspond to denominators in upsample chain.
    sc = t._spatial_cache
    # Dump scale 16 = coarsest = first C2S input (decoder scale (1,1,1)).
    # Each subsequent C2S halves scale. Map dump_scale -> input fraction = 16/dump_scale.
    for dump_scale, c in caches.items():
        denom = 16 // dump_scale  # 16->1, 8->2, 4->4, 2->8
        f = Fraction(1, denom)
        sk = str((f, f, f))
        sc.setdefault(sk, {})['channel2spatial_2'] = c
    return t

# NO pre-warmup. Time every rep starting from rep 0 (cold).
torch.cuda.synchronize()
times = []
with torch.no_grad():
    for i in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = dec(make())
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        print(f"rep {i}: {dt:.1f} ms", flush=True)

cold = times[0]; warm = times[1:]
print(f"\ncold rep 0  : {cold:.1f} ms")
if warm:
    print(f"warm rep 1+ : min={min(warm):.1f}  median={sorted(warm)[len(warm)//2]:.1f}  max={max(warm):.1f}  ms")
    print(f"cold-warm gap: {cold - min(warm):.1f} ms")
