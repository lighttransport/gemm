"""Cross-check of the C packer (PRAXIS_PACK_CACHE blob) against the praxis nvfp4.py recipe re-implemented in torch (CPU), on one real expert.
  ~/.venv/bin/python -I cuda/praxis/check_pack.py --cache /mnt/disk1/praxis/praxis-1-droid.nvfp4.packcache --ckpt /mnt/disk1/praxis/praxis-1-droid --block 2 --expert 3
Also confirms the e4m3 scale clamp [2^-6,448], global = amax/2688 and reports the fraction of identical codes / scale bytes."""
import argparse, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_attr as M
M.dev = "cpu"
M.GRID = M.GRID.cpu()
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--block", type=int, default=2); ap.add_argument("--expert", type=int, default=3)
a = ap.parse_args()
E, C, Hm = 128, 3072, 1280
al = lambda x: (x + 255) & ~255
gu_q = 0; gu_sf = al(E * 2 * Hm * C // 2); dn_q = al(gu_sf + E * 2 * Hm * C // 16); dn_sf = al(dn_q + E * C * Hm // 2)
gu_g = al(dn_sf + E * C * Hm // 16); dn_g = al(gu_g + E * 4); total = al(dn_g + E * 4)
moe_id = a.block - 2
f = open(a.cache, "rb"); base = 4096 + moe_id * total
def rd(off, n):
    f.seek(base + off); return np.frombuffer(f.read(n), dtype=np.uint8)
ck = M.Ckpt(a.ckpt)
for nm, N, K, q_off, sf_off, g_off in (("gate_up_weight", 2 * Hm, C, gu_q, gu_sf, gu_g), ("down_weight", C, Hm, dn_q, dn_sf, dn_g)):
    W = ck.get(f"backbone.encoder.core.blocks.{a.block}.ffn.routed.{nm}")[a.expert].float()   # [K][N]
    g = W.abs().max().clamp_min(1e-12) * (1.0 / 2688.0)
    q = rd(q_off + a.expert * N * (K // 2), N * K // 2).reshape(N, K // 2)
    sf = rd(sf_off + a.expert * N * (K // 16), N * K // 16).reshape(N, K // 16)
    glob = np.frombuffer(f.read(0), dtype=np.float32)
    f.seek(base + g_off + a.expert * 4); cg = np.frombuffer(f.read(4), dtype=np.float32)[0]
    # torch recipe: blocks along K for each output channel n
    Wt = W.t().contiguous()                              # [N][K]
    blocks = Wt.reshape(N, K // 16, 16)
    bs = (blocks.abs().amax(-1) * (1.0 / 6.0) / g).clamp(2.0 ** -6, 448.0).to(torch.float8_e4m3fn)
    rcp = (1.0 / g) / bs.float()
    scaled = (blocks * rcp.unsqueeze(-1)).clamp(-6, 6)
    codes = torch.sign(scaled) * M.e2m1_rne(scaled.abs())
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    lo = q & 15; hi = q >> 4
    cc = np.stack([lo, hi], -1).reshape(N, K)           # code per k
    cval = np.where(cc & 8, -1.0, 1.0) * lut[cc & 7]
    same_codes = (cval.reshape(N, K // 16, 16) == codes.numpy()).mean()
    same_sf = (sf == bs.view(torch.uint8).numpy()).mean()
    print(f"{nm}: block {a.block} expert {a.expert}: global C={cg:.8e} torch={float(g):.8e} | scale bytes identical {same_sf:.6f} | codes identical {same_codes:.6f}")
