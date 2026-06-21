#!/usr/bin/env python3
"""Pure-Python (no numpy/torch) reference for the exact DS4F mHC math.

Mirrors common/ds4f.h ds4f_hc_sinkhorn / ds4f_hc_pre / ds4f_hc_post / ds4f_hc_head
and the deterministic synthetic fill (ds4f_fill_worker F32 branch + ds4f_hc_fill_meta).
The matching C harness (ds4f_mhc_test.c) writes mhc_c.txt; this writes mhc_py.txt.
Compare with: paste mhc_py.txt mhc_c.txt | awk '{d=$2-$4;a=d<0?-d:d;...}'

Tiny config: hc=4, d=8  ->  hd=32, mix_hc=24. Validates the algorithm (indexing,
iteration count, formulas), NOT bit-exactness (this is float64 vs C float32)."""
import math

HC   = 4
D    = 8
HD   = HC * D          # 32
MIX  = (2 + HC) * HC   # 24
ITERS = 20
EPS  = 1e-6
NORM_EPS = 1e-6

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

# ---- deterministic synthetic weights (identical integer hashes to the C fills) ----
def fill_fn(rows, cols):
    return [[float((((i*131 + j*17) % 97) - 48)) * (0.02/48.0) for j in range(cols)]
            for i in range(rows)]

def fill_base(n, seed):
    return [float((((j + seed)*13) % 17) - 8) * (0.1/8.0) for j in range(n)]

def fill_scale(n):
    return [0.5 + 0.1*s for s in range(n)]

# ---- mHC kernels (exact) ----
def hc_sinkhorn(mixes, scale, base):
    pre  = [sigmoid(mixes[j]*scale[0] + base[j]) + EPS for j in range(HC)]
    post = [2.0*sigmoid(mixes[j+HC]*scale[1] + base[j+HC]) for j in range(HC)]
    comb = [[mixes[j*HC + k + 2*HC]*scale[2] + base[j*HC + k + 2*HC]
             for k in range(HC)] for j in range(HC)]
    # comb = softmax(-1) + eps  (per row)
    for j in range(HC):
        mx = max(comb[j])
        ex = [math.exp(comb[j][k] - mx) for k in range(HC)]
        s = sum(ex)
        comb[j] = [ex[k]/s + EPS for k in range(HC)]
    # comb /= (colsum + eps)
    for k in range(HC):
        cs = sum(comb[j][k] for j in range(HC)) + EPS
        for j in range(HC): comb[j][k] /= cs
    for _ in range(ITERS-1):
        for j in range(HC):
            rs = sum(comb[j]) + EPS
            comb[j] = [comb[j][k]/rs for k in range(HC)]
        for k in range(HC):
            cs = sum(comb[j][k] for j in range(HC)) + EPS
            for j in range(HC): comb[j][k] /= cs
    return pre, post, comb

def hc_pre(x4, fn, scale, base):
    flat = [x4[k][d] for k in range(HC) for d in range(D)]
    rsq = 1.0 / math.sqrt(sum(v*v for v in flat)/HD + NORM_EPS)
    mixes = [sum(fn[m][i]*flat[i] for i in range(HD))*rsq for m in range(MIX)]
    pre, post, comb = hc_sinkhorn(mixes, scale, base)
    y = [sum(pre[k]*x4[k][d] for k in range(HC)) for d in range(D)]
    return y, post, comb

def hc_post(resid, f, post, comb):
    return [[post[k]*f[d] + sum(comb[j][k]*resid[j][d] for j in range(HC))
             for d in range(D)] for k in range(HC)]

def hc_head(x4, fn, scale, base):
    flat = [x4[k][d] for k in range(HC) for d in range(D)]
    rsq = 1.0 / math.sqrt(sum(v*v for v in flat)/HD + NORM_EPS)
    mixes = [sum(fn[k][i]*flat[i] for i in range(HD))*rsq for k in range(HC)]
    pre = [sigmoid(mixes[k]*scale[0] + base[k]) + EPS for k in range(HC)]
    y = [sum(pre[k]*x4[k][d] for k in range(HC)) for d in range(D)]
    return y

# ---- test vectors (identical formulas to the C harness) ----
x4 = [[float((((k*23 + d*5) % 31) - 15))/15.0 for d in range(D)] for k in range(HC)]
f  = [float((((d*7) % 13) - 6))/6.0 for d in range(D)]

afn, abase, ascale = fill_fn(MIX, HD), fill_base(MIX, 0), fill_scale(3)        # attn (layer-0 seed)
hfn, hbase, hscale = fill_fn(HC, HD), fill_base(HC, 4096), fill_scale(1)       # head (seed 4096)

y_pre, post, comb = hc_pre(x4, afn, ascale, abase)
y_post = hc_post(x4, f, post, comb)     # residual = x4, block output = f
y_head = hc_head(x4, hfn, hscale, hbase)

vals = []
vals += [("y_pre%d"  % d, y_pre[d])  for d in range(D)]
vals += [("post%d"   % k, post[k])   for k in range(HC)]
vals += [("comb%d_%d" % (j,k), comb[j][k]) for j in range(HC) for k in range(HC)]
vals += [("ypost%d_%d" % (k,d), y_post[k][d]) for k in range(HC) for d in range(D)]
vals += [("yhead%d"  % d, y_head[d]) for d in range(D)]

with open("mhc_py.txt", "w") as fp:
    for name, v in vals:
        fp.write("%-10s %.7e\n" % (name, v))
print("wrote mhc_py.txt (%d values)" % len(vals))
