#!/usr/bin/env python3
"""Pure-Python (no numpy/torch) reference for the Tier-B2 activation-quant / rotate
kernels added to common/ds4f.h:

  * ds4f_round_scale_pow2     <- kernel.py fast_round_scale (ue8m0 power-of-2 scale)
  * ds4f_fp4_e2m1_snap        <- RNE round to the float4_e2m1 grid {0,.5,1,1.5,2,3,4,6}
  * ds4f_fp4_act_quant_inplace<- kernel.py fp4_quant (block-32 FP4 QAT quant->dequant)
  * ds4f_rotate_activation    <- model.py rotate_activation (Sylvester FWHT * n^-0.5)

The matching C harness (ds4f_q2_test.c) writes q2_c.txt with the SAME line order;
this writes q2_py.txt. Compare with:
  paste q2_py.txt q2_c.txt | awk '{d=$2-$4;a=d<0?-d:d;if(a>m){m=a;w=$1}}END{print "max-abs",m,"@",w}'

Every op is forced to float32 (struct round-trip) so this is bit-faithful to the C
(which is genuinely float32). Expect max-abs ~0 (a few ULP at most)."""
import math, struct

def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def round_scale_pow2(amax, max_inv):
    t = f32(f32(amax) * f32(max_inv))
    b = struct.unpack('<I', struct.pack('<f', t))[0]
    e = ((b >> 23) & 0xFF) - 127 + (1 if (b & 0x7FFFFF) else 0)
    sb = ((e + 127) & 0xFF) << 23
    return struct.unpack('<f', struct.pack('<I', sb))[0]


def bf16_round(f):
    u = struct.unpack('<I', struct.pack('<f', f32(f)))[0]
    if (u & 0x7FFFFFFF) >= 0x7F800000:
        return f32(f)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', r))[0]


G  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
EV = [1, 0, 1, 0, 1, 0, 1, 0]

def fp4_e2m1_snap(v):
    sign = -1.0 if v < 0.0 else 1.0
    a = f32(sign * v)
    best = G[0]; bd = f32(a if a >= 0 else -a); bi = 0
    for i in range(1, 8):
        d = f32(a - G[i]); d = f32(d if d >= 0 else -d)
        if d < bd - 1e-12 or (d <= bd + 1e-12 and EV[i] and not EV[bi]):
            bd = d; bi = i; best = G[i]
    return f32(sign * best)


def clampf(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def fp4_act_quant_inplace(x, block):
    n = len(x)
    for b0 in range(0, n, block):
        bn = min(block, n - b0)
        amax = f32(6.0 * (2.0 ** -126))
        for j in range(bn):
            a = f32(x[b0+j]); a = f32(a if a >= 0 else -a)
            if a > amax: amax = a
        s = round_scale_pow2(amax, 1.0/6.0); inv = f32(1.0/s)
        for j in range(bn):
            q = clampf(f32(x[b0+j]*inv), -6.0, 6.0)
            x[b0+j] = bf16_round(f32(fp4_e2m1_snap(q) * s))
    return x


def rotate_activation(x):
    n = len(x); h = 1
    while h < n:
        i = 0
        while i < n:
            for j in range(i, i + h):
                a = x[j]; b = x[j+h]
                x[j] = f32(a + b); x[j+h] = f32(a - b)
            i += 2 * h
        h <<= 1
    sc = f32(1.0 / math.sqrt(float(n)))
    return [f32(v * sc) for v in x]


# ---- same test vectors / order as ds4f_q2_test.c ----
vals = []
amaxes = [0.3, 1.0, 5.7, 12.3, 448.0, 449.0]
for i, a in enumerate(amaxes):
    vals.append(("rs6_%d" % i,   round_scale_pow2(f32(a), 1.0/6.0)))
    vals.append(("rs448_%d" % i, round_scale_pow2(f32(a), 1.0/448.0)))

for i in range(49):
    v = f32((i - 24) * 0.25)
    vals.append(("snap%d" % i, fp4_e2m1_snap(v)))

NFP4 = 64; BLK = 32
x = [f32((((i*37 + 11) % 97) - 48) / 7.0) for i in range(NFP4)]
fp4_act_quant_inplace(x, BLK)
for i in range(NFP4):
    vals.append(("fq%d" % i, x[i]))

NHAD = 64
h = [f32((((i*29 + 5) % 101) - 50) / 25.0) for i in range(NHAD)]
h = rotate_activation(h)
for i in range(NHAD):
    vals.append(("had%d" % i, h[i]))

with open("q2_py.txt", "w") as fp:
    for name, val in vals:
        fp.write("%-12s %.7e\n" % (name, val))
print("wrote q2_py.txt (%d values)" % len(vals))
