#!/usr/bin/env python3
"""Pure-Python (no numpy/torch) reference for the exact DS4F forward math added
behind DS4F_EXACT in common/ds4f.h:

  * ds4f_rope_table   <- model.py precompute_freqs_cis (YaRN gated on orig>0)
  * ds4f_rope_apply   <- model.py apply_rotary_emb     (fwd + inverse/conjugate)
  * ds4f_topk_exact   <- model.py Gate.forward (sqrtsoftplus, bias selects only,
                          unbiased weights, /sum, *route_scale)

The matching C harness (ds4f_exact_test.c) writes exact_c.txt with the SAME line
order; this writes exact_py.txt. Compare numerically with:
  paste exact_py.txt exact_c.txt | awk '{d=$2-$4;a=d<0?-d:d;if(a>m){m=a;w=$1}}END{print "max-abs",m,"@",w}'

float64 here vs float32 in C => expect ~1e-5, not bit-exact. Validates the
algorithm (indexing, YaRN ramp, conjugate, sqrtsoftplus, bias/weight split)."""
import math

# ---- config mirrors ds4f_default_config() rope/gate fields ----
DIM      = 64            # qk_rope_dim
HALF     = DIM // 2      # 32
MAXP     = 32            # positions 0..31 (harness operating point)
FACTOR   = 16
BETA_F   = 32
BETA_S   = 1
THETA_D  = 10000.0       # dense layers (YaRN off)
THETA_C  = 160000.0      # compressed/sparse layers (YaRN on)
ORIG     = 65536         # original_seq_len (comp); 0 for dense
N_EXP    = 8
TOPK     = 3
ROUTE    = 1.5


def rope_table(dim, max_pos, base, factor, beta_fast, beta_slow, original_seq_len):
    half = dim // 2
    freq = [1.0 / base ** ((2.0 * k) / dim) for k in range(half)]
    if original_seq_len > 0:
        lo_d = dim * math.log(original_seq_len / (beta_fast * 2.0 * math.pi)) / (2.0 * math.log(base))
        hi_d = dim * math.log(original_seq_len / (beta_slow * 2.0 * math.pi)) / (2.0 * math.log(base))
        low = math.floor(lo_d);  low = max(low, 0)
        high = math.ceil(hi_d);  high = min(high, dim - 1)
        if low == high:
            high += 0.001
        for k in range(half):
            lin = (float(k) - low) / (high - low)
            ramp = 0.0 if lin < 0.0 else (1.0 if lin > 1.0 else lin)
            smooth = 1.0 - ramp
            freq[k] = freq[k] / factor * (1.0 - smooth) + freq[k] * smooth
    cos = [0.0] * (max_pos * half)
    sin = [0.0] * (max_pos * half)
    for p in range(max_pos):
        for k in range(half):
            ang = p * freq[k]
            cos[p * half + k] = math.cos(ang)
            sin[p * half + k] = math.sin(ang)
    return cos, sin


def rope_apply(v, cos, sin, pos, half, inverse):
    out = list(v)
    for k in range(half):
        a = v[2 * k]; b = v[2 * k + 1]
        c = cos[pos * half + k]; s = sin[pos * half + k]
        if not inverse:
            out[2 * k] = a * c - b * s; out[2 * k + 1] = a * s + b * c
        else:
            out[2 * k] = a * c + b * s; out[2 * k + 1] = -a * s + b * c
    return out


def topk_exact(logits, bias, k, route_scale):
    n = len(logits)
    sc = []
    for z in logits:
        sp = (z + math.log1p(math.exp(-z))) if z > 0.0 else math.log1p(math.exp(z))
        sc.append(math.sqrt(sp if sp > 0.0 else 0.0))
    key = [sc[e] + (bias[e] if bias else 0.0) for e in range(n)]
    order = sorted(range(n), key=lambda e: key[e], reverse=True)[:k]
    ssum = sum(sc[e] for e in order)
    if ssum <= 0.0:
        ssum = 1.0
    sel = sorted(order)                       # sort by eid for order-independent compare
    return [(e, sc[e] / ssum * route_scale) for e in sel]


# ---- deterministic test vectors (identical integer hashes to ds4f_exact_test.c) ----
v = [float((((i * 37 + 11) % 97) - 48)) / 48.0 for i in range(DIM)]
logits = [float((((e * 53 + 7) % 29) - 14)) / 7.0 for e in range(N_EXP)]
bias   = [float((((e * 19 + 3) % 11) - 5)) / 10.0 for e in range(N_EXP)]

cd, sd = rope_table(DIM, MAXP, THETA_D, FACTOR, BETA_F, BETA_S, 0)        # dense
cc, sc_ = rope_table(DIM, MAXP, THETA_C, FACTOR, BETA_F, BETA_S, ORIG)    # comp

POS = [1, 7, 31]
KS  = [0, 1, 15, 31]
vals = []
for p in POS:
    for k in KS:
        vals.append(("dcos%d_%d" % (p, k), cd[p * HALF + k]))
        vals.append(("dsin%d_%d" % (p, k), sd[p * HALF + k]))
for p in POS:
    for k in KS:
        vals.append(("ccos%d_%d" % (p, k), cc[p * HALF + k]))
        vals.append(("csin%d_%d" % (p, k), sc_[p * HALF + k]))

fwd = rope_apply(v, cc, sc_, 7, HALF, False)
inv = rope_apply(v, cc, sc_, 7, HALF, True)
for i in range(DIM):
    vals.append(("fwd%d" % i, fwd[i]))
for i in range(DIM):
    vals.append(("inv%d" % i, inv[i]))

# gate WITH bias, and WITHOUT bias (must equal plain topk path)
gb = topk_exact(logits, bias, TOPK, ROUTE)
g0 = topk_exact(logits, None, TOPK, ROUTE)
for i, (e, w) in enumerate(gb):
    vals.append(("gb%d_eid" % i, float(e)))
    vals.append(("gb%d_wt" % i, w))
for i, (e, w) in enumerate(g0):
    vals.append(("g0%d_eid" % i, float(e)))
    vals.append(("g0%d_wt" % i, w))

with open("exact_py.txt", "w") as fp:
    for name, val in vals:
        fp.write("%-12s %.7e\n" % (name, val))
print("wrote exact_py.txt (%d values)" % len(vals))
