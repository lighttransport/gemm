#!/usr/bin/env python3
"""Pure-Python (math/struct only, no numpy/torch) f32-faithful reference for the DS4F
Tier-B2 prefill math added to common/ds4f.h. Mirrors model.py / kernel.py op-for-op,
forcing every arithmetic op to float32 (struct round-trip) so the agreement with the
genuinely-float32 C kernels is a few ULP at worst (transcendentals: expf vs exp).

Sections (same line order as a64fx/llm/ds4f_tierb2_test.c):
  Part 1  wi_*   get_window_topk_idxs   (prefill causal window indices)     [int]
          ci_*   get_compress_topk_idxs (prefill causal compress indices)   [int]
          sa_*   sparse_attn            (gather + online-softmax + sink)     [float]
          hash_* hash routing           (tid2eid table lookup)              [int]

Compare:  paste tierb2_py.txt tierb2_c.txt |
          awk '{d=$2-$4;a=d<0?-d:d;if(a>m){m=a;w=$1}}END{print "max-abs",m,"@",w}'
Run from this tools/ directory (writes tierb2_py.txt here)."""
import math, struct, sys


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def gv(seed, i):
    """Deterministic f32 value in ~[-2,2), identical to gv() in the C test."""
    t = (i * 1315423911 + seed * 2654435761) % 1000003
    return f32((t - 500001) / 250000.0)


vals = []  # list of (name, value); ints written as floats so one compare covers both


# ---------- 1. get_window_topk_idxs (prefill, start_pos==0) ----------
def window_idx_prefill(window, seqlen, s):
    wq = min(seqlen, window)
    b = max(0, s - window + 1)
    return [(-1 if (b + c) > s else b + c) for c in range(wq)]


WIN, WSEQ = 4, 10
for s in range(WSEQ):
    row = window_idx_prefill(WIN, WSEQ, s)
    for c in range(min(WSEQ, WIN)):
        vals.append(("wi_%d_%d" % (s, c), row[c]))


# ---------- 2. get_compress_topk_idxs (prefill, start_pos==0) ----------
def compress_idx_prefill(ratio, seqlen, offset, s):
    ncol = seqlen // ratio
    thr = (s + 1) // ratio
    return [(-1 if t >= thr else t + offset) for t in range(ncol)]


CRATIO, CSEQ, COFF = 2, 8, 100
for s in range(CSEQ):
    row = compress_idx_prefill(CRATIO, CSEQ, COFF, s)
    for t in range(CSEQ // CRATIO):
        vals.append(("ci_%d_%d" % (s, t), row[t]))


# ---------- 3. sparse_attn (gather + online-softmax + sink) ----------
def sparse_attn(q, kv, sink, topk_idxs, scale):
    m, h, d = len(q), len(q[0]), len(q[0][0])
    out = []
    for s in range(m):
        oh = []
        for hd in range(h):
            qd = q[s][hd]
            sc, vi, mx = [], [], -1e30
            for idx in topk_idxs[s]:
                if idx < 0:
                    continue
                kvr = kv[idx]
                acc = 0.0
                for dd in range(d):
                    acc = f32(acc + f32(qd[dd] * kvr[dd]))
                sv = f32(acc * scale)
                sc.append(sv); vi.append(idx)
                if sv > mx:
                    mx = sv
            denom = f32(math.exp(f32(sink[hd] - mx)))
            for k in range(len(sc)):
                e = f32(math.exp(f32(sc[k] - mx)))
                sc[k] = e; denom = f32(denom + e)
            inv = f32(1.0 / denom)
            od = [0.0] * d
            for k in range(len(vi)):
                w = f32(sc[k] * inv)
                kvr = kv[vi[k]]
                for dd in range(d):
                    od[dd] = f32(od[dd] + f32(w * kvr[dd]))
            oh.append(od)
        out.append(oh)
    return out


SA_M, SA_H, SA_D, SA_N, SA_TOPK = 6, 3, 8, 12, 5
q = [[[gv(1 + s * 100 + hd * 10, dd) for dd in range(SA_D)] for hd in range(SA_H)] for s in range(SA_M)]
kv = [[gv(2 + n * 10, dd) for dd in range(SA_D)] for n in range(SA_N)]
sink = [gv(3, hd) for hd in range(SA_H)]
# topk_idxs: distinct valid positions + one explicit -1 skip (matches the C test)
topk = [[((s + t * 2) % SA_N) for t in range(SA_TOPK - 1)] + [-1] for s in range(SA_M)]
scale = f32(1.0 / math.sqrt(SA_D))
o = sparse_attn(q, kv, sink, topk, scale)
for s in range(SA_M):
    for hd in range(SA_H):
        for dd in range(SA_D):
            vals.append(("sa_%d_%d_%d" % (s, hd, dd), o[s][hd][dd]))


# ---------- 4. hash routing (tid2eid lookup, first n_hash_layers) ----------
def tid2eid(tid, k, n_exp):
    return (tid * 7 + k * 3) % n_exp


HVOCAB, HNACT, HNEXP = 20, 4, 16
toks = [3, 17, 0, 9]
for ti, tok in enumerate(toks):
    for k in range(HNACT):
        vals.append(("hash_%d_%d" % (ti, k), tid2eid(tok, k, HNEXP)))


# ---------- f32-faithful primitives reused by the compressor (Part 2) ----------
def bf16_trunc(f):  # ds4f_bf16(ds4f_f32_bf16(f)) — truncating bf16 (matches norm weight)
    u = struct.unpack('<I', struct.pack('<f', f32(f)))[0]
    return struct.unpack('<f', struct.pack('<I', u & 0xFFFF0000))[0]


def bf16_round(f):  # RNE bf16 (ds4f_bf16_round)
    u = struct.unpack('<I', struct.pack('<f', f32(f)))[0]
    if (u & 0x7FFFFFFF) >= 0x7F800000:
        return f32(f)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', r))[0]


def round_scale_pow2(amax, max_inv):
    t = f32(f32(amax) * f32(max_inv))
    b = struct.unpack('<I', struct.pack('<f', t))[0]
    e = ((b >> 23) & 0xFF) - 127 + (1 if (b & 0x7FFFFF) else 0)
    return struct.unpack('<f', struct.pack('<I', ((e + 127) & 0xFF) << 23))[0]


_G = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_EV = [1, 0, 1, 0, 1, 0, 1, 0]


def fp4_e2m1_snap(v):
    sign = -1.0 if v < 0.0 else 1.0
    a = f32(sign * v)
    best = _G[0]; bd = f32(a if a >= 0 else -a); bi = 0
    for i in range(1, 8):
        dd = f32(a - _G[i]); dd = f32(dd if dd >= 0 else -dd)
        if dd < bd - 1e-12 or (dd <= bd + 1e-12 and _EV[i] and not _EV[bi]):
            bd = dd; bi = i; best = _G[i]
    return f32(sign * best)


def clampf(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def fp4_act_quant_inplace(x, block):
    n = len(x); y = list(x)
    for b0 in range(0, n, block):
        bn = min(block, n - b0)
        amax = f32(6.0 * (2.0 ** -126))
        for j in range(bn):
            a = f32(y[b0 + j]); a = f32(a if a >= 0 else -a)
            if a > amax:
                amax = a
        s = round_scale_pow2(amax, 1.0 / 6.0); inv = f32(1.0 / s)
        for j in range(bn):
            q = clampf(f32(y[b0 + j] * inv), -6.0, 6.0)
            y[b0 + j] = bf16_round(f32(fp4_e2m1_snap(q) * s))
    return y


def rotate_activation(x):
    n = len(x); y = list(x); h = 1
    while h < n:
        i = 0
        while i < n:
            for j in range(i, i + h):
                a = y[j]; b = y[j + h]
                y[j] = f32(a + b); y[j + h] = f32(a - b)
            i += 2 * h
        h <<= 1
    sc = f32(1.0 / math.sqrt(float(n)))
    return [f32(v * sc) for v in y]


def rope_table(seqlen, rd, base=10000.0):  # matches the C test's table build
    half = rd // 2
    cosb = [0.0] * (seqlen * half); sinb = [0.0] * (seqlen * half)
    for pos in range(seqlen):
        for k in range(half):
            freq = 1.0 / (base ** ((2.0 * k) / rd))
            ang = pos * freq
            cosb[pos * half + k] = f32(math.cos(ang))
            sinb[pos * half + k] = f32(math.sin(ang))
    return cosb, sinb


def rope_apply(v, cosb, sinb, pos, half):  # in-place, non-inverse (ds4f_rope_apply)
    for k in range(half):
        a = v[2 * k]; b = v[2 * k + 1]
        c = cosb[pos * half + k]; s = sinb[pos * half + k]
        v[2 * k] = f32(f32(a * c) - f32(b * s))
        v[2 * k + 1] = f32(f32(a * s) + f32(b * c))


def rmsnorm(x, w_bf16, eps):  # double accumulation, f32 inv (ds4f_rmsnorm)
    n = len(x); ss = 0.0
    for i in range(n):
        ss += float(x[i]) * float(x[i])
    inv = f32(1.0 / f32(math.sqrt(f32(f32(ss / n) + eps))))
    return [f32(f32(x[i] * inv) * w_bf16[i]) for i in range(n)]


# ---------- 5. Compressor (prefill gated pooling) ----------
def compress_prefill(x, seqlen, dim, d, rd, ratio, wkv, wgate, ape, norm_w,
                     cosb, sinb, eps, rotate):
    overlap = (ratio == 4); coff = 2 if overlap else 1; W = coff * d
    cutoff = seqlen - seqlen % ratio; nwin = cutoff // ratio
    kvl = [[0.0] * W for _ in range(cutoff)]; scl = [[0.0] * W for _ in range(cutoff)]
    for pos in range(cutoff):
        xp = x[pos]
        for o in range(W):
            a = 0.0; b = 0.0
            for i in range(dim):
                a = f32(a + f32(wkv[o][i] * xp[i])); b = f32(b + f32(wgate[o][i] * xp[i]))
            kvl[pos][o] = a; scl[pos][o] = b
    P = 2 * ratio if overlap else ratio
    out = []
    for w in range(nwin):
        ow = [0.0] * d
        for e in range(d):
            ksub = [0.0] * P; ssub = [0.0] * P
            if overlap:
                for p in range(ratio):
                    if w == 0:
                        ksub[p] = 0.0; ssub[p] = -1e30
                    else:
                        pos = (w - 1) * ratio + p
                        ksub[p] = kvl[pos][e]; ssub[p] = f32(scl[pos][e] + ape[p][e])
                for r in range(ratio):
                    pos = w * ratio + r; p = ratio + r
                    ksub[p] = kvl[pos][d + e]; ssub[p] = f32(scl[pos][d + e] + ape[r][d + e])
            else:
                for p in range(ratio):
                    pos = w * ratio + p
                    ksub[p] = kvl[pos][e]; ssub[p] = f32(scl[pos][e] + ape[p][e])
            mx = -1e30
            for p in range(P):
                if ssub[p] > mx:
                    mx = ssub[p]
            den = 0.0
            for p in range(P):
                ex = f32(math.exp(f32(ssub[p] - mx))); ssub[p] = ex; den = f32(den + ex)
            acc = 0.0
            for p in range(P):
                acc = f32(acc + f32(ksub[p] * f32(ssub[p] / den)))
            ow[e] = acc
        ow = rmsnorm(ow, norm_w, eps)
        slc = ow[d - rd:]
        rope_apply(slc, cosb, sinb, w * ratio, rd // 2)
        ow = ow[:d - rd] + slc
        if rotate:
            ow = rotate_activation(ow)
            ow = fp4_act_quant_inplace(ow, 32)
        out.append(ow)
    return out


def run_compress(tag, base, seqlen, dim, d, rd, ratio, rotate, eps=1e-6):
    overlap = (ratio == 4); coff = 2 if overlap else 1; W = coff * d
    wkv = [[gv(base + o, i) for i in range(dim)] for o in range(W)]
    wgate = [[gv(base + 1000 + o, i) for i in range(dim)] for o in range(W)]
    ape = [[gv(base + 2000 + r, c) for c in range(W)] for r in range(ratio)]
    norm_w = [bf16_trunc(gv(base + 3000, e)) for e in range(d)]
    x = [[gv(base + 4000 + pos, i) for i in range(dim)] for pos in range(seqlen)]
    cosb, sinb = rope_table(seqlen, rd)
    out = compress_prefill(x, seqlen, dim, d, rd, ratio, wkv, wgate, ape, norm_w,
                           cosb, sinb, eps, rotate)
    nwin = (seqlen - seqlen % ratio) // ratio
    for w in range(nwin):
        for e in range(d):
            vals.append(("%s_%d_%d" % (tag, w, e), out[w][e]))


run_compress("c1", 5000, 10, 6, 8, 4, 4, 1)   # overlap + rotate (indexer compressor)
run_compress("c2", 5000, 10, 6, 8, 4, 4, 0)   # overlap, no rotate (layer compressor)
run_compress("c3", 6000, 8, 6, 8, 4, 2, 0)    # non-overlap (ratio!=4), no rotate


# ---------- 6. Indexer (prefill scoring + masked top-k) ----------
def linear(W, x):  # W[out][in], x[in] -> y[out], f32-faithful
    out = []
    for o in range(len(W)):
        a = 0.0
        for i in range(len(x)):
            a = f32(a + f32(W[o][i] * x[i]))
        out.append(a)
    return out


def index_score_row(qheads, kvc, weights, H, hd, T):
    score = [0.0] * T
    for t in range(T):
        kt = kvc[t]; acc = 0.0
        for h in range(H):
            qh = qheads[h]; dot = 0.0
            for d in range(hd):
                dot = f32(dot + f32(qh[d] * kt[d]))
            if dot < 0.0:
                dot = 0.0
            acc = f32(acc + f32(dot * weights[h]))
        score[t] = acc
    return score


def index_topk(score, T, thr, k, offset):
    if thr > T:
        thr = T
    npick = k if k < thr else thr
    used = [False] * thr; chosen = []
    for _ in range(npick):
        best = -1; bv = -1e30
        for t in range(thr):
            if not used[t] and score[t] > bv:
                bv = score[t]; best = t
        used[best] = True; chosen.append(best)
    chosen.sort()
    return [(chosen[n] + offset) if n < npick else -1 for n in range(k)]


def run_indexer(tag, base, seqlen, dim, qlora, H, hd, rd, ratio, k, offset, eps=1e-6):
    wqb = [[gv(base + o, i) for i in range(qlora)] for o in range(H * hd)]
    wproj = [[gv(base + 100 + h, i) for i in range(dim)] for h in range(H)]
    cwkv = [[gv(base + 200 + o, i) for i in range(dim)] for o in range(2 * hd)]   # overlap coff=2
    cwgate = [[gv(base + 300 + o, i) for i in range(dim)] for o in range(2 * hd)]
    cape = [[gv(base + 400 + r, c) for c in range(2 * hd)] for r in range(ratio)]
    cnorm = [bf16_trunc(gv(base + 500, e)) for e in range(hd)]
    x = [[gv(base + 600 + pos, i) for i in range(dim)] for pos in range(seqlen)]
    qr = [[gv(base + 700 + pos, i) for i in range(qlora)] for pos in range(seqlen)]
    cosb, sinb = rope_table(seqlen, rd)
    kvc = compress_prefill(x, seqlen, dim, hd, rd, ratio, cwkv, cwgate, cape, cnorm,
                           cosb, sinb, eps, 1)
    T = (seqlen - seqlen % ratio) // ratio
    sm_scale = f32(1.0 / math.sqrt(hd))
    wscale = f32(sm_scale * f32(1.0 / math.sqrt(H)))
    for s in range(seqlen):
        qflat = linear(wqb, qr[s])
        qheads = [qflat[h * hd:(h + 1) * hd] for h in range(H)]
        for h in range(H):
            qh = qheads[h]
            slc = qh[hd - rd:]
            rope_apply(slc, cosb, sinb, s, rd // 2)
            qh = qh[:hd - rd] + slc
            qh = rotate_activation(qh)
            qh = fp4_act_quant_inplace(qh, 32)
            qheads[h] = qh
        wlin = linear(wproj, x[s])
        weights = [f32(wlin[h] * wscale) for h in range(H)]
        score = index_score_row(qheads, kvc, weights, H, hd, T)
        thr = (s + 1) // ratio
        for t in range(min(thr, T)):
            vals.append(("%s_is_%d_%d" % (tag, s, t), score[t]))
        sel = index_topk(score, T, thr, k, offset)
        for n in range(k):
            vals.append(("%s_it_%d_%d" % (tag, s, n), sel[n]))


run_indexer("ix", 8000, 12, 6, 6, 4, 8, 4, 4, 2, 100)


# ---------- 7. window/compress index helpers (decode, start_pos>0) ----------
def window_idx_decode(window, start_pos):
    if start_pos >= window - 1:
        sp = start_pos % window
        return list(range(sp + 1, window)) + list(range(0, sp + 1))
    row = list(range(start_pos + 1))
    return row + [-1] * (window - start_pos - 1)


def compress_idx_decode(ratio, start_pos, offset):
    return [t + offset for t in range((start_pos + 1) // ratio)]


WD_WIN = 4
for sp in range(1, 10):                       # exercises the ring wraparound (sp>=win-1)
    row = window_idx_decode(WD_WIN, sp)
    for c in range(WD_WIN):
        vals.append(("wd_%d_%d" % (sp, c), row[c]))
for sp in range(1, 10):
    row = compress_idx_decode(3, sp, 100)
    for t in range((sp + 1) // 3):
        vals.append(("cd_%d_%d" % (sp, t), row[t]))


# ---------- 8. stateful incremental Compressor (decode) ----------
class CompState:
    """Mirrors ds4f_compress_step: incremental kv_state/score_state ring + per-step compress."""
    def __init__(self, ratio, d):
        self.ratio, self.d = ratio, d
        self.overlap = (ratio == 4); self.coff = 2 if self.overlap else 1
        self.W = self.coff * d; self.rows = self.coff * ratio
        self.kv_state = [[0.0] * self.W for _ in range(self.rows)]
        self.score_state = [[-1e30] * self.W for _ in range(self.rows)]

    def step(self, x, dim, rd, start_pos, wkv, wgate, ape, norm_w, cosb, sinb, eps, rotate):
        W, d, ratio, overlap = self.W, self.d, self.ratio, self.overlap
        kv = [0.0] * W; score = [0.0] * W
        for o in range(W):
            a = 0.0; b = 0.0
            for i in range(dim):
                a = f32(a + f32(wkv[o][i] * x[i])); b = f32(b + f32(wgate[o][i] * x[i]))
            kv[o] = a; score[o] = b
        if start_pos == 0:
            offset = ratio if overlap else 0
            for o in range(W):
                self.kv_state[offset][o] = kv[o]
                self.score_state[offset][o] = f32(score[o] + ape[0][o])
            return None
        should = ((start_pos + 1) % ratio) == 0
        apr = start_pos % ratio
        for o in range(W):
            score[o] = f32(score[o] + ape[apr][o])
        out = [0.0] * d
        if overlap:
            slot = ratio + apr
            for o in range(W):
                self.kv_state[slot][o] = kv[o]; self.score_state[slot][o] = score[o]
            if should:
                P = 2 * ratio
                for e in range(d):
                    ksub = [0.0] * P; ssub = [0.0] * P
                    for p in range(ratio):
                        ksub[p] = self.kv_state[p][e]; ssub[p] = self.score_state[p][e]
                    for p in range(ratio):
                        ksub[ratio + p] = self.kv_state[ratio + p][d + e]
                        ssub[ratio + p] = self.score_state[ratio + p][d + e]
                    out[e] = _wsum(ksub, ssub, P)
                for p in range(ratio):                       # shift current -> overlap
                    for o in range(W):
                        self.kv_state[p][o] = self.kv_state[ratio + p][o]
                        self.score_state[p][o] = self.score_state[ratio + p][o]
        else:
            slot = apr
            for o in range(W):
                self.kv_state[slot][o] = kv[o]; self.score_state[slot][o] = score[o]
            if should:
                P = ratio
                for e in range(d):
                    ksub = [self.kv_state[p][e] for p in range(ratio)]
                    ssub = [self.score_state[p][e] for p in range(ratio)]
                    out[e] = _wsum(ksub, ssub, P)
        if not should:
            return None
        out = rmsnorm(out, norm_w, eps)
        slc = out[d - rd:]
        rope_apply(slc, cosb, sinb, start_pos + 1 - ratio, rd // 2)
        out = out[:d - rd] + slc
        if rotate:
            out = rotate_activation(out)
            out = fp4_act_quant_inplace(out, 32)
        return out


def _wsum(ksub, ssub, P):
    mx = -1e30
    for p in range(P):
        if ssub[p] > mx:
            mx = ssub[p]
    den = 0.0
    for p in range(P):
        ex = f32(math.exp(f32(ssub[p] - mx))); ssub[p] = ex; den = f32(den + ex)
    acc = 0.0
    for p in range(P):
        acc = f32(acc + f32(ksub[p] * f32(ssub[p] / den)))
    return acc


def run_compress_decode(tag, base, npos, dim, d, rd, ratio, rotate, eps=1e-6):
    overlap = (ratio == 4); coff = 2 if overlap else 1; W = coff * d
    wkv = [[gv(base + o, i) for i in range(dim)] for o in range(W)]
    wgate = [[gv(base + 1000 + o, i) for i in range(dim)] for o in range(W)]
    ape = [[gv(base + 2000 + r, c) for c in range(W)] for r in range(ratio)]
    norm_w = [bf16_trunc(gv(base + 3000, e)) for e in range(d)]
    x = [[gv(base + 4000 + pos, i) for i in range(dim)] for pos in range(npos)]
    cosb, sinb = rope_table(npos, rd)
    st = CompState(ratio, d)
    for pos in range(npos):
        out = st.step(x[pos], dim, rd, pos, wkv, wgate, ape, norm_w, cosb, sinb, eps, rotate)
        if out is not None:
            for e in range(d):
                vals.append(("%s_%d_%d" % (tag, pos, e), out[e]))


# same bases as the prefill compressors -> incremental output must equal batched windows
run_compress_decode("c1d", 5000, 10, 6, 8, 4, 4, 1)   # overlap + rotate (indexer compressor)
run_compress_decode("c2d", 5000, 10, 6, 8, 4, 4, 0)   # overlap, no rotate (layer compressor)
run_compress_decode("c3d", 6000, 8, 6, 8, 4, 2, 0)    # non-overlap (ratio!=4)


# ---------- 9. stateful incremental Indexer (decode) ----------
def run_indexer_decode(tag, base, npos, dim, qlora, H, hd, rd, ratio, k, offset, eps=1e-6):
    wqb = [[gv(base + o, i) for i in range(qlora)] for o in range(H * hd)]
    wproj = [[gv(base + 100 + h, i) for i in range(dim)] for h in range(H)]
    cwkv = [[gv(base + 200 + o, i) for i in range(dim)] for o in range(2 * hd)]
    cwgate = [[gv(base + 300 + o, i) for i in range(dim)] for o in range(2 * hd)]
    cape = [[gv(base + 400 + r, c) for c in range(2 * hd)] for r in range(ratio)]
    cnorm = [bf16_trunc(gv(base + 500, e)) for e in range(hd)]
    x = [[gv(base + 600 + pos, i) for i in range(dim)] for pos in range(npos)]
    qr = [[gv(base + 700 + pos, i) for i in range(qlora)] for pos in range(npos)]
    cosb, sinb = rope_table(npos, rd)
    sm = f32(1.0 / math.sqrt(hd)); wscale = f32(sm * f32(1.0 / math.sqrt(H)))
    st = CompState(ratio, hd)
    idx_kv = [[0.0] * hd for _ in range(npos // ratio + 1)]
    for pos in range(npos):
        out = st.step(x[pos], dim, rd, pos, cwkv, cwgate, cape, cnorm, cosb, sinb, eps, 1)
        if out is not None:
            idx_kv[pos // ratio] = out
        if pos == 0:
            continue                                         # seed only (start_pos==0)
        qflat = linear(wqb, qr[pos])
        qheads = [qflat[h * hd:(h + 1) * hd] for h in range(H)]
        for h in range(H):
            qh = qheads[h]; slc = qh[hd - rd:]
            rope_apply(slc, cosb, sinb, pos, rd // 2)
            qh = qh[:hd - rd] + slc
            qh = rotate_activation(qh); qh = fp4_act_quant_inplace(qh, 32)
            qheads[h] = qh
        wlin = linear(wproj, x[pos])
        weights = [f32(wlin[h] * wscale) for h in range(H)]
        T = (pos + 1) // ratio
        if T == 0:
            continue
        score = index_score_row(qheads, idx_kv, weights, H, hd, T)
        for t in range(T):
            vals.append(("%s_is_%d_%d" % (tag, pos, t), score[t]))
        sel = index_topk(score, T, T, k, offset)             # decode: no causal mask (thr=T)
        for n in range(k):
            vals.append(("%s_it_%d_%d" % (tag, pos, n), sel[n]))


run_indexer_decode("ixd", 8000, 12, 6, 6, 4, 8, 4, 4, 2, 100)


# independent check: incremental decode tokens must bit-match the batched prefill windows
# (same gv bases) -> proves compress_step reproduces compress_prefill, not just the C mirror.
_vd = dict(vals)
def _chk(dtag, dpos, ptag, pwin, d):
    return sum(1 for e in range(d) if _vd["%s_%d_%d" % (dtag, dpos, e)] != _vd["%s_%d_%d" % (ptag, pwin, e)])
_bad = (_chk("c1d", 3, "c1", 0, 8) + _chk("c1d", 7, "c1", 1, 8) +
        _chk("c2d", 3, "c2", 0, 8) + _chk("c2d", 7, "c2", 1, 8) +
        _chk("c3d", 1, "c3", 0, 8) + _chk("c3d", 3, "c3", 1, 8) +
        _chk("c3d", 5, "c3", 2, 8) + _chk("c3d", 7, "c3", 3, 8))
sys.stderr.write("incremental==batched compress mismatches: %d (expect 0)\n" % _bad)


with open("tierb2_py.txt", "w") as fp:
    for name, val in vals:
        fp.write("%-16s %.7e\n" % (name, float(val)))
print("wrote tierb2_py.txt (%d values)" % len(vals))
