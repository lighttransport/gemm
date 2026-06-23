#!/usr/bin/env python3
import argparse
import csv
import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rel_l2(a, b):
    se = sum((x - y) * (x - y) for x, y in zip(a, b))
    r2 = sum(x * x for x in a)
    return math.sqrt(se / (r2 + 1e-30))


def cosine(a, b):
    ab = dot(a, b)
    aa = dot(a, a)
    bb = dot(b, b)
    return ab / (math.sqrt(aa * bb) + 1e-30)


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def kl_div(p, q):
    s = 0.0
    for a, b in zip(p, q):
        if a > 0.0:
            s += a * math.log(a / max(b, 1e-30))
    return s


def fwht_inplace(x):
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    scale = 1.0 / math.sqrt(n)
    for i in range(n):
        x[i] *= scale


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def qsym_vec(x, bits):
    denom = (1 << (bits - 1)) - 1
    mx = max((abs(v) for v in x), default=0.0)
    s = mx / denom if mx else 1.0
    out = []
    for v in x:
        q = int(round(v / s))
        q = max(-denom, min(denom, q))
        out.append(q * s)
    return out


def fp4_e2m1_value(x):
    codebook = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    sign = -1.0 if x < 0.0 else 1.0
    ax = abs(x)
    return sign * min(codebook, key=lambda y: abs(ax - y))


def fp4_vec(x):
    mx = max((abs(v) for v in x), default=0.0)
    s = mx / 6.0 if mx else 1.0
    return [fp4_e2m1_value(v / s) * s for v in x]


def turbo_mse_vec(x, bits, signs):
    y = [v * s for v, s in zip(x, signs)]
    fwht_inplace(y)
    y = qsym_vec(y, bits)
    fwht_inplace(y)
    return [v * s for v, s in zip(y, signs)]


def make_cache(seq, dim, seed):
    rng = random.Random(seed)
    keys = []
    vals = []
    for t in range(seq):
        k = []
        v = []
        for i in range(dim):
            # Mild position structure plus sparse outliers to mimic cache stress.
            base = rng.gauss(0.0, 1.0) + 0.05 * math.sin((t + 1) * (i + 3) * 0.017)
            if rng.randrange(256) == 0:
                base *= 5.0
            k.append(base)
            val = rng.gauss(0.0, 0.7)
            if rng.randrange(512) == 0:
                val *= 4.0
            v.append(val)
        keys.append(k)
        vals.append(v)
    queries = []
    for _ in range(16):
        q = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        queries.append(q)
    return queries, keys, vals


def quant_cache(keys, vals, method, signs):
    if method == "int4":
        return [qsym_vec(k, 4) for k in keys], [qsym_vec(v, 4) for v in vals]
    if method == "fp4":
        return [fp4_vec(k) for k in keys], [fp4_vec(v) for v in vals]
    if method == "turbo4":
        return [turbo_mse_vec(k, 4, signs) for k in keys], [turbo_mse_vec(v, 4, signs) for v in vals]
    if method == "turbo3":
        return [turbo_mse_vec(k, 3, signs) for k in keys], [turbo_mse_vec(v, 3, signs) for v in vals]
    raise ValueError(method)


def attention_outputs(queries, keys, vals):
    scale = 1.0 / math.sqrt(len(keys[0]))
    score_rows = []
    prob_rows = []
    outs = []
    for q in queries:
        scores = [dot(q, k) * scale for k in keys]
        probs = softmax(scores)
        out = [0.0] * len(vals[0])
        for p, v in zip(probs, vals):
            for i, x in enumerate(v):
                out[i] += p * x
        score_rows.append(scores)
        prob_rows.append(probs)
        outs.append(out)
    return score_rows, prob_rows, outs


def evaluate(seq, dim, seed, methods):
    if not is_pow2(dim):
        raise SystemExit("--dim must be a power of two for the FWHT TurboQuant proxy")
    queries, keys, vals = make_cache(seq, dim, seed)
    ref_scores, ref_probs, ref_outs = attention_outputs(queries, keys, vals)
    rng = random.Random(seed ^ 0x5eed)
    signs = [1.0 if rng.randrange(2) else -1.0 for _ in range(dim)]
    rows = []
    for method in methods:
        qk, qv = quant_cache(keys, vals, method, signs)
        scores, probs, outs = attention_outputs(queries, qk, qv)
        score_rel = sum(rel_l2(a, b) for a, b in zip(ref_scores, scores)) / len(scores)
        prob_kl = sum(kl_div(a, b) for a, b in zip(ref_probs, probs)) / len(probs)
        out_rel = sum(rel_l2(a, b) for a, b in zip(ref_outs, outs)) / len(outs)
        out_cos = sum(cosine(a, b) for a, b in zip(ref_outs, outs)) / len(outs)
        k_rel = sum(rel_l2(a, b) for a, b in zip(keys, qk)) / len(keys)
        v_rel = sum(rel_l2(a, b) for a, b in zip(vals, qv)) / len(vals)
        rows.append({
            "method": method,
            "seq": seq,
            "dim": dim,
            "bits_per_value": 4 if method in ("int4", "fp4", "turbo4") else 3,
            "key_rel_l2": k_rel,
            "value_rel_l2": v_rel,
            "score_rel_l2": score_rel,
            "softmax_kl": prob_kl,
            "attn_out_rel_l2": out_rel,
            "attn_out_cosine": out_cos,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--methods", default="int4,fp4,turbo4,turbo3")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = evaluate(args.seq, args.dim, args.seed, methods)
    fields = list(rows[0].keys())
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[f]) for f in fields))


if __name__ == "__main__":
    main()
