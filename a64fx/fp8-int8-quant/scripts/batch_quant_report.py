#!/usr/bin/env python3
import argparse
import csv
import json
import math
import mmap
import os
import re
import struct
import sys


def fp8_e4m3fn_to_f32(x):
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    mant = x & 7
    if exp == 0:
        if mant == 0:
            bits = sign << 31
        else:
            sh = 0
            while (mant & 4) == 0:
                mant <<= 1
                sh += 1
            mant &= 3
            bits = (sign << 31) | ((127 - 7 - sh) << 23) | (mant << 20)
    elif exp == 15 and mant == 7:
        bits = (sign << 31) | (0xFF << 23) | (1 << 22)
    else:
        bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def e8m0_to_f32(x):
    return struct.unpack("<f", struct.pack("<I", x << 23))[0]


def bf16_to_f32(h):
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def f32_to_f16(x):
    return struct.unpack("<e", struct.pack("<e", float(x)))[0]


class SafeFile:
    def __init__(self, path):
        self.path = path
        self.fd = open(path, "rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack_from("<Q", self.mm, 0)[0]
        header = json.loads(self.mm[8:8 + header_len])
        self.data_base = 8 + header_len
        self.tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            self.tensors[name] = meta

    def close(self):
        self.mm.close()
        self.fd.close()

    def elem_f32(self, meta, r, c):
        dtype = meta["dtype"]
        shape = meta["shape"]
        cols = shape[1] if len(shape) > 1 else 1
        off0 = self.data_base + meta["data_offsets"][0]
        idx = r * cols + c
        if dtype == "BF16":
            return bf16_to_f32(struct.unpack_from("<H", self.mm, off0 + idx * 2)[0])
        if dtype in ("F16", "FP16"):
            return struct.unpack_from("<e", self.mm, off0 + idx * 2)[0]
        if dtype == "F32":
            return struct.unpack_from("<f", self.mm, off0 + idx * 4)[0]
        if dtype in ("F8_E4M3", "F8_E4M3FN"):
            return fp8_e4m3fn_to_f32(self.mm[off0 + idx])
        raise ValueError("unsupported dtype " + dtype)

    def scale_f32(self, meta, r, c, rows_block, cols_block):
        dtype = meta["dtype"]
        shape = meta["shape"]
        off0 = self.data_base + meta["data_offsets"][0]
        sr = r // rows_block
        sc = c // cols_block
        if len(shape) == 2:
            idx = sr * shape[1] + sc
        else:
            idx = sc
        if dtype == "F32":
            return struct.unpack_from("<f", self.mm, off0 + idx * 4)[0]
        if dtype == "BF16":
            return bf16_to_f32(struct.unpack_from("<H", self.mm, off0 + idx * 2)[0])
        if dtype in ("F8_E8M0",):
            return e8m0_to_f32(self.mm[off0 + idx])
        return 1.0


def load_sample(sf, name, rows_cap, cols_cap, max_elems):
    meta = sf.tensors[name]
    rows, cols = meta["shape"][:2]
    sr = min(rows, rows_cap)
    sc = min(cols, cols_cap)
    if max_elems and sr * sc > max_elems:
        sr = max(1, min(sr, max_elems // sc))
        if sr * sc > max_elems:
            sc = max(1, max_elems // sr)
    scale = sf.tensors.get(name + "_scale_inv") or sf.tensors.get(name + ".scale_inv") or sf.tensors.get(name + ".scale")
    out = []
    for r in range(sr):
        base = []
        for c in range(sc):
            v = sf.elem_f32(meta, r, c)
            if meta["dtype"] in ("F8_E4M3", "F8_E4M3FN") and scale is not None:
                # GLM FP8 uses 128x128 F32 scale_inv. M3 FP8 commonly uses row x 32-col blocks.
                if len(scale["shape"]) == 2 and scale["shape"][0] == (rows + 127) // 128:
                    v *= sf.scale_f32(scale, r, c, 128, 128)
                elif len(scale["shape"]) == 2 and scale["shape"][0] == rows:
                    v *= sf.scale_f32(scale, r, c, 1, 32)
                else:
                    v *= sf.scale_f32(scale, r, c, 128, 128)
            base.append(v)
        out.extend(base)
    return out, sr, sc, scale is not None


def metrics(ref, got):
    n = len(ref)
    se = ae = mx = r2 = g2 = dot = 0.0
    for a, b in zip(ref, got):
        e = b - a
        se += e * e
        ae += abs(e)
        mx = max(mx, abs(e))
        r2 += a * a
        g2 += b * b
        dot += a * b
    return {
        "rmse": math.sqrt(se / n) if n else 0.0,
        "mae": ae / n if n else 0.0,
        "max_abs": mx,
        "rel_l2": math.sqrt(se / (r2 + 1e-30)),
        "cosine": dot / (math.sqrt(r2 * g2) + 1e-30),
        "sqnr_db": 10.0 * math.log10((r2 + 1e-30) / (se + 1e-30)),
    }


def qsym(v, denom, scale):
    if scale == 0.0:
        return 0
    q = int(round(v / scale))
    if q > denom:
        q = denom
    if q < -denom:
        q = -denom
    return q


def quant_tensor(w, denom):
    mx = max((abs(x) for x in w), default=0.0)
    s = mx / denom if mx else 1.0
    return [qsym(x, denom, s) * s for x in w]


def quant_row(w, rows, cols, denom):
    out = [0.0] * len(w)
    for r in range(rows):
        row = w[r * cols:(r + 1) * cols]
        mx = max((abs(x) for x in row), default=0.0)
        s = mx / denom if mx else 1.0
        for c, x in enumerate(row):
            out[r * cols + c] = qsym(x, denom, s) * s
    return out


def quant_block(w, rows, cols, block, denom, col_scale=None):
    out = [0.0] * len(w)
    for rb in range(0, rows, block):
        re = min(rows, rb + block)
        for cb in range(0, cols, block):
            ce = min(cols, cb + block)
            mx = 0.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    cs = col_scale[c] if col_scale else 1.0
                    mx = max(mx, abs(w[off + c] * cs))
            s = mx / denom if mx else 1.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    cs = col_scale[c] if col_scale else 1.0
                    out[off + c] = qsym(w[off + c] * cs, denom, s) * s / cs
    return out


def fp4_e2m1_value(x):
    codebook = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    sign = -1.0 if x < 0.0 else 1.0
    ax = abs(x)
    best = min(codebook, key=lambda y: abs(ax - y))
    return sign * best


def quant_fp4_block(w, rows, cols, block):
    out = [0.0] * len(w)
    for rb in range(0, rows, block):
        re = min(rows, rb + block)
        for cb in range(0, cols, block):
            ce = min(cols, cb + block)
            mx = 0.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    mx = max(mx, abs(w[off + c]))
            s = mx / 6.0 if mx else 1.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    out[off + c] = fp4_e2m1_value(w[off + c] / s) * s
    return out


def awq_scale(w, rows, cols):
    best_out = None
    best_m = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        cs = []
        for c in range(cols):
            vals = [abs(w[r * cols + c]) for r in range(rows)]
            wmax = max(vals) if vals else 0.0
            rms = math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0
            s = ((rms + 1e-12) ** alpha) / ((wmax + 1e-12) ** (1.0 - alpha))
            cs.append(s if math.isfinite(s) and s > 0 else 1.0)
        out = quant_block(w, rows, cols, 128, 127, cs)
        m = metrics(w, out)
        if best_m is None or m["rel_l2"] < best_m["rel_l2"]:
            best_m = m
            best_out = out
    return best_out


def features(w):
    n = len(w)
    if not n:
        return {}
    abs_vals = [abs(x) for x in w]
    s2 = sum(x * x for x in w)
    return {
        "sample_absmax": max(abs_vals),
        "sample_mean_abs": sum(abs_vals) / n,
        "sample_rms": math.sqrt(s2 / n),
        "sample_zero_pct": 100.0 * sum(1 for x in w if x == 0.0) / n,
    }


def write_row(writer, model_name, shard, name, meta, rows, cols, has_scale, w, methods):
    f = features(w)
    base = {
        "model": model_name,
        "shard": shard,
        "tensor": name,
        "dtype": meta["dtype"],
        "rows": meta["shape"][0],
        "cols": meta["shape"][1],
        "sample_rows": rows,
        "sample_cols": cols,
        "sample_elems": len(w),
        "has_scale": int(has_scale),
        **f,
    }
    variants = {}
    if "i8_tensor" in methods:
        variants["i8_tensor"] = quant_tensor(w, 127)
    if "i8_row" in methods:
        variants["i8_row"] = quant_row(w, rows, cols, 127)
    if "i8_block128" in methods:
        variants["i8_block128"] = quant_block(w, rows, cols, 128, 127)
    if "i8_awq" in methods:
        variants["i8_awq"] = awq_scale(w, rows, cols)
    if "int4_block128" in methods:
        variants["int4_block128"] = quant_block(w, rows, cols, 128, 7)
    if "fp4_block128" in methods:
        variants["fp4_block128"] = quant_fp4_block(w, rows, cols, 128)
    if "i16_block128" in methods:
        variants["i16_block128"] = quant_block(w, rows, cols, 128, 32767)
    if "fp16" in methods:
        variants["fp16"] = [f32_to_f16(x) for x in w]
    for key, got in variants.items():
        m = metrics(w, got)
        for mk, mv in m.items():
            base[f"{key}_{mk}"] = mv
    writer.writerow(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--max-elements-per-tensor", type=int, default=32768)
    ap.add_argument("--include", default=r".*\.weight$")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--methods", default="i8_tensor,i8_row,i8_block128,i8_awq,int4_block128,fp4_block128,i16_block128,fp16")
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    model_name = os.path.basename(model.rstrip("/"))
    include = re.compile(args.include)
    shards = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
    fields = [
        "model", "shard", "tensor", "dtype", "rows", "cols", "sample_rows", "sample_cols",
        "sample_elems", "has_scale", "sample_absmax", "sample_mean_abs", "sample_rms",
        "sample_zero_pct",
    ]
    methods = set(x.strip() for x in args.methods.split(",") if x.strip())
    for q in ("i8_tensor", "i8_row", "i8_block128", "i8_awq", "int4_block128", "fp4_block128", "i16_block128", "fp16"):
        for m in ("rmse", "mae", "max_abs", "rel_l2", "cosine", "sqnr_db"):
            fields.append(f"{q}_{m}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    count = 0
    with open(args.out, "w", newline="") as of:
        writer = csv.DictWriter(of, fieldnames=fields)
        writer.writeheader()
        for shard in shards:
            sf = SafeFile(os.path.join(model, shard))
            try:
                for name, meta in sf.tensors.items():
                    if args.max_tensors and count >= args.max_tensors:
                        return
                    if not include.match(name):
                        continue
                    if name.endswith("_scale_inv") or name.endswith(".scale") or name.endswith(".scale_inv"):
                        continue
                    if len(meta.get("shape", [])) != 2:
                        continue
                    if meta["dtype"] not in ("BF16", "F16", "FP16", "F32", "F8_E4M3", "F8_E4M3FN"):
                        continue
                    try:
                        w, sr, sc, has_scale = load_sample(sf, name, args.rows, args.cols, args.max_elements_per_tensor)
                        write_row(writer, model_name, shard, name, meta, sr, sc, has_scale, w, methods)
                        count += 1
                        if count % 100 == 0:
                            of.flush()
                            print(f"{model_name}: processed {count}", file=sys.stderr)
                    except Exception as e:
                        print(f"skip {model_name}:{name}: {e}", file=sys.stderr)
            finally:
                sf.close()
    print(f"{model_name}: wrote {count} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
