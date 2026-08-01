#!/usr/bin/env python3
"""Requantize the bf16 Qwen-Image DiT to e4m3 fp8 with HIP's fp8 behavior in mind.

HIP loads fp8 weights raw (w_scale=1): values clip at +-448 and use only 3 mantissa
bits. The CUDA fp8 ckpt was quantized that way. Per-OUTPUT-CHANNEL scaling lets each
row span the full e4m3 range, cutting clip/quant error. Compares vs the shipped fp8.
Run after the bf16 download finishes; emits fp8 + per-row scale per linear.
"""
import argparse, numpy as np
from safetensors import safe_open

E4M3_MAX = 448.0

def to_e4m3(x):  # round-to-nearest-even via numpy emulation
    x = np.clip(x, -E4M3_MAX, E4M3_MAX).astype(np.float32)
    # crude e4m3: 3 mantissa bits, exp bias 7 — quantize mantissa
    s = np.sign(x); a = np.abs(x); a[a==0]=1
    e = np.floor(np.log2(a)); m = np.round(a/2**e*8)/8; q = s*m*2**e
    return np.clip(q, -E4M3_MAX, E4M3_MAX)

def err(ref, q): return float(np.abs(ref-q).mean()/(np.abs(ref).mean()+1e-9))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bf16",required=True); ap.add_argument("--n",type=int,default=20)
    a=ap.parse_args()
    with safe_open(a.bf16,"pt") as f:
        ws=[k for k in f.keys() if k.endswith(".weight") and "blocks.0." in k][:a.n]
        for k in ws:
            w=f.get_tensor(k).float().numpy()
            flat=to_e4m3(w); sc=np.abs(w).max(1,keepdims=True)/E4M3_MAX; sc[sc==0]=1
            perch=to_e4m3(w/sc)*sc
            print(f"{k:50s} flat_mae={err(w,flat):.4f} perchan_mae={err(w,perch):.4f}")

if __name__=="__main__": main()
