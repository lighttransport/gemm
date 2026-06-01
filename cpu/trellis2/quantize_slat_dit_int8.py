#!/usr/bin/env python3
"""Quantize TRELLIS.2 Stage-2 SLAT/shape DiT block linears to INT8 W8A8.

The emitted format mirrors the tex DiT/Qwen-Image SmoothQuant path:

  <base>.weight        I8   [out, in]
  <base>.weight_scale  F32  [out]
  <base>.smooth_scale  F32  [in]   (omitted with --no-smooth)

Top-level linears, adaLN modulation, norms, and biases pass through unchanged.
"""
import argparse
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


SLAT_BLOCK_LINEARS = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "mlp.mlp.0",
    "mlp.mlp.2",
)


def load_calib(paths):
    merged = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".amax"):
                    continue
                t = f.get_tensor(key).float()
                merged[key] = t if key not in merged else torch.maximum(merged[key], t)
    return merged


def quant_linear(weight, amax, alpha, clip):
    w = weight.float()
    wmax = w.abs().amax(dim=0).clamp(min=1.0e-8)
    if amax is None or alpha <= 0.0:
        smooth = torch.ones_like(wmax)
    else:
        smooth = (amax.float().clamp(min=1.0e-8).pow(alpha) /
                  wmax.pow(1.0 - alpha)).clamp(1.0, clip)
    w_smooth = w * smooth.unsqueeze(0)
    wscale = (w_smooth.abs().amax(dim=1) / 127.0).clamp(min=1.0e-12)
    wq = torch.round(w_smooth / wscale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return wq.contiguous(), wscale.float().contiguous(), smooth.float().contiguous()


def check_linear(weight, amax, alpha, clip, n_probe):
    wq, wscale, smooth = quant_linear(weight, amax, alpha, clip)
    out_ch, in_ch = weight.shape
    act_scale = amax.float().clamp(min=1.0e-6) if amax is not None else torch.ones(in_ch)
    x = torch.randn(n_probe, in_ch) * act_scale.unsqueeze(0)
    x_eff = x / smooth.unsqueeze(0)
    xscale = (x_eff.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1.0e-12)
    xq = torch.round(x_eff / xscale).clamp(-127, 127)
    y_q = (xq @ wq.float().t()) * (xscale * wscale.unsqueeze(0))
    y_ref = x @ weight.float().t()
    rel = (y_q - y_ref).norm().item() / (y_ref.norm().item() + 1.0e-12)
    cos = torch.nn.functional.cosine_similarity(y_q.flatten(), y_ref.flatten(), dim=0).item()
    return cos, rel, out_ch, in_ch


def infer_blocks(keys):
    blocks = []
    prefix = "blocks."
    for key in keys:
        if not key.startswith(prefix):
            continue
        block_s = key[len(prefix):].split(".", 1)[0]
        if block_s.isdigit():
            blocks.append(int(block_s))
    return sorted(set(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="BF16/F32 SLAT/shape DiT safetensors")
    ap.add_argument("out", help="Output INT8 SmoothQuant safetensors")
    ap.add_argument("--calib", nargs="*", default=[], help="T2_SLAT_CALIB_DUMP safetensors")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0e3)
    ap.add_argument("--blocks", default="all", help="'all' or comma-separated block ids")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    calib = {} if args.no_smooth else load_calib(args.calib)
    if not args.no_smooth and not calib:
        print("[warn] no calibration stats loaded; smooth_scale will be all ones", file=sys.stderr)

    out = {}
    rels = []
    n_quant = 0
    n_pass = 0
    src_bytes = 0
    dst_bytes = 0

    with safe_open(args.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if args.blocks == "all":
            block_ids = set(infer_blocks(keys))
        else:
            block_ids = {int(x) for x in args.blocks.split(",") if x.strip()}

        quant_keys = {
            f"blocks.{b}.{suffix}.weight"
            for b in block_ids
            for suffix in SLAT_BLOCK_LINEARS
        }

        for key in keys:
            tensor = f.get_tensor(key)
            if key in quant_keys:
                base = key[:-len(".weight")]
                amax = None if args.no_smooth else calib.get(base + ".amax")
                wq, wscale, smooth = quant_linear(tensor, amax, args.alpha, args.clip)
                out[key] = wq
                out[base + ".weight_scale"] = wscale
                if not args.no_smooth:
                    out[base + ".smooth_scale"] = smooth
                n_quant += 1
                src_bytes += tensor.numel() * tensor.element_size()
                dst_bytes += wq.numel() + wscale.numel() * 4
                if not args.no_smooth:
                    dst_bytes += smooth.numel() * 4
                if args.check:
                    cos, rel, out_ch, in_ch = check_linear(tensor, amax, args.alpha, args.clip, args.probe)
                    rels.append(rel)
                    print(f"{base:40s} out={out_ch:5d} in={in_ch:5d} cos={cos:.6f} relL2={rel:.5f}",
                          file=sys.stderr)
            else:
                out[key] = tensor
                n_pass += 1

    metadata = {
        "format": "trellis2-slat-dit-int8-smoothquant-w8a8",
        "alpha": str(args.alpha),
        "smooth": "0" if args.no_smooth else "1",
        "linears": ",".join(SLAT_BLOCK_LINEARS),
    }
    save_file(out, args.out, metadata=metadata)

    print(f"Quantized {n_quant} SLAT DiT block linears to INT8.", file=sys.stderr)
    print(f"Pass-through tensors: {n_pass}.", file=sys.stderr)
    if src_bytes:
        print(f"Quantized payload: {src_bytes/1e6:.1f} MB -> {dst_bytes/1e6:.1f} MB "
              f"({100.0 * dst_bytes / src_bytes:.1f}%).", file=sys.stderr)
    if rels:
        print(f"[check] mean relL2={np.mean(rels):.5f} max relL2={np.max(rels):.5f}",
              file=sys.stderr)
    print(f"Wrote {args.out} in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
