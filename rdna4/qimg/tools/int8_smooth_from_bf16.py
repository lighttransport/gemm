#!/usr/bin/env python3
"""int8_smooth_from_bf16.py — W8A8 SmoothQuant of the bf16 Qwen-Image DiT.

Fixes the int4 flat-region "wobble": int8 keeps both activation outliers and bulk values precise
once SmoothQuant migrates the per-input-channel activation outliers into the weights. Port of
cuda/qimg int8-smoothquant.md (alpha=0.5). Per quantized linear:

  s[j]   = amax[j]^alpha / wmax[j]^(1-alpha)     (per-input-channel smooth; amax from calibration)
  What   = W * diag(s)                            (outliers migrate into weights)
  ws[o]  = max_j|What[o,j]| / 127                 (per-output-channel int8 scale)
  Wq     = round(What / ws[:,None]).clip(-127,127)  int8

Emits per linear:  <key>.weight (I8 [out,in])  <key>.weight_scale (F32 [out])
                   <key>.smooth_scale (F32 [in] = s)  <key>.bias (bf16 passthrough)
The runner divides the activation by smooth_scale (== s) before per-token int8 quant, so the GEMM
math is unchanged: (X/s)·(W·s)^T = X·W^T. Globals/norms pass through bf16. ~20 GB output.
"""
import argparse, sys, time, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
        "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2"]
MOD = ["img_mod.1", "txt_mod.1"]
NORMS = ["attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k"]
GLOBALS = ["img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias", "txt_norm.weight",
           "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias",
           "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias",
           "norm_out.linear.weight", "norm_out.linear.bias", "proj_out.weight", "proj_out.bias"]


def load_calib(paths):
    merged = {}
    for p in paths:
        with safe_open(p, "pt", "cpu") as f:
            for k in f.keys():
                if not k.endswith(".amax"):
                    continue
                t = f.get_tensor(k).float()
                merged[k] = t if k not in merged else torch.maximum(merged[k], t)
    return merged


def quant_lin(W, amax, alpha, clip):
    wmax = W.abs().amax(0).clamp(min=1e-8)                                   # [in]
    if amax is None or alpha <= 0:
        lam = torch.ones_like(wmax)
    else:
        # Clamp lam >= 1: only migrate activation outliers INTO weights (smooth>=1 => x/smooth<=x),
        # never amplify a low-activation channel. lam<1 amplifies x by 1/lam, and when the real
        # activation exceeds the (few-prompt) calib estimate the per-token int8 scale explodes ->
        # velocity blowup. >=1 keeps the outlier taming (the wobble fix) without that risk.
        lam = (amax.clamp(min=1e-8).pow(alpha) / wmax.pow(1.0 - alpha)).clamp(1.0, clip)
    What = W * lam.unsqueeze(0)                                              # [out,in]
    wscale = (What.abs().amax(1) / 127.0).clamp(min=1e-12)                   # [out]
    Wq = torch.round(What / wscale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return Wq, wscale.float().contiguous(), lam.float().contiguous()


def check(W, amax, alpha, clip, group_probe=8):
    """Per-linear output relL2 vs f32 on amax-scaled probes (runtime: x/s int8-quantized too)."""
    Wq, ws, lam = quant_lin(W, amax, alpha, clip)
    out, k = W.shape
    scale = amax.clamp(min=1e-6) if amax is not None else torch.ones(k)
    x = torch.randn(group_probe, k) * scale.unsqueeze(0)
    # simulate per-token int8 activation quant of (x/lam)
    xe = x / lam.unsqueeze(0)
    xs = (xe.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
    xq = torch.round(xe / xs).clamp(-127, 127)
    y_q = (xq @ Wq.float().t()) * (xs * ws.unsqueeze(0))                     # dequant
    y_ref = x @ W.t()
    return (y_q - y_ref).norm().item() / (y_ref.norm().item() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="/mnt/disk1/models/qwen-image/_bf16dl/split_files/diffusion_models/qwen_image_bf16.safetensors")
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1e3)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    calib = {} if a.no_smooth else load_calib(a.calib)
    if not a.no_smooth and not calib:
        print("[warn] no calib -> smooth=1 (plain int8, will wobble)", file=sys.stderr)
    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())
    if a.blocks == "all":
        nblk = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("transformer_blocks."))
        blocks = list(range(nblk))
    else:
        blocks = [int(x) for x in a.blocks.split(",")]
    print(f"int8-smooth: {len(blocks)} blocks, alpha={a.alpha}, smooth={'off' if a.no_smooth else ('calib' if calib else '1')}", file=sys.stderr)
    out = {}; t0 = time.time(); rels = []
    for bi, b in enumerate(blocks):
        p = f"transformer_blocks.{b}."
        for suf in MAIN + MOD:
            W = src.get_tensor(p + suf + ".weight").float()
            amax = calib.get(p + suf + ".amax")
            Wq, ws, lam = quant_lin(W, amax, a.alpha, a.clip)
            k = p + suf
            out[k + ".weight"] = Wq
            out[k + ".weight_scale"] = ws
            if not a.no_smooth:
                out[k + ".smooth_scale"] = lam
            out[k + ".bias"] = src.get_tensor(p + suf + ".bias").to(torch.bfloat16)
            if a.check:
                r = check(W, amax, a.alpha, a.clip); rels.append(r)
                print(f"  b{b:02d} {suf:18s} out={W.shape[0]:5d} in={W.shape[1]:5d} relL2={r:.4f}", file=sys.stderr)
        for suf in NORMS:
            out[p + suf + ".weight"] = src.get_tensor(p + suf + ".weight").to(torch.bfloat16)
        if (bi + 1) % 10 == 0 or bi + 1 == len(blocks):
            print(f"  block {b} ({bi+1}/{len(blocks)}, {time.time()-t0:.0f}s)", file=sys.stderr)
    for g in GLOBALS:
        if g in keys:
            out[g] = src.get_tensor(g).to(torch.bfloat16)
    if rels:
        print(f"[check] mean relL2 over {len(rels)} linears = {np.mean(rels):.4f}", file=sys.stderr)
    print(f"writing {len(out)} tensors -> {a.out}", file=sys.stderr)
    save_file(out, a.out, metadata={"format": "qwen-image-int8-smoothquant-w8a8", "alpha": str(a.alpha)})
    print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
