#!/usr/bin/env python3
"""Repack the NVIDIA ModelOpt NVFP4 FLUX.2-Klein checkpoint into the runner's
W4A4 OMMA layout (cuda/flux2 native FP4 path).

The source (e.g. black-forest-labs/FLUX.2-klein-4b-nvfp4) stores per quantized
nn.Linear (80 of them: 5 dbl × 8 + 20 sgl × 2):

    <base>.weight          U8         [out, in/2]    FP4 e2m1 packed (low nib = even k)
    <base>.weight_scale    F8_E4M3    [out, in/16]   per-block-16 micro scales (linear, NOT swizzled)
    <base>.weight_scale_2  F32        scalar         per-tensor outer scale
    <base>.input_scale     F32        scalar         (static act calib, IGNORED — we re-quantize dynamically)

Dequant: W[o,i] = e2m1(code) * e4m3(weight_scale[o, i//16]) * weight_scale_2

We re-emit:
    <base>.qweight_fp4  int32      [out, in/8]    8 e2m1 codes / uint32 (low nib = low k)
    <base>.wscales_fp4  uint8      [out, in/16]   raw e4m3 bytes (copy)
    <base>.wcwt         float32    [out]          weight_scale_2 broadcast → per-row scale

Modes:
    default          OMMA W4A4 layout (Stage 2 target).
    --dequant-bf16   plain dense BF16 with BFL fp8-style names (no .weight_scale) — loads through
                     the existing flux2 BF16 path for Stage-1 validation vs the on-disk fp8 model.
    --selftest       repack→decode round-trip vs direct ModelOpt dequant on a few real tensors,
                     assert rel_L2 < 1e-6. Stage-0 guard before any GPU work.

Fused vs split:
    qkv [3H,H], linear1 [3H+2*n_ff, H]: kept fused (the runner row-slices q/k/v/gate_up at
        runtime; row-slices are contiguous in the [out, in/8] row-major layout).
    linear2 [H, H+n_ff]: SPLIT into two separate FP4 sub-linears `linear2_attn [H,H]` (first H
        cols) and `linear2_mlp [H, n_ff]` (last n_ff cols). Column-slicing at runtime would
        change the row stride, which the FP4 GEMM doesn't support. H=3072 is a multiple of 16
        → clean block boundary so we can slice codes/scales without re-quantizing.

Usage:
    python3 modelopt_fp4_repack.py SRC DST [--dequant-bf16] [--selftest [--n-tensors N]]
"""
import os, sys, argparse, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# E2M1 sign-magnitude code → value (codes 0..7 positive, 8..15 negative; -0 == 0 numerically)
E2M1 = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.,
                     -0., -0.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)


def unpack_nibbles_u8(packed_u8, dev):
    """packed uint8 [out, in/2] → int32 [out, in] codes 0..15.

    ModelOpt NVFP4 convention (verified bit-exact against
    modelopt.torch.quantization.qtensor.NVFP4QTensor.dequantize on the BFL nvfp4
    file): EVEN k uses LOW nibble, ODD k uses HIGH nibble — i.e.
        unpacked[..., 0::2] = input & 0x0F
        unpacked[..., 1::2] = input >> 4
    The unpacked codes are then re-packed 8/uint32 (low-nib-low-k) for the
    runtime, which expects the same convention the OMMA W4A4 kernel decodes.
    """
    p = packed_u8.to(dev)
    O, K2 = p.shape
    lo = (p & 0x0F).to(torch.int32)
    hi = ((p >> 4) & 0x0F).to(torch.int32)
    # Interleave: even k = lo nibble, odd k = hi nibble.
    out = torch.stack([lo, hi], dim=-1).reshape(O, K2 * 2)
    return out


def pack_codes_u32(codes):
    """[O,I] int32 0..15 → [O, I/8] int32 (8 codes/uint, low nib = low k). Matches runtime decode."""
    O, I = codes.shape
    assert I % 8 == 0, f"in={I} not divisible by 8"
    c = codes.to(torch.int32).reshape(O, I // 8, 8)
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=codes.device)
    return (c << sh).sum(dim=-1).to(torch.int32)


def repack_omma(weight_u8, ws_e4m3, ws2_f32, dev):
    """ModelOpt (uint8[out,in/2], e4m3[out,in/16], f32 scalar) → OMMA (i32[out,in/8],
    u8[out,in/16], f32[out]). Lossless re-layout (codes unchanged, scales rebroadcast)."""
    codes = unpack_nibbles_u8(weight_u8, dev)              # [O, I]
    O = codes.shape[0]
    qw = pack_codes_u32(codes)                              # [O, I/8] i32
    ws_u8 = ws_e4m3.to(dev).view(torch.uint8).contiguous() # raw bytes [O, I/16]
    wcwt = torch.full((O,), float(ws2_f32), dtype=torch.float32, device=dev)
    return qw.cpu(), ws_u8.cpu(), wcwt.cpu()


def modelopt_dequant(weight_u8, ws_e4m3, ws2_f32, dev):
    """Reference: W[o,i] = e2m1(code) * e4m3(weight_scale[o, i//16]) * weight_scale_2."""
    e2 = E2M1.to(dev)
    codes = unpack_nibbles_u8(weight_u8, dev)                              # [O, I]
    O, I = codes.shape
    GS = 16
    ws_f32 = ws_e4m3.to(dev).view(torch.float8_e4m3fn).float().view(O, I // GS)
    gidx = torch.arange(I, device=dev) // GS
    return e2[codes.long()] * ws_f32[:, gidx] * float(ws2_f32)


def omma_decode(qw, ws_u8, wcwt, dev, GS=16):
    """Inverse of repack_omma: reconstruct W. Same formula as the runtime FP4 kernels.
    Computes: W[o,i] = E2M1[code(o,i)] * e4m3_decode(ws[o, i//GS]) * wcwt[o]."""
    O = qw.shape[0]
    I = qw.shape[1] * 8
    e2 = E2M1.to(dev)
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)
    codes = ((qw.to(dev).view(O, I // 8, 1) >> sh) & 0xF).reshape(O, I)
    ws_f32 = ws_u8.to(dev).view(torch.float8_e4m3fn).float().view(O, I // GS)
    gidx = torch.arange(I, device=dev) // GS
    return e2[codes.long()] * ws_f32[:, gidx] * wcwt.to(dev).unsqueeze(1)


def quantized_bases(keys):
    """Set of base names that are NVFP4 (have .weight_scale_2)."""
    return sorted({k[: -len(".weight_scale_2")] for k in keys if k.endswith(".weight_scale_2")})


def bf16_passthrough_keys(keys, quant_bases):
    """All keys that are NOT one of the four (.weight/.weight_scale/.weight_scale_2/.input_scale)
    of a quantized linear — i.e. norms, embedders, modulation, final_layer, etc."""
    quant_suffixes = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
    bases = set(quant_bases)
    out = []
    for k in keys:
        is_quant_tensor = any(k.endswith(s) and k[: -len(s)] in bases for s in quant_suffixes)
        if not is_quant_tensor:
            out.append(k)
    return out


def self_test(src_path, dev, n_tensors):
    """Stage-0 guard: round-trip a few real tensors through repack→decode vs the direct
    ModelOpt dequant. Asserts the layout/packing is correct before any GPU work."""
    f = safe_open(src_path, "pt")
    keys = list(f.keys())
    bases = quantized_bases(keys)
    sample = []
    # Pick a variety: qkv, mlp.0, mlp.2, linear1, linear2 from block 0
    wanted_kinds = ["img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
                    "linear1", "linear2", "txt_attn.qkv"]
    for kind in wanted_kinds:
        for b in bases:
            if b.endswith(kind) and ("blocks.0." in b):
                sample.append(b)
                break
    sample = sample[:n_tensors] if n_tensors > 0 else sample
    print(f"[selftest] running on {len(sample)} representative tensors:")
    max_rel = 0.0
    for base in sample:
        w8 = f.get_tensor(base + ".weight")
        ws = f.get_tensor(base + ".weight_scale")
        s2 = f.get_tensor(base + ".weight_scale_2")
        Wmo = modelopt_dequant(w8, ws, s2, dev)
        qw, ws_u8, wcwt = repack_omma(w8, ws, s2, dev)
        Wom = omma_decode(qw, ws_u8, wcwt, dev)
        rel = ((Wmo - Wom).norm() / Wmo.norm().clamp_min(1e-12)).item()
        max_rel = max(max_rel, rel)
        print(f"  {base:50s} w{tuple(w8.shape)} s{tuple(ws.shape)} s2{tuple(s2.shape)}  "
              f"rel_L2 = {rel:.3e}")
    print(f"[selftest] max rel_L2 = {max_rel:.3e}  ({'OK' if max_rel < 1e-6 else 'FAIL'})")
    assert max_rel < 1e-6, f"OMMA decode does not match ModelOpt dequant (max rel_L2 {max_rel:.3e})"


def emit_omma(src_path, dst_path, dev, verbose):
    """Default mode: produce W4A4 OMMA layout."""
    f = safe_open(src_path, "pt")
    keys = sorted(f.keys())
    bases = quantized_bases(keys)
    out = {}
    H = 3072  # FLUX2_HIDDEN_DIM
    GS = 16
    t0 = time.time()
    for i, base in enumerate(bases, 1):
        w8 = f.get_tensor(base + ".weight")
        ws = f.get_tensor(base + ".weight_scale")
        s2 = f.get_tensor(base + ".weight_scale_2")
        O = w8.shape[0]
        I = w8.shape[1] * 2  # uint8 holds 2 nibbles
        if base.endswith("linear2"):
            # Column-split [H, H+n_ff] at K=H into l2_attn [H,H] + l2_mlp [H,n_ff].
            # H is multiple of GS=16 → clean block boundary. wcwt is shared (same scale_2).
            assert I % GS == 0 and H % GS == 0
            codes = unpack_nibbles_u8(w8, dev)                       # [O, I]
            ws_u8 = ws.to(dev).view(torch.uint8).contiguous()       # [O, I/GS]
            wcwt_full = torch.full((O,), float(s2), dtype=torch.float32, device=dev)
            # Slice codes/scales by column then re-pack codes to int32
            codes_a = codes[:, :H].contiguous()
            codes_m = codes[:, H:].contiguous()
            ws_a = ws_u8[:, : H // GS].contiguous()
            ws_m = ws_u8[:, H // GS:].contiguous()
            qw_a = pack_codes_u32(codes_a)
            qw_m = pack_codes_u32(codes_m)
            prefix = base.rsplit(".linear2", 1)[0]
            for nm, qw, wsx in [
                (prefix + ".linear2_attn", qw_a, ws_a),
                (prefix + ".linear2_mlp",  qw_m, ws_m),
            ]:
                out[nm + ".qweight_fp4"] = qw.cpu()
                out[nm + ".wscales_fp4"] = wsx.cpu()
                out[nm + ".wcwt"] = wcwt_full.cpu()
        else:
            # Fused or single qkv/proj/mlp/linear1 — straight repack, runner row-slices fused linears.
            qw, ws_u8, wcwt = repack_omma(w8, ws, s2, dev)
            out[base + ".qweight_fp4"] = qw
            out[base + ".wscales_fp4"] = ws_u8
            out[base + ".wcwt"] = wcwt
        if verbose and (i % 20 == 0 or i == 1 or i == len(bases)):
            print(f"  [omma] {i}/{len(bases)} {base}  -> {O}x{I}")
        if dev == "cuda":
            torch.cuda.empty_cache()
    # Pass-through everything else (BF16 globals + per-block norms)
    pt = bf16_passthrough_keys(keys, bases)
    for k in pt:
        out[k] = f.get_tensor(k)
    print(f"[omma] {len(bases)} FP4 linears, {len(pt)} pass-through tensors  "
          f"({time.time()-t0:.1f}s)")
    print(f"[omma] saving {len(out)} tensors -> {dst_path}")
    save_file(out, dst_path, metadata={"format": "pt", "stage": "modelopt-nvfp4 -> W4A4 OMMA"})
    print("[omma] done")


def emit_dequant_bf16(src_path, dst_path, dev, verbose):
    """Stage-1 mode: dequantize NVFP4 to dense BF16 with BFL fp8-style names so the
    EXISTING flux2 BF16 path loads it unchanged. Validates parse/layout vs the known-good
    fp8 model — emit one image with this checkpoint and compare to the fp8-model image."""
    f = safe_open(src_path, "pt")
    keys = sorted(f.keys())
    bases = quantized_bases(keys)
    out = {}
    t0 = time.time()
    for i, base in enumerate(bases, 1):
        w8 = f.get_tensor(base + ".weight")
        ws = f.get_tensor(base + ".weight_scale")
        s2 = f.get_tensor(base + ".weight_scale_2")
        W = modelopt_dequant(w8, ws, s2, dev).to(torch.bfloat16).cpu()
        out[base + ".weight"] = W
        if verbose and (i % 20 == 0 or i == 1 or i == len(bases)):
            print(f"  [dequant-bf16] {i}/{len(bases)} {base}  -> {tuple(W.shape)}")
        if dev == "cuda":
            torch.cuda.empty_cache()
    pt = bf16_passthrough_keys(keys, bases)
    for k in pt:
        out[k] = f.get_tensor(k)
    print(f"[dequant-bf16] {len(bases)} dequant + {len(pt)} pass-through  "
          f"({time.time()-t0:.1f}s)")
    print(f"[dequant-bf16] saving {len(out)} tensors -> {dst_path}")
    save_file(out, dst_path, metadata={"format": "pt", "stage": "modelopt-nvfp4 dequant->BF16"})
    print("[dequant-bf16] done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="source nvfp4 .safetensors")
    ap.add_argument("dst", nargs="?", help="destination .safetensors")
    ap.add_argument("--dequant-bf16", action="store_true",
                    help="Stage-1 mode: emit dense BF16 (no FP4 layout)")
    ap.add_argument("--selftest", action="store_true",
                    help="Stage-0: round-trip vs ModelOpt dequant, exit")
    ap.add_argument("--n-tensors", type=int, default=8,
                    help="number of representative tensors for --selftest (0=all)")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu (cpu is slow but doesn't need a GPU)")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("[modelopt-fp4] cuda not available, falling back to cpu", file=sys.stderr)
        a.device = "cpu"
    if a.selftest:
        if not a.src:
            ap.error("--selftest needs SRC path")
        self_test(a.src, a.device, a.n_tensors)
        return
    if not a.src or not a.dst:
        ap.error("SRC and DST required (or pass --selftest)")
    if a.dequant_bf16:
        emit_dequant_bf16(a.src, a.dst, a.device, a.verbose)
    else:
        emit_omma(a.src, a.dst, a.device, a.verbose)


if __name__ == "__main__":
    main()
