#!/usr/bin/env python3
"""fp8_smooth_from_bf16.py — bf16 -> FP8 E4M3 (per-tensor scale) + optional SmoothQuant.

Builds a comfy-style FP8 single-file from the bf16 transformer, optionally folding a
SmoothQuant lambda (per-K-channel; lam≥1 like int8) into the weights:

  lam[in]   = clamp(amax^alpha / wmax^(1-alpha), min=1.0)
  W_smooth  = W * lam[None, :]
  ws        = max(|W_smooth|) / 448                          (PER-TENSOR scalar)
  Wq        = (W_smooth / ws) -> e4m3 (round-to-nearest, sat ±448)

Emits per linear: <key>.weight (F8_E4M3), <key>.weight_scale (F32 scalar),
                  <key>.smooth (BF16 [in], only when --calib).
Runtime: existing fp8 path divides X by .smooth (via flux2_div_by_smooth_f32)
before flux2_quant_act_clamp, then pipe32 FP8×FP8 proceeds unchanged.

NOTE on theoretical risk: fp8 weight_scale is PER-TENSOR. SmoothQuant inflates the
max-abs of W (the most-amplified column dominates), so the per-tensor ws grows by
~max(lam)x. All NON-outlier elements get represented at lower fp8 precision via the
larger ws. May REGRESS quality. Compare to bf16-ref render to verify.

CLI:
  python fp8_smooth_from_bf16.py --bf16 ... --calib ... --out ...
"""
import argparse, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_DBLK = ["img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
             "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2"]
MAIN_SBLK = ["linear1", "linear2"]


def smooth_lambda(W_bf16, amax, alpha=0.5):
    W = W_bf16.float()
    wmax = W.abs().amax(0).clamp(min=1e-8)
    am   = amax.clamp(min=1e-8)
    return (am.pow(alpha) / wmax.pow(1.0 - alpha)).clamp(min=1.0)


def load_calib(path):
    out = {}; halves = {}
    f = safe_open(path, "pt", "cpu")
    for k in f.keys():
        if not k.endswith(".amax"): continue
        name = k[:-len(".amax")]
        if name.endswith(".linear2_attn") or name.endswith(".linear2_mlp"):
            base = name.rsplit(".", 1)[0] + ".linear2"
            tag  = "attn" if name.endswith("attn") else "mlp"
            halves.setdefault(base, {})[tag] = f.get_tensor(k).float()
        else:
            out[name] = f.get_tensor(k).float()
    for base, hv in halves.items():
        if "attn" in hv and "mlp" in hv:
            out[base] = torch.cat([hv["attn"], hv["mlp"]], dim=0)
    return out


def quant_fp8_e4m3(W_eff_bf16):
    """W_eff bf16 [out, in] -> (Wq u8 fp8-e4m3 bytes [out, in], ws scalar f32)."""
    W = W_eff_bf16.float()
    ws = (W.abs().max() / 448.0).clamp(min=1e-12).item()
    Wn = (W / ws).clamp(-448.0, 448.0)
    # Use PyTorch fp8 dtype if available, else manual encode (torch.float8_e4m3fn is in 2.1+)
    if hasattr(torch, "float8_e4m3fn"):
        Wq = Wn.to(torch.float8_e4m3fn)
    else:
        Wq = manual_e4m3(Wn)
    return Wq.contiguous(), float(ws)


def manual_e4m3(x_f32):
    """Fallback if torch doesn't have float8_e4m3fn."""
    out = torch.empty(x_f32.shape, dtype=torch.uint8)
    flat = x_f32.flatten().tolist()
    for i, v in enumerate(flat):
        out.view(-1)[i] = _f32_to_e4m3(v)
    return out


def _f32_to_e4m3(f):
    import struct
    if f > 448.0: f = 448.0
    if f < -448.0: f = -448.0
    b = struct.unpack("<I", struct.pack("<f", f))[0]
    sign = (b >> 31) & 1
    e = ((b >> 23) & 0xFF) - 127
    mant = b & 0x7FFFFF
    fp8_exp = e + 7
    if e < -9: return sign << 7
    if fp8_exp <= 0:
        full = mant | 0x800000
        shift = 1 - fp8_exp + 20
        if shift >= 24: return sign << 7
        m8 = (full + (1 << (shift-1))) >> shift
        if m8 > 7: m8 = 7
        return (sign << 7) | (m8 & 7)
    if fp8_exp >= 15: return (sign << 7) | (15 << 3) | 6
    m8 = (mant + (1 << 19)) >> 20
    if m8 > 7: m8 = 0; fp8_exp += 1
    if fp8_exp >= 15: return (sign << 7) | (15 << 3) | 6
    return (sign << 7) | (fp8_exp << 3) | (m8 & 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None, help="Enable SmoothQuant (lam≥1). Omit for plain fp8.")
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    calib = load_calib(a.calib) if a.calib else {}
    print(f"fp8{'+smooth' if calib else ''}: alpha={a.alpha}, {len(calib)} calib entries", file=sys.stderr)

    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())
    n_dbl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")), default=-1)
    n_sgl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.")), default=-1)

    out = {}; quant_keys = set(); t0 = time.time(); n_done = 0

    def quant_one(prefix, suf):
        nonlocal n_done
        k = f"{prefix}.{suf}.weight"
        if k not in keys: return
        W = src.get_tensor(k)
        name = f"{prefix}.{suf}"
        amax = calib.get(name)
        if amax is not None:
            lam = smooth_lambda(W, amax, a.alpha)
            W_eff = (W.float() * lam.unsqueeze(0)).to(torch.bfloat16)
            out[f"{name}.smooth"] = lam.to(torch.bfloat16)
        else:
            W_eff = W
            lam = None
        Wq, ws = quant_fp8_e4m3(W_eff)
        out[f"{name}.weight"]       = Wq
        out[f"{name}.weight_scale"] = torch.tensor(ws, dtype=torch.float32)
        bk = f"{name}.bias"
        if bk in keys: out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)
        quant_keys.add(k); n_done += 1
        if n_done % 10 == 0 or n_done == 80:
            tag = ""
            if lam is not None:
                lo, me, hi = lam.min().item(), lam.median().item(), lam.max().item()
                tag = f"  lam[min={lo:.2f} med={me:.2f} max={hi:.2f}]"
            print(f"  [{n_done}/80] {name}  W{tuple(W.shape)} ws={ws:.4g}{tag}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    for bi in range(n_dbl):
        for suf in MAIN_DBLK: quant_one(f"double_blocks.{bi}", suf)
    for bi in range(n_sgl):
        for suf in MAIN_SBLK: quant_one(f"single_blocks.{bi}", suf)
    print(f"quantized {n_done} linears in {time.time()-t0:.1f}s", file=sys.stderr)

    drop = {f"{k.rsplit('.weight',1)[0]}.weight_scale" for k in quant_keys if k.endswith(".weight")}
    n_pass = 0
    for k in keys:
        if k in quant_keys or k in drop: continue
        out[k] = src.get_tensor(k); n_pass += 1

    save_file(out, a.out, metadata={"format": "flux2-klein-fp8-smooth", "alpha": str(a.alpha)})
    sz = os.path.getsize(a.out) / 1e9
    print(f"wrote {a.out} ({sz:.2f} GB, {n_pass} passthrough)", file=sys.stderr)


if __name__ == "__main__":
    main()
