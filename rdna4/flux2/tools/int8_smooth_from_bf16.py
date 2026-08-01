#!/usr/bin/env python3
"""int8_smooth_from_bf16.py — W8A8 SmoothQuant quantization of FLUX.2 Klein 4B.

Quantize the bf16 4B single-file to per-row INT8 + per-K-channel SmoothQuant lambda.
Per linear:
   lam   = clamp(amax^alpha / wmax^(1-alpha), min=1.0)         [in] f32, lam ≥ 1 only
   W_sm  = W * lam[None, :]
   ws[o] = max|W_sm[o]| / 127
   Wq    = round(W_sm / ws[:, None]).clip(-127, 127) int8
Runtime: quant_act_perrow_int8 divides X by smooth before per-token int8 quant; W8A8
GEMM is exact; the kernel dequants via ws[o] * x_scale[t] + bias. Math:
   (Wq * ws[None]) @ (Xq * x_scale[:,None]) = W_sm @ (X / lam) = (W * lam) @ (X / lam) = W @ X.

The lam ≥ 1 clamp is CRITICAL — per [[project_qimg_int8_smoothquant]] (commit dfa106e):
without it, channels with amax < wmax get lam < 1, which AMPLIFIES the per-token int8
activation scale and blows the output up (cos -0.87 → 0.9998 with the clamp). Different
from the int4 SVDQ tool which uses the full [1/clip, clip] range (LoRA absorbs outliers).

Emits (replaces <key>.weight in the original bf16 file):
   <key>.weight       I8   [out, in]
   <key>.weight_scale F32  [out, 1]    (per-row, matches the ModelOpt int8 format we already load)
   <key>.smooth       BF16 [in]
   <key>.bias         BF16 [out]       (passthrough)
Globals/norms stay bf16.

CLI:
   python int8_smooth_from_bf16.py \\
       --bf16  /mnt/disk1/models/klein2-4b/bf16/flux-2-klein-4b.safetensors \\
       --calib /mnt/disk1/models/klein2-4b/calib/flux2_klein_4b_calib.safetensors \\
       --out   /mnt/disk1/models/klein2-4b/int8_smooth/flux-2-klein-4b-int8-smooth.safetensors
"""
import argparse, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_DBLK = ["img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
             "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2"]
MAIN_SBLK = ["linear1", "linear2"]


def smooth_lambda_int8(W_bf16: torch.Tensor, amax: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """lam ≥ 1 only — see qimg int8 note for why clamping at 1 (not [1/clip, clip]) is required."""
    W = W_bf16.float()
    wmax = W.abs().amax(0).clamp(min=1e-8)               # [in]
    am   = amax.clamp(min=1e-8)
    lam  = (am.pow(alpha) / wmax.pow(1.0 - alpha)).clamp(min=1.0)
    return lam


def quant_int8_perrow(W_eff_bf16: torch.Tensor):
    """W_eff (after smoothing) bf16 [out, in] -> (Wq i8[out,in], ws f32[out,1])."""
    W = W_eff_bf16.float()
    ws = (W.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)    # [out, 1]
    Wq = torch.round(W / ws).clamp(-127, 127).to(torch.int8)
    return Wq.contiguous(), ws.float().contiguous()


def load_calib(path):
    """Same merge logic as the int4 SVDQ tool: combine linear2_attn + linear2_mlp halves
    into a single [H+n_ff] amax for each single_blocks.N.linear2."""
    out = {}
    halves = {}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    calib = load_calib(a.calib)
    print(f"calib: {len(calib)} amax entries from {a.calib} (alpha={a.alpha}, lam ≥ 1)", file=sys.stderr)

    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())
    n_dbl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")), default=-1)
    n_sgl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.")), default=-1)
    print(f"int8-smooth: {n_dbl} double_blocks, {n_sgl} single_blocks", file=sys.stderr)

    out = {}; quant_keys = set(); t0 = time.time(); n_done = 0

    def quant_one(prefix: str, suf: str):
        nonlocal n_done
        k = f"{prefix}.{suf}.weight"
        if k not in keys:
            print(f"[warn] missing {k}", file=sys.stderr); return
        W = src.get_tensor(k)
        name = f"{prefix}.{suf}"
        amax = calib.get(name)
        if amax is None:
            print(f"[warn] no calib for {name} -> lam=1 (plain int8)", file=sys.stderr)
            lam = torch.ones(W.shape[1])
        else:
            lam = smooth_lambda_int8(W, amax, a.alpha)
        W_eff = (W.float() * lam.unsqueeze(0)).to(torch.bfloat16)
        Wq, ws = quant_int8_perrow(W_eff)
        out[f"{name}.weight"]        = Wq
        out[f"{name}.weight_scale"]  = ws
        out[f"{name}.smooth"]        = lam.to(torch.bfloat16)
        bk = f"{name}.bias"
        if bk in keys: out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)
        quant_keys.add(k)
        n_done += 1
        if n_done % 10 == 0 or n_done == 80:
            lo, me, hi = lam.min().item(), lam.median().item(), lam.max().item()
            print(f"  [{n_done}/80] {name}  W{tuple(W.shape)}  lam[min={lo:.2f} med={me:.2f} max={hi:.2f}]  ({time.time()-t0:.1f}s)", file=sys.stderr)

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

    save_file(out, a.out, metadata={"format": "flux2-klein-int8-smoothquant-w8a8", "alpha": str(a.alpha)})
    sz = os.path.getsize(a.out) / 1e9
    print(f"wrote {a.out} ({sz:.2f} GB, {len(out)} tensors, {n_pass} passthrough)", file=sys.stderr)


if __name__ == "__main__":
    main()
