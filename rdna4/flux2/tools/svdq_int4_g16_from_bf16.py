#!/usr/bin/env python3
"""svdq_int4_g16_from_bf16.py — SVDQuant (LoRA only, no smooth/calib) of FLUX.2 Klein 4B.

Upgrade over rtn_int4_g16_from_bf16.py: for each main linear, decompose
    W [out, in] = U[:, :r] @ diag(S[:r]) @ V[:r, :]    (top-r SVD)
into a low-rank LoRA branch + an INT4-RTN-g16 residual:
    lora_up   [out, r]   bf16  = U[:, :r] * S[:r]
    lora_down [r, in]    bf16  = V[:r, :]
    R         [out, in]  bf16  = W - lora_up @ lora_down
    qint4 / wscale = RTN-g16(R)                         (same packing as the RTN tool)

Runtime forward: y = R_dequant @ x + lora_up @ (lora_down @ x) + bias.
NO smooth, NO calibration (lambda=1) — qimg's "weight-only SVDQuant" form. Quality lift
vs plain RTN comes from the LoRA absorbing the principal singular components so the
residual quantizes better. Full SmoothQuant+activation-calib is a separate upgrade.

Per quantized linear emits (in addition to .qint4 + .wscale + .bias from RTN tool):
    <key>.lora_up   bf16 [out, r]
    <key>.lora_down bf16 [r, in]

CLI:
    python svdq_int4_g16_from_bf16.py \\
        --bf16 /mnt/disk1/models/klein2-4b/bf16/flux-2-klein-4b.safetensors \\
        --out  /mnt/disk1/models/klein2-4b/int4/flux-2-klein-4b-int4-svdq-r32-g16.safetensors \\
        --rank 32
"""
import argparse, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_DBLK = ["img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
             "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2"]
MAIN_SBLK = ["linear1", "linear2"]


def rtn_pack(R_f32: torch.Tensor, group: int = 16):
    """Per-(out, group-of-K) symmetric RTN int4. Returns (qint4 u8[out, in/2],
    wscale bf16[out, in/group]). group=16 pairs with gemm_int4w_g16_*; group=64
    pairs with the smooth-aware gemm_int4w_bf16a_wmma_t (better SmoothQuant composability)."""
    out, n_in = R_f32.shape
    assert n_in % group == 0, f"n_in={n_in} must be divisible by group={group}"
    Rg = R_f32.view(out, n_in // group, group)
    scale = Rg.abs().amax(dim=2).clamp(min=1e-8) / 7.0
    q = torch.round(R_f32 / scale.repeat_interleave(group, dim=1)).clamp(-7, 7).to(torch.int8)
    qu = (q.to(torch.int16) & 0xF).to(torch.uint8)
    lo = qu[:, 0::2]; hi = qu[:, 1::2]
    return (lo | (hi << 4)).contiguous(), scale.to(torch.bfloat16).contiguous()


# Backward-compat alias (older calls)
def rtn_g16_pack(R_f32):
    return rtn_pack(R_f32, group=16)


def svd_split(W_bf16: torch.Tensor, rank: int):
    """W bf16 [out, in] -> (lora_up bf16[out, r], lora_down bf16[r, in], residual f32[out, in])."""
    W = W_bf16.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    lora_up = (U[:, :r] * S[:r]).contiguous()
    lora_dn = Vh[:r, :].contiguous()
    R = W - lora_up @ lora_dn
    return lora_up.to(torch.bfloat16), lora_dn.to(torch.bfloat16), R


def smooth_lambda(W_bf16: torch.Tensor, amax: torch.Tensor, alpha: float = 0.5,
                  clip: float = 1e3) -> torch.Tensor:
    """SmoothQuant lambda[in] = amax^a / wmax^(1-a). lam>1 migrates activation outliers INTO
    weights (kernel-side x/lam shrinks them). lam<1 amplifies low-activation channels which
    qimg's int4 SVDQuant TOLERATES (the rank-r LoRA absorbs the outliers); see qimg's
    [[project_qimg_int4_from_bf16]] note (lam>=1 is for int8, not int4)."""
    W = W_bf16.float()
    wmax = W.abs().amax(0).clamp(min=1e-8)
    am = amax.clamp(min=1e-8)
    lam = (am.pow(alpha) / wmax.pow(1.0 - alpha)).clamp(1.0 / clip, clip)
    return lam


def load_calib(path: str | None):
    """Load per-linear amax dump and merge linear2 halves into a single concat. Returns
    {linear_name: amax F32 [in]} keyed by the SVDQ-quant linear name (NOT linear2_attn/mlp)."""
    if not path: return {}
    out = {}
    f = safe_open(path, "pt", "cpu")
    # collect halves to merge
    halves = {}    # base -> {'attn': tensor, 'mlp': tensor}
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (default 32 = same as the Nunchaku-FLUX2-klein-9B checkpoint)")
    ap.add_argument("--calib", default=None, help="Activation amax safetensors from FLUX2_CALIB_DUMP. Enables SmoothQuant lambda.")
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha (default 0.5)")
    ap.add_argument("--clip", type=float, default=4.0,
                    help="SmoothQuant lambda clip [1/clip, clip] (default 4.0; raise to 1e3 for g64 + smooth)")
    ap.add_argument("--group-size", type=int, default=None, choices=[16, 64],
                    help="Int4 group size. Default: 64 when --calib (g64+smooth-aware kernel), else 16.")
    a = ap.parse_args()
    if a.group_size is None:
        a.group_size = 64 if a.calib else 16
    calib = load_calib(a.calib)
    if a.calib:
        print(f"calib: {len(calib)} amax entries from {a.calib} (alpha={a.alpha})", file=sys.stderr)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())
    n_dbl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")), default=-1)
    n_sgl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.")), default=-1)
    print(f"svdq-int4-g16 r={a.rank}: {n_dbl} double_blocks, {n_sgl} single_blocks", file=sys.stderr)

    out = {}; quant_keys = set()
    t0 = time.time(); n_done = 0

    def quant_one(prefix: str, suf: str):
        nonlocal n_done
        k = f"{prefix}.{suf}.weight"
        if k not in keys:
            print(f"[warn] missing {k}", file=sys.stderr); return
        W = src.get_tensor(k)
        name = f"{prefix}.{suf}"
        amax = calib.get(name)
        lam = None
        if amax is not None:
            lam = smooth_lambda(W, amax, a.alpha, a.clip)          # [in] F32
            W_eff = (W.float() * lam.unsqueeze(0)).to(torch.bfloat16)
            out[f"{name}.smooth"] = lam.to(torch.bfloat16)
        else:
            if calib:                                              # had calib but missing this key
                print(f"[warn] no calib for {name} -> lam=1", file=sys.stderr)
            W_eff = W
        lu, ld, R = svd_split(W_eff, a.rank)
        if lam is not None and a.group_size == 64:
            # Pre-divide lora_down by smooth so the runtime LoRA branch can use raw X:
            #   lora_down_eff @ x = (lora_down / lam) @ x = lora_down @ (x/lam) = lora_down @ x_smooth
            ld = (ld.float() / lam.unsqueeze(0)).to(torch.bfloat16)
        q, s = rtn_pack(R, group=a.group_size)
        bk = f"{name}.bias"
        out[f"{name}.qint4"]     = q
        out[f"{name}.wscale"]    = s
        out[f"{name}.lora_up"]   = lu
        out[f"{name}.lora_down"] = ld
        if bk in keys:
            out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)
        quant_keys.add(k)
        n_done += 1
        if n_done % 10 == 0 or n_done == 80:
            tag = "smooth" if amax is not None else "no-smooth"
            print(f"  [{n_done}/80] {name}  W{tuple(W.shape)} {tag}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    for bi in range(n_dbl):
        for suf in MAIN_DBLK: quant_one(f"double_blocks.{bi}", suf)
    for bi in range(n_sgl):
        for suf in MAIN_SBLK: quant_one(f"single_blocks.{bi}", suf)

    print(f"quantized {n_done} main linears in {time.time()-t0:.1f}s", file=sys.stderr)

    # passthrough non-quantized tensors (globals/norms/biases of non-main linears)
    drop = {f"{k.rsplit('.weight',1)[0]}.weight_scale" for k in quant_keys if k.endswith(".weight")}
    n_pass = 0
    for k in keys:
        if k in quant_keys or k in drop: continue
        out[k] = src.get_tensor(k); n_pass += 1

    save_file(out, a.out, metadata={
        "format": "flux2-klein-int4-svdq-rtn-g16",
        "scheme": f"svd_lora_r{a.rank}_plus_rtn_g16",
        "rank": str(a.rank),
    })
    sz = os.path.getsize(a.out) / 1e9
    print(f"wrote {a.out}  ({sz:.2f} GB, {len(out)} tensors, {n_pass} passthrough)", file=sys.stderr)


if __name__ == "__main__":
    main()
