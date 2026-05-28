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


def rtn_g16_pack(R_f32: torch.Tensor):
    out, n_in = R_f32.shape
    assert n_in % 16 == 0
    Rg = R_f32.view(out, n_in // 16, 16)
    scale = Rg.abs().amax(dim=2).clamp(min=1e-8) / 7.0
    q = torch.round(R_f32 / scale.repeat_interleave(16, dim=1)).clamp(-7, 7).to(torch.int8)
    qu = (q.to(torch.int16) & 0xF).to(torch.uint8)
    lo = qu[:, 0::2]; hi = qu[:, 1::2]
    return (lo | (hi << 4)).contiguous(), scale.to(torch.bfloat16).contiguous()


def svd_split(W_bf16: torch.Tensor, rank: int):
    """W bf16 [out, in] -> (lora_up bf16[out, r], lora_down bf16[r, in], residual f32[out, in]).

    F32 SVD on the bf16 weight (low-rank branch keeps the principal singular values; residual
    has flatter spectrum -> RTN g16 sees less per-group dynamic range -> lower error).
    """
    W = W_bf16.float()
    # full_matrices=False gives U[m, k], S[k], Vh[k, n] with k=min(m,n) — we slice top-r.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    lora_up = (U[:, :r] * S[:r]).contiguous()           # [out, r]
    lora_dn = Vh[:r, :].contiguous()                    # [r, in]
    R = W - lora_up @ lora_dn                           # f32 residual
    return lora_up.to(torch.bfloat16), lora_dn.to(torch.bfloat16), R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (default 32 = same as the Nunchaku-FLUX2-klein-9B checkpoint)")
    a = ap.parse_args()

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
        lu, ld, R = svd_split(W, a.rank)
        q, s = rtn_g16_pack(R)
        bk = f"{prefix}.{suf}.bias"
        out[f"{prefix}.{suf}.qint4"]     = q
        out[f"{prefix}.{suf}.wscale"]    = s
        out[f"{prefix}.{suf}.lora_up"]   = lu
        out[f"{prefix}.{suf}.lora_down"] = ld
        if bk in keys:
            out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)
        quant_keys.add(k)
        n_done += 1
        if n_done % 10 == 0 or n_done == 80:
            print(f"  [{n_done}/80] {prefix}.{suf}  W{tuple(W.shape)}  ({time.time()-t0:.1f}s)", file=sys.stderr)

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
