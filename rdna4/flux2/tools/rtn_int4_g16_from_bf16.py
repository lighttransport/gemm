#!/usr/bin/env python3
"""rtn_int4_g16_from_bf16.py — simple RTN int4-g16 quantization of FLUX.2 Klein 4B.

The 4B bf16 single-file (black-forest-labs/FLUX.2-klein-4B, flux-2-klein-4b.safetensors)
has 80 "main" linears: 5 double_blocks × {img,txt}_{attn.qkv,attn.proj,mlp.0,mlp.2}
plus 20 single_blocks × {linear1,linear2}. We quantize those 80 to RTN int4-g16 (no
SVDQuant LoRA, no smooth, no calibration) — the minimal-risk int4 path matching qimg's
shipped gemm_int4w_g16_bf16a_wmma_t kernel (renders cleanly). Globals/norms pass through
as bf16. Output safetensors stays in the official FLUX.2 single-file naming so the
existing runner can adopt it with only a load+dispatch addition.

Per quantized linear (replaces `<key>.weight` BF16 [out, in]):
    <key>.qint4   u8   [out, in/2]     two nibbles per byte (kp even -> low, odd -> high)
    <key>.wscale  bf16 [out, in/16]    per-(out, group-of-16) scale (max|W_g|/7)
    <key>.bias    bf16 [out]           passthrough (if present)
The matching `<key>.weight_scale` (ModelOpt artifact) is dropped.

CLI:
    python rtn_int4_g16_from_bf16.py \\
        --bf16 /mnt/disk1/models/klein2-4b/bf16/flux-2-klein-4b.safetensors \\
        --out  /mnt/disk1/models/klein2-4b/int4/flux-2-klein-4b-int4-g16.safetensors
"""
import argparse, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_DBLK = ["img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
             "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2"]
MAIN_SBLK = ["linear1", "linear2"]


def quant_rtn_g16(W_bf16: torch.Tensor):
    """W bf16 [out, in] -> (qint4 u8[out, in/2], wscale bf16[out, in/16]).

    Per-group scale = max|W_g| / 7; q = clip(round(W/scale), -7, 7) (matches the kernel's
    signed nibble decode nib>=8 ? nib-16 : nib, restricted to [-7,7] for symmetric range).
    """
    out, n_in = W_bf16.shape
    assert n_in % 16 == 0, f"n_in={n_in} must be %16==0"
    W = W_bf16.float()
    Wg = W.view(out, n_in // 16, 16)
    scale = Wg.abs().amax(dim=2).clamp(min=1e-8) / 7.0          # [out, n_in/16]
    q = torch.round(W / scale.repeat_interleave(16, dim=1)).clamp(-7, 7).to(torch.int8)
    # pack nibbles: even col -> low, odd col -> high
    qu = (q.to(torch.int16) & 0xF).to(torch.uint8)              # 0..15 representation
    lo = qu[:, 0::2]                                            # [out, in/2]
    hi = qu[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()                       # u8[out, in/2]
    return packed, scale.to(torch.bfloat16).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())

    # detect block counts
    n_dbl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")), default=-1)
    n_sgl = 1 + max((int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.")), default=-1)
    print(f"rtn-int4-g16: {n_dbl} double_blocks, {n_sgl} single_blocks, {len(keys)} src tensors", file=sys.stderr)

    out = {}
    quant_keys = set()
    t0 = time.time()
    nbytes_w = 0; nbytes_q = 0

    # quantize main linears
    for bi in range(n_dbl):
        for suf in MAIN_DBLK:
            k = f"double_blocks.{bi}.{suf}.weight"
            if k not in keys:
                print(f"[warn] missing {k}", file=sys.stderr); continue
            W = src.get_tensor(k)
            q, s = quant_rtn_g16(W)
            out[f"double_blocks.{bi}.{suf}.qint4"]  = q
            out[f"double_blocks.{bi}.{suf}.wscale"] = s
            nbytes_w += W.numel() * 2; nbytes_q += q.numel() + s.numel() * 2
            quant_keys.add(k)
            bk = f"double_blocks.{bi}.{suf}.bias"
            if bk in keys: out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)
    for bi in range(n_sgl):
        for suf in MAIN_SBLK:
            k = f"single_blocks.{bi}.{suf}.weight"
            if k not in keys:
                print(f"[warn] missing {k}", file=sys.stderr); continue
            W = src.get_tensor(k)
            q, s = quant_rtn_g16(W)
            out[f"single_blocks.{bi}.{suf}.qint4"]  = q
            out[f"single_blocks.{bi}.{suf}.wscale"] = s
            nbytes_w += W.numel() * 2; nbytes_q += q.numel() + s.numel() * 2
            quant_keys.add(k)
            bk = f"single_blocks.{bi}.{suf}.bias"
            if bk in keys: out[bk] = src.get_tensor(bk).to(torch.bfloat16); quant_keys.add(bk)

    print(f"quantized {len([k for k in out if k.endswith('.qint4')])} main linears in {time.time()-t0:.1f}s; "
          f"bf16 main {nbytes_w/1e9:.2f} GB -> int4+scales {nbytes_q/1e9:.2f} GB", file=sys.stderr)

    # passthrough everything else as-is (norms / embedders / modulation / bias / etc.) — these are small
    # and keep the official FLUX.2 single-file naming the runner already reads.
    drop = {f"{k.rsplit('.weight',1)[0]}.weight_scale" for k in quant_keys if k.endswith(".weight")}
    n_pass = 0
    for k in keys:
        if k in quant_keys: continue
        if k in drop:       continue        # ModelOpt artifact, irrelevant for bf16 source
        out[k] = src.get_tensor(k)
        n_pass += 1
    print(f"passthrough: {n_pass} tensors  ->  total out: {len(out)}", file=sys.stderr)

    save_file(out, a.out, metadata={"format": "flux2-klein-int4-rtn-g16", "scheme": "rtn_g16_signed"})
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e9:.2f} GB)", file=sys.stderr)


if __name__ == "__main__":
    main()
