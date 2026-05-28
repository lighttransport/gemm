#!/usr/bin/env python3
"""svdquant_from_bf16.py — port of nunchaku/deepcompressor SVDQuant (W4A16), applied directly to the
bf16 Qwen-Image DiT, emitting the RDNA4 "logical" INT4 layout that hip_qimg_load_dit_int4 already consumes.

For each of the 12 per-block main linears, given the bf16 weight W [out,in] and (optionally) the per-input
activation max-abs `amax[in]` collected by the runner's QIMG_CALIB_DUMP calibration pass, we do:

  1. Smoothing (SmoothQuant/SVDQuant outlier migration, alpha=0.5):
        lambda_j = amax_j^alpha / max_out|W_:,j|^(1-alpha)      (lambda=1 when --no-smooth / no calib)
        What = W * diag(lambda)                                 (so the kernel computes What @ (x/lambda) == W@x)
  2. Low-rank branch via SVD of the *smoothed* weight What = U S V^T (rank r, default 128):
        lora_up = U[:, :r] * S[:r]   [out, r]   (bf16)
        lora_down = V[:, :r]^T       [r, in]    (bf16)
        R = What - lora_up @ lora_down
  3. INT4-g64 symmetric quant of the residual R (matches the validated kernel nibble decode [-8,7]):
        per (out, group-of-64): scale = max|R_g| / 7 ; q = clip(round(R/scale), -7, 7)
        qint4 [out, in/2] u8 (even col -> low nibble, odd -> high) ; wscale [out, in/64] bf16

The emitted per-linear bundle (loader<->kernel contract; see hip_qimg_runner.c:627 qimg_upload_int4_linear):
    <key>.qint4 u8[out,in/2]  .wscale bf16[out,in/64]  .smooth f32[in]  .lora_down bf16[r,in]
    <key>.lora_up bf16[out,r] .bias f32[out]
Modulation (img_mod.1/txt_mod.1) is weight-only RTN g64 (rank 0, no smooth/lora). BF16 globals/norms pass through.
"""
import argparse, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

# The 12 SVDQuant main linears per block (order/keys must match qimg_int4_linear_suffix in the runner).
MAIN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
        "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2"]
MOD = ["img_mod.1", "txt_mod.1"]                              # weight-only RTN g64 (rank 0)
NORMS = ["attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k"]  # bf16 passthrough (.weight)
GLOBALS = ["img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias", "txt_norm.weight",
           "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias",
           "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias",
           "norm_out.linear.weight", "norm_out.linear.bias", "proj_out.weight", "proj_out.bias"]


def pack_nibbles(q_int8):                                     # q_int8 [out,in] in [-7,7] -> u8 [out,in/2]
    u = (q_int8.to(torch.int16) & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def quant_int4_g64(R, group=64):                             # R [out,in] f32 -> (qint4 u8[out,in/2], wscale bf16[out,in/g])
    out, k = R.shape
    g = R.reshape(out, k // group, group)
    scale = (g.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-12)         # [out, k/g, 1]
    q = torch.round(g / scale).clamp(-7, 7).to(torch.int8).reshape(out, k)
    return pack_nibbles(q), scale.reshape(out, k // group).to(torch.bfloat16)


def smoothing_lambda(W, amax, alpha, clip):                  # W [out,in] f32, amax [in] f32 or None -> lambda [in] f32
    wmax = W.abs().amax(dim=0).clamp(min=1e-8)                                  # [in]
    if amax is None:
        return torch.ones_like(wmax)
    a = amax.clamp(min=1e-8)
    # NOTE: unlike int8 (int8_smooth_from_bf16 clamps lam>=1 to avoid an activation-amplification
    # blowup), INT4 keeps the full [1/clip, clip] range: the rank-128 low-rank branch absorbs the
    # migrated outliers, so lam<1 is safe here and tested marginally BETTER (final-latent cos vs fp8
    # 0.936 with lam<1 vs 0.929 with lam>=1). The residual int4 grain is inherent 4-bit precision.
    lam = (a.pow(alpha) / wmax.pow(1.0 - alpha)).clamp(min=1.0 / clip, max=clip)
    return lam


def svd_lowrank_branch(What, rank, oversample=32, niter=6):  # What [out,in] f32 -> lora_up[out,r], lora_down[r,in], R
    q = min(rank + oversample, min(What.shape))
    U, S, V = torch.svd_lowrank(What, q=q, niter=niter)                         # What ~ U diag(S) V^T ; U[out,q] V[in,q]
    r = min(rank, S.numel())
    lora_up = (U[:, :r] * S[:r]).contiguous()                                  # [out, r]
    lora_down = V[:, :r].t().contiguous()                                      # [r, in]
    R = What - lora_up @ lora_down
    return lora_up, lora_down, R, r


def load_calib(paths):                                       # merge K calib safetensors by element-wise max -> {key: amax}
    if not paths:
        return {}
    merged = {}
    for p in paths:
        with safe_open(p, "pt", "cpu") as f:
            for k in f.keys():
                if not k.endswith(".amax"):
                    continue
                t = f.get_tensor(k).float()
                merged[k] = t if k not in merged else torch.maximum(merged[k], t)
    return merged


def reconstruct_and_check(W, lam, lora_up, lora_down_emit, qint4, wscale, group=64, n_probe=8):
    """Cosine vs bf16 W, modeling the RUNTIME exactly: the fused int4 kernel divides the
    activation by `smooth` (x/lam) for the int4 residual, while the LoRA branch consumes the
    RAW activation x with the emitted (pre-divided) lora_down. So lora_down_emit must already
    be lora_down/lam for the two paths to recombine to W@x."""
    out, k = W.shape
    u = (qint4 & 0xF).to(torch.int16); u -= 16 * (u >= 8)
    hi = (qint4 >> 4).to(torch.int16); hi -= 16 * (hi >= 8)
    nib = torch.empty(out, k, dtype=torch.float32); nib[:, 0::2] = u.float(); nib[:, 1::2] = hi.float()
    R = nib * wscale.float().repeat_interleave(group, dim=1)                    # int4 residual of W*diag(lam)
    x = torch.randn(n_probe, k)
    y_ref = x @ W.t()
    y_q = (x / lam) @ R.t() + (x @ lora_down_emit.float().t()) @ lora_up.float().t()
    cos = torch.nn.functional.cosine_similarity(y_q.flatten(), y_ref.flatten(), dim=0).item()
    rel = (y_q - y_ref).norm().item() / (y_ref.norm().item() + 1e-12)
    return cos, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="/mnt/disk1/models/qwen-image/_bf16dl/split_files/diffusion_models/qwen_image_bf16.safetensors")
    ap.add_argument("--calib", nargs="*", default=[], help="QIMG_CALIB_DUMP safetensors (merged by max). Empty => no smoothing.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="all", help="'all' or comma-separated indices (dev subset)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1e3, help="clamp range for smoothing lambda")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--mod-rtn", action="store_true", help="quantize modulation with plain RTN (old behavior) instead of SVDQuant")
    ap.add_argument("--check", action="store_true", help="print per-linear reconstruction cosine")
    a = ap.parse_args()
    torch.manual_seed(0)

    calib = {} if a.no_smooth else load_calib(a.calib)
    if not a.no_smooth and not calib:
        print("[warn] no calibration stats loaded -> smoothing lambda=1 (weight-only SVDQuant)", file=sys.stderr)

    src = safe_open(a.bf16, "pt", "cpu")
    keys = set(src.keys())
    if a.blocks == "all":
        nblk = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("transformer_blocks."))
        blocks = list(range(nblk))
    else:
        blocks = [int(x) for x in a.blocks.split(",")]
    print(f"quantizing {len(blocks)} block(s), rank={a.rank} group={a.group} "
          f"smooth={'off' if a.no_smooth else ('calib' if calib else 'lambda=1')}", file=sys.stderr)

    out_t = {}
    t0 = time.time()
    cos_acc = []
    for bi, b in enumerate(blocks):
        p = f"transformer_blocks.{b}."
        for suf in MAIN:
            W = src.get_tensor(p + suf + ".weight").float()                    # [out,in]
            amax = calib.get(p + suf + ".amax")
            lam = torch.ones(W.shape[1]) if a.no_smooth else smoothing_lambda(W, amax, a.alpha, a.clip)
            What = W * lam.unsqueeze(0)
            lora_up, lora_down, R, r = svd_lowrank_branch(What, a.rank)
            qint4, wscale = quant_int4_g64(R, a.group)
            # The runtime LoRA branch sees the RAW activation x (not x/lam), so fold the
            # smoothing into lora_down: lora_down_emit[r,j] = lora_down[r,j]/lam[j]. With lam=1
            # (--no-smooth) this is a no-op. Matches nunchaku's "unsmoothed" lora_down.
            lora_down_emit = lora_down / lam.unsqueeze(0)
            k = p + suf
            out_t[k + ".qint4"] = qint4
            out_t[k + ".wscale"] = wscale
            out_t[k + ".lora_down"] = lora_down_emit.to(torch.bfloat16)
            out_t[k + ".lora_up"] = lora_up.to(torch.bfloat16)
            out_t[k + ".bias"] = src.get_tensor(p + suf + ".bias").float()
            if not a.no_smooth:
                out_t[k + ".smooth"] = lam.contiguous()
            if a.check:
                cos, rel = reconstruct_and_check(W, lam, lora_up, lora_down_emit, qint4, wscale, a.group)
                cos_acc.append(cos)
                print(f"  b{b:02d} {suf:18s} out={W.shape[0]:5d} in={W.shape[1]:5d} r={r} cos={cos:.5f} rel={rel:.4f}", file=sys.stderr)
        for suf in MOD:                                                        # SVDQuant rank-r (weight-only) — was RTN
            W = src.get_tensor(p + suf + ".weight").float()                    # [6*hidden, hidden]
            if a.mod_rtn:
                qint4, wscale = quant_int4_g64(W, a.group)
                out_t[p + suf + ".qint4"] = qint4; out_t[p + suf + ".wscale"] = wscale
            else:
                # Modulation drives per-token adaLN scale/shift for every token+block; plain RTN
                # injects high-frequency speckle. Give it the rank-r low-rank + INT4-g64 residual
                # treatment (weight-only; no calib stats for the time-embedding input).
                lu, ld, R, _ = svd_lowrank_branch(W, a.rank)
                qint4, wscale = quant_int4_g64(R, a.group)
                out_t[p + suf + ".qint4"] = qint4
                out_t[p + suf + ".wscale"] = wscale
                out_t[p + suf + ".lora_down"] = ld.to(torch.bfloat16)
                out_t[p + suf + ".lora_up"] = lu.to(torch.bfloat16)
            out_t[p + suf + ".bias"] = src.get_tensor(p + suf + ".bias").float()
        for suf in NORMS:                                                      # bf16 passthrough
            out_t[p + suf + ".weight"] = src.get_tensor(p + suf + ".weight").to(torch.bfloat16)
        print(f"  block {b} done ({bi+1}/{len(blocks)}, {time.time()-t0:.0f}s)", file=sys.stderr)

    for g in GLOBALS:                                                          # bf16 passthrough
        if g in keys:
            out_t[g] = src.get_tensor(g).to(torch.bfloat16)

    if cos_acc:
        print(f"[check] mean per-linear cosine over {len(cos_acc)} linears = {np.mean(cos_acc):.5f} "
              f"(min {np.min(cos_acc):.5f})", file=sys.stderr)
    print(f"writing {len(out_t)} tensors -> {a.out}", file=sys.stderr)
    save_file(out_t, a.out, metadata={"format": "rdna4-logical-int4-w4a16", "group_size": str(a.group),
                                      "rank": str(a.rank), "source": "bf16-svdquant", "alpha": str(a.alpha)})
    print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
