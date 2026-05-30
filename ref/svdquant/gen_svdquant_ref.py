#!/usr/bin/env python3
r"""gen_svdquant_ref.py — self-contained SVDQuant reference generator.

Produces a deterministic, model-free ground-truth dump that pins the SVDQuant
forward for the cpu/svdquant and cuda/svdquant unit tests. One synthetic Linear
(OUT x IN) with seeded random weight/activation is quantized four ways:

    A  int4_w4a16   group-64 signed[-7,7] residual,   activation f32
    B  int4_w4a4    group-64 signed[-7,7] residual,   activation int4 g64
    C  nvfp4_w4a16  e2m1 g16 + e4m3 scales residual,  activation f32
    D  nvfp4_w4a4   e2m1 g16 + e4m3 scales residual,  activation e2m1 g16

The SVDQuant decomposition (SmoothQuant lambda + rank-128 SVD low-rank branch)
is SHARED across all four cases; only the residual quantizer and the activation
path differ. The forward every implementation must reproduce is

    y = act(x / lam) @ R_dec^T  +  (x @ lora_down_emit^T) @ lora_up^T  +  bias

with the residual term using x/lam (smoothed) and the low-rank term using the
RAW x; lora_down_emit = lora_down / lam folds the smoothing back in so the two
branches recombine to W@x (see the cancellation in the module docstring of
rdna4/qimg/tools/svdquant_from_bf16.py).

Everything is dumped as .npy in {float32, int32, uint8} ONLY, because the C
loader common/npy_io.h reads exactly f4/i4/u1. y_svdq is computed by DECODING
the dumped quantized tensors (not the pre-quant floats), so a C/CUDA reader of
the same bytes with the same decode formula matches to f32 rounding.

The math helpers are vendored (with credit) from in-repo references so this file
has no dependency on those modules or on deepcompressor:
  - rdna4/qimg/tools/svdquant_from_bf16.py : smoothing_lambda, svd_lowrank_branch,
                                             quant_int4_g64, pack_nibbles
  - nunchaku_fp4_repack_omma.py            : quant_nvfp4, pack_codes_u32,
                                             _E2M1_THR, _decode_nvfp4

Usage:
    python3 gen_svdquant_ref.py --out dumps                 # synthetic (default)
    python3 gen_svdquant_ref.py --out dumps --no-smooth     # lambda = 1 baseline
    python3 gen_svdquant_ref.py --out dumps --cross-check   # optional deepcompressor check
    python3 gen_svdquant_ref.py --out dumps --real <ckpt> --layer <dotted>
"""
import argparse, json, hashlib, os, sys
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Vendored INT4 SVDQuant math (credit: rdna4/qimg/tools/svdquant_from_bf16.py)
# ---------------------------------------------------------------------------
def pack_nibbles(q_int8):                       # [out,in] in [-7,7] -> u8 [out,in/2]
    u = (q_int8.to(torch.int16) & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def quant_int4_g64(R, group=64):                # R [out,in] -> (qint4 u8[out,in/2], wscale [out,in/g])
    out, k = R.shape
    g = R.reshape(out, k // group, group)
    scale = (g.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-7, 7).to(torch.int8).reshape(out, k)
    return pack_nibbles(q), scale.reshape(out, k // group)


def smoothing_lambda(W, amax, alpha, clip):     # W[out,in], amax[in] or None -> lambda[in]
    wmax = W.abs().amax(dim=0).clamp(min=1e-8)
    if amax is None:
        return torch.ones_like(wmax)
    a = amax.clamp(min=1e-8)
    lam = (a.pow(alpha) / wmax.pow(1.0 - alpha)).clamp(min=1.0 / clip, max=clip)
    return lam


def svd_lowrank_branch(What, rank, oversample=32, niter=6):
    q = min(rank + oversample, min(What.shape))
    U, S, V = torch.svd_lowrank(What, q=q, niter=niter)
    r = min(rank, S.numel())
    lora_up = (U[:, :r] * S[:r]).contiguous()           # [out, r]
    lora_down = V[:, :r].t().contiguous()               # [r, in]
    R = What - lora_up @ lora_down
    return lora_up, lora_down, R, r


# ---------------------------------------------------------------------------
# Vendored NVFP4 quant/decode (credit: nunchaku_fp4_repack_omma.py)
# ---------------------------------------------------------------------------
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6],
                    dtype=torch.float32)
_E2M1_MAG = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_THR = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def pack_codes_u32(codes):                      # [O,I] int32 0..15 -> [O,I/8] int32
    O, I = codes.shape
    c = codes.reshape(O, I // 8, 8)
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=codes.device)
    return (c.to(torch.int32) << sh).sum(-1).to(torch.int32)


def quant_nvfp4(W, dev, GS=16):                 # W[O,I] -> (qw i32[O,I/8], ws u8[O,I/GS], wcwt f32[O])
    O, I = W.shape
    W = W.to(dev).float()
    Wg = W.reshape(O, I // GS, GS)
    amax_g = Wg.abs().amax(dim=2)
    wcwt = (amax_g.amax(dim=1) / 6.0).clamp_min(1e-12)
    gscale_rel = (amax_g / 6.0) / wcwt.unsqueeze(1)
    ws_e4m3 = gscale_rel.to(torch.float8_e4m3fn)
    eff = (ws_e4m3.float() * wcwt.unsqueeze(1)).clamp_min(1e-12)
    Wn = (Wg / eff.unsqueeze(2))
    mag = Wn.abs()
    thr = _E2M1_THR.to(dev)
    midx = torch.bucketize(mag, thr).to(torch.int32)
    codes = (midx + (Wn < 0).to(torch.int32) * 8).reshape(O, I)
    qw = pack_codes_u32(codes)
    return qw.contiguous().cpu(), ws_e4m3.view(torch.uint8).contiguous().cpu(), wcwt.float().cpu()


def decode_nvfp4(qw, ws_u8, wcwt, O, I, dev, GS=16):    # inverse of quant_nvfp4 -> R[O,I] f32
    e2 = E2M1.to(dev)
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)
    codes = ((qw.to(dev).view(O, I // 8, 1) >> sh) & 0xF).reshape(O, I)
    vals = e2[codes.long()]
    ws = ws_u8.to(dev).view(torch.float8_e4m3fn).float().view(O, I // GS)
    gidx = torch.arange(I, device=dev) // GS
    return vals * ws[:, gidx] * wcwt.to(dev).unsqueeze(1)


# ---------------------------------------------------------------------------
# Decode + activation-quant helpers (mirror the C/CUDA side exactly)
# ---------------------------------------------------------------------------
def decode_int4_residual(qint4, wscale, out, inn, group=64):
    """Signed [-8,7] nibble decode x per-group-64 scale -> R[out,in] f32.
    Matches sq_unpack_int4_residual in cpu/svdquant/svdquant_cpu.h."""
    lo = (qint4 & 0xF).to(torch.int16); lo -= 16 * (lo >= 8)
    hi = (qint4 >> 4).to(torch.int16); hi -= 16 * (hi >= 8)
    nib = torch.empty(out, inn, dtype=torch.float32)
    nib[:, 0::2] = lo.float(); nib[:, 1::2] = hi.float()
    return nib * wscale.float().repeat_interleave(group, dim=1)


def quant_act_int4(xr, group=64):
    """Per-token per-group-64 symmetric int4 quant -> dequantized xr_dq[tok,in] f32.
    torch.round is round-half-to-even; the C side uses rintf (same mode)."""
    tok, inn = xr.shape
    g = xr.reshape(tok, inn // group, group)
    scale = (g.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-7, 7)
    return (q * scale).reshape(tok, inn)


def quant_act_nvfp4(xr, GS=16):
    """Per-token per-group-16 NVFP4 activation quant -> dequantized xr_dq[tok,in] f32.
    Mirrors fp4_quant_act in cuda/fp4_w4a4.h (amax/6 -> e4m3 scale -> nearest e2m1)."""
    tok, inn = xr.shape
    g = xr.reshape(tok, inn // GS, GS)
    amax = g.abs().amax(dim=2)                                  # [tok, ng]
    sc = (amax / 6.0).to(torch.float8_e4m3fn).float()          # e4m3-quantized scale
    inv = torch.where(sc > 0, 1.0 / sc, torch.zeros_like(sc))
    Wn = g * inv.unsqueeze(2)
    midx = torch.bucketize(Wn.abs(), _E2M1_THR)
    val = torch.sign(Wn) * _E2M1_MAG[midx] * sc.unsqueeze(2)
    return val.reshape(tok, inn)


# ---------------------------------------------------------------------------
# Dump helper (.npy + manifest.json; f32/i32/u8 only — npy_io.h contract)
# ---------------------------------------------------------------------------
class Dumper:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.manifest = {}

    def __call__(self, name, t):
        if torch.is_tensor(t):
            if t.dtype in (torch.bfloat16, torch.float16, torch.float64):
                t = t.float()
            arr = t.detach().cpu().contiguous().numpy()
        else:
            arr = np.ascontiguousarray(t)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        assert arr.dtype in (np.float32, np.int32, np.uint8), f"{name}: bad dtype {arr.dtype}"
        np.save(os.path.join(self.outdir, name + ".npy"), arr)
        self.manifest[name] = {
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "mean": float(arr.mean()) if arr.size else 0.0,
        }

    def write_manifest(self, extra=None):
        m = {"_meta": extra or {}, **self.manifest}
        with open(os.path.join(self.outdir, "manifest.json"), "w") as f:
            json.dump(m, f, indent=1)


# ---------------------------------------------------------------------------
def f64gemm(A, B):
    """A[m,k] @ B[k,n] in float64, return float32. (B already transposed by caller.)"""
    return (A.double() @ B.double()).float()


def forward_svdq(x_act, x_raw, R_dec, lora_up, lora_down_emit, bias):
    """y = x_act @ R_dec^T + (x_raw @ ld_emit^T) @ lora_up^T + bias  (f64 accum)."""
    resid = f64gemm(x_act, R_dec.t())                          # [tok,out]
    la = f64gemm(x_raw, lora_down_emit.t())                    # [tok,rank]
    lo = f64gemm(la, lora_up.t())                              # [tok,out]
    return resid + lo + bias.unsqueeze(0)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(),
                                                 b.flatten().double(), dim=0).item()


def rel_l2(a, b):
    return ((a - b).double().norm() / (b.double().norm() + 1e-30)).item()


# ---------------------------------------------------------------------------
def _nunchaku_weight_perm(out, K):
    """Inverse of Nunchaku pack_weight byte layout (int4, warp_n=128); credit
    rdna4/qimg/tools/nunchaku_int4_decode.py."""
    return (torch.arange(out * K).reshape(out, K)
            .reshape(out // 128, 8, 2, 8, 1, K // 64, 1, 2, 4, 8)
            .permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous().reshape(-1).numpy())


def _nunchaku_scale_outperm(n):
    return (torch.arange(n).reshape(n // 128, 1, 8, 2, 4, 2)
            .permute(0, 1, 2, 4, 3, 5).reshape(-1).numpy())


def decode_nunchaku_dump(path):
    """Derive a real DiT-magnitude dense [out,in] weight + real activation from a
    Nunchaku SVDQuant ground-truth dump (e.g. nunchaku_ref_dump.*.safetensors).

    The rank-128 low-rank is decoded exactly via the installed Nunchaku packer
    (`unpack_lowrank_weight`); the int4-main residual is de-swizzled with the
    in-repo perms. NOTE: the int4-main byte-swizzle is Nunchaku-version-specific,
    so the reconstructed Ŵ may not bit-match the original layer — we report the
    Ŵ@(x/smooth)+b vs real-y fidelity so the caller sees the decode quality. For
    a unit test the point is a realistic, non-Gaussian [out,in] weight (real
    per-channel outliers) + real activation, not an exact layer recovery."""
    from safetensors import safe_open
    h = safe_open(path, "pt", "cpu"); g = lambda n: h.get_tensor(n)
    qw = g("weight.qweight").numpy().astype(np.uint8)            # [out, in/2]
    wsc = g("weight.wscales").float().numpy()                    # [in/64, out]
    real_x = g("x").float()                                      # [N, in]
    out, Khalf = qw.shape; K = Khalf * 2
    bf = qw.ravel()
    nib = np.empty(out * K, np.int16); nib[0::2] = bf & 0xF; nib[1::2] = bf >> 4; nib -= 16 * (nib >= 8)
    nibbles = np.empty(out * K, np.float32)
    nibbles[_nunchaku_weight_perm(out, K)] = nib.astype(np.float32)
    main = nibbles.reshape(out, K) * np.repeat(wsc[:, _nunchaku_scale_outperm(out)].T, 64, axis=1)

    lowrank = np.zeros((out, K), np.float32)
    rel_y = None
    try:
        from nunchaku.lora.flux.packer import NunchakuWeightPacker
        pk = NunchakuWeightPacker(bits=4, warp_n=128)
        pu = pk.unpack_lowrank_weight(g("weight.proj_up").to(torch.bfloat16), down=False).float().numpy()
        pd = pk.unpack_lowrank_weight(g("weight.proj_down").to(torch.bfloat16), down=True).float().numpy()
        lowrank = pu @ pd
        try:                                                    # fidelity vs the operational oracle y
            smf = g("weight.smooth_factor").float().numpy(); bias = g("weight.bias").float().numpy()
            y = g("y").float().numpy(); xs = real_x.numpy() / smf[None, :]
            rel_y = float(np.linalg.norm((xs @ (main + lowrank).T + bias) - y) / np.linalg.norm(y))
        except Exception:
            pass
    except Exception as e:
        print(f"[real-nunchaku] Nunchaku packer unavailable ({str(e).splitlines()[-1][:60]}); "
              f"using int4-main only", file=sys.stderr)

    W = torch.from_numpy((main + lowrank).astype(np.float32))
    msg = f"[real-nunchaku] W[{out},{K}] from {os.path.basename(path)}"
    if rel_y is not None:
        msg += (f"  (Ŵ@(x/smooth)+b vs real y: rel_L2={rel_y:.3f} — int4-main swizzle is "
                f"nunchaku-version-specific; low-rank is exact)")
    print(msg, file=sys.stderr)
    return W, real_x


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dumps", help="output directory")
    ap.add_argument("--out-features", type=int, default=256)
    ap.add_argument("--in-features", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1e3)
    ap.add_argument("--no-smooth", action="store_true", help="force lambda = 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="torch device for quant (cpu/cuda)")
    ap.add_argument("--cross-check", action="store_true",
                    help="optionally cross-check the low-rank+4bit decomposition vs deepcompressor")
    ap.add_argument("--real", default=None, help="safetensors path to pull a real dense [out,in] weight from")
    ap.add_argument("--layer", default="transformer_blocks.0.attn.to_out.0.weight",
                    help="tensor key inside --real safetensors (a [out,in] weight)")
    ap.add_argument("--real-nunchaku", default=None, dest="real_nunchaku",
                    help="Nunchaku SVDQuant dump (weight.qweight/proj_*/x/y) -> derive a real "
                         "DiT-magnitude [out,in] weight + real activation")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = a.device

    OUT, IN, TOK, R = a.out_features, a.in_features, a.tokens, a.rank
    assert IN % 64 == 0, "IN must be %64 (int4 g64 and nvfp4 BK=64)"
    assert OUT % 8 == 0 and TOK % 16 == 0, "OUT%8, TOK%16 for the naive-GEMM tiles"

    # ---- inputs (synthetic, seeded) or a real layer weight + activation ----
    real_x = None
    if a.real_nunchaku:
        W, real_x = decode_nunchaku_dump(a.real_nunchaku)
        OUT, IN = W.shape
    elif a.real:
        from safetensors import safe_open
        with safe_open(a.real, "pt", "cpu") as f:
            W = f.get_tensor(a.layer).float()
        OUT, IN = W.shape
        if IN % 64:                                    # pad IN to a multiple of 64
            W = torch.nn.functional.pad(W, (0, 64 - IN % 64)); IN = W.shape[1]
        print(f"[real] {a.layer}: W[{OUT},{IN}] from {a.real}", file=sys.stderr)
    else:
        W = (torch.randn(OUT, IN) * 0.02)
    assert IN % 64 == 0 and OUT % 8 == 0, f"need IN%64==0, OUT%8==0 (got {OUT}x{IN})"
    if real_x is not None:
        if real_x.shape[0] < TOK:
            TOK = (real_x.shape[0] // 16) * 16         # clamp to available rows, keep %16
        x = real_x[:TOK].float().contiguous()
    else:
        x = torch.randn(TOK, IN)
    bias = (torch.randn(OUT) * 0.1)
    W, x, bias = W.float(), x.float(), bias.float()

    # synthetic per-input activation max-abs (the calibration our own probe gives)
    amax = None if a.no_smooth else x.abs().amax(dim=0)
    lam = smoothing_lambda(W, amax, a.alpha, a.clip).float()    # [IN]

    # ---- shared SVDQuant decomposition (smooth + rank-R SVD low-rank) ----
    What = (W * lam.unsqueeze(0)).float()
    lora_up, lora_down, R_res, r = svd_lowrank_branch(What.to(dev), R)
    lora_up = lora_up.float().cpu(); lora_down = lora_down.float().cpu()
    R_res = R_res.float().cpu()
    lora_down_emit = (lora_down / lam.unsqueeze(0)).float()     # fold smoothing into low-rank

    print(f"shape OUT={OUT} IN={IN} TOK={TOK} rank={r} smooth={'off' if a.no_smooth else 'on'} "
          f"lam[min={lam.min():.3f} max={lam.max():.3f}]", file=sys.stderr)

    d = Dumper(a.out)
    # ---- shared inputs + full-precision oracle ----
    d("x", x); d("W", W); d("bias", bias)
    d("dims", torch.tensor([OUT, IN, TOK, r], dtype=torch.int32))
    y_fp = (x.double() @ W.double().t() + bias.double()).float()
    d("y_fp", y_fp)

    # ---- per-format quantized residual + decode ----
    qint4, wscale = quant_int4_g64(R_res, 64)
    wscale_f32 = wscale.float()
    R_int4 = decode_int4_residual(qint4, wscale_f32, OUT, IN, 64)

    qw, ws, wcwt = quant_nvfp4(R_res, dev, GS=16)
    R_nvfp4 = decode_nvfp4(qw, ws, wcwt, OUT, IN, dev, GS=16).float().cpu()

    summary = []

    def emit(prefix, R_dec, scope, fmt):
        """Dump one case and report its accuracy vs the f32 oracle."""
        # per-case shared tensors (duplicated for a uniform C loader)
        d(prefix + "_smooth", lam)
        d(prefix + "_lora_up", lora_up)
        d(prefix + "_lora_down", lora_down_emit)
        if fmt == "int4":
            d(prefix + "_qint4", qint4); d(prefix + "_wscale", wscale_f32)
        else:
            d(prefix + "_qw", qw.to(torch.int32)); d(prefix + "_ws", ws); d(prefix + "_wcwt", wcwt)

        xr = x / lam.unsqueeze(0)
        if scope == "w4a16":
            x_act = xr
        elif fmt == "int4":
            x_act = quant_act_int4(xr, 64)
            d(prefix + "_xr_dq", x_act)
        else:
            x_act = quant_act_nvfp4(xr, 16)
            d(prefix + "_xr_dq", x_act)

        y_svdq = forward_svdq(x_act, x, R_dec, lora_up, lora_down_emit, bias)
        d(prefix + "_y_svdq", y_svdq)
        c = cosine(y_svdq, y_fp); rl = rel_l2(y_svdq, y_fp)
        summary.append((prefix, c, rl))
        print(f"  {prefix:14s} cos(y_svdq,y_fp)={c:.5f}  rel_L2(y_svdq,y_fp)={rl:.4f}", file=sys.stderr)

    emit("int4_w4a16", R_int4, "w4a16", "int4")
    emit("int4_w4a4", R_int4, "w4a4", "int4")
    emit("nvfp4_w4a16", R_nvfp4, "w4a16", "nvfp4")
    emit("nvfp4_w4a4", R_nvfp4, "w4a4", "nvfp4")

    d.write_manifest(extra={"out": OUT, "in": IN, "tokens": TOK, "rank": r,
                            "smooth": (not a.no_smooth), "seed": a.seed,
                            "cases": [s[0] for s in summary]})
    print(f"wrote {len(d.manifest)} tensors + manifest.json -> {a.out}", file=sys.stderr)

    # sanity: every case must track the full-precision output well
    worst = min(c for _, c, _ in summary)
    if worst < 0.99:
        print(f"[warn] worst cosine {worst:.5f} < 0.99 — decomposition may be off", file=sys.stderr)

    if a.cross_check:
        cross_check(What, lora_up, lora_down, r)


def cross_check(What, lora_up, lora_down, rank):
    """Drive deepcompressor's authoritative SVD low-rank branch and diff it against ours.

    deepcompressor's `LowRankBranch` factorizes the (smoothed) weight with EXACT
    `torch.linalg.svd`; ours uses randomized `torch.svd_lowrank`. We compare the
    residual energy, the effective-weight agreement, and a forward pass. The 4-bit
    RTN residual quant (deepcompressor's `simple_quantize`) is attempted too, but it
    pulls in a CUDA C-extension build — skipped gracefully if that fails. Never on
    the default path; graceful skip if deepcompressor isn't importable."""
    dc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepcompressor")
    if dc_dir not in sys.path:
        sys.path.insert(0, dc_dir)
    try:
        from deepcompressor.nn.patch.lowrank import LowRankBranch
    except Exception as e:
        print(f"[cross-check] deepcompressor.LowRankBranch not importable "
              f"({str(e).splitlines()[-1][:80]}); skipping", file=sys.stderr)
        return
    out, inn = What.shape
    branch = LowRankBranch(inn, out, rank, weight=What.float())       # exact torch.linalg.svd
    eff_dc = branch.get_effective_weight().detach().float()           # b@a = lu@ld  [out,in]
    eff_ours = (lora_up @ lora_down).float()
    res_dc = ((What - eff_dc).norm() / What.norm()).item()
    res_ours = ((What - eff_ours).norm() / What.norm()).item()
    agree = ((eff_dc - eff_ours).norm() / (eff_dc.norm() + 1e-30)).item()
    xp = torch.randn(8, inn)
    yb = branch(xp.float())
    fwd = ((((xp @ lora_down.t()) @ lora_up.t()) - yb).norm() / (yb.norm() + 1e-30)).item()
    print(f"[cross-check] low-rank vs deepcompressor LowRankBranch (exact SVD, rank={rank}):", file=sys.stderr)
    print(f"  residual ||What-lu@ld||/||What||: ours(svd_lowrank)={res_ours:.4e}  "
          f"deepcompressor(exact)={res_dc:.4e}  (exact is the optimal lower bound)", file=sys.stderr)
    print(f"  effective-weight agreement ||eff_dc-eff_ours||/||eff_dc|| = {agree:.4e}", file=sys.stderr)
    print(f"  low-rank forward agreement rel_L2 = {fwd:.4e}", file=sys.stderr)
    try:
        from deepcompressor.quantizer.impl.simple import simple_quantize  # noqa: F401
        print("  4-bit RTN cross-check: simple_quantize importable (available)", file=sys.stderr)
    except Exception as e:
        print(f"  4-bit RTN cross-check: skipped — simple_quantize needs the CUDA C-ext build "
              f"({str(e).splitlines()[-1][:60]})", file=sys.stderr)


if __name__ == "__main__":
    main()
