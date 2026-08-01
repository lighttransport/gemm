#!/usr/bin/env python3
"""
nunchaku_int4_decode.py — reference decoder for Nunchaku SVDQuant INT4 (Qwen-Image) linear weights.

This is the deterministic *weight-side* decode spec that the RDNA4 Phase-1 (W4A16) loader and the
in-LDS dequant kernel mirror. It assembles a linear's logical representation entirely from pure-Python
generators that were each validated against the checkpoint / Nunchaku source:

  - the int4 `qweight` byte-swizzle  — inverse of Nunchaku's `pack_weight` (bit-exact-proven round-trip);
  - the `wscales` output-axis fragment order — inverse of `pack_scale` (the group axis stays logical-contiguous);
  - the rank-128 low-rank factors — Nunchaku's own `unpack_lowrank_weight`;
  - the per-input smoothing diagonal `smooth_factor` (the runtime divides the activation by it);
  - group_size 64, signed int4 [-8,7].

SVDQuant composition (read from ~/work/nunchaku/src CUDA): the stored int4 main quantizes the SMOOTHED-weight
residual, so the smoothed effective weight is  Ŵ = (deswizzled_int4 × wscale_per(out,input-group)) + proj_up·proj_down.
At inference  y = Ŵ @ (x / smooth_factor) + bias.  (The activation int4 layout — Phase-2/W4A4 only — is NOT needed here.)

Phase-1 W4A16 keeps the int4 weights packed in VRAM and dequantizes per-group to BF16 in-kernel; this reference
computes the equivalent on the host as the trusted spec / correctness oracle.
"""
import numpy as np, torch, torch.nn.functional as Fn
from safetensors import safe_open

# Nunchaku packer constants for the int4 / warp_n=128 path (see nunchaku/lora/flux/packer.py).
def _weight_deswizzle_perm(out, K):
    """Inverse of pack_weight's byte layout: stored-nibble-order index -> logical (out,in) flat index."""
    return (torch.arange(out * K).reshape(out, K)
            .reshape(out // 128, 8, 2, 8, 1, K // 64, 1, 2, 4, 8)
            .permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous().reshape(-1).numpy())

def _wscale_out_to_logical(n):
    """Inverse of pack_scale's output-axis swizzle (group_size 64, warp_n 128): stored-out-pos -> logical-out idx."""
    return (torch.arange(n).reshape(n // 128, 1, 8, 2, 4, 2)
            .permute(0, 1, 2, 4, 3, 5).reshape(-1).numpy())

# Nunchaku's authoritative low-rank unpack (isolated from the cloned packer.py, helper imports stubbed).
def _load_packer():
    src = open("/tmp/nunchaku_src/nunchaku/lora/flux/packer.py").read()
    src = (src.replace("from ...utils import ceil_divide", "def ceil_divide(a,b):\n return -(-a//b)")
              .replace("from .utils import pad", "def pad(t, divisor, dim):\n return t"))
    ns = {"torch": torch, "Fn": Fn}; exec(src, ns)
    return ns["NunchakuWeightPacker"](bits=4, warp_n=128)

def decode_linear(ckpt_path, prefix, packer=None):
    """Return the logical decoded pieces for one SVDQuant linear (prefix e.g. 'transformer_blocks.0.img_mlp.net.0.proj')."""
    packer = packer or _load_packer()
    g = lambda h, nm: h.get_tensor(prefix + "." + nm)
    with safe_open(ckpt_path, "pt", "cpu") as h:
        qw  = g(h, "qweight").numpy().astype(np.uint8)            # [out, in/2] swizzled int4
        wsc = g(h, "wscales").float().numpy()                    # [in/64, out] (out axis fragment-ordered)
        smf = g(h, "smooth_factor").float().numpy()              # [in]
        bias = g(h, "bias").float().numpy()                      # [out]
        pdn = g(h, "proj_down").to(torch.bfloat16)               # packed low-rank
        pup = g(h, "proj_up").to(torch.bfloat16)
    out, Khalf = qw.shape; K = Khalf * 2
    # int4 weight -> logical (out, in), signed
    perm = _weight_deswizzle_perm(out, K)
    bytes_flat = qw.ravel()
    nib = np.empty(out * K, np.int16); nib[0::2] = bytes_flat & 0xF; nib[1::2] = bytes_flat >> 4; nib -= 16 * (nib >= 8)
    nibbles = np.empty(out * K, np.float32); nibbles[perm] = nib.astype(np.float32)
    nibbles = nibbles.reshape(out, K)
    # wscales -> logical (in_group, out); group axis already contiguous, only the out axis is permuted
    wsc_log = wsc[:, _wscale_out_to_logical(out)]                # [in/64, out] logical out order
    # low-rank logical factors
    proj_down = packer.unpack_lowrank_weight(pdn, down=True).float().numpy()    # [rank, in]
    proj_up   = packer.unpack_lowrank_weight(pup, down=False).float().numpy()   # [out, rank]
    return dict(nibbles=nibbles, wscales_logical=wsc_log, proj_down=proj_down, proj_up=proj_up,
                smooth_factor=smf, bias=bias, out=out, K=K, group_size=64)

def smoothed_effective_weight(dec):
    """Ŵ (smoothed-basis effective weight) = int4-main dequant + low-rank residual, shape [out, in]."""
    scale_per_elt = np.repeat(dec["wscales_logical"].T, 64, axis=1)             # [out, in]: wscale[i//64, o]
    main = dec["nibbles"] * scale_per_elt
    lowrank = dec["proj_up"] @ dec["proj_down"]                                  # [out, in]
    return main + lowrank

if __name__ == "__main__":
    INT4 = "/mnt/disk1/models/qwen-image/diffusion_models/svdq-int4_r128-qwen-image.safetensors"
    FP8  = "/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
    L = "transformer_blocks.0.img_mlp.net.0.proj"
    dec = decode_linear(INT4, L)
    print(f"decoded {L}: out={dec['out']} in={dec['K']} group_size={dec['group_size']}; "
          f"nibbles int range [{int(dec['nibbles'].min())},{int(dec['nibbles'].max())}], "
          f"low-rank shape proj_up{dec['proj_up'].shape} proj_down{dec['proj_down'].shape}")
    # self-check: the int4-main sign pattern should agree with the same model's FP8 weights (~0.80 — the
    # disagreement is the rank-128 low-rank that the int4 main excludes). Confirms the deterministic decode is coherent.
    with safe_open(FP8, "pt", "cpu") as h:
        ref = h.get_tensor(L + ".weight").float().numpy()        # [out, in], natural order
    sa = np.sign(dec["nibbles"]).ravel(); sb = np.sign(ref).ravel(); m = (sa != 0) & (sb != 0)
    print(f"sign-match int4-main vs FP8 weights = {float(np.mean(sa[m] == sb[m])):.3f}  (expect ~0.80 — decode coherent)")
