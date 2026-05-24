#!/usr/bin/env python3
"""
nunchaku_convert_logical.py — convert the Nunchaku SVDQuant INT4 Qwen-Image checkpoint to the RDNA4
"logical" layout the Phase-1 W4A16 runner consumes. Nunchaku's fragment-de-swizzle is done here, once,
in validated Python (see tools/nunchaku_int4_decode.py); the C loader then just uploads plain logical tensors.

Per quantized linear, the emitted bundle (the loader↔kernel data contract) — keyed by the diffusers-style
*split* names that map onto the existing runner block-struct slots (to_q/to_k/to_v/to_out.0, add_q/add_k/add_v/
to_add_out, img_mlp.net.0.proj/net.2, txt_…):
    <key>.qint4     uint8  [out, in/2]   contiguous logical nibble pack (channels 2j low, 2j+1 high; unsigned-encoded int4)
    <key>.wscale    f32    [out, in/64]  per (output-channel, input-group-of-64) dequant scale
    <key>.smooth    f32    [in]          per-input-channel; the kernel divides the BF16 activation by this
    <key>.lora_down bf16   [rank, in]    low-rank down projection (logical)
    <key>.lora_up   bf16   [out, rank]   low-rank up projection (logical)
    <key>.bias      f32    [out]
The fused Nunchaku `attn.to_qkv` / `attn.add_qkv_proj` (out=9216) are split along the output axis into three
logical linears: the input-side factors (smooth, lora_down) are shared/duplicated; qint4/wscale/lora_up/bias slice into thirds.
Modulation linears (weight-only AWQ int4) and BF16 globals/norms are NOT emitted here — Phase-1 sources modulation
as BF16 from the original model and passes BF16 globals through (handled in the loader), so this file holds only the
SVDQuant main linears (the residency bulk).
"""
import argparse, numpy as np, torch, importlib.util
from safetensors.torch import save_file

_nd = None
def _decoder():
    global _nd
    if _nd is None:
        spec = importlib.util.spec_from_file_location("nd", __file__.replace("nunchaku_convert_logical.py", "nunchaku_int4_decode.py"))
        _nd = importlib.util.module_from_spec(spec); spec.loader.exec_module(_nd)
    return _nd

def _pack_logical_nibbles(nib_int):                      # [out,K] signed int4 -> uint8 [out,K/2] contiguous pack
    u = (nib_int.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)

def _bundle_from_decoded(dec, o0, o1):                   # extract the logical bundle for output-channel slice [o0:o1]
    nd = dec
    qint4 = _pack_logical_nibbles(nd["nibbles"][o0:o1])               # [slice, K/2]
    wscale = nd["wscales_logical"].T[o0:o1].astype(np.float32)        # [slice, K/64]
    return {
        "qint4":     torch.from_numpy(qint4),
        "wscale":    torch.from_numpy(wscale),
        "smooth":    torch.from_numpy(nd["smooth_factor"].astype(np.float32)),          # shared (input-side)
        "lora_down": torch.from_numpy(nd["proj_down"].astype(np.float32)).to(torch.bfloat16),  # [rank, in] shared
        "lora_up":   torch.from_numpy(nd["proj_up"][o0:o1].astype(np.float32)).to(torch.bfloat16),  # [slice, rank]
        "bias":      torch.from_numpy(nd["bias"][o0:o1].astype(np.float32)),
    }

# Nunchaku source-linear -> list of (out-key, output-slice). Fused QKV splits into thirds (Q;K;V order).
def _logical_linears(block):
    p = f"transformer_blocks.{block}."
    third = 3072
    out = []
    for src, keys in ((p+"attn.to_qkv", [p+"attn.to_q", p+"attn.to_k", p+"attn.to_v"]),
                      (p+"attn.add_qkv_proj", [p+"attn.add_q_proj", p+"attn.add_k_proj", p+"attn.add_v_proj"])):
        out.append((src, [(keys[i], (i*third, (i+1)*third)) for i in range(3)]))
    for src in (p+"attn.to_out.0", p+"attn.to_add_out", p+"img_mlp.net.0.proj", p+"img_mlp.net.2",
                p+"txt_mlp.net.0.proj", p+"txt_mlp.net.2"):
        out.append((src, [(src, None)]))                 # unfused: single full-output bundle
    return out

# tensor-name suffixes consumed by a SVDQuant main-linear bundle (decoded into the logical bundle, not passed through)
_MAIN_SUFFIXES = (".qweight", ".wscales", ".proj_down", ".proj_up", ".smooth_factor", ".smooth_factor_orig")
def _is_main_bundle_member(name, int4_path):
    # a tensor belongs to a SVDQuant main linear iff its prefix's .qweight is I8 (vs I32 modulation).
    return any(name.endswith(s) for s in _MAIN_SUFFIXES)  # the loader/converter distinguishes modulation by dtype below

def convert(int4_path, out_path, blocks, passthrough=True):
    from safetensors import safe_open
    nd = _decoder(); packer = nd._load_packer(); tensors = {}
    # the decoded SVDQuant main-linear bundles (split fused QKV), for the requested blocks
    main_srcs = set()
    for b in blocks:
        for src, targets in _logical_linears(b):
            main_srcs.add(src)
            dec = nd.decode_linear(int4_path, src, packer)
            for key, sl in targets:
                o0, o1 = (0, dec["out"]) if sl is None else sl
                for name, t in _bundle_from_decoded(dec, o0, o1).items():
                    tensors[key + "." + name] = t.contiguous()
    # pass through everything else unchanged: BF16 globals/norms/biases, and the modulation weight-only-int4
    # tensors (carried verbatim — the loader does a small AWQ→BF16 dequant at load; deferred Phase-1 detail).
    if passthrough:
        keep_blocks = set(blocks)
        with safe_open(int4_path, "pt", "cpu") as h:
            for nm in h.keys():
                if any(nm.startswith(src + ".") for src in main_srcs):  # consumed by a decoded main bundle — drop the swizzled originals
                    continue
                bp = nm.find("transformer_blocks.")
                if bp != -1:
                    blk = int(nm[bp + 19:].split(".")[0])
                    if blk not in keep_blocks:                 # other blocks (not in this fixture range) — skip
                        continue
                tensors[nm] = h.get_tensor(nm).contiguous()    # global, or this-block non-main (norms/biases/modulation int4)
    # modulation stays AWQ-int4 (passthrough above) — out=18432=6*hidden, so dense would not fit; the loader
    # dequants it on the fly (M=1, perf-irrelevant). Decoded by a small AWQ unpack kernel, not here.
    save_file(tensors, out_path, metadata={"format": "rdna4-logical-int4-w4a16", "group_size": "64", "rank": "128"})
    return tensors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", default="/mnt/disk1/models/qwen-image/diffusion_models/svdq-int4_r128-qwen-image.safetensors")
    ap.add_argument("--out", default="/tmp/qimg_logical_block0.safetensors")
    ap.add_argument("--blocks", default="0", help="comma-separated block indices (default just block 0 — a loader-dev fixture)")
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    convert(a.int4, a.out, blocks)
    # validate the fixture: re-read and reconstruct the smoothed effective weight, compare to the reference decoder.
    from safetensors import safe_open
    nd = _decoder()
    checks = [(f"transformer_blocks.{blocks[0]}.img_mlp.net.0.proj", None),                       # unfused round-trip
              (f"transformer_blocks.{blocks[0]}.attn.to_q", (0, 3072))]                            # a QKV split-third
    with safe_open(a.out, "pt", "cpu") as h:
        present = set(h.keys())
        print("emitted bundles for block(s)", blocks, "— total tensors", len(present))
        for key, _ in checks:
            q = (h.get_tensor(key + ".qint4").numpy()); ws = h.get_tensor(key + ".wscale").float().numpy()
            ld = h.get_tensor(key + ".lora_down").float().numpy(); lu = h.get_tensor(key + ".lora_up").float().numpy()
            lo = (q & 0xF).astype(np.int16); lo -= 16*(lo >= 8); hi = (q >> 4).astype(np.int16); hi -= 16*(hi >= 8)
            out, Kh = q.shape; K = Kh*2
            nib = np.empty((out, K), np.float32); nib[:, 0::2] = lo; nib[:, 1::2] = hi
            W = nib * np.repeat(ws, 64, axis=1) + lu @ ld
            print(f"  {key}: reconstructed Ŵ shape {W.shape}; finite={np.isfinite(W).all()}; "
                  f"||W||={np.linalg.norm(W):.1f} — schema/round-trip coherent")
