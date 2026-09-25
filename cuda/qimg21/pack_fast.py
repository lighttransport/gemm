#!/usr/bin/env python3
"""Pack Qwen-Image 2.1 transformer blocks for test_cuda_qimg21_fast.

--kind int8-smooth writes SmoothQuant W8A8 blocks: for each linear group that
shares an input (fused QKV, out, fused gate|proj, mlp.out) the per-input-channel
factor s = amax(X)^alpha / amax(W)^(1-alpha) moves activation outliers into the
weights, W*s is quantized per output row to symmetric INT8, and the runtime
multiplies activations by 1/s before per-token INT8 quantization. amax(X) comes
from one or more --calib files written by test_cuda_qimg21_fast --calib-dump.

The output is blocks.bin (32 fixed-size blobs in the runner's device layout)
and manifest.json. Non-block layers stay BF16 and are read from the model.

--kind nvfp4-svd writes SVDQuant W4A4 blocks: after smoothing, the rank-R SVD
of W*s is kept in BF16 (the down factor has 1/s folded in, so the runtime feeds
it the unsmoothed activation) and the residual is rounded to NVFP4: E2M1 codes
with one E4M3 scale per 16 inputs and an F32 scale per output row that puts
the largest group scale at 448. The runtime quantizes activations the same way
per token.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

D, F, BLOCKS = 4096, 12288, 32
FORMAT = "qimg21-fast-int8-smooth-v1"
FORMAT_FP4 = "qimg21-fast-nvfp4-svd-v1"
RANK = 128
GROUPS = {"qkv": (3 * D, D), "out": (D, D), "gp": (2 * F, D), "mo": (D, F)}
E2M1_THR = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def layout_fp4():
    """Byte offsets per part; mirrors q21f_fp4_layout in the runner."""
    sizes = []
    for g, (n, k) in GROUPS.items():
        sizes += [(g + "_codes", n * k // 2), (g + "_gs", n * k // 16), (g + "_wc", n * 4),
                  (g + "_down", RANK * k * 2), (g + "_up", n * RANK * 2), (g + "_inv", k * 4)]
    sizes += [("nq", 128 * 4), ("nk", 128 * 4)]
    parts, offset = {}, 0
    for name, size in sizes:
        parts[name] = (offset, size)
        offset += (size + 255) & ~255
    return parts, offset


def quantize_nvfp4(r, per_matrix=False):
    """r [n, k] float32 -> (codes u8 [n, k/2] low nibble = even k, group e4m3 bytes [n, k/16],
    row scale f32 [n], dequantized [n, k]). per_matrix uses one F32 scale for all rows so a
    block-scaled GEMM can apply it as a scalar alpha."""
    n, k = r.shape
    g = r.reshape(n, k // 16, 16)
    amax = g.abs().amax(2)
    if per_matrix:
        wc = (amax.max() / (6.0 * 448.0)).clamp_min(1e-30).expand(n).contiguous()
    else:
        wc = (amax.amax(1) / (6.0 * 448.0)).clamp_min(1e-30)
    gs = (amax / 6.0 / wc[:, None]).clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = (gs.float() * wc[:, None]).clamp_min(1e-30)
    x = g / eff[:, :, None]
    code = torch.bucketize(x.abs(), E2M1_THR.to(r.device)).to(torch.uint8) + (x < 0).to(torch.uint8) * 8
    code = code.reshape(n, k)
    deq = E2M1.to(r.device)[code.long()].reshape(n, k // 16, 16) * eff[:, :, None]
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, gs.view(torch.uint8), wc, deq.reshape(n, k)


def layout():
    """Byte offsets of each part; mirrors q21f_int8_layout in the runner."""
    sizes = [
        ("qkv", 3 * D * D), ("out", D * D), ("gp", 2 * F * D), ("mo", D * F),
        ("s_qkv", 3 * D * 4), ("s_out", D * 4), ("s_gp", 2 * F * 4), ("s_mo", D * 4),
        ("i_qkv", D * 4), ("i_out", D * 4), ("i_gp", D * 4), ("i_mo", F * 4),
        ("nq", 128 * 4), ("nk", 128 * 4),
    ]
    parts, offset = {}, 0
    for name, size in sizes:
        parts[name] = (offset, size)
        offset += (size + 255) & ~255
    return parts, offset


def tensors(model):
    index = json.loads((model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    handles = {}
    def get(name):
        shard = index["weight_map"][name]
        if shard not in handles:
            handles[shard] = safe_open(str(model / "transformer" / shard), framework="pt")
        return handles[shard].get_tensor(name)
    return get


def quantize(weight, amax_x, alpha):
    """weight [out, in] float32, amax_x [in]; returns int8, row scale, 1/s."""
    amax_w = weight.abs().amax(dim=0)
    s = torch.ones_like(amax_w)
    ok = (amax_x > 0) & (amax_w > 0)
    s[ok] = (amax_x[ok] ** alpha / amax_w[ok] ** (1 - alpha)).clamp(1e-4, 1e4)
    smoothed = weight * s[None, :]
    scale = smoothed.abs().amax(dim=1) / 127.0
    scale[scale == 0] = 1.0
    q = torch.round(smoothed / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale, 1.0 / s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--calib", type=Path, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--kind", choices=("int8-smooth", "nvfp4-svd"), default="int8-smooth")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp4-scale", choices=("row", "matrix"), default="matrix",
                    help="nvfp4-svd outer weight scale: per output row, or one per matrix (CUTLASS alpha)")
    args = ap.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        ap.error("--alpha must be in [0, 1]")
    amax = None
    for path in args.calib:
        a = np.load(path).astype(np.float32).reshape(BLOCKS, 4, F)
        amax = a if amax is None else np.maximum(amax, a)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    get = tensors(args.model)
    if args.kind == "nvfp4-svd":
        return pack_fp4(args, amax, get)
    parts, blob_bytes = layout()
    dev = torch.device(args.device)
    stats = []
    with open(args.out_dir / "blocks.bin", "wb") as out:
        for b in range(BLOCKS):
            blob = bytearray(blob_bytes)
            def put(name, array):
                data = np.ascontiguousarray(array).tobytes()
                offset, size = parts[name]
                assert len(data) == size, (name, len(data), size)
                blob[offset:offset + size] = data
            pre = f"transformer_blocks.{b}."
            groups = {
                "qkv": (["attn.to_q", "attn.to_k", "attn.to_v"], 0, D),
                "out": (["attn.to_out.0"], 1, D),
                "gp": (["img_mlp.gate_layer", "img_mlp.proj"], 2, D),
                "mo": (["img_mlp.out"], 3, F),
            }
            for name, (members, which, width) in groups.items():
                weight = torch.cat([get(pre + m + ".weight") for m in members]).to(dev, torch.float32)
                amax_x = torch.from_numpy(amax[b, which, :width]).to(dev)
                if not (amax_x > 0).any():
                    raise SystemExit(f"block {b} {name}: calibration has no activations")
                q, scale, inverse = quantize(weight, amax_x, args.alpha)
                recon = q.float() * scale[:, None] / (1.0 / inverse)[None, :]
                rel = float((recon - weight).norm() / weight.norm())
                stats.append({"block": b, "group": name, "weight_rel_l2": rel})
                put(name, q.cpu().numpy())
                put("s_" + name, scale.cpu().numpy().astype(np.float32))
                put("i_" + name, inverse.cpu().numpy().astype(np.float32))
            put("nq", get(pre + "attn.norm_q.weight").float().numpy())
            put("nk", get(pre + "attn.norm_k.weight").float().numpy())
            out.write(blob)
            print(f"block {b:02d}: " + " ".join(f"{s['group']}={s['weight_rel_l2']:.4f}"
                                                 for s in stats[-4:]), flush=True)
    manifest = {
        "format": FORMAT, "kind": args.kind, "alpha": args.alpha, "blocks": BLOCKS,
        "blob_bytes": blob_bytes, "parts": {k: list(v) for k, v in parts.items()},
        "calibration": [{"path": str(p.resolve()),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.calib],
        "model": str(args.model.resolve()), "weight_error": stats,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_dir} ({blob_bytes * BLOCKS / 2**30:.2f} GiB)")


def pack_fp4(args, amax, get):
    parts, blob_bytes = layout_fp4()
    dev = torch.device(args.device)
    stats = []
    members = {"qkv": (["attn.to_q", "attn.to_k", "attn.to_v"], 0), "out": (["attn.to_out.0"], 1),
               "gp": (["img_mlp.gate_layer", "img_mlp.proj"], 2), "mo": (["img_mlp.out"], 3)}
    with open(args.out_dir / "blocks.bin", "wb") as out:
        for b in range(BLOCKS):
            blob = bytearray(blob_bytes)
            def put(name, array):
                data = np.ascontiguousarray(array).tobytes()
                offset, size = parts[name]
                assert len(data) == size, (name, len(data), size)
                blob[offset:offset + size] = data
            pre = f"transformer_blocks.{b}."
            for name, (names, which) in members.items():
                n, k = GROUPS[name]
                weight = torch.cat([get(pre + m + ".weight") for m in names]).to(dev, torch.float32)
                amax_x = torch.from_numpy(amax[b, which, :k]).to(dev)
                amax_w = weight.abs().amax(dim=0)
                s = torch.ones_like(amax_w)
                ok = (amax_x > 0) & (amax_w > 0)
                s[ok] = (amax_x[ok] ** args.alpha / amax_w[ok] ** (1 - args.alpha)).clamp(1e-4, 1e4)
                smoothed = weight * s[None, :]
                u, sv, vh = torch.linalg.svd(smoothed, full_matrices=False)
                root = sv[:RANK].sqrt()
                up = (u[:, :RANK] * root).to(torch.bfloat16)
                down = (root[:, None] * vh[:RANK]).to(torch.bfloat16)
                residual = smoothed - up.float() @ down.float()
                codes, gs, wc, deq = quantize_nvfp4(residual, args.fp4_scale == "matrix")
                recon = (deq + up.float() @ down.float()) / s[None, :]
                stats.append({"block": b, "group": name,
                              "weight_rel_l2": float((recon - weight).norm() / weight.norm()),
                              "residual_rel_l2": float((deq - residual).norm() / residual.norm())})
                put(name + "_codes", codes.cpu().numpy())
                put(name + "_gs", gs.cpu().numpy())
                put(name + "_wc", wc.cpu().numpy().astype(np.float32))
                # Fold 1/s into the down factor: x @ (down/s)^T == (x/s) @ down^T.
                put(name + "_down", (down.float() / s[None, :]).to(torch.bfloat16).view(torch.int16).cpu().numpy())
                put(name + "_up", up.view(torch.int16).cpu().numpy())
                put(name + "_inv", (1.0 / s).cpu().numpy().astype(np.float32))
            put("nq", get(pre + "attn.norm_q.weight").float().numpy())
            put("nk", get(pre + "attn.norm_k.weight").float().numpy())
            out.write(blob)
            print(f"block {b:02d}: " + " ".join(f"{s['group']}={s['weight_rel_l2']:.4f}/{s['residual_rel_l2']:.4f}"
                                                 for s in stats[-4:]), flush=True)
    manifest = {
        "format": FORMAT_FP4, "kind": args.kind, "alpha": args.alpha, "rank": RANK, "blocks": BLOCKS,
        "fp4_scale": args.fp4_scale,
        "blob_bytes": blob_bytes, "parts": {k: list(v) for k, v in parts.items()},
        "calibration": [{"path": str(p.resolve()),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.calib],
        "model": str(args.model.resolve()), "weight_error": stats,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_dir} ({blob_bytes * BLOCKS / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
