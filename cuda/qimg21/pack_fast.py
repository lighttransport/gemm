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
    ap.add_argument("--kind", choices=("int8-smooth",), default="int8-smooth")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        ap.error("--alpha must be in [0, 1]")
    amax = None
    for path in args.calib:
        a = np.load(path).astype(np.float32).reshape(BLOCKS, 4, F)
        amax = a if amax is None else np.maximum(amax, a)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    get = tensors(args.model)
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


if __name__ == "__main__":
    main()
