#!/usr/bin/env python3
"""Export optional per-row symmetric INT8 transformer matrices, one file each.

Original weights are never modified. Norms and biases remain in the original
snapshot. This is weight compression, not an INT8 activation/GEMM backend.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def quantize_rows(weight):
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or not weight.size or not np.isfinite(weight).all():
        raise ValueError("expected finite, nonempty matrix")
    maximum = np.max(np.abs(weight), axis=1)
    scale = np.where(maximum > 0, maximum / np.float32(127), np.float32(1)).astype(np.float32)
    if np.any(scale <= 0):
        raise ValueError("row scale underflow")
    quantized = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
    return quantized, scale


def main():
    from safetensors import safe_open
    from safetensors.numpy import save_file

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="New directory (must not exist)")
    args = ap.parse_args()
    sources = sorted((args.model / "transformer").glob("diffusion_pytorch_model-*.safetensors"))
    if not sources:
        raise ValueError("no transformer shards found")
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sources:
        with safe_open(str(source), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if len(shard.get_slice(name).get_shape()) != 2:
                    continue
                if "/" in name or "\\" in name:
                    raise ValueError("invalid matrix name")
                weight = shard.get_tensor(name).float().numpy()
                quantized, scale = quantize_rows(weight)
                save_file({"weight": quantized, "scale": scale}, str(args.out / f"{name}.safetensors"))
                records.append({"name": name, "shape": list(weight.shape),
                                "bytes": int(quantized.nbytes + scale.nbytes)})
                print(f"exported {name} {weight.shape}", flush=True)
                del weight, quantized, scale
    if not records:
        raise ValueError("no matrices exported")
    (args.out / "manifest.json").write_text(json.dumps({
        "format": "qimg21-int8-row-v1", "source": str(args.model.resolve()),
        "matrices": records, "compute": "dequantize-to-bf16",
        "model_quality_validated": False}, indent=2) + "\n")
    # Written last: the native loader refuses interrupted exports.
    (args.out / "format.txt").write_text("qimg21-int8-row-v1\n")


if __name__ == "__main__":
    main()
