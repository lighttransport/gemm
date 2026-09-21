#!/usr/bin/env python3
"""Compare a native attention stage with PyTorch using identical Q/K/V."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=("default", "math", "flash"), default="default")
    args = ap.parse_args()
    folder = args.stage_dir
    prefix = np.load(folder / "txt_input.npy").shape[0]

    def load(name):
        a = np.load(folder / f"{name}.npy")
        return torch.from_numpy(a).to(device="cuda", dtype=torch.bfloat16).reshape(
            1, -1, 32, 128).transpose(1, 2)

    q, k, v = load("rope_q"), load("rope_k"), load("v")
    mask = torch.ones(prefix, prefix, device="cuda", dtype=torch.bool).tril()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = nullcontext() if args.backend == "default" else sdpa_kernel(
        SDPBackend.MATH if args.backend == "math" else SDPBackend.FLASH_ATTENTION)
    with torch.inference_mode(), backend:
        text = F.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix],
                                              v[:, :, :prefix], attn_mask=mask)
        image = F.scaled_dot_product_attention(q[:, :, prefix:], k, v)
        ref = torch.cat([text, image], dim=2).transpose(1, 2).flatten(2)
    ref = ref.float().cpu().numpy()[0]
    native = np.load(folder / "attn_raw.npy")
    if native.shape != ref.shape:
        raise ValueError(f"shape mismatch: {native.shape} vs {ref.shape}")
    a, b = native.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    result = dict(backend=args.backend, torch=torch.__version__, cosine=cosine,
                  relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                  equal_fraction=float(np.mean(a==b)), finite=finite)
    np.save(folder / f"pytorch_attn_matched_{args.backend}.npy", ref)
    (folder / f"attention_probe_{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if finite and cosine >= 0.99996 else 1


if __name__ == "__main__":
    raise SystemExit(main())
