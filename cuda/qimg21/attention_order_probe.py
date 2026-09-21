#!/usr/bin/env python3
"""CPU replay of softmax tile order against a saved CUDA PyTorch output.

This infers a promising native arithmetic order; it does not prove which
kernel PyTorch dispatched or replace native full-denoiser acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--reference-name", default="pytorch_attn_matched.npy")
    args = ap.parse_args()
    folder = args.stage_dir
    prefix = np.load(folder / "txt_input.npy").shape[0]

    def read(name):
        return torch.from_numpy(np.load(folder / f"{name}.npy")).bfloat16().float().reshape(
            -1, 32, 128).transpose(0, 1)

    q, k, v = [read(name) for name in ("rope_q", "rope_k", "v")]
    n = q.shape[1]
    scores = q @ k.transpose(-1, -2) / np.sqrt(128)
    mask = torch.arange(n)[None, :] <= torch.arange(n)[:, None]
    mask[prefix:] = True
    scores.masked_fill_(~mask, -torch.inf)
    reference = np.load(folder / args.reference_name)
    results = []

    def report(name, result):
        array = result.bfloat16().float().transpose(0, 1).reshape(n, -1).numpy()
        if array.shape != reference.shape or not np.isfinite(array).all() or not np.isfinite(reference).all():
            raise ValueError("invalid attention comparison tensors")
        for label, section in (("text", slice(0, prefix)), ("image", slice(prefix, None))):
            a, b = [value[section].astype(np.float64).ravel() for value in (array, reference)]
            results.append(dict(mode=name, section=label,
                                cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                                exact=float(np.mean(a == b))))

    p = torch.exp(scores - scores.amax(-1, keepdim=True))
    report("math", p @ v / p.sum(-1, keepdim=True))
    report("global_bf16_prob", p.bfloat16().float() @ v / p.sum(-1, keepdim=True))
    for block in (32, 64, 128, 256):
        for reverse in (False, True):
            output, denom = torch.zeros_like(q), torch.zeros(32, n, 1)
            maximum = torch.full((32, n, 1), -torch.inf)
            tiles = list(range(0, n, block))
            for start in reversed(tiles) if reverse else tiles:
                end = min(n, start + block)
                tile = scores[:, :, start:end]
                next_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
                alpha = torch.nan_to_num(torch.exp(maximum - next_max), nan=1.)
                p = torch.nan_to_num(torch.exp(tile - next_max), nan=0.)
                output = output * alpha + p.bfloat16().float() @ v[:, start:end]
                denom = denom * alpha + p.sum(-1, keepdim=True)
                maximum = next_max
            report(f"{'reverse' if reverse else 'forward'}{block}", output / denom)
    (folder / "attention_order_probe.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
