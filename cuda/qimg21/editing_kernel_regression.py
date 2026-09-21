#!/usr/bin/env python3
"""Synthetic CUDA editing primitives versus official layout/RoPE and torch math.

This is not end-to-end editing or model parity. No checkpoint is loaded.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope, QwenImage21Transformer2DModel, apply_rotary_emb_qwen,
)

from compare import _cosine_error, NONQUANTIZED_COSINE_THRESHOLD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-edit-kernels"))
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    folder = args.work_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if not args.compare_only:
        subprocess.run([str(Path(__file__).with_name("test_edit_kernels").resolve()), str(folder)], check=True)

    def load(name):
        return torch.from_numpy(np.load(folder / f"{name}.npy"))

    # Same fixture as the C harness: two adjacent condition-image blocks,
    # text before/after them, then a separate target image.
    slots = torch.tensor([False, True, True, False, True])
    repeats = torch.where(slots, 4, 1)
    image_mask = slots.repeat_interleave(repeats)
    shapes = [(1, 2, 2)] * 3
    ids, _ = QwenImage21Transformer2DModel.build_token_metadata(image_mask, shapes)
    expected = torch.cat([load("text"), torch.zeros(1, 256)]).repeat_interleave(repeats, dim=0)
    expected[image_mask] = load("image")
    results = {}

    def compare(name, reference, *, exact=False):
        actual = load(name).numpy()
        reference = reference.float().numpy().reshape(actual.shape)
        cosine, relative_l2 = _cosine_error(reference, actual)
        equal = bool(np.array_equal(reference, actual))
        results[name] = dict(cosine=cosine, relative_l2=relative_l2, exact=equal,
                             passed=equal if exact else cosine >= NONQUANTIZED_COSINE_THRESHOLD)

    compare("scatter", expected, exact=True)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    freq = rope(shapes, image_mask, torch.device("cpu"))
    cos, sin = freq.real.repeat_interleave(2, dim=-1), freq.imag.repeat_interleave(2, dim=-1)
    for name in ("q", "k"):
        value = load(f"{name}_input").reshape(1, -1, 2, 128).transpose(1, 2)
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        normalized = normalized * load(f"{name}w").bfloat16()
        rotated = apply_rotary_emb_qwen(normalized, (cos, sin)).transpose(1, 2).reshape(-1, 256)
        compare(f"rope_{name}", rotated)

    q, k, v = [load(name).reshape(-1, 2, 128).transpose(0, 1)
               for name in ("rope_q", "rope_k", "v_input")]
    n = len(ids)
    idx = torch.arange(n)
    allowed = (idx[:, None] >= idx[None, :]) | ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :]))
    scores = (q @ k.transpose(-1, -2)) / np.sqrt(128)
    probability = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    attention = (probability @ v).bfloat16().transpose(0, 1).reshape(n, 256)
    compare("attention", attention)
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
