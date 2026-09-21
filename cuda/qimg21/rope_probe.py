#!/usr/bin/env python3
"""Replay real Q projection through official RMSNorm and complex RoPE."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope, apply_rotary_emb_qwen

from compare import _cosine_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--height-tokens", type=int, required=True)
    ap.add_argument("--width-tokens", type=int, required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--export-replay", type=Path, help="Write normalized Q, official frequencies and expected output")
    args = ap.parse_args()
    folder = args.stage_dir
    q = torch.from_numpy(np.load(folder / "q.npy")).cuda().bfloat16().reshape(1, -1, 32, 128)
    nt = np.load(folder / "txt_input.npy").shape[0]
    ni = args.height_tokens * args.width_tokens
    if q.shape[1] != nt + ni:
        raise ValueError("expected text-to-image stage layout")
    index = json.loads((args.model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    name = f"transformer_blocks.{args.block}.attn.norm_q.weight"
    with safe_open(args.model / "transformer" / index["weight_map"][name], framework="pt") as weights:
        weight = weights.get_tensor(name)
    norm = RMSNorm(128, eps=1e-6).cuda().bfloat16()
    norm.weight.data.copy_(weight)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_mask = torch.arange(nt+ni) >= nt
    frequencies = rope([(1, args.height_tokens, args.width_tokens)], image_mask, torch.device("cuda"))
    with torch.inference_mode():
        normalized = norm(q)
        expected = apply_rotary_emb_qwen(normalized, frequencies, use_real=False).float().cpu().numpy().reshape(-1, 4096)
        # Reproduce the native 128-thread pairwise RMS reduction, but retain
        # official frequencies/complex rotation to isolate reduction error.
        square = q.float().square()
        while square.shape[-1] > 1:
            half = square.shape[-1] // 2
            square = square[..., :half] + square[..., half:]
        tree_normalized = (q.float() * torch.rsqrt(square / 128 + 1e-6)).bfloat16() * norm.weight
        # PyTorch's contiguous mean reduction combines four adjacent values
        # per lane, then reduces the 32 lanes in descending offset order.
        grouped = q.float().square().reshape(*q.shape[:-1], 32, 4)
        vector_sum = ((grouped[..., 0] + grouped[..., 1]) + grouped[..., 2]) + grouped[..., 3]
        while vector_sum.shape[-1] > 1:
            half = vector_sum.shape[-1] // 2
            vector_sum = vector_sum[..., :half] + vector_sum[..., half:]
        vector_normalized = (q.float() * torch.rsqrt(vector_sum / 128 + 1e-6)).bfloat16() * norm.weight
        vector_mismatch = int(torch.count_nonzero(normalized != vector_normalized).item())
        tree_rotated = apply_rotary_emb_qwen(tree_normalized, frequencies, use_real=False).float().cpu().numpy().reshape(-1, 4096)
        norm_mismatch = int(torch.count_nonzero(normalized != tree_normalized).item())
    if args.export_replay:
        args.export_replay.mkdir(parents=True, exist_ok=False)
        np.save(args.export_replay / "normalized.npy", np.ascontiguousarray(normalized.float().cpu().numpy().reshape(-1, 4096)))
        np.save(args.export_replay / "frequencies.npy", np.ascontiguousarray(torch.view_as_real(frequencies).cpu().numpy().reshape(-1, 128)))
        np.save(args.export_replay / "expected.npy", np.ascontiguousarray(expected))
    native = np.load(folder / "rope_q.npy")
    cosine, relative_l2 = _cosine_error(expected, native)
    mismatch = (expected != native).reshape(-1, 32, 128)
    result = {"cosine": cosine, "relative_l2": relative_l2, "exact_fraction": float(np.mean(~mismatch)),
              "mismatches": int(mismatch.sum()), "per_channel": mismatch.sum(axis=(0, 1)).tolist(),
              "rms_reduction_mismatches": norm_mismatch,
              "vector4_rms_reduction_mismatches": vector_mismatch,
              "tree_rms_official_rope_mismatches": int(np.count_nonzero(tree_rotated != expected)),
              "native_vs_tree_rms_official_rope_mismatches": int(np.count_nonzero(native != tree_rotated))}
    (folder / "rope_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_channel"}, indent=2))


if __name__ == "__main__":
    main()
