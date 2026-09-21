#!/usr/bin/env python3
"""Export Qwen-Image 2.1's exact CPU-generated complex RoPE frequencies."""
import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("cuda/qimg21/qwen21_rope_freqs.npy"))
    args = parser.parse_args()
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    frame = torch.view_as_real(rope.freqs[0]).flatten(1)
    spatial = torch.view_as_real(rope.freqs[1]).flatten(1)
    table = torch.cat((frame, spatial, spatial), dim=1).contiguous().numpy()
    if table.shape != (9216, 128) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise RuntimeError("unexpected RoPE frequency table")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table, allow_pickle=False)
    print(f"saved {args.out} {table.shape}")


if __name__ == "__main__":
    main()
