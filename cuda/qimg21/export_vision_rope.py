#!/usr/bin/env python3
"""Export the exact CUDA Qwen3-VL vision RoPE cosine/sine rows."""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-coordinate", type=int, default=256)
    args = ap.parse_args()
    config = Qwen3VLVisionConfig(hidden_size=1152, num_heads=16, rope_theta=10000)
    rope = Qwen3VLVisionRotaryEmbedding(config).cuda()
    coords = torch.arange(args.max_coordinate, device="cuda", dtype=torch.long)
    position_ids = torch.stack((coords, coords), dim=1)
    dummy = torch.empty(args.max_coordinate, 1152, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        cos, sin = rope(dummy, position_ids)
    table = torch.stack((cos[:, :18], sin[:, :18]), dim=-1).float().cpu().numpy()
    if table.shape != (args.max_coordinate, 18, 2) or not np.isfinite(table).all():
        raise RuntimeError("unexpected vision RoPE table")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.ascontiguousarray(table, dtype=np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
