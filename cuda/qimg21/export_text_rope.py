#!/usr/bin/env python3
"""Export the exact BF16 Qwen3-VL text RoPE cos/sin table."""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    config = Qwen3VLTextConfig(head_dim=128, rope_theta=5_000_000,
                               rope_scaling={"rope_type": "default",
                                             "mrope_section": [24, 20, 20]})
    rope = Qwen3VLTextRotaryEmbedding(config).cuda()
    positions = torch.arange(args.max_tokens, device="cuda", dtype=torch.long)
    position_ids = positions[None, None, :].expand(3, 1, -1)
    dummy = torch.empty(1, args.max_tokens, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        cos, sin = rope(dummy, position_ids)
    table = torch.stack((cos[0], sin[0]), dim=-1).float().cpu().numpy()
    np.save(args.out, np.ascontiguousarray(table, dtype=np.float32))


if __name__ == "__main__":
    main()
