#!/usr/bin/env python3
"""Create the small .npy inputs consumed by test_cuda_qimg21_native."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--prompt-embeds", required=True, help="F32 prompt_embeds.npy from --test-text")
ap.add_argument("--height-tokens", type=int, default=16)
ap.add_argument("--width-tokens", type=int, default=16)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
np.save(out / "prompt_embeds.npy", np.load(a.prompt_embeds).astype(np.float32, copy=False))
g = np.random.default_rng(a.seed)
np.save(out / "latents.npy", g.standard_normal((a.height_tokens * a.width_tokens, 64), dtype=np.float32))
print(out)
