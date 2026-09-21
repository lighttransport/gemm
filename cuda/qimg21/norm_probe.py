#!/usr/bin/env python3
"""Replay first-block BF16 normalization with identical inputs in PyTorch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare import _cosine_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    folder = args.stage_dir
    x = torch.from_numpy(np.load(folder / "hidden0.npy")).cuda().bfloat16()
    modulation = torch.from_numpy(np.load(folder / "mod.npy")).cuda().bfloat16().reshape(2, 16384)
    nt = np.load(folder / "txt_input.npy").shape[0]
    if x.ndim != 2 or x.shape[1] != 4096 or not 0 < nt < len(x):
        raise ValueError("expected text-to-image first-block fixture")
    scale = torch.cat([modulation[1, :4096].expand(nt, -1),
                       modulation[0, :4096].expand(len(x)-nt, -1)])
    with torch.inference_mode():
        target = (torch.nn.functional.layer_norm(x, (4096,), eps=1e-6) * (1+scale)).float().cpu().numpy()
    candidate = np.load(args.candidate, allow_pickle=False)
    if candidate.shape != target.shape:
        raise ValueError("candidate must match the full normalized tensor shape")
    cosine, relative_l2 = _cosine_error(target, candidate)
    result = {"cosine": cosine, "relative_l2": relative_l2, "torch": torch.__version__,
              "exact": bool(np.array_equal(target, candidate)),
              "mismatches": int(np.count_nonzero(target != candidate)),
              "elements": target.size}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
