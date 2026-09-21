#!/usr/bin/env python3
"""Compare isolated denoiser text projection stages, using original weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TextProjection
from compare import _cosine_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--native-dir", required=True, type=Path)
    args = ap.parse_args()
    model = QwenImage21TextProjection(4096, 4096).cuda().bfloat16()
    folder = args.model / "transformer"
    index = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for name in model.state_dict():
        key = "txt_in." + name
        with safe_open(folder / index[key], framework="pt") as weights:
            state[name] = weights.get_tensor(key)
    model.load_state_dict(state)
    x = torch.from_numpy(np.load(args.input)).cuda().bfloat16().reshape(-1, 4096)
    results = {}
    with torch.inference_mode():
        for name, module in (("norm", model.text_norm), ("in", model.in_layer),
                             ("gelu", model.act), ("out", model.out_layer)):
            x = module(x)
            target = x.float().cpu().numpy()
            native = np.load(args.native_dir / f"{name}.npy")
            cosine, relative_l2 = _cosine_error(target, native)
            results[name] = {"cosine": cosine, "relative_l2": relative_l2,
                             "mismatches": int(np.count_nonzero(target != native)),
                             "exact_fraction": float(np.mean(target == native))}
    (args.native_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
