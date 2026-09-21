#!/usr/bin/env python3
"""Compare a native attention stage with PyTorch using identical Q/K/V."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from attention_replay_compare import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=("default", "math", "flash", "efficient", "cudnn"),
                    default="default")
    ap.add_argument("--editing-reference", type=Path,
                    help="Use official segmented editing attention from a positive-branch capture")
    args = ap.parse_args()
    folder = args.stage_dir
    prefix = None if args.editing_reference else np.load(folder / "txt_input.npy").shape[0]

    def load(name):
        a = np.load(folder / f"{name}.npy")
        return torch.from_numpy(a).to(device="cuda", dtype=torch.bfloat16).reshape(
            1, -1, 32, 128).transpose(1, 2)

    q, k, v = load("rope_q"), load("rope_k"), load("v")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    selected = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    target_backend = nullcontext() if args.backend == "default" else sdpa_kernel(selected[args.backend])
    with torch.inference_mode():
        if args.editing_reference:
            from diffusers.models.transformers.transformer_qwenimage21 import (
                QwenImage21Transformer2DModel, _qwenimage21_prefix_segments,
            )
            reference = args.editing_reference
            slots = torch.from_numpy(np.load(reference / "positive_img_mask.npy"))[0].bool()
            shapes = json.loads((reference / "positive_layout.json").read_text())["img_shapes"][0]
            ids, target = QwenImage21Transformer2DModel.build_token_metadata(
                slots.repeat_interleave(torch.where(slots, 4, 1)), shapes)
            prefix = int((~target).sum())
            valid = np.load(reference / "positive_encoder_hidden_states_mask.npy")
            if not (valid == 1).all() or len(ids) != q.shape[2]:
                raise ValueError("editing probe requires matching unpadded capture")
            outputs = []
            for start, end, is_text in _qwenimage21_prefix_segments(ids, prefix):
                mask = None
                if is_text:
                    mask = torch.cat([torch.ones(end-start, start, device="cuda", dtype=torch.bool),
                                      torch.ones(end-start, end-start, device="cuda", dtype=torch.bool).tril()], dim=1)
                outputs.append(F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, :end],
                                                              v[:, :, :end], attn_mask=mask))
            with target_backend:
                outputs.append(F.scaled_dot_product_attention(q[:, :, prefix:], k, v))
            ref = torch.cat(outputs, dim=2).transpose(1, 2).flatten(2)
        else:
            mask = torch.ones(prefix, prefix, device="cuda", dtype=torch.bool).tril()
            text = F.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix],
                                              v[:, :, :prefix], attn_mask=mask)
            with target_backend:
                image = F.scaled_dot_product_attention(q[:, :, prefix:], k, v)
            ref = torch.cat([text, image], dim=2).transpose(1, 2).flatten(2)
    ref = ref.float().cpu().numpy()[0]
    native = np.load(folder / "attn_raw.npy")
    if native.ndim == 3 and native.shape[0] == 1:
        native = native[0]
    if native.shape != ref.shape:
        raise ValueError(f"shape mismatch: {native.shape} vs {ref.shape}")
    result = metrics(ref, native, len(ref)-prefix)
    # Preserve the existing top-level aggregate fields for older notebooks.
    aggregate = result["regions"]["all"]
    result.update(backend=args.backend, torch=torch.__version__, finite=True,
                  cosine=aggregate["cosine"], relative_l2=aggregate["relative_l2"],
                  equal_fraction=aggregate["equal_fraction"])
    np.save(folder / f"pytorch_attn_matched_{args.backend}.npy", ref)
    (folder / f"attention_probe_{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
