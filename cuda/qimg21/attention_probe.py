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


def native_editing_segments(path, tokens):
    """Expand the native slot-mask wire format into block-causal prefix runs."""
    values = [int(x) for x in path.read_text().split()]
    if len(values) < 3:
        raise ValueError("editing layout is incomplete")
    slots, text_slots, images = values[:3]
    if slots < 1 or not 0 <= text_slots < slots or images < 1 or len(values) != 3 + slots + 2 * images:
        raise ValueError("invalid editing layout dimensions")
    mask = values[3:3 + slots]
    sizes = values[3 + slots:]
    target_tokens = sizes[-2] * sizes[-1]
    if (any(x not in (0, 1) for x in mask) or any(x < 1 for x in sizes) or
        any(sizes[2 * i] * sizes[2 * i + 1] % 4 for i in range(images)) or
        slots - text_slots != target_tokens // 4 or any(x != 1 for x in mask[text_slots:])):
        raise ValueError("invalid editing layout values")
    ids = []
    image = 0
    remaining = sizes[0] * sizes[1]
    for marked in mask:
        if marked:
            if image >= images or remaining < 4:
                raise ValueError("invalid editing image extent")
            ids.extend([image] * 4)
            remaining -= 4
            if remaining == 0:
                image += 1
                if image < images:
                    remaining = sizes[2 * image] * sizes[2 * image + 1]
        else:
            if image < images and remaining != sizes[2 * image] * sizes[2 * image + 1]:
                raise ValueError("text splits an image latent block")
            ids.append(-1)
    prefix = len(ids) - target_tokens
    if image != images or len(ids) != tokens or prefix < 1 or any(x != images - 1 for x in ids[prefix:]):
        raise ValueError("editing layout does not match attention tensors")
    segments = []
    start = 0
    for end in range(1, prefix + 1):
        if end == prefix or ids[end] != ids[start]:
            segments.append((start, end, ids[start] < 0))
            start = end
    return prefix, segments


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=("default", "math", "flash", "efficient", "cudnn"),
                    default="default")
    ap.add_argument("--prefix-backend", choices=("default", "math", "flash", "efficient", "cudnn"),
                    default="default", help="force each segmented editing-prefix call")
    editing = ap.add_mutually_exclusive_group()
    editing.add_argument("--editing-reference", type=Path,
                        help="Use official segmented editing attention from a positive-branch capture")
    editing.add_argument("--editing-layout", type=Path,
                        help="Use a native editing layout without importing Diffusers")
    args = ap.parse_args()
    folder = args.stage_dir
    prefix = None if args.editing_reference or args.editing_layout else np.load(folder / "txt_input.npy").shape[0]

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
    def backend_context(name):
        return nullcontext() if name == "default" else sdpa_kernel(selected[name])
    with torch.inference_mode():
        if args.editing_reference or args.editing_layout:
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
                segments = _qwenimage21_prefix_segments(ids, prefix)
            else:
                prefix, segments = native_editing_segments(args.editing_layout, q.shape[2])
            outputs = []
            for start, end, is_text in segments:
                mask = None
                if is_text:
                    mask = torch.cat([torch.ones(end-start, start, device="cuda", dtype=torch.bool),
                                      torch.ones(end-start, end-start, device="cuda", dtype=torch.bool).tril()], dim=1)
                with backend_context(args.prefix_backend):
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
    result.update(backend=args.backend, prefix_backend=args.prefix_backend,
                  torch=torch.__version__, finite=True,
                  cosine=aggregate["cosine"], relative_l2=aggregate["relative_l2"],
                  equal_fraction=aggregate["equal_fraction"])
    suffix = args.backend if args.prefix_backend == "default" else f"{args.backend}_prefix_{args.prefix_backend}"
    np.save(folder / f"pytorch_attn_matched_{suffix}.npy", ref)
    (folder / f"attention_probe_{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
