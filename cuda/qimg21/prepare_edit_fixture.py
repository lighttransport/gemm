#!/usr/bin/env python3
"""Prepare exact editing inputs from an official transformer-call capture.

Condition latents are already normalized in input_NNN.npy. This bridge
does not implement native VAE encoding or vision-language text encoding.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(reference: Path, output: Path, step: int = 0, branch: str = "positive"):
    if step < 0 or branch not in ("positive", "negative"):
        raise ValueError("invalid step/branch")
    metadata = json.loads((reference / f"{branch}_layout.json").read_text())
    shapes = metadata["img_shapes"]
    if len(shapes) != 1 or len(shapes[0]) < 2:
        raise ValueError("expected batch-one editing layout with condition and target images")
    shapes = shapes[0]
    for shape in shapes:
        if (len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape) or
                shape[0] != 1 or max(shape[1:]) > 1024 or (shape[1] * shape[2]) % 4):
            raise ValueError("invalid single-frame image shape")
    nt = metadata["text_slots"]
    target = shapes[-1][1] * shapes[-1][2]
    if metadata["target_tokens"] != target:
        raise ValueError("inconsistent target token count")
    mask = np.load(reference / f"{branch}_img_mask.npy", allow_pickle=False)
    if (type(nt) is not int or nt < 1 or nt + target // 4 > 262144 or
            mask.shape != (1, nt + target // 4)):
        raise ValueError("invalid slot mask shape")
    if not np.isin(mask, [0, 1]).all() or not mask[0, nt:].all():
        raise ValueError("invalid image slot mask")
    keys = np.load(reference / f"{branch}_encoder_hidden_states_mask.npy", allow_pickle=False)
    if keys.shape != (1, nt) or not (keys == 1).all():
        raise ValueError("native editing currently requires unpadded text")
    total = sum(h * w for _, h, w in shapes)
    if total > 1048576:
        raise ValueError("image token count exceeds native layout limit")
    if 4 * np.count_nonzero(mask) != total:
        raise ValueError("image slots do not match latent shapes")
    block, consumed = 0, 0
    for is_image in mask[0]:
        if not is_image:
            if consumed:
                raise ValueError("text splits a condition-image block")
            continue
        consumed += 4
        if block >= len(shapes) or consumed > shapes[block][1] * shapes[block][2]:
            raise ValueError("invalid image block boundary")
        if consumed == shapes[block][1] * shapes[block][2]:
            block += 1
            consumed = 0
    if block != len(shapes) or consumed:
        raise ValueError("incomplete image block")

    name = "prompt_embeds.npy" if branch == "positive" else "negative_prompt_embeds.npy"
    prompt = np.load(reference / name, allow_pickle=False)
    image = np.load(reference / f"input_{step:03d}.npy", allow_pickle=False)
    if prompt.shape == (1, nt, 4096):
        prompt = prompt[0]
    if image.shape == (1, total, 64):
        image = image[0]
    if prompt.shape != (nt, 4096) or image.shape != (total, 64):
        raise ValueError("prompt or packed-latent shape mismatch")
    for value in (prompt, image):
        if value.dtype.kind != "f" or np.any(np.abs(value) > np.finfo(np.float32).max):
            raise ValueError("model inputs must be representable as float32")
    timestep = np.load(reference / f"timestep_{step:03d}.npy", allow_pickle=False)
    if (not np.isfinite(prompt).all() or not np.isfinite(image).all() or timestep.size != 1 or
            not np.isfinite(timestep).all() or not 0 <= float(timestep.reshape(-1)[0]) <= 1):
        raise ValueError("invalid/nonfinite model inputs")

    output.mkdir(parents=True, exist_ok=False)
    values = [mask.shape[1], nt, len(shapes), *mask[0].astype(int).tolist()]
    values += [v for _, h, w in shapes for v in (h, w)]
    (output / "layout.txt").write_text(" ".join(map(str, values)) + "\n")
    np.save(output / "prompt_embeds.npy", np.ascontiguousarray(prompt, dtype=np.float32))
    np.save(output / "condition_latents.npy", np.ascontiguousarray(image[:-target], dtype=np.float32))
    np.save(output / "target_latents.npy", np.ascontiguousarray(image[-target:], dtype=np.float32))
    result = {"reference": str(reference.resolve()), "step": step, "branch": branch,
              "timestep": float(timestep.reshape(-1)[0]), "condition_tokens": total - target,
              "target_tokens": target, "height_tokens": shapes[-1][1], "width_tokens": shapes[-1][2],
              "text_slots": nt, "native_editing_validated": False}
    (output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--branch", choices=("positive", "negative"), default="positive")
    args = ap.parse_args()
    print(json.dumps(prepare(args.reference_dir, args.out_dir, args.step, args.branch), indent=2))


if __name__ == "__main__":
    main()
