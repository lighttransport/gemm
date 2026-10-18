#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert facebook/wav2vec2-lv-60-espeak-cv-ft (Apache-2.0) into the flat safetensors layout of w2v2.h.

Input : the HF repository folder (pytorch_model.bin, config.json, vocab.json)
Output: en_align.safetensors with
  - encoder tensors without the "wav2vec2." prefix ("feature_extractor.*", "feature_projection.*",
    "encoder.layers.N.*", "encoder.layer_norm.*"); the "layer" feature extractor keeps every
    conv bias and LayerNorm
  - pos_conv weight-norm folded into "encoder.pos_conv_embed.conv.weight" [1024, 64, 128]
  - "phoneme_head.*" = lm_head (392 classes, <pad> = CTC blank at index 0)
  - metadata: inter_ctc_layer = "final" (the head reads the normed final output), the vocabulary
    as a JSON list in id order, and the source name
plus en_align.vocab.txt (one label per line, id order) next to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/wav2vec2-lv-60-espeak-cv-ft")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folder = Path(args.model)
    config = json.loads((folder / "config.json").read_text())
    if config.get("feat_extract_norm") != "layer" or not config.get("do_stable_layer_norm"):
        raise SystemExit("expected the lv60 'layer' feature extractor with stable layer norm")
    sd = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
    out: dict[str, torch.Tensor] = {}
    g = v = None
    for k, t in sd.items():
        if k.endswith("masked_spec_embed") or k.startswith("quantizer") or k.startswith("project_"):
            continue
        name = k[len("wav2vec2."):] if k.startswith("wav2vec2.") else k
        if name.endswith("pos_conv_embed.conv.weight_g") or name.endswith("parametrizations.weight.original0"):
            g = t; continue
        if name.endswith("pos_conv_embed.conv.weight_v") or name.endswith("parametrizations.weight.original1"):
            v = t; continue
        if name.startswith("lm_head."):
            name = "phoneme_head." + name[len("lm_head."):]
        out[name] = t.contiguous()
    if g is None or v is None:
        raise SystemExit("positional conv weight norm parameters not found")
    g32, v32 = g.float(), v.float()
    norm = v32.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()
    out["encoder.pos_conv_embed.conv.weight"] = (g32 * v32 / norm).contiguous()
    vocab = json.loads((folder / "vocab.json").read_text())
    size = int(config["vocab_size"])
    labels = ["<unk>"] * size
    for tok, idx in vocab.items():
        if 0 <= idx < size:
            labels[idx] = tok
    if labels[0] != "<pad>" or config.get("pad_token_id", 0) != 0:
        raise SystemExit("expected <pad> (the CTC blank) at index 0")
    if out["phoneme_head.weight"].shape[0] != size:
        raise SystemExit("lm_head size does not match vocab_size")
    meta = {"inter_ctc_layer": "final", "source": "facebook/wav2vec2-lv-60-espeak-cv-ft (Apache-2.0)",
            "vocab": json.dumps(labels, ensure_ascii=False)}
    save_file(out, args.out, metadata=meta)
    Path(args.out).with_suffix(".vocab.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(out)} tensors, {size} labels")


if __name__ == "__main__":
    main()
