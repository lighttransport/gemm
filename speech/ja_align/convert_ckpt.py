#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert the hiragana-ctc PyTorch checkpoint into a flat safetensors file for ja_align.

Input : best-medium-ep5-inference.pt from sakasegawa/japanese-wav2vec2-large-hiragana-ctc
        (Apache-2.0; wav2vec2-large encoder + phoneme head @ layer 12 + kana head @ final layer)
Output: ja_align.safetensors with
  - all encoder tensors, prefix "encoder." removed ("feature_extractor.*", "encoder.layers.N.*", ...)
  - pos_conv weight-norm folded: "encoder.pos_conv_embed.conv.weight" [1024, 64, 128] F32
  - kana_head.*, phoneme_head.* (F32)
  - metadata: inter_ctc_layer, source checkpoint name
BF16 tensors stay BF16 (the C loader widens to F32, exactly like the reference's model.float()).
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/best-medium-ep5-inference.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    out: dict[str, torch.Tensor] = {}
    g = v = None
    for k, t in sd.items():
        if k == "encoder.masked_spec_embed":
            continue
        if k.endswith("pos_conv_embed.conv.parametrizations.weight.original0"):
            g = t
            continue
        if k.endswith("pos_conv_embed.conv.parametrizations.weight.original1"):
            v = t
            continue
        name = k[len("encoder."):] if k.startswith("encoder.") else k
        out[name] = t.contiguous()
    # torch weight_norm(dim=2): w = g * v / ||v|| with the norm over dims (0, 1) per kernel tap
    g32, v32 = g.float(), v.float()
    norm = v32.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()
    out["encoder.pos_conv_embed.conv.weight"] = (g32 * v32 / norm).contiguous()
    meta = {"inter_ctc_layer": str(ck.get("inter_ctc_layer", 12)),
            "pretrained": str(ck.get("pretrained", "")),
            "source": "sakasegawa/japanese-wav2vec2-large-hiragana-ctc best-medium-ep5-inference.pt"}
    save_file(out, args.out, metadata=meta)
    print(f"wrote {args.out}: {len(out)} tensors, inter_ctc_layer={meta['inter_ctc_layer']}")


if __name__ == "__main__":
    main()
