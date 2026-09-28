#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Teacher-forced codec decoder reference: codes.npy -> per-stage .npy + wav.npy.

If --codes is omitted, deterministic random codes of --frames frames are generated
(and saved as codes.npy) so the C decoder can be validated without a TTS run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice/speech_tokenizer")
    ap.add_argument("--codes", default="")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
    model = AutoModel.from_pretrained(args.tokenizer, dtype=torch.float32, attn_implementation="eager").eval()

    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.codes:
        codes = np.load(args.codes).astype(np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        codes = rng.integers(0, 2048, size=(args.frames, 16)).astype(np.int64)
    np.save(out / "codes.npy", codes.astype(np.int32))

    dec = model.decoder
    stages: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_m, _i, o):
            if name not in stages:
                t = o.last_hidden_state if hasattr(o, "last_hidden_state") else o
                stages[name] = t[0].float().numpy()
        return fn

    orig = dec.quantizer.decode

    def qdecode(c):
        r = orig(c)
        stages.setdefault("codec_rvq", r[0].float().numpy())
        return r

    dec.quantizer.decode = qdecode
    dec.pre_conv.register_forward_hook(hook("codec_preconv"))
    dec.pre_transformer.register_forward_hook(hook("codec_pretf"))
    for i, blk in enumerate(dec.upsample):
        blk[1].register_forward_hook(hook(f"codec_up{i}"))
    for i, blk in enumerate(dec.decoder):
        blk.register_forward_hook(hook(f"codec_dec{i}"))

    with torch.no_grad():
        wav = model.decode(torch.from_numpy(codes)[None]).audio_values[0].float().numpy()
    np.save(out / "wav.npy", wav.astype(np.float32))
    for k, v in stages.items():
        np.save(out / f"{k}.npy", v.astype(np.float32))
    print(f"frames={codes.shape[0]} samples={wav.shape[0]} stages={sorted(stages)}")


if __name__ == "__main__":
    main()
