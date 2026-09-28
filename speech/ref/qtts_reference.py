#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dump a PyTorch Qwen3-TTS (CustomVoice) reference run as .npy fixtures.

Uses the Apache-2.0 `qwen_tts` package as the oracle. Greedy decoding
(do_sample=False for both the talker and the code predictor) makes the run
deterministic so the C runner can be compared code-for-code.

Outputs (in --dump-dir):
  input_ids.npy       int32 [N]    assistant-formatted text token ids
  instruct_ids.npy    int32 [M]    (only when --instruct is given)
  prefill_embeds.npy  f32 [L, H]   talker prefill input embeddings
  trailing_text.npy   f32 [K, H]   per-step text embeddings added to the codec sum
  tts_pad_embed.npy   f32 [H]
  step_logits.npy     f32 [T+1, V] talker codebook-0 logits per generation step
  talker_hidden.npy   f32 [T+1, H] talker last hidden state per step
  codes.npy           int32 [T, 16]
  codec_*.npy         codec decoder intermediate stages (first chunk)
  wav.npy / out.wav   f32 24 kHz waveform
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--text", default="今日はいい天気ですね。散歩に行きましょう。")
    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--speaker", default="Ono_Anna")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--sample", action="store_true", help="sampled run (seeded); skips per-step fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--streaming", action="store_true", help="non_streaming_mode=False")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(os.cpu_count() or 16)
    torch.manual_seed(args.seed)

    from qwen_tts import Qwen3TTSModel

    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=dtype,
                                        attn_implementation="eager")
    model = tts.model
    talker = model.talker

    rec: dict[str, list] = {"logits": [], "hidden": [], "prefill": None, "trailing": None, "pad": None}
    def talker_hook(_m, _a, kw, res):
        emb = kw.get("inputs_embeds")
        if emb is not None and emb.shape[1] > 1:
            rec["prefill"] = emb[0].float().cpu().numpy()
            rec["trailing"] = kw["trailing_text_hidden"][0].float().cpu().numpy()
            rec["pad"] = kw["tts_pad_embed"].reshape(-1).float().cpu().numpy()
        rec["logits"].append(res.logits[0, -1].float().cpu().numpy())
        rec["hidden"].append(res.past_hidden[0, -1].float().cpu().numpy())

    talker.register_forward_hook(talker_hook, with_kwargs=True)

    # Codec decoder stage hooks (first call of each module = first chunk).
    dec = model.speech_tokenizer.model.decoder
    stages: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_m, _inp, outp):
            if name not in stages:
                t = outp.last_hidden_state if hasattr(outp, "last_hidden_state") else outp
                stages[name] = t[0].float().cpu().numpy()
        return fn

    codes_rec = []
    orig_st_decode = model.speech_tokenizer.decode

    def st_decode(encoded):
        c = encoded[0]["audio_codes"] if isinstance(encoded, list) else encoded["audio_codes"]
        codes_rec.append(np.asarray(c.cpu() if hasattr(c, "cpu") else c).astype(np.int32))
        return orig_st_decode(encoded)

    model.speech_tokenizer.decode = st_decode

    orig_qdecode = dec.quantizer.decode

    def qdecode(codes):
        r = orig_qdecode(codes)
        stages.setdefault("codec_rvq", r[0].float().cpu().numpy())
        return r

    dec.quantizer.decode = qdecode
    dec.pre_conv.register_forward_hook(hook("codec_preconv"))
    dec.pre_transformer.register_forward_hook(hook("codec_pretf"))
    for i, blk in enumerate(dec.upsample):
        blk[1].register_forward_hook(hook(f"codec_up{i}"))
    for i, blk in enumerate(dec.decoder):
        blk.register_forward_hook(hook(f"codec_dec{i}"))

    gen = dict(max_new_tokens=args.max_new_tokens)
    if args.sample:
        gen.update(do_sample=True, subtalker_dosample=True)
    else:
        gen.update(do_sample=False, subtalker_dosample=False)
    with torch.no_grad():
        wavs, sr = tts.generate_custom_voice(
            text=args.text, language=args.language, speaker=args.speaker,
            instruct=args.instruct or None, non_streaming_mode=not args.streaming, **gen)
    wav = np.asarray(wavs[0], dtype=np.float32)

    # Re-run tokenization so the ids are recorded exactly as the wrapper builds them.
    ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0][0].cpu().numpy().astype(np.int32)
    np.save(out / "input_ids.npy", ids)
    if args.instruct:
        iid = tts._tokenize_texts([tts._build_instruct_text(args.instruct)])[0][0].cpu().numpy()
        np.save(out / "instruct_ids.npy", iid.astype(np.int32))

    np.save(out / "codes.npy", codes_rec[0])
    np.save(out / "wav.npy", wav)
    import soundfile as sf
    sf.write(out / "out.wav", wav, sr)
    if not args.sample:
        np.save(out / "prefill_embeds.npy", rec["prefill"])
        np.save(out / "trailing_text.npy", rec["trailing"])
        np.save(out / "tts_pad_embed.npy", rec["pad"])
        np.save(out / "step_logits.npy", np.stack(rec["logits"]))
        np.save(out / "talker_hidden.npy", np.stack(rec["hidden"]))
    for k, v in stages.items():
        np.save(out / f"{k}.npy", v)
    meta = dict(text=args.text, language=args.language, speaker=args.speaker, instruct=args.instruct,
                sample_rate=sr, n_samples=int(wav.shape[0]), dtype=args.dtype, sample=args.sample,
                seed=args.seed, streaming=args.streaming)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
