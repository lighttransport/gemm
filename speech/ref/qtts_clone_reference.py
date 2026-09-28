#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PyTorch reference for Qwen3-TTS voice cloning (Base model) -> .npy fixtures.

Uses the Apache-2.0 qwen_tts package: create_voice_clone_prompt + generate_voice_clone,
greedy decoding. The reference clip must already be 24 kHz mono (no resampling here, so
the C runner sees identical samples). Dumps (in --dump-dir):
  ref_wav.npy          f32 [N]        reference waveform (24 kHz)
  spk_mel.npy          f32 [T, 128]   speaker-encoder log-mel input
  spk_emb.npy          f32 [2048]     x-vector
  enc_seanet.npy       f32 [T25, 512] Mimi SEANet encoder output
  enc_tf.npy           f32 [T25, 512] encoder transformer output
  enc_down.npy         f32 [T12, 512] downsampled embeddings
  ref_codes.npy        i32 [T12, 16]  reference codes (ICL mode)
  input_ids.npy / ref_ids.npy         token ids
  prefill_embeds.npy / step_logits.npy / talker_hidden.npy / codes.npy / wav.npy
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
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--text", default="今日はいい天気ですね。散歩に行きましょう。")
    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--xvec-only", action="store_true")
    ap.add_argument("--non-streaming", action="store_true")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(os.cpu_count() or 16)
    torch.manual_seed(args.seed)
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(args.ref_wav, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    assert sr == 24000, "reference clip must be 24 kHz"
    np.save(out / "ref_wav.npy", wav)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32,
                                        attn_implementation="eager")
    model = tts.model
    rec: dict = {"logits": [], "hidden": []}

    def talker_hook(_m, _a, kw, res):
        emb = kw.get("inputs_embeds")
        if emb is not None and emb.shape[1] > 1:
            rec["prefill"] = emb[0].float().cpu().numpy()
        rec["logits"].append(res.logits[0, -1].float().cpu().numpy())
        rec["hidden"].append(res.past_hidden[0, -1].float().cpu().numpy())

    model.talker.register_forward_hook(talker_hook, with_kwargs=True)
    spk = model.speaker_encoder
    def spk_hook(_m, i, _o):
        rec.setdefault("mel", i[0][0].float().cpu().numpy())

    spk.register_forward_hook(spk_hook)
    enc = model.speech_tokenizer.model.encoder

    def hk(name, transpose):
        def f(_m, _i, o):
            t = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, tuple) else o)
            t = t[0].float().cpu().numpy()
            rec.setdefault(name, t.T if transpose else t)
        return f

    enc.encoder.register_forward_hook(hk("enc_seanet", True))
    enc.encoder_transformer.register_forward_hook(hk("enc_tf", False))
    enc.downsample.register_forward_hook(hk("enc_down", True))
    codes_rec = []
    orig_decode = model.speech_tokenizer.decode

    def st_decode(encoded):
        c = encoded[0]["audio_codes"] if isinstance(encoded, list) else encoded["audio_codes"]
        codes_rec.append(np.asarray(c.cpu()).astype(np.int32))
        return orig_decode(encoded)

    model.speech_tokenizer.decode = st_decode

    with torch.no_grad():
        items = tts.create_voice_clone_prompt(ref_audio=(wav, 24000), ref_text=args.ref_text or None,
                                              x_vector_only_mode=args.xvec_only)
        gen = dict(max_new_tokens=args.max_new_tokens)
        gen.update(do_sample=args.sample, subtalker_dosample=args.sample)
        wavs, osr = tts.generate_voice_clone(text=args.text, language=args.language, voice_clone_prompt=items,
                                             non_streaming_mode=args.non_streaming, **gen)
    it = items[0]
    np.save(out / "spk_emb.npy", it.ref_spk_embedding.float().cpu().numpy())
    if it.ref_code is not None:
        np.save(out / "ref_codes.npy", it.ref_code.cpu().numpy().astype(np.int32))
    for k in ("mel", "enc_seanet", "enc_tf", "enc_down"):
        if k in rec:
            np.save(out / f"{'spk_mel' if k == 'mel' else k}.npy", rec[k])
    ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0][0].cpu().numpy().astype(np.int32)
    np.save(out / "input_ids.npy", ids)
    if args.ref_text:
        rid = tts._tokenize_texts([tts._build_ref_text(args.ref_text)])[0][0].cpu().numpy().astype(np.int32)
        np.save(out / "ref_ids.npy", rid)
    full = codes_rec[0]
    nref = 0 if it.ref_code is None else it.ref_code.shape[0]
    np.save(out / "codes.npy", full[nref:])
    if not args.sample:
        np.save(out / "prefill_embeds.npy", rec["prefill"])
        np.save(out / "step_logits.npy", np.stack(rec["logits"]))
        np.save(out / "talker_hidden.npy", np.stack(rec["hidden"]))
    w = np.asarray(wavs[0], dtype=np.float32)
    np.save(out / "wav.npy", w)
    sf.write(out / "out.wav", w, osr)
    meta = dict(text=args.text, ref_text=args.ref_text, xvec_only=args.xvec_only, non_streaming=args.non_streaming,
                n_ref_codes=int(nref), n_codes=int(full.shape[0] - nref), n_samples=int(w.shape[0]))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
