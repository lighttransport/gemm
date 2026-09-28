#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Intelligibility eval: official PyTorch Qwen3-TTS vs the C/CUDA runner.

For each sentence and seed:
  - torch : qwen_tts generate_custom_voice (bf16, CUDA, sampling with torch seed)
  - c     : speech/build/qwen3_tts_cuda --backend cuda --seed N
Both wavs are transcribed with the hiragana-ctc model (HF wav2vec2, greedy CTC) and
scored by kana CER against the pyopenjtalk pronunciation reading (pyopenjtalk is used
only as an evaluation oracle). Prints a table and writes eval.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from align_reference import KANA, load_audio  # noqa: E402


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def norm_kana(s: str) -> str:
    s = kata2hira(s)
    return "".join(c for c in s if c in KANA)


def cer(ref: str, hyp: str) -> float:
    d = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(hyp)] / max(len(ref), 1)


class Recognizer:
    def __init__(self, ckpt: str, config: str, device: str):
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg = Wav2Vec2Config.from_pretrained(config)
        cfg.mask_time_prob = 0.0
        self.enc = Wav2Vec2Model(cfg)
        sd = {k[8:]: v for k, v in ck["model_state_dict"].items() if k.startswith("encoder.") and "masked_spec" not in k}
        self.enc.load_state_dict(sd, strict=False)
        self.enc = self.enc.float().eval().to(device)
        self.head = torch.nn.Linear(1024, 83).to(device)
        self.head.weight.data = ck["model_state_dict"]["kana_head.weight"].float().to(device)
        self.head.bias.data = ck["model_state_dict"]["kana_head.bias"].float().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, wav_path: str) -> str:
        x = load_audio(wav_path)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        h = self.enc(torch.from_numpy(x.astype(np.float32))[None].to(self.device)).last_hidden_state
        ids = self.head(h)[0].argmax(-1).cpu().numpy()
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(KANA[i - 1])
            prev = i
        return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--ckpt", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/best-medium-ep5-inference.pt")
    ap.add_argument("--config", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large")
    ap.add_argument("--sentences", default=str(Path(__file__).parent.parent / "tests" / "ja_sentences.txt"))
    ap.add_argument("--runner", default=str(Path(__file__).parent.parent / "build" / "qwen3_tts_cuda"))
    ap.add_argument("--speaker", default="Ono_Anna")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import pyopenjtalk

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sents = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()]
    if args.limit:
        sents = sents[: args.limit]

    tts = None
    if not args.skip_torch:
        from qwen_tts import Qwen3TTSModel
        tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16,
                                            attn_implementation="eager")
    rows = []
    for si, text in enumerate(sents):
        ref = norm_kana(pyopenjtalk.g2p(text, kana=True))
        for seed in args.seeds:
            item = {"id": si, "seed": seed, "text": text, "ref": ref}
            cw = out / f"c_{si:02d}_{seed}.wav"
            t0 = time.time()
            r = subprocess.run([args.runner, "--backend", "cuda", "--quiet", "--seed", str(seed), "--text", text,
                                "--speaker", args.speaker, "--out", str(cw)], capture_output=True, text=True)
            item["c_time"] = time.time() - t0
            if r.returncode:
                print(r.stderr, file=sys.stderr)
            if tts is not None:
                import soundfile as sf
                torch.manual_seed(seed)
                t0 = time.time()
                wavs, sr = tts.generate_custom_voice(text=text, language="Japanese", speaker=args.speaker)
                item["torch_time"] = time.time() - t0
                tw = out / f"t_{si:02d}_{seed}.wav"
                sf.write(tw, wavs[0], sr)
            rows.append(item)
    del tts
    torch.cuda.empty_cache()
    rec = Recognizer(args.ckpt, args.config, "cuda")
    for item in rows:
        for sysname in ("c", "t"):
            w = out / f"{sysname}_{item['id']:02d}_{item['seed']}.wav"
            if not w.exists():
                continue
            import soundfile as sf
            info = sf.info(str(w))
            hyp = rec(str(w))
            item[f"{sysname}_hyp"] = hyp
            item[f"{sysname}_cer"] = cer(item["ref"], hyp)
            item[f"{sysname}_dur"] = info.duration
    (out / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for sysname, label in (("c", "C/CUDA runner"), ("t", "PyTorch qwen_tts")):
        v = [r[f"{sysname}_cer"] for r in rows if f"{sysname}_cer" in r]
        if v:
            print(f"{label:18s} kana CER mean {np.mean(v) * 100:5.1f}%  median {np.median(v) * 100:5.1f}%  "
                  f"n={len(v)}  >20%: {sum(x > 0.2 for x in v)}")
    for r in rows:
        print(f"[{r['id']:2d}/{r['seed']}] c={r.get('c_cer', -1) * 100:5.1f}% t={r.get('t_cer', -1) * 100:5.1f}%  "
              f"{r['text']}\n      ref {r['ref']}\n      c   {r.get('c_hyp', '')}\n      t   {r.get('t_hyp', '')}")


if __name__ == "__main__":
    main()
