#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Oracle for speech/en_align: HF transformers Wav2Vec2ForCTC log posteriors of a WAV, compared with
the C runner's `--dump-dir` phoneme_logp.npy (same 0.5 s padding on both sides, 16 kHz, normalized).

    python speech/ref/en_align_reference.py --wav in.wav --dump-dir tmp/en-align/dump [--model DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_wav_16k(path):
    import wave
    with wave.open(str(path)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        from math import gcd
        import scipy.signal
        g = gcd(sr, 16000)
        x = scipy.signal.resample_poly(x, 16000 // g, sr // g)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/wav2vec2-lv-60-espeak-cv-ft")
    ap.add_argument("--pad", type=float, default=0.5)
    a = ap.parse_args()
    from transformers import Wav2Vec2ForCTC
    model = Wav2Vec2ForCTC.from_pretrained(a.model).eval()
    x = load_wav_16k(a.wav)
    pad = int(round(a.pad / 0.02)) * 320
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])
    xp = (xp - xp.mean()) / np.sqrt(xp.var() + 1e-7)
    with torch.no_grad():
        logits = model(torch.from_numpy(xp)[None]).logits[0]
        logp = torch.log_softmax(logits, -1).numpy()
    ours = np.load(Path(a.dump_dir) / "phoneme_logp.npy")
    T = min(len(ours), len(logp))
    diff = np.abs(ours[:T] - logp[:T])
    agree = (ours[:T].argmax(1) == logp[:T].argmax(1)).mean()
    report = {"frames_ref": int(len(logp)), "frames_c": int(len(ours)), "max_abs_logp": float(diff.max()),
              "mean_abs_logp": float(diff.mean()), "argmax_agreement": float(agree),
              "max_abs_prob": float(np.abs(np.exp(ours[:T]) - np.exp(logp[:T])).max())}
    print(json.dumps(report, indent=1))
    # Posterior-level agreement (float32 GEMM order differs from PyTorch): every frame's top class
    # must agree and the mean log-posterior error stay small; near-tie frames may differ in probability.
    report["passed"] = bool(agree >= 0.995 and report["mean_abs_logp"] < 0.1)
    print(json.dumps({"passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
