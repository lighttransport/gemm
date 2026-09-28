#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Voice-clone quality: speaker similarity + intelligibility of the C/CUDA runner.

Reference speakers: JSUT (female) clips resampled to 24 kHz. For each target sentence the
runner synthesizes speech in ICL mode (reference transcript + codes) and x-vector mode.
Metrics (evaluation oracles only, not part of the runtime):
  - speaker similarity: cosine of SpeechBrain ECAPA (VoxCeleb) embeddings between the
    reference clip and the output. Anchors: another utterance of the same JSUT speaker
    (same-speaker ceiling) and a CustomVoice preset voice (different speaker).
  - kana CER: hiragana-ctc greedy transcription (0.5 s padding) vs pyopenjtalk reading.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import asr_eval as A  # noqa: E402
import align_reference as R  # noqa: E402


def resample_to(path_in: str, path_out: str, sr_out: int = 24000) -> None:
    import torchaudio.functional as F
    x, sr = sf.read(path_in, dtype="float32")
    sf.write(path_out, F.resample(torch.from_numpy(x), sr, sr_out).numpy(), sr_out, subtype="FLOAT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=str(Path(__file__).parent.parent / "build" / "qwen3_tts_cuda"))
    ap.add_argument("--base", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--custom", default="/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--jsut", default="/mnt/nvme01/datasets/jsut_ver1.1/basic5000")
    ap.add_argument("--ref-id", default="BASIC5000_0002")
    ap.add_argument("--anchor-id", default="BASIC5000_0003")
    ap.add_argument("--sentences", default=str(Path(__file__).parent.parent / "tests" / "ja_sentences.txt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import pyopenjtalk
    from speechbrain.inference.speaker import EncoderClassifier

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trans = dict(l.split(":", 1) for l in (Path(args.jsut) / "transcript_utf8.txt").read_text().splitlines()[:200])
    ref = out / "ref24.wav"
    anchor = out / "anchor24.wav"
    resample_to(str(Path(args.jsut) / "wav" / f"{args.ref_id}.wav"), str(ref))
    resample_to(str(Path(args.jsut) / "wav" / f"{args.anchor_id}.wav"), str(anchor))
    ref_text = trans[args.ref_id]
    sents = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()][: args.n]

    def run(tag: str, i: int, text: str, extra: list[str], model: str) -> Path:
        w = out / f"{tag}_{i:02d}.wav"
        r = subprocess.run([args.runner, "--backend", "cuda", "--quiet", "--model", model, "--seed", str(i + 1),
                            "--text", text, "--out", str(w)] + extra, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr, file=sys.stderr)
        return w

    rows = []
    for i, text in enumerate(sents):
        rows.append({"text": text, "ref_kana": A.norm_kana(pyopenjtalk.g2p(text, kana=True)), "wav": {
            "icl": run("icl", i, text, ["--ref-wav", str(ref), "--ref-text", ref_text], args.base),
            "xvec": run("xvec", i, text, ["--ref-wav", str(ref), "--xvec-only"], args.base),
            "preset": run("preset", i, text, ["--speaker", "Ono_Anna"], args.custom),
        }})
    spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/mnt/nvme01/models/speech/spkrec-ecapa-voxceleb")

    def emb(path: Path) -> np.ndarray:
        x = R.load_audio(str(path))  # 16 kHz mono
        e = spk.encode_batch(torch.from_numpy(x)[None]).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    e_ref = emb(ref)
    rec = A.Recognizer("/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/best-medium-ep5-inference.pt",
                       "/mnt/nvme01/models/speech/japanese-wav2vec2-large", "cuda")
    orig = R.load_audio
    pad = lambda p: np.concatenate([np.zeros(8000, np.float32), orig(p), np.zeros(8000, np.float32)])
    res = {k: {"sim": [], "cer": []} for k in ("icl", "xvec", "preset")}
    for r in rows:
        for k, w in r["wav"].items():
            if not w.exists():
                continue
            res[k]["sim"].append(float(emb(w) @ e_ref))
            A.load_audio = pad
            res[k]["cer"].append(A.cer(r["ref_kana"], rec(str(w))))
            A.load_audio = orig
    anchor_sim = float(emb(anchor) @ e_ref)
    print(f"same-speaker anchor ({args.anchor_id}) similarity: {anchor_sim:.3f}")
    for k, label in (("icl", "clone, ICL (ref text+codes)"), ("xvec", "clone, x-vector only"),
                     ("preset", "CustomVoice Ono_Anna")):
        s, c = res[k]["sim"], res[k]["cer"]
        if s:
            print(f"{label:28s} speaker sim {np.mean(s):.3f} (min {np.min(s):.3f})  kana CER {np.mean(c) * 100:5.1f}%  n={len(s)}")
    (out / "clone_eval.json").write_text(json.dumps({"anchor": anchor_sim, "res": res}, indent=1))


if __name__ == "__main__":
    main()
