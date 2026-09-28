#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pitch-accent check for Japanese speech (Tokyo-dialect accent phrases).

Inputs: TTS outputs from asr_eval.py (--eval-dir, systems c/t) and/or natural speech
(--jsut DIR --jsut-n N) as a human reference measured the same way.

Per utterance:
  1. pyopenjtalk full-context labels (evaluation oracle only): phonemes, accent phrases
     (mora count n, accent type k) and each phoneme's mora index.
  2. ja_align force-aligns the phonemes and measures F0 (YIN, 10 ms).
  3. Each mora gets the median F0 in semitones over [vowel start, next mora start).
Per accent phrase with >= 2 voiced moras:
  - accented (1 <= k < n): the largest adjacent F0 drop must sit after mora k or k+1
    (one mora of lag tolerance: the F0 fall is realised late) and be >= 1 semitone
  - unaccented (k == 0 or k == n): no adjacent drop >= 1.5 semitones inside the phrase
  - initial: mora 1 -> 2 rises for k != 1, falls for k == 1 (|step| >= 0.3 st)
The chance baseline scores the same F0 against accent types shuffled across phrases of
equal length.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

VOWELS = set("aiueoAIUEO") | {"N", "cl"}


def parse_labels(text: str):
    import pyopenjtalk
    phones, moras = [], []
    for lab in pyopenjtalk.extract_fullcontext(text):
        p = re.search(r"-(.+?)\+", lab).group(1)
        if p in ("sil", "pau"):
            continue
        a2 = int(re.search(r"/A:[^+]+\+(\d+)\+", lab).group(1))
        f = re.search(r"/F:(\d+)_(\d+)#", lab)
        phrase = (re.search(r"/I:[^@]+@(\d+)", lab).group(1), re.search(r"/F:[^@]+@(\d+)_", lab).group(1))
        idx = len(phones)
        phones.append(p)
        if moras and moras[-1]["phrase"] == phrase and moras[-1]["i"] == a2:
            moras[-1]["ph"].append(idx)
        else:
            moras.append({"phrase": phrase, "i": a2, "k": int(f.group(2)), "n": int(f.group(1)), "ph": [idx]})
    return phones, moras


def mora_f0(aux: dict, moras, phones):
    segs = aux["phones"]
    if len(segs) != len(phones):
        return None
    f0 = np.asarray(aux["prosody"]["f0_hz"], dtype=np.float64)
    hop = aux["prosody"]["hop"]
    starts = [s["start"] for s in segs]
    out = []
    for mi, m in enumerate(moras):
        last = m["ph"][-1]
        a = starts[last] if phones[last] in VOWELS else starts[m["ph"][0]]
        b = starts[moras[mi + 1]["ph"][0]] if mi + 1 < len(moras) else segs[last]["end"] + 0.08
        b = max(b, a + 0.04)
        v = f0[int(a / hop):int(np.ceil(b / hop)) + 1]
        v = v[v > 0]
        out.append(float(12 * np.log2(np.median(v) / 100.0)) if len(v) >= 2 and phones[last] not in "AIUEO" else None)
    return out


def phrases(moras, st):
    groups = {}
    for m, s in zip(moras, st):
        groups.setdefault(m["phrase"], []).append((m, s))
    return list(groups.values())


def score_phrase(ph, k):
    n = ph[0][0]["n"]
    st = [s for _, s in ph]
    idx = [m["i"] for m, _ in ph]
    drops = [(st[j] - st[j + 1], idx[j]) for j in range(len(ph) - 1)
             if st[j] is not None and st[j + 1] is not None and idx[j + 1] == idx[j] + 1]
    r = {}
    if drops:
        dmax, at = max(drops)
        if 1 <= k < n:
            r["nucleus"] = dmax >= 1.0 and at in (k, k + 1)
        else:
            r["flat"] = dmax < 1.5
    if len(st) >= 2 and st[0] is not None and st[1] is not None and idx[0] == 1 and idx[1] == 2:
        d = st[1] - st[0]
        r["initial"] = d <= -0.3 if k == 1 else d >= 0.3
    return r


def summarize(items, rng):
    acc = {"nucleus": [0, 0], "flat": [0, 0], "initial": [0, 0]}
    chance = {"nucleus": [0, 0], "flat": [0, 0], "initial": [0, 0]}
    by_n = {}
    for moras, st in items:
        for ph in phrases(moras, st):
            by_n.setdefault(ph[0][0]["n"], []).append(ph[0][0]["k"])
    for moras, st in items:
        for ph in phrases(moras, st):
            for tgt, k in ((acc, ph[0][0]["k"]), (chance, rng.choice(by_n[ph[0][0]["n"]]))):
                for key, ok in score_phrase(ph, k).items():
                    tgt[key][0] += ok
                    tgt[key][1] += 1
    return acc, chance


def run_align(ja_align, model, wav, phones, extra, tmpdir):
    with tempfile.NamedTemporaryFile(suffix=".json", dir=tmpdir) as tf:
        subprocess.run([ja_align, "--model", model, "--wav", str(wav), "--phonemes", " ".join(phones),
                        "--out", tf.name] + extra, capture_output=True, check=True)
        return json.loads(Path(tf.name).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--jsut", default="", help="jsut_ver1.1 directory (basic5000)")
    ap.add_argument("--jsut-n", type=int, default=40)
    ap.add_argument("--ja-align", default=str(Path(__file__).parent.parent / "build" / "ja_align"))
    ap.add_argument("--model", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--tmp", default="tmp/speech")
    args = ap.parse_args()
    extra = args.extra_args.split()
    systems: dict[str, list] = {}
    if args.eval_dir:
        ed = Path(args.eval_dir)
        for r in json.loads((ed / "eval.json").read_text()):
            phones, moras = parse_labels(r["text"])
            for s in ("c", "t"):
                w = ed / f"{s}_{r['id']:02d}_{r['seed']}.wav"
                if w.exists():
                    systems.setdefault(s, []).append((w, phones, moras))
    if args.jsut:
        d = Path(args.jsut) / "basic5000"
        lines = (d / "transcript_utf8.txt").read_text().splitlines()[: args.jsut_n]
        for line in lines:
            uid, text = line.split(":", 1)
            w = d / "wav" / f"{uid}.wav"
            if w.exists():
                phones, moras = parse_labels(text)
                systems.setdefault("jsut", []).append((w, phones, moras))
    names = {"c": "C/CUDA runner", "t": "PyTorch qwen_tts", "jsut": "JSUT (human)"}
    for s, lst in systems.items():
        items = []
        for w, phones, moras in lst:
            aux = run_align(args.ja_align, args.model, w, phones, extra, args.tmp)
            st = mora_f0(aux, moras, phones) if aux["mode"] == "forced" else None
            if st is not None:
                items.append((moras, st))
        acc, ch = summarize(items, random.Random(0))
        fmt = lambda a: f"{a[0] / max(a[1], 1) * 100:5.1f}% ({a[0]}/{a[1]})"
        print(f"{names[s]:18s} n={len(items):3d} | nucleus {fmt(acc['nucleus'])} chance {ch['nucleus'][0] / max(ch['nucleus'][1], 1) * 100:4.1f}%"
              f" | unaccented-flat {fmt(acc['flat'])} | initial rise/fall {fmt(acc['initial'])}"
              f" chance {ch['initial'][0] / max(ch['initial'][1], 1) * 100:4.1f}%")


if __name__ == "__main__":
    main()
