"""English alignment (speech/en_align) driving the facial rig."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from .rig import speech

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/mnt/nvme01/models/speech/wav2vec2-lv-60-espeak-cv-ft/en_align.safetensors")
BINARY = ROOT / "speech" / "build" / "en_align"
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def _aux(phones, duration=1.0, fps=30.0):
    names = list(speech.VISEMES)
    n = int(duration * fps) + 1
    frames = []
    for i in range(n):
        t = i / fps
        row = [0.0] * len(names)
        active = [p for p in phones if p["start"] <= t < p["end"]]
        if active:
            row[names.index(active[0]["viseme"])] = 1.0
        else:
            row[0] = 1.0
        frames.append(row)
    return {"format": "en_align.v1", "duration": duration, "mode": "forced", "phoneme_text": " ".join(p["s"] for p in phones),
            "unknown_words": 0, "phones": phones, "intervals": phones, "words": [],
            "visemes": {"fps": fps, "names": names, "frames": frames},
            "prosody": {"hop": 0.01, "rms_db": [-20.0] * (int(duration * 100) + 1), "f0_hz": [120.0] * (int(duration * 100) + 1),
                        "aperiodicity": [0.2] * (int(duration * 100) + 1)}}


class EnglishFramesTest(unittest.TestCase):
    def test_en_align_drives_the_rig(self):
        phones = [{"s": "m", "start": 0.1, "end": 0.25, "conf": 0.9, "viseme": "PP"},
                  {"s": "ɑː", "start": 0.25, "end": 0.5, "conf": 0.9, "viseme": "aa"},
                  {"s": "l", "start": 0.5, "end": 0.6, "conf": 0.9, "viseme": "nn"},
                  {"s": "θ", "start": 0.6, "end": 0.7, "conf": 0.9, "viseme": "TH"}]
        frames = speech.build_frames(_aux(phones), secondary_strength=0)
        at = lambda t: next(f["v"] for f in frames if abs(f["t"] - t) < 1e-6)
        self.assertGreater(at(0.2).get("mouthClose", 0), 0.3)          # the bilabial seals the lips
        self.assertLess(at(0.2).get("jawOpen", 0), 0.05)
        self.assertGreater(at(0.4).get("jawOpen", 0), 0.3)             # the open vowel
        self.assertGreater(at(0.5333333333333333).get("tongueUp", 0), 0.2)   # l: tongue contact
        self.assertGreater(at(0.6333333333333333).get("tongueOut", 0), 0.05)  # th: tongue tip forward

    def test_dominance_coarticulation(self):
        import numpy as np
        times = np.arange(0, 1.0, 1 / 30)
        intervals = [{"s": "ɑː", "start": 0.1, "end": 0.4, "viseme": "aa"},
                     {"s": "m", "start": 0.4, "end": 0.46, "viseme": "PP"},
                     {"s": "ɑː", "start": 0.46, "end": 0.8, "viseme": "aa"},
                     {"s": "k", "start": 0.8, "end": 0.85, "viseme": "kk"}]
        w = speech.coarticulate(intervals, times)
        self.assertTrue(np.allclose(w.sum(1), 1.0))
        pp, aa, kk, sil = (speech.VISEMES.index(v) for v in ("PP", "aa", "kk", "sil"))
        centre = int(round(0.43 * 30))
        self.assertGreater(w[centre, pp], w[centre, aa])              # the lip closure dominates the vowels around it
        self.assertGreater(w[int(round(0.25 * 30)), aa], 0.95)         # a long vowel is pure in its middle
        k = int(round(0.825 * 30))
        self.assertLess(w[k, kk], w[k, aa] + w[k, sil] + 0.5)          # a short velar yields to its surroundings
        self.assertEqual(w[0, sil], 1.0)                               # silence before speech
        self.assertLess(abs(w[int(round(0.1 * 30)), aa] - w[int(round(0.1 * 30)) - 1, aa]), 0.6)   # flanks are smooth

    @unittest.skipUnless(MODEL.is_file() and BINARY.is_file(), "English aligner model or binary unavailable")
    def test_forced_alignment_of_a_synthesized_sentence(self):
        wav = ROOT / "tmp" / "en-align" / "fox.wav"
        if not wav.is_file():
            self.skipTest("no synthesized English clip (speech/build/qwen3_tts --language English)")
        out = ROOT / "tmp" / "en-align" / "test_forced.json"
        env = dict(os.environ, OMP_NUM_THREADS="8")
        subprocess.run([str(BINARY), "--model", str(MODEL), "--wav", str(wav), "--text", TEXT, "--out", str(out)],
                       check=True, capture_output=True, env=env, timeout=600)
        aux = json.loads(out.read_text())
        self.assertEqual(aux["format"], "en_align.v1")
        self.assertEqual(aux["mode"], "forced")
        words = [w["s"].lower() for w in aux["words"]]
        self.assertEqual(words, [w.strip(".").lower() for w in TEXT.split()])
        starts = [w["start"] for w in aux["words"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(0.0 < w["start"] < w["end"] <= aux["duration"] for w in aux["words"]))
        self.assertGreater(sum(w["conf"] for w in aux["words"]) / len(aux["words"]), 0.6)
        self.assertEqual(aux["unknown_words"], 0)
        frames = speech.build_frames(aux)
        self.assertTrue(any(f["v"].get("mouthClose", 0) > 0.3 for f in frames))   # b, m, p close the lips


if __name__ == "__main__":
    unittest.main()
