"""SenseVoice adapter tests with a tiny stand-in runtime (no model download)."""
from __future__ import annotations

import math
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from .rig import emotion
from .service import ROOT


class EmotionTests(unittest.TestCase):
    def test_tags_and_missing_tags(self):
        self.assertEqual(emotion.parse_tag("log\n<|ja|><|HAPPY|><|Speech|>こんにちは"), ("HAPPY", "joy"))
        self.assertEqual(emotion.parse_tag("<|EMO_UNKNOWN|>"), ("EMO_UNKNOWN", "neutral"))
        with self.assertRaisesRegex(ValueError, "no emotion tag"):
            emotion.parse_tag("<|ja|>こんにちは")

    def test_japanese_window_timeline_and_resampling(self):
        scratch = ROOT / "tmp/vhuman/test-emotion"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"fake-model")
            runner = root / "sensevoice"
            runner.write_text("#!/usr/bin/env python3\n"
                              "import sys, wave\n"
                              "with wave.open(sys.argv[sys.argv.index('-a') + 1]) as w:\n"
                              "    assert w.getframerate() == 16000 and w.getnchannels() == 1\n"
                              "tag = 'HAPPY' if 'window-0' in sys.argv[sys.argv.index('-a') + 1] else 'ANGRY'\n"
                              "print(f'<|ja|><|{tag}|><|Speech|>test')\n")
            runner.chmod(0o755)
            wav = root / "source.wav"
            rate, seconds = 24000, 6
            frames = (struct.pack("<h", int(4000 * math.sin(2 * math.pi * 220 * i / rate)))
                      for i in range(rate * seconds))
            with wave.open(str(wav), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(b"".join(frames))
            result = emotion.extract(wav, seconds, root, threading.Event(), lambda *_: None,
                                     runner=runner, model=model)
            self.assertEqual([w["label"] for w in result["windows"]], ["joy", "anger"])
            self.assertEqual([k["t"] for k in result["emotion_keyframes"]], [0, 2.8, 3.2])
            self.assertEqual(result["emotion_keyframes"][-1]["weights"], {"anger": 1.0})
            self.assertEqual(list(root.glob("emotion-window-*.wav")), [])


if __name__ == "__main__":
    unittest.main()
