"""Speech-to-rig timeline and take export tests; no model weights required."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from . import gpu
from .rig import rigdef, speech
from .service import EyeService, ROOT, ServiceError
from .test_rig import _toy_rig


def _aux(duration=.5, fps=30):
    n = math.ceil(duration * fps) + 1
    frames = [[0.0] * len(speech.VISEMES) for _ in range(n)]
    for i, row in enumerate(frames):
        row[speech.VISEMES.index("aa" if 3 <= i < n - 3 else "sil")] = 1.0
    return {"format": "ja_align.v1", "duration": duration,
            "visemes": {"fps": fps, "names": speech.VISEMES, "frames": frames}}


class SpeechTimelineTests(unittest.TestCase):
    def test_all_visemes_have_bounded_rig_poses(self):
        for label in speech.VISEMES:
            aux = _aux()
            for row in aux["visemes"]["frames"]:
                row[:] = [float(name == label) for name in speech.VISEMES]
            frames = speech.build_frames(aux)
            self.assertEqual(len(frames), 16)
            self.assertEqual(frames[0]["t"], 0)
            self.assertEqual(frames[-1]["v"], {})
            self.assertTrue(all(set(f["v"]) <= set(rigdef.CONTROLS) for f in frames))
            self.assertTrue(all(0 <= v <= 1 for f in frames for v in f["v"].values()))

    def test_speech_and_emotion_layers(self):
        frames = speech.build_frames(_aux(), [{"t": 0, "weights": {"joy": 1}},
                                                {"t": .5, "weights": {"anger": 1}}])
        self.assertGreater(frames[0]["v"]["cheekSquintLeft"], 0)
        self.assertLessEqual(frames[0]["v"]["mouthSmileLeft"], .35 * .6 + 1e-5)
        self.assertGreater(frames[8]["v"].get("jawOpen", 0), .5)
        self.assertGreater(frames[-2]["v"].get("browDownLeft", 0), 0)
        self.assertEqual(frames[-1]["v"], {})
        from_provider = speech.build_frames(_aux(), emotion_provider=lambda t: {"joy": 1 if t < .4 else 0})
        self.assertGreater(from_provider[0]["v"]["cheekSquintLeft"], 0)

    def test_invalid_inputs(self):
        aux = _aux()
        with self.assertRaisesRegex(ValueError, "order"):
            speech.build_frames({**aux, "visemes": {**aux["visemes"], "names": list(reversed(speech.VISEMES))}})
        with self.assertRaisesRegex(ValueError, "increasing"):
            speech.build_frames(aux, [{"t": .3, "weights": {}}, {"t": .2, "weights": {}}])
        with self.assertRaisesRegex(ValueError, "unknown emotion"):
            speech.build_frames(aux, [{"t": 0, "weights": {"unknown": 1}}])
        with self.assertRaisesRegex(ValueError, "speech_strength"):
            speech.build_frames(aux, speech_strength=float("nan"))


class SpeechTakeTests(unittest.TestCase):
    def test_subprocess_cancellation(self):
        cancel = threading.Event()
        timer = threading.Timer(.2, cancel.set)
        timer.start()
        try:
            with self.assertRaises(gpu.Cancelled):
                speech._run([sys.executable, "-c", "import time; time.sleep(30)"], cancel, lambda *_: None)
        finally:
            timer.join()

    def test_rebuild_take_and_guard_files(self):
        scratch = ROOT / "tmp" / "vhuman" / "test-speech"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as d:
            service = EyeService(Path(d))
            rig_dir = service.work / "heads" / "h1" / "rig"
            rig_dir.mkdir(parents=True)
            (rig_dir / "rig.json").write_text(json.dumps(_toy_rig()))
            (rig_dir / "rig.usda").write_text("#usda 1.0\n")
            src = rig_dir / "takes" / ("a" * 12)
            src.mkdir(parents=True)
            (src / "manifest.json").write_text(json.dumps({"id": src.name, "head_id": "h1", "duration": .5,
                                                            "text": "こんにちは。", "speaker": "Ono_Anna", "seed": 7}))
            (src / "align.json").write_text(json.dumps(_aux()))
            with wave.open(str(src / "audio.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(bytes(24000))
            report = speech.speech_job(service, {"head_id": "h1", "source_take": src.name,
                                                 "emotion_keyframes": [{"t": 0, "weights": {"joy": 1}}]},
                                       lambda *_: None, threading.Event(), backend="cpu")
            self.assertEqual(report["frames"], 16)
            self.assertEqual(report["text"], "こんにちは。")
            self.assertEqual(report["backend"], "reused")
            self.assertEqual(len(service.list_takes("h1")), 2)
            self.assertEqual(set(report["urls"]), {"manifest", "audio", "align", "animation", "usd", "lightrig"})
            animation = json.loads(service.take_file("h1", report["id"], "animation.json").read_text())
            self.assertEqual(animation["format"], "vhuman.performance.v1")
            track = service.take_file("h1", report["id"], "lightrig.txt")
            times, controls = rigdef.read_track(track)
            self.assertEqual(len(times), 16)
            self.assertEqual(len(controls), 16)
            self.assertIn("Track", service.take_file("h1", report["id"], "animation.usda").read_text())
            (rig_dir / "rig.json").write_text(json.dumps({**_toy_rig(), "changed": True}))
            self.assertTrue(service.take_summary("h1", report["id"])["rig_stale"])
            with self.assertRaises(ServiceError):
                service.take_file("h1", report["id"], "../rig.json")
            with self.assertRaises(ServiceError):
                service.take_file("h1", "../" + report["id"], "audio.wav")


if __name__ == "__main__":
    unittest.main()
