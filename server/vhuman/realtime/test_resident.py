"""Optional real-model integration checks; absent local weights skip explicitly."""
from pathlib import Path
import queue
import time
import unittest
from .src.avatar.provenance import sha256
from .src.tts.resident import ResidentTTS

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "tmp/vhuman-realtime/resident-tests"
MODEL = Path("/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice")
RUNNER = ROOT / "tmp/vhuman-realtime/speech/qwen3_tts_cuda"


class ResidentIntegrationTests(unittest.TestCase):
    def test_repeated_requests_cancel_and_reuse(self):
        if not RUNNER.exists() or not (MODEL / "model.safetensors").exists():
            self.skipTest("requires locally built CUDA runner and Qwen0.6B weights")
        worker = ResidentTTS(RUNNER, MODEL, sha256(MODEL / "model.safetensors"), WORK,
                             max_frames=16, threads=4)
        try:
            pid = worker.process.pid
            timings = []
            for epoch in (0, 1):
                worker.submit("こんにちは。", epoch)
                self._drain(worker, epoch)
                timings.append((worker.first_audio_ns - worker.started_ns) / 1e6)
                self.assertEqual(worker.process.pid, pid)
            worker.submit("今日はいい天気ですね。", 2)
            deadline = time.monotonic() + 10
            while worker.first_audio_ns is None and time.monotonic() < deadline:
                worker.check(); time.sleep(.005)
            self.assertIsNotNone(worker.first_audio_ns)
            worker.cancel()
            self.assertTrue(worker.audio.empty() and worker.features.empty())
            worker.submit("ありがとう。", 3)
            self._drain(worker, 3)
            self.assertEqual(worker.process.pid, pid)
            print(f"resident warm first PCM ms: {timings}")
        finally:
            worker.close()

    def _drain(self, worker, epoch):
        starts = [[], []]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            worker.check()
            for index, destination in enumerate((worker.audio, worker.features)):
                while True:
                    try: record = destination.get_nowait()
                    except queue.Empty: break
                    self.assertEqual(record.epoch, epoch)
                    starts[index].append(record.sample_start)
            if worker.done_audio.is_set() and worker.done_features.is_set() and worker.audio.empty() and worker.features.empty():
                self.assertTrue(starts[0])
                self.assertEqual(starts[0], starts[1])
                self.assertEqual(starts[0], list(range(0, len(starts[0]) * 1920, 1920)))
                return
            time.sleep(.005)
        self.fail("resident request did not finish within20 seconds")


if __name__ == "__main__": unittest.main()
