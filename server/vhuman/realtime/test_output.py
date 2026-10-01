"""Sample-timed CFR recording and exact PCM sidecar checks without a GPU."""
import shutil
import subprocess
from pathlib import Path
import wave
import unittest
import numpy as np
from .src.ui.output import FrameOutput

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class RecordingTests(unittest.TestCase):
    def test_playout_audio_mux_and_missed_frame_slots(self):
        work = ROOT / "tmp/vhuman-realtime/recording-tests"
        work.mkdir(parents=True, exist_ok=True)
        video = work / "sample.mp4"
        sink = FrameOutput(size=(16, 16), fps=10, video=video)
        sink.prepare = lambda handle: np.full((16, 16, 3), 128, np.uint8)
        pcm = np.sin(np.arange(12000)*2*np.pi*440/24000).astype(np.float32)*.1
        try:
            sink.write(None, 0)
            sink.audio(pcm[:2400]); sink.write(None, 2400)
            sink.audio(pcm[2400:9600]); sink.write(None, 9600)
            sink.audio(pcm[9600:]); sink.write(None, 12000)
        finally: sink.close()
        with wave.open(str(video.with_name("sample.playout.wav")), "rb") as wav:
            self.assertEqual(wav.getnframes(), 12000)
            self.assertEqual((wav.getnchannels(), wav.getframerate()), (1, 24000))
            actual = np.frombuffer(wav.readframes(12000), "<i2")
        np.testing.assert_array_equal(actual, (pcm*32767).round().astype("<i2"))
        decoded = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-map", "0:a:0",
            "-f", "s16le", "-"], check=True, capture_output=True).stdout
        self.assertGreaterEqual(len(decoded)//2, 12000)
        self.assertLess(len(decoded)//2, 12000+1024)
        frames = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
        self.assertEqual(len(frames)//(16*16*3), 6)


if __name__ == "__main__": unittest.main()
