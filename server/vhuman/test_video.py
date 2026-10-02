"""Video job, packaging, cancellation, and artifact boundary tests (no GPU)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from PIL import Image
from . import video, gpu
from .service import ROOT, EyeService, ServiceError
from cuda.hunyuan_video15 import native_generate as native
from ref.hunyuan_video15.compare import compare
import numpy as np

class VideoTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "tmp/vhuman/test-video"
        base.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=base))
        self.service = EyeService(self.work)
        portrait = self.work / "heads/testhead/portrait.png"
        portrait.parent.mkdir(parents=True)
        Image.new("RGB", (128, 128), (80, 100, 120)).save(portrait)

    def tearDown(self):
        shutil.rmtree(self.work)

    def test_invalid_requests(self):
        for invalid in ({"frames": 80}, {"frames": True}, {"seed": -1}, {"seed": 2**63},
                        {"preset": "fast"}, {"preset": []}, {"expression": {}},
                        {"prompt": 2}, {"head_id": "../outside"}):
            with self.subTest(invalid=invalid), self.assertRaises(ServiceError):
                video.validate({"head_id": "testhead", **invalid})

    def test_repository_backend_explicitly_disables_vendor_fallback(self):
        from .video_backend import RepositoryBackend
        backend = RepositoryBackend()
        result = {'metrics': {'repo_gemm_calls': 20, 'cublas_gemm_calls': 0, 'fallback_gemm_calls': 0}}
        with patch.object(backend.module, 'generate', return_value=result) as run:
            actual = backend.generate(frames=81, preset='fast12')
            self.assertEqual(run.call_args.kwargs['gemm'], 'repo')
            self.assertEqual(run.call_args.kwargs['gemm_fallback'], 'error')
            self.assertEqual((actual['steps'], actual['cfg'], actual['flow_shift']), (12, 1, 7))
            with self.assertRaisesRegex(ValueError, '81 frames'):
                backend.generate(frames=121)
            result['metrics']['fallback_gemm_calls'] = 1
            with self.assertRaisesRegex(RuntimeError, 'fallback'):
                backend.generate(frames=81)

    def test_repository_capabilities_do_not_claim_legacy_lengths(self):
        self.assertEqual(video.availability()['frames'], [81])
        self.assertEqual(video.availability(backend='legacy')['frames'], [81, 121])
        with self.assertRaisesRegex(ServiceError, 'frame count'):
            video.video_job(self.service, {'head_id': 'testhead', 'frames': 121},
                            lambda *_: None, threading.Event())

    def test_mock_packaging_and_file_boundary(self):
        result = video.video_job(self.service, {"head_id": "testhead", "expression": "blink"},
            lambda *_: None, threading.Event(), mock=True)
        clips = video.list_videos(self.service, "testhead")
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["backend"], "mock")
        path = video.video_file(self.service, "testhead", result["id"], "clip.mp4")
        self.assertIn(b"ftyp", path.read_bytes()[:32])
        decoded = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(path), "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1"], text=True)
        counters = dict(line.split("=", 1) for line in decoded.splitlines() if "=" in line)
        self.assertEqual(counters["frame"], "81")
        self.assertEqual(counters["out_time_us"], "3375000")
        metadata = json.loads(video.video_file(self.service, "testhead", result["id"], "manifest.json").read_text())
        self.assertEqual(metadata["request"]["frames"], 81)
        for run, name in (("..", "clip.mp4"), (result["id"], "runner.log"),
                          (result["id"], "../../portrait.png")):
            with self.assertRaises(ServiceError):
                video.video_file(self.service, "testhead", run, name)

    def test_cancel_removes_unpublished_outputs(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(gpu.Cancelled):
            video.video_job(self.service, {"head_id": "testhead"}, lambda *_: None, cancel, mock=True)
        self.assertEqual(video.list_videos(self.service, "testhead"), [])
        self.assertEqual(list((self.work / "heads/testhead/videos").iterdir()), [])

    def test_failure_cleans_outputs(self):
        with patch.object(native, "run_process", side_effect=RuntimeError("encoder failed")):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                video.video_job(self.service, {"head_id": "testhead"}, lambda *_: None,
                                threading.Event(), mock=True)
        self.assertEqual(list((self.work / "heads/testhead/videos").iterdir()), [])

    def test_cancel_silent_subprocess(self):
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        try:
            started = time.monotonic()
            with self.assertRaises(native.Cancelled):
                native.run_process([sys.executable, "-c", "import time; time.sleep(30)"], cancel=cancel)
            self.assertLess(time.monotonic() - started, 6)
        finally:
            timer.cancel()

    def test_cancel_partial_line_subprocess(self):
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        try:
            with self.assertRaises(native.Cancelled):
                native.run_process([sys.executable, "-c",
                    "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)"], cancel=cancel)
        finally:
            timer.cancel()

    def test_portrait_pixels_and_model_boundary(self):
        image = self.work / "heads/testhead/portrait.png"
        prepared, pixels = native.prepare_portrait(image, self.work, 480, 848)
        with Image.open(prepared) as decoded:
            self.assertEqual(decoded.size, (480, 848))
        values = np.fromfile(pixels, dtype="<f4")
        self.assertEqual(values.size, 384 * 384 * 3)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((np.abs(values) <= 1).all())
        model = self.work / "model"
        model.mkdir()
        names = {key: "../input.png" for key in ("vae", "qwen", "byt5", "vision", "tokenizer")}
        (model / "model.json").write_text(json.dumps({"schema": "hunyuan_video15.model.v1",
            "checkpoints": {"quality_i2v": "../input.png"}, "components": names, "sources": {"test": {}}}))
        with self.assertRaises(ValueError):
            native.load_manifest(model, "i2v", "quality")

    def test_transparent_portrait_composites_hidden_rgb(self):
        image = self.work / "transparent.png"
        Image.new("RGBA", (32, 32), (255, 0, 255, 0)).save(image)
        prepared, _ = native.prepare_portrait(image, self.work, 480, 848)
        with Image.open(prepared) as decoded:
            self.assertEqual(decoded.getpixel((0, 0)), (96, 96, 96))

    def test_native_opt_in_and_incomplete_frame_cleanup(self):
        model = self.work / "native-model"
        model.mkdir()
        (model / "weights.safetensors").write_bytes(b"fixture")
        manifest = {"schema": "hunyuan_video15.model.v1", "checkpoints": {"quality_i2v": "weights.safetensors"},
            "components": {k: "weights.safetensors" for k in ("vae", "qwen", "byt5", "vision", "tokenizer")},
            "sources": {"weights.safetensors": {"fixture": True}}}
        (model / "model.json").write_text(json.dumps(manifest))
        out = self.work / "native-output"
        kwargs = {"model": model, "image": self.work / "heads/testhead/portrait.png",
                  "prompt": "smile", "out": out}
        with self.assertRaisesRegex(ValueError, "unverified"):
            native.generate(**kwargs)
        self.assertFalse(out.exists())
        with patch.object(native, "run_process") as process:
            with self.assertRaisesRegex(RuntimeError, "incomplete frame sequence"):
                native.generate(**kwargs, allow_experimental=True)
            command = process.call_args.args[0]
            self.assertIn("--vision-pixels", command)
            self.assertIn("--allow-experimental", command)
        self.assertFalse(out.exists())
        with self.assertRaises(KeyError):
            native.load_manifest(model, "i2v", "fast12")

    def test_parity_fails_closed(self):
        a, b = self.work / "reference", self.work / "actual"
        a.mkdir(); b.mkdir()
        with self.assertRaises(FileNotFoundError):
            compare(a, b, ["dit_first"])
        np.save(a / "dit_first.npy", np.array([1, 2, 3], dtype=np.float32))
        np.save(b / "dit_first.npy", np.array([1, 2, 3], dtype=np.float32))
        self.assertTrue(compare(a, b, ["dit_first"])["dit_first"]["pass"])
        np.save(b / "dit_first.npy", np.array([3, 2, 1], dtype=np.float32))
        self.assertFalse(compare(a, b, ["dit_first"])["dit_first"]["pass"])
        np.save(b / "dit_first.npy", np.array([np.nan, 2, 3], dtype=np.float32))
        with self.assertRaises(ValueError):
            compare(a, b, ["dit_first"])

if __name__ == "__main__":
    unittest.main()
