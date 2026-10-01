import io
import json
from pathlib import Path
import struct
import threading
import time
import unittest
import numpy as np
from .src.audio.clock import SampleClock
from .src.audio.ring import PcmRing, build
from .src.animation.timeline import MotionTimeline, retarget
from .src.avatar.bundle import GaussianAvatar, bind, deform
from .src.avatar.provenance import validate_receipts
from .src.pipeline.protocol import AudioChunk, MotionFrame
from .src.pipeline.session import Session
from .src.tts.features import read_features

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "tmp/vhuman-realtime/tests"


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.library = build(WORK / "libpcm.so")

    def test_ring_wrap_backpressure_and_silence(self):
        ring = PcmRing(self.library, 8)
        try:
            ring.write(np.arange(6))
            np.testing.assert_array_equal(ring.read(4)[0], np.arange(4))
            ring.write(np.arange(6, 12))
            with self.assertRaises(BufferError): ring.write([12])
            x, n = ring.read(8)
            np.testing.assert_array_equal(x, np.arange(4, 12))
            self.assertEqual(n, 8)
            x, n = ring.read(3)
            self.assertEqual(n, 0)
            np.testing.assert_array_equal(x, np.zeros(3))
        finally: ring.close()

    def test_spsc_order_under_contention(self):
        ring = PcmRing(self.library, 257)
        result, errors = [], []
        def producer():
            try:
                for start in range(0, 10000, 20):
                    while True:
                        try: ring.write(np.arange(start, start + 20)); break
                        except BufferError: time.sleep(0)
            except Exception as error: errors.append(error)
        thread = threading.Thread(target=producer)
        thread.start()
        deadline = time.monotonic() + 10
        while len(result) < 10000 and time.monotonic() < deadline:
            x, n = ring.read(17)
            result.extend(x[:n])
            if not n: time.sleep(0)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        np.testing.assert_array_equal(result, np.arange(10000))
        ring.close()

    def test_underrun_does_not_advance_speech(self):
        clock = SampleClock()
        clock.submit(1920, 0); clock.submit(960); clock.submit(1920, 1920)
        clock.observe(4800, 48000, 10**9)
        self.assertEqual(clock.position(), 1920)
        clock.observe(6720, 48000, 10**9 + 20_000_000)
        self.assertEqual(clock.position(), 2400)
        with self.assertRaises(ValueError): clock.observe(1, 48000, 10**9)

    def test_clock_long_rational_no_drift(self):
        clock = SampleClock()
        clock.submit(24_000 * 3600, 0)
        for device in (1, 44101, 44100 * 1234 + 7, 44100 * 3600):
            clock.observe(device, 44100, device * 10**9 // 44100)
            self.assertEqual(clock.position(), device * 24000 // 44100)

    def test_offline_playout_follows_time_during_frame_stalls(self):
        from .src.audio.offline import OfflinePlayout
        clock = OfflinePlayout(100)
        self.assertEqual(clock.due(100), 0)
        self.assertEqual(clock.due(100 + 10_000_000), 240)
        self.assertEqual(clock.due(100 + 510_000_000), 12000)
        self.assertEqual(clock.due(100 + 510_000_000), 0)
        self.assertEqual(clock.due(100 + 3600 * 10**9), 86400000-12240)
        with self.assertRaises(ValueError): clock.due(100)

    def test_first_pcm_is_not_first_speech(self):
        from .src.tts.startup import StartupMeter
        meter = StartupMeter()
        meter.observe(AudioChunk(0, 0, 0, np.zeros(1920)), 100)
        self.assertIsNone(meter.first_speech_ns)
        pcm = np.zeros(1920, np.float32); pcm[960:] = .05
        meter.observe(AudioChunk(0, 1, 1920, pcm), 200)
        self.assertEqual(meter.first_speech_ns, 200)
        self.assertEqual(meter.leading_low_energy_samples, 2880)

    def test_motion_bounds_epoch_and_ease(self):
        motion = MotionTimeline(["jawOpen", "headYaw"], [[0, 1], [-1, 1]], 2)
        motion.push(MotionFrame(0, 0, [2, -2]))
        motion.push(MotionFrame(0, 240, [0, 1]))
        np.testing.assert_allclose(motion.sample(120), [.5, 0])
        with self.assertRaises(BufferError): motion.push(MotionFrame(0, 480, [0, 0]))
        np.testing.assert_allclose(motion.sample(6000), [0, 0])
        motion.reset(1)
        self.assertFalse(motion.push(MotionFrame(0, 8000, [1, 1])))
        np.testing.assert_allclose(retarget([.2, .9], ["tongueOut", "jawOpen"], ["jawOpen", "tongueOut", "eyeBlinkLeft"]), [.9, .2, 0])

    def test_session_cancel_and_retry(self):
        ring = PcmRing(self.library, 8)
        session = Session(ring, ["jawOpen"], [[0, 1]])
        session.ingest(AudioChunk(0, 0, 0, [1, 2, 3]))
        pcm, _ = session.pull(5, 1)
        np.testing.assert_array_equal(pcm, [1, 2, 3, 0, 0])
        self.assertEqual(session.clock.position(), 3)
        session.cancel()
        self.assertFalse(session.ingest(AudioChunk(0, 1, 3, [4])))
        session.ingest(AudioChunk(1, 0, 0, [5]))
        self.assertEqual(session.pull(1, 2)[0][0], 5)
        ring.close()

    def test_gaussian_rigid_covariance_and_roundtrip(self):
        vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32)
        triangles = np.array([[0, 1, 2]], np.int32)
        avatar = bind(vertices, triangles, ["jawOpen"], count=10)
        avatar.save(WORK / "avatar.npz")
        restored = GaussianAvatar.load(WORK / "avatar.npz", triangles)
        p, cov, opacity, _ = deform(restored, vertices, triangles)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
        moved, rotated, _, _ = deform(restored, vertices @ rotation.T + .2, triangles)
        np.testing.assert_allclose(moved, p @ rotation.T + .2, atol=1e-7)
        np.testing.assert_allclose(rotated, rotation @ cov @ rotation.T, atol=1e-10)
        _, _, opacity, _ = deform(restored, np.zeros_like(vertices), triangles)
        np.testing.assert_array_equal(opacity, 0)
        with self.assertRaises(ValueError): restored.validate(triangles[:, [0, 2, 1]])

    def test_trace_geometry_has_finite_gradients_and_cpu_parity(self):
        import torch
        from .src.avatar.geometry import deform_torch
        vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32)
        triangles = np.array([[0, 1, 2]], np.int32)
        avatar = bind(vertices, triangles, ["jawOpen"], count=10)
        avatar.metadata["covariance_policy"] = "trace-v1"
        arrays = {k: torch.tensor(v, dtype=torch.long if k == "triangle" else torch.float32)
                  for k, v in avatar.arrays.items()}
        # Repeated eigenvalues used to make eigendecomposition gradients fragile.
        arrays["covariance_local"] = torch.eye(3).repeat(10, 1, 1).requires_grad_()
        actual = deform_torch(torch.tensor(vertices), torch.tensor(triangles, dtype=torch.long), arrays, policy="trace-v1")
        actual[1].square().sum().backward()
        self.assertTrue(torch.isfinite(arrays["covariance_local"].grad).all())
        avatar.arrays["covariance_local"] = arrays["covariance_local"].detach().numpy()
        expected = deform(avatar, vertices, triangles)
        for gpu, cpu in zip(actual, expected):
            np.testing.assert_allclose(gpu.detach().numpy(), cpu, atol=2e-7, rtol=1e-4)

    def test_research_receipt_rejected(self):
        with self.assertRaises(ValueError): validate_receipts([dict(license="qwen-research")])
        with self.assertRaises(ValueError): validate_receipts([])

    def test_feature_stream_exact_intervals(self):
        blob = b"VHFEAT1\0" + struct.pack("<i", 3)
        for start in (0, 1920):
            blob += struct.pack("<q", start) + np.arange(16, dtype="<i4").tobytes() + np.ones(3, "<f4").tobytes()
        frames = list(read_features(io.BytesIO(blob), "test-revision"))
        self.assertEqual([f.sample_start for f in frames], [0, 1920])
        with self.assertRaises(ValueError): list(read_features(io.BytesIO(blob[:-1]), "test-revision"))

    def test_pcm_stream_fragmented_reads(self):
        from .src.tts.pcm import read_pcm
        blob = b"VHPCM1\0\0" + struct.pack("<i", 24000)
        x = np.arange(1920, dtype="<f4") / 1920
        blob += struct.pack("<qi", 0, 1920) + x.tobytes()
        class Fragmented(io.BytesIO):
            def read(self, count=-1): return super().read(min(7, count))
        frames = list(read_pcm(Fragmented(blob)))
        np.testing.assert_array_equal(frames[0].pcm, x)
        with self.assertRaises(ValueError): list(read_pcm(Fragmented(blob[:-1])))

    def test_named_performance_and_codec_corpus(self):
        from .src.animation.performance import load_performance
        from .src.animation.corpus import capture
        path = WORK / "performance.json"
        path.write_text(json.dumps(dict(format="vhuman.performance.v1", controls=["jawOpen"],
            frames=[dict(t=0, v={"jawOpen": 0}), dict(t=.08, v={"jawOpen": 1})])))
        _, positions, values = load_performance(path)
        np.testing.assert_array_equal(positions, [0, 1920])
        blob = b"VHFEAT1\0" + struct.pack("<i", 3) + struct.pack("<q", 0)
        blob += np.arange(16, dtype="<i4").tobytes() + np.ones(3, "<f4").tobytes()
        features = WORK / "capture.bin"; features.write_bytes(blob)
        capture(features, path, WORK / "capture.npz", "test", ["jawOpen"], [[0, 1]])
        with np.load(WORK / "capture.npz") as z:
            np.testing.assert_allclose(z["controls"][0, :, 0], np.arange(8) / 8)

    def test_resident_end_markers_preserve_next_utterance(self):
        from .src.tts.pcm import read_pcm
        pcm = b"VHPCM1\0\0" + struct.pack("<i", 24000) + struct.pack("<qi", -1, 0)
        feature = b"VHFEAT1\0" + struct.pack("<i", 3) + struct.pack("<q", -1) + bytes(76)
        for blob, reader, args in ((pcm, read_pcm, ()), (feature, read_features, ("test",))):
            stream = io.BytesIO(blob + blob)
            self.assertEqual(list(reader(stream, *args, allow_end_marker=True)), [])
            self.assertEqual(stream.tell(), len(blob))
            self.assertEqual(list(reader(stream, *args, allow_end_marker=True)), [])
            with self.assertRaises(ValueError):
                list(reader(io.BytesIO(blob[:12]), *args, allow_end_marker=True))

    def test_model_revision_is_weight_hash(self):
        from .src.tts.identity import verify_model
        from .src.avatar.provenance import sha256
        model = WORK / "model"; model.mkdir(exist_ok=True)
        (model / "model.safetensors").write_bytes(b"test weights")
        verify_model(model, sha256(model / "model.safetensors"))
        with self.assertRaises(ValueError): verify_model(model, "0" * 64)

    def test_cancel_drains_already_finished_utterance(self):
        import queue
        from .src.tts.resident import ResidentTTS
        worker = ResidentTTS.__new__(ResidentTTS)
        worker.audio, worker.features = queue.Queue(4), queue.Queue(4)
        worker.audio.put(1); worker.features.put(2)
        worker.done_audio, worker.done_features = threading.Event(), threading.Event()
        worker.done_audio.set(); worker.done_features.set()
        worker.discard, worker.threads = threading.Event(), []
        worker.check = lambda: None
        worker.cancel()
        self.assertTrue(worker.audio.empty() and worker.features.empty())

    def test_imported_rig_mouth_namespace(self):
        from ..rig.native import append_surface
        exterior = np.ones((3, 3), np.float32)
        source = np.arange(18, dtype=np.float32).reshape(6, 3)
        mapping = np.array([5, 3, 5], np.int32)
        result = append_surface(exterior, {"jaw": exterior * 2}, np.zeros((3, 4), np.int32), np.ones((3, 4), np.float32),
                                mapping, source, {"jaw": source * 2}, np.zeros((6, 4), np.int32), np.ones((6, 4), np.float32))
        rest, shapes, _, _, remap = result
        self.assertEqual(len(rest), 5)
        np.testing.assert_array_equal(rest[remap], source[mapping])
        np.testing.assert_array_equal(shapes["jaw"][remap], source[mapping] * 2)

    def test_causal_step_matches_full_sequence(self):
        import torch
        from .src.animation.causal import CausalMotion
        torch.manual_seed(7)
        model = CausalMotion(12, ["jawOpen"], [[0, 1]]).eval()
        hidden, codes = torch.randn(1, 5, 12), torch.randint(2048, (1, 5, 16))
        with torch.no_grad():
            full, _ = model(hidden, codes)
            state, steps = None, []
            for i in range(5):
                x, state = model(hidden[:, i:i+1], codes[:, i:i+1], state)
                steps.append(x)
        torch.testing.assert_close(full, torch.cat(steps, 1), atol=1e-6, rtol=1e-5)


if __name__ == "__main__": unittest.main()
