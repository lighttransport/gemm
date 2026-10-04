"""Public request validation and owned-artifact cleanup contracts."""
import ctypes
from contextlib import nullcontext
import os
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h3_generate", Path(__file__).with_name("generate.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


class Config(ctypes.Structure):
    _fields_ = [("model_dir", ctypes.c_char_p), ("device", ctypes.c_int),
                ("vram_budget_mib", ctypes.c_int), ("bf16_hipblas", ctypes.c_int), ("convrot_hipblas", ctypes.c_int),
                ("aotriton_bridge", ctypes.c_char_p), ("vae_hipblas", ctypes.c_int)]


class Request(ctypes.Structure):
    _fields_ = [(n, ctypes.c_char_p) for n in ("prompt", "noise_file", "audio_noise_file", "dump_dir")]
    _fields_ += [(n, ctypes.c_int) for n in ("width", "height", "frames", "steps")]
    _fields_ += [("seed", ctypes.c_int64)]


class H3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(os.environ.get("H3_TEST_LIBRARY", str(ROOT / "tmp/video-rocm/h3-build/libh3_rocm.so")))
        cls.lib.h3_validate.argtypes = [ctypes.POINTER(Request), ctypes.c_char_p, ctypes.c_size_t]
        cls.lib.h3_request_defaults.argtypes = [ctypes.POINTER(Request)]
        cls.lib.h3_config_defaults.argtypes = [ctypes.POINTER(Config)]
        cls.lib.h3_load.argtypes = [ctypes.POINTER(Config), ctypes.c_char_p, ctypes.c_size_t]
        cls.lib.h3_load.restype = ctypes.c_void_p
    def setUp(self):
        self.request = Request()
        self.lib.h3_request_defaults(ctypes.byref(self.request))
        self.request.prompt = b"A red ball rolling on a wooden table."
    def validate(self):
        error = ctypes.create_string_buffer(1024)
        return self.lib.h3_validate(ctypes.byref(self.request), error, len(error))
    def test_convrot_config_rejects_invalid_backend_before_gpu_load(self):
        config = Config()
        self.lib.h3_config_defaults(ctypes.byref(config))
        self.assertEqual(config.convrot_hipblas, 1)
        self.assertEqual(config.bf16_hipblas, 1)
        config.convrot_hipblas = 2
        error = ctypes.create_string_buffer(1024)
        self.assertFalse(self.lib.h3_load(ctypes.byref(config), error, len(error)))
        self.assertIn(b"ConvRot", error.value)

    def test_bf16_config_rejects_invalid_backend_before_gpu_load(self):
        config = Config()
        self.lib.h3_config_defaults(ctypes.byref(config))
        config.bf16_hipblas = 2
        error = ctypes.create_string_buffer(1024)
        self.assertFalse(self.lib.h3_load(ctypes.byref(config), error, len(error)))
        self.assertIn(b"BF16", error.value)

    def test_missing_attention_bridge_rejected_before_gpu_load(self):
        config = Config()
        self.lib.h3_config_defaults(ctypes.byref(config))
        self.assertIsNone(config.aotriton_bridge)
        config.aotriton_bridge = b"nonexistent-aotriton-bridge.so"
        error = ctypes.create_string_buffer(1024)
        self.assertFalse(self.lib.h3_load(ctypes.byref(config), error, len(error)))
        self.assertIn(b"AOTriton bridge", error.value)

    def test_invalid_vae_backend_rejected_before_gpu_load(self):
        config = Config()
        self.lib.h3_config_defaults(ctypes.byref(config))
        self.assertEqual(config.vae_hipblas, 0)
        config.vae_hipblas = 2
        error = ctypes.create_string_buffer(1024)
        self.assertFalse(self.lib.h3_load(ctypes.byref(config), error, len(error)))
        self.assertIn(b"VAE backend", error.value)

    def test_fp32_setter_rejects_missing_context(self):
        self.lib.h3_set_fp32_hipblas.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_char_p, ctypes.c_size_t]
        error = ctypes.create_string_buffer(1024)
        self.assertNotEqual(self.lib.h3_set_fp32_hipblas(None, 1, error, len(error)), 0)
        self.assertIn(b"missing H3 context", error.value)
        self.assertEqual(ctypes.sizeof(Config), 40)

    def test_fp32_python_backend_rejects_invalid_values(self):
        for value in (True, False, 2, -1, 1.0):
            with self.assertRaisesRegex(ValueError, "fp32_hipblas"):
                gen.generate(out=ROOT / "tmp/video-rocm/fp32-invalid-test-unused",
                             prompt="A ball.", allow_experimental=True, fp32_hipblas=value)

    def test_supported_geometry_and_seed(self):
        self.assertEqual(self.validate(), 0)
        self.request.seed = 2**63 - 1
        self.assertEqual(self.validate(), 0)
        for frames in (5, 22, 124):
            self.request.frames = frames
            self.assertEqual(self.validate(), 0)
    def test_invalid_geometry_and_schedule(self):
        for field, value in (("width", 65), ("height", 16), ("frames", 125), ("steps", 1), ("seed", -1)):
            old = getattr(self.request, field)
            setattr(self.request, field, value)
            self.assertNotEqual(self.validate(), 0)
            setattr(self.request, field, old)
        self.request.prompt = b""
        self.assertNotEqual(self.validate(), 0)
    def test_failure_cleans_only_owned_output(self):
        scratch = ROOT / "tmp/video-rocm/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            model = root / "model"
            for name in gen.COMPONENTS:
                file = model / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture")
            keep = root / "keep"
            keep.write_text("unrelated")
            out = root / "video"
            with patch.object(gen.video, "run_process", side_effect=RuntimeError("native failure")), patch.object(gen.video, "MemorySampler"), patch.object(gen.video, "device_lock", return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    gen.generate(model=model, out=out, prompt="test", allow_experimental=True)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name("video.partial").exists())
            self.assertEqual(keep.read_text(), "unrelated")
    def test_cancellation_during_device_wait(self):
        import threading
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(gen.video.Cancelled), gen.video.device_lock(0, cancelled):
            self.fail("cancelled request acquired the device")


if __name__ == "__main__":
    unittest.main()
