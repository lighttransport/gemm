import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from server.qwen_image21.app import Demo, ROOT


class QwenImage21RoutingTest(unittest.TestCase):
    def make_demo(self, root: Path) -> Demo:
        return Demo(root / "model", root / "quant", root / "cuda-python",
                    root / "work", root / "cuda-native", "127.0.0.1", 0,
                    native_rocm=root / "rocm-native",
                    python_rocm=root / "rocm-python")

    def test_legacy_and_explicit_backend_requests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            legacy = demo._validate({"prompt": "apple", "mode": "cuda"})
            self.assertEqual((legacy["backend"], legacy["mode"]), ("cuda", "native"))
            rocm = demo._validate({"prompt": "apple", "backend": "rocm"})
            self.assertEqual((rocm["backend"], rocm["mode"]), ("rocm", "native"))
            compare = demo._validate({"prompt": "apple", "backend": "rocm", "mode": "compare"})
            self.assertEqual((compare["backend"], compare["mode"]), ("rocm", "compare"))

    def test_native_components_include_attention_plugin(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            self.assertEqual(demo.native_components("rocm")["attention"].name,
                             "libq21_hip_attention.so")
            self.assertEqual(demo.native_components("cuda")["attention"].name,
                             "libq21_cutlass_attention.so")

    def test_native_command_selects_rocm_driver(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "rocm", "width": 256,
                                  "height": 256, "steps": 1})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
                demo._native(cfg, root / "out")
            command = commands[0]
            self.assertIn("--backend", command)
            self.assertEqual(command[command.index("--backend") + 1], "rocm")
            self.assertEqual(command[command.index("--native-bin") + 1], str(root / "rocm-native"))
            self.assertEqual(command[command.index("--native-attention") + 1], "wmma-fused")
            self.assertEqual(command[0], sys.executable)

    def test_rocm_quantized_uses_bf16_wmma_not_cuda_int8(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            (root / "quant").mkdir()
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "rocm", "quantized": True})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
                demo._native(cfg, root / "out")
            self.assertIn("--quantized-transformer", commands[0])
            self.assertNotIn("--int8-tensor-core", commands[0])

    def test_cuda_quantized_keeps_int8_tensor_core(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            (root / "quant").mkdir()
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "cuda", "quantized": True})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
                demo._native(cfg, root / "out")
            self.assertIn("--int8-tensor-core", commands[0])
            self.assertIn("--int8-bf16-tail-blocks", commands[0])


if __name__ == "__main__":
    unittest.main()
