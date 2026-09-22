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
            self.assertEqual(command[command.index("--native-attention") + 1], "wmma")
            self.assertEqual(command[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
