#!/usr/bin/env python3
"""Native CLI boundary tests. Run after make -C cpu/rmbg; no ML dependencies."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "cpu/rmbg/swin_backbone"


class RunnerContract(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="swin-test-", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.f32"
        self.input.write_bytes(struct.pack("<3f", 0, 0, 0))
        self.model = self.root / "wrong.safetensors"
        header = json.dumps({"bb.patch_embed.proj.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        self.model.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))

    def run_cli(self, extra=()):
        return subprocess.run([str(RUNNER), "--model", str(self.model), "--input", str(self.input),
                               "--width", "1", "--height", "1", "--output-dir", str(self.root),
                               *extra], capture_output=True, text=True)

    def test_shape_mismatch(self):
        run = self.run_cli()
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertIn("incompatible FP32 tensor", run.stderr)
        self.assertFalse(list(self.root.glob("feature_*")))

    def test_input_contract(self):
        for payload in (bytes(8), bytes(16), struct.pack("<3f", 0, float("nan"), 0),
                        struct.pack("<3f", 0, float("inf"), 0)):
            with self.subTest(payload=repr(payload)):
                self.input.write_bytes(payload)
                run = self.run_cli()
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn("finite F32 CHW input", run.stderr)

    def test_invalid_arguments(self):
        for args in (("--width", "1025"), ("--height", "0"), ("--height", "-1"),
                     ("--width", "1x"), ("--threads", "129"), ("--device", "-1"),
                     ("--backend", "onnx"), ("--unknown", "x"), ("--input",)):
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(args).returncode, 2)

    def test_cpu_build_rejects_cuda(self):
        run = self.run_cli(("--backend", "cuda"))
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("cuda/rmbg/swin_backbone", run.stderr)


if __name__ == "__main__":
    unittest.main()
