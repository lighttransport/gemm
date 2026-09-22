"""CPU check of exported INT8 rows and the native BF16 reconstruction."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

import numpy as np
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file
import torch

from quantize_weights import quantize_rows
import quantize_weights


class QuantizedWeightsTest(unittest.TestCase):
    def test_export_disk_preflight_does_not_create_partial_package(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-int8-space-") as work:
            model = Path(work)
            (model / "transformer").mkdir()
            save_file({"matrix": np.ones((8, 8), dtype=np.float32)},
                      str(model / "transformer/diffusion_pytorch_model-00001-of-00001.safetensors"))
            output = model / "export"
            with patch.object(sys, "argv", ["quantize_weights.py", "--model", str(model), "--out", str(output)]), \
                 patch("quantize_weights.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "export needs"):
                    quantize_weights.main()
            self.assertFalse(output.exists())

    def test_quantize_on_load_matches_export(self):
        root = Path(__file__).resolve().parents[2]
        binary = root / "cuda/qimg21/test_quant_weights"
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-int8-load-") as work:
            folder = Path(work)
            for dtype in (torch.float32, torch.bfloat16):
                weight = np.random.default_rng(123).standard_normal((5, 129), dtype=np.float32)
                weight[0] = 0
                weight[1] = 0
                weight[1, :7] = [.5, 1.5, 2.5, -.5, -1.5, -2.5, 127]
                tensor = torch.from_numpy(weight).to(dtype)
                save_torch_file({"weight": tensor}, str(folder / "original.safetensors"))
                q, scale = quantize_rows(tensor.float().numpy())
                save_file({"weight": q, "scale": scale}, str(folder / "package.safetensors"))
                for source, name, flags in (("original", "stream", ["--quantize"]),
                                            ("package", "export", [])):
                    subprocess.run([str(binary), str(folder / f"{source}.safetensors"), "5", "129",
                                    str(folder / f"{name}.bin"), *flags], check=True)
                np.testing.assert_array_equal(np.fromfile(folder / "stream.bin", dtype=np.uint16),
                                              np.fromfile(folder / "export.bin", dtype=np.uint16))

    def test_native_reconstruction_and_rejections(self):
        root = Path(__file__).resolve().parents[2]
        binary = root / "cuda/qimg21/test_quant_weights"
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-int8-") as work:
            folder = Path(work)
            weight = np.random.default_rng(17).standard_normal((5, 129), dtype=np.float32)
            weight[0] = 0
            weight[1, 64] = 30  # outlier row exercises per-row scale.
            quantized, scale = quantize_rows(weight)
            self.assertTrue(np.all(quantized[0] == 0))
            self.assertEqual(scale[0], 1)
            source, output = folder / "w.safetensors", folder / "bf16.bin"

            def invoke(q=quantized, s=scale, rows=5):
                save_file({"weight": q, "scale": s}, str(source))
                return subprocess.run([str(binary), str(source), str(rows), "129", str(output)],
                                      capture_output=True).returncode

            self.assertEqual(invoke(), 0)
            expected = torch.from_numpy(quantized.astype(np.float32) * scale[:, None]).bfloat16()
            np.testing.assert_array_equal(np.fromfile(output, dtype=np.uint16).reshape(weight.shape),
                                          expected.view(torch.uint16).numpy())
            subprocess.run([str(binary), str(source), "5", "129", str(output), "--fat"], check=True)
            raw = output.read_bytes()
            np.testing.assert_array_equal(np.frombuffer(raw[:20], dtype=np.float32), scale)
            np.testing.assert_array_equal(np.frombuffer(raw[20:], dtype=np.int8).reshape(weight.shape),
                                          quantized)
            self.assertNotEqual(invoke(rows=4), 0)
            for bad in (np.nan, np.inf, 0, -1):
                invalid = scale.copy(); invalid[2] = bad
                self.assertNotEqual(invoke(s=invalid), 0)
            invalid = quantized.copy(); invalid[2, 0] = -128
            self.assertNotEqual(invoke(q=invalid), 0)
            with self.assertRaises(ValueError):
                quantize_rows(np.array([[np.nan]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
