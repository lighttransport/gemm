"""CPU check of exported INT8 rows and the native BF16 reconstruction."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
import torch

from quantize_weights import quantize_rows


class QuantizedWeightsTest(unittest.TestCase):
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
