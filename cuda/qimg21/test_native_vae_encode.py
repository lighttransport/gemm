"""Encoder input validation must reject unsupported data before opening CUDA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class NativeVaeEncodeTest(unittest.TestCase):
    def test_invalid_inputs_before_cuda(self):
        binary = Path(__file__).with_name("test_cuda_qimg21_vae_encode")
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="qimg21-encoder-cli-") as work:
            path = Path(work)
            for value in (np.zeros((3, 64, 64), np.float32),
                          np.zeros((4, 63, 64), np.float32),
                          np.zeros((4, 1040, 16), np.float32),
                          np.full((4, 16, 16), np.nan, np.float32),
                          np.full((4, 16, 16), 1.1, np.float32)):
                with self.subTest(shape=value.shape, first=value.flat[0]):
                    np.save(path / "input.npy", value)
                    result = subprocess.run([str(binary), "--model", "missing-model", "--image",
                                             str(path / "input.npy"), "--out", str(path / "output.npy")],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("NVIDIA", result.stderr)
                    self.assertFalse((path / "output.npy").exists())


if __name__ == "__main__":
    unittest.main()
