"""Editing CLI rejects unsupported combinations before touching CUDA."""
from pathlib import Path
import subprocess
import unittest


class NativeEditCliTest(unittest.TestCase):
    def test_reject_unsupported_combinations(self):
        binary = Path(__file__).with_name("test_cuda_qimg21_native")
        for args in (["--editing-layout", "missing.txt"],
                     ["--condition-latents", "missing.npy"],
                     ["--editing-layout", "missing.txt", "--condition-latents", "missing.npy",
                      "--negative-prompt-embeds", "negative.npy"]):
            with self.subTest(args=args):
                result = subprocess.run([str(binary), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("editing requires", result.stderr)
                self.assertNotIn("NVIDIA", result.stderr)


if __name__ == "__main__":
    unittest.main()
