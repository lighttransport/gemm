"""Editing CLI rejects unsupported combinations before touching CUDA."""
from pathlib import Path
import os
import subprocess
import unittest


class NativeEditCliTest(unittest.TestCase):
    def test_hidden_replay_guards_before_cuda(self):
        binary = Path(__file__).with_name("test_cuda_qimg21_native")
        base = {key: value for key, value in os.environ.items() if not key.startswith("QIMG21_STAGE_")}
        base["QIMG21_REPLAY_HIDDEN"] = "missing.npy"
        for additions, args in (
            ({}, ["--timestep", "1"]),
            ({"QIMG21_STAGE_DIR": "unused"}, ["--timestep", "1"]),
            ({"QIMG21_STAGE_DIR": "unused", "QIMG21_STAGE_BLOCK": "bad"}, ["--timestep", "1"]),
            ({"QIMG21_STAGE_DIR": "unused", "QIMG21_STAGE_BLOCK": "32"}, ["--timestep", "1"]),
            ({"QIMG21_STAGE_DIR": "unused", "QIMG21_STAGE_BLOCK": "17"}, []),
            ({"QIMG21_STAGE_DIR": "unused", "QIMG21_STAGE_BLOCK": "17"}, ["--timestep", "nan"]),
            ({"QIMG21_STAGE_DIR": "unused", "QIMG21_STAGE_BLOCK": "17"}, ["--timestep", "1", "--steps", "2"]),
        ):
            with self.subTest(additions=additions, args=args):
                result = subprocess.run([str(binary), *args], env={**base, **additions}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                if "--steps" not in args:
                    self.assertIn("hidden replay requires", result.stderr)
                self.assertNotIn("NVIDIA", result.stderr)

    def test_reject_unsupported_combinations(self):
        binary = Path(__file__).with_name("test_cuda_qimg21_native")
        for args in (["--editing-layout", "missing.txt"],
                     ["--condition-latents", "missing.npy"],
                     ["--editing-layout", "missing.txt", "--condition-latents", "missing.npy",
                      "--negative-prompt-embeds", "negative.npy"],
                     ["--negative-editing-layout", "missing.txt"],
                     ["--editing-layout", "missing.txt", "--condition-latents", "missing.npy",
                      "--negative-editing-layout", "negative.txt"]):
            with self.subTest(args=args):
                result = subprocess.run([str(binary), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("editing requires", result.stderr)
                self.assertNotIn("NVIDIA", result.stderr)


if __name__ == "__main__":
    unittest.main()
