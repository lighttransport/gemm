"""CPU-only coverage for native fixture output, including delayed flush failure."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class NativeOutputTest(unittest.TestCase):
    binary = Path(__file__).with_name("test_scheduler")

    def run_writer(self, path):
        return subprocess.run(
            [str(self.binary), "--schedule", "2", "256", str(path)],
            capture_output=True, text=True,
        )

    def test_round_trip_and_missing_parent(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="qimg21-output-test-") as directory:
            path = Path(directory) / "schedule.npy"
            result = self.run_writer(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = np.load(path)
            self.assertEqual(values.shape, (3, 1))
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(values[-1, 0], 0)
            result = self.run_writer(Path(directory) / "missing" / "schedule.npy")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot write", result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires Linux /dev/full")
    def test_buffered_flush_failure(self):
        # This small output fits stdio's buffer: fwrite can succeed, fclose cannot.
        result = self.run_writer("/dev/full")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed writing", result.stderr)


if __name__ == "__main__":
    unittest.main()
