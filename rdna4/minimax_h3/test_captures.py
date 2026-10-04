import gzip
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "capture checks require NumPy")
class CaptureTests(unittest.TestCase):
    def test_lossless_and_bounded(self):
        from ref.minimax_h3_native.captures import compress, read_f32, write_npy, read_npy
        root = Path(__file__).resolve().parents[2] / "tmp/video-rocm"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            directory = Path(temporary)
            values = np.array([-0., 0., 1.25, -17.5, 1.e-20], dtype="<f4")
            values.tofile(directory / "state.f32")
            with self.assertRaises(InterruptedError):
                compress(directory, cancelled=lambda: True)
            self.assertTrue((directory / "state.f32").exists())
            self.assertFalse(list(directory.glob("*.partial")))
            expected = read_f32(directory, "state", values.size)
            for compressed in (False, True):
                stored = write_npy(directory / "independent.npy", values, compressed)
                self.assertEqual(read_npy(stored).tobytes(), values.tobytes())
            compress(directory)
            actual = read_f32(directory, "state", values.size)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertFalse((directory / "state.f32").exists())
            for count in (0, values.size - 1, values.size + 1):
                with self.assertRaises(ValueError):
                    read_f32(directory, "state", count)
            with gzip.open(directory / "oversized.f32.gz", "wb") as stream:
                stream.write(values.tobytes() * 2)
            with self.assertRaises(ValueError):
                read_f32(directory, "oversized", values.size)


if __name__ == "__main__":
    unittest.main()
