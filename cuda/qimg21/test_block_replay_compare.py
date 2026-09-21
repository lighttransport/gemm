from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from block_replay_compare import compare_stages


class BlockReplayCompareTest(unittest.TestCase):
    def compare(self, a, b):
        with patch("block_replay_compare.np.load", side_effect=[a, b] * 6):
            return compare_stages(Path("reference"), Path("candidate"), 17)

    def test_batch_one_and_bit_exact(self):
        a = np.ones((2, 4), np.float32)
        self.assertTrue(all(x["bit_exact"] for x in self.compare(a[None], a).values()))
        b = a.copy()
        a[0, 0], b[0, 0] = 0., -0.
        self.assertFalse(self.compare(a, b)["block"]["bit_exact"])

    def test_invalid_shapes_dtypes_and_nonfinite(self):
        a = np.ones((2, 4), np.float32)
        for b in (a.T, a.astype(np.float64), a * np.nan):
            with self.subTest(shape=b.shape, dtype=b.dtype):
                with self.assertRaises(ValueError):
                    self.compare(a, b)


if __name__ == "__main__":
    unittest.main()
