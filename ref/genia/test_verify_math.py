"""Native oracle buffer safety regression tests; no GPU required."""
import unittest

import numpy as np

from ref.genia.verify_math import validate_arrays


class OracleBufferTest(unittest.TestCase):
    def arrays(self):
        arrays = {name: np.zeros((2, 8), np.float32) for name in
                  ("fused", "latent", "gradient", "clean", "guided")}
        arrays.update(coords=np.zeros((2, 3), np.int32),
                      cameras=np.zeros((1, 3), np.float32),
                      occupancy=np.zeros((64, 64, 64), np.uint8),
                      visibility=np.zeros((1, 2), np.float32),
                      velocities=np.zeros((1, 2, 8), np.float32),
                      weights=np.zeros((1, 2), np.float32))
        return arrays

    def test_contract(self):
        validate_arrays(self.arrays())

    def test_short_wrong_dtype_and_nonfinite_buffers(self):
        for bad in (np.zeros((1, 8), np.float32), np.zeros((2, 8), np.float64),
                    np.full((2, 8), np.nan, np.float32)):
            arrays = self.arrays()
            arrays["gradient"] = bad
            with self.assertRaises(ValueError):
                validate_arrays(arrays)

    def test_invalid_coordinate(self):
        arrays = self.arrays()
        arrays["coords"][0, 0] = 64
        with self.assertRaises(ValueError):
            validate_arrays(arrays)


if __name__ == "__main__":
    unittest.main()
