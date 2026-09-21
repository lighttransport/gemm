"""A native replay must use exact captured model timesteps, not estimates."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regression import _load_timesteps


class RegressionInputsTest(unittest.TestCase):
    def test_exact_and_invalid_timesteps(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-timesteps-") as work:
            folder = Path(work)
            np.save(folder / "timestep_000.npy", np.array([1.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "fixture set"):
                _load_timesteps(folder, 2)
            np.save(folder / "timestep_001.npy", np.array([0.02001953125], dtype=np.float32))
            self.assertEqual(_load_timesteps(folder, 2), [1., 0.02001953125])
            for invalid in ([np.nan], [np.inf], [-.1], [1.1], [0., 1.]):
                np.save(folder / "timestep_001.npy", np.array(invalid, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "invalid captured"):
                    _load_timesteps(folder, 2)
            np.save(folder / "timestep_002.npy", np.array([0.]))
            with self.assertRaisesRegex(ValueError, "fixture set"):
                _load_timesteps(folder, 2)


if __name__ == "__main__":
    unittest.main()
