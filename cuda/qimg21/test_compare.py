"""CPU regression for fail-closed denoiser/trajectory/prompt acceptance."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import compare


class CompareTest(unittest.TestCase):
    def test_nonfinite_arrays_rejected(self):
        for invalid in (np.array([np.nan]), np.array([np.inf]), np.array([])):
            with self.assertRaisesRegex(ValueError, "empty or non-finite"):
                compare._cosine_error(invalid, invalid)

    def test_fixture_gates(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-compare-") as work:
            ref, got = Path(work) / "ref", Path(work) / "got"
            ref.mkdir(); got.mkdir()
            fixture = np.ones((4, 64), dtype=np.float32)
            for folder in (ref, got):
                np.save(folder / "step_000.npy", fixture)
                np.save(folder / "prompt_embeds.npy", fixture)

            def run():
                argv = ["compare.py", "--reference-dir", str(ref), "--runner-dir", str(got)]
                with patch.object(sys, "argv", argv), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    return compare.main()

            self.assertEqual(run(), 0)
            np.save(got / "prompt_embeds.npy", np.full_like(fixture, np.nan))
            self.assertEqual(run(), 1)  # NaN < threshold is false; must not pass.
            np.save(got / "prompt_embeds.npy", fixture)
            np.save(got / "step_001.npy", fixture)
            self.assertEqual(run(), 1)  # Extra stale trajectory checkpoint.
            (got / "step_001.npy").unlink()
            np.save(got / "step_000.npy", -fixture)
            self.assertEqual(run(), 1)
            (got / "steps").mkdir()
            (got / "step_000.npy").unlink()
            np.save(got / "steps/step_000.npy", fixture)
            self.assertEqual(run(), 0)


if __name__ == "__main__":
    unittest.main()
