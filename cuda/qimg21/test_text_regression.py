"""CPU checks for strict text parity fixture validation and acceptance."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from text_regression import compare_hidden, load_fixture


class TextRegressionTest(unittest.TestCase):
    def test_acceptance(self):
        ref = np.arange(24, dtype=np.float32).reshape(3, 8) + 1
        self.assertTrue(compare_hidden(ref, ref)["passed"])
        self.assertFalse(compare_hidden(ref, -ref)["passed"])
        bad = ref.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare_hidden(ref, bad)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare_hidden(ref, ref[None])

    def test_fixture_contract(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-text-gate-") as work:
            folder = Path(work)
            capture = folder / "text_positive"
            capture.mkdir()
            (capture / "capture.json").write_text(json.dumps({
                "hidden_boundary": "before_final_rmsnorm", "calls": 1, "norm_calls": 1, "drop_idx": 1}))
            ids = np.array([[151643, 42, 43]], dtype=np.int64)
            np.save(capture / "input_ids.npy", ids)
            np.save(capture / "attention_mask.npy", np.ones_like(ids))
            hidden = np.ones((1, 3, 4096), dtype=np.float32)
            np.save(capture / "hidden_prenorm.npy", hidden)
            np.save(folder / "prompt_embeds.npy", hidden[:, 1:])
            got_ids, full, cropped, drop = load_fixture(folder, "positive")
            np.testing.assert_array_equal(got_ids, ids[0])
            self.assertEqual(full.shape, (3, 4096))
            self.assertEqual(cropped.shape, (2, 4096))
            self.assertEqual(drop, 1)
            np.save(folder / "prompt_embeds.npy", hidden[:, 1:] * 2)
            with self.assertRaisesRegex(ValueError, "not the captured"):
                load_fixture(folder, "positive")
            np.save(folder / "prompt_embeds.npy", hidden[:, 1:])
            np.save(capture / "attention_mask.npy", np.array([[0, 1, 1]]))
            with self.assertRaisesRegex(ValueError, "unpadded"):
                load_fixture(folder, "positive")


if __name__ == "__main__":
    unittest.main()
