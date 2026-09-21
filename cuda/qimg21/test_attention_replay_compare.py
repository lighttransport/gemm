import unittest

import numpy as np

from attention_replay_compare import metrics


class AttentionReplayCompareTest(unittest.TestCase):
    def test_target_failure_cannot_hide_in_prefix(self):
        reference = np.ones((100001, 1), dtype=np.float32)
        candidate = reference.copy()
        candidate[-1] = -1
        result = metrics(reference, candidate, 1)
        self.assertTrue(result["regions"]["all"]["passed"])
        self.assertFalse(result["regions"]["target"]["passed"])
        self.assertFalse(result["passed"])

    def test_exact_and_invalid_inputs(self):
        reference = np.ones((4, 2), dtype=np.float32)
        self.assertTrue(metrics(reference, reference.copy(), 2)["passed"])
        for candidate, count in ((reference, 0), (reference, 5), (reference[:, :1], 2),
                                 (np.full_like(reference, np.nan), 2)):
            with self.assertRaises(ValueError):
                metrics(reference, candidate, count)


if __name__ == "__main__":
    unittest.main()
