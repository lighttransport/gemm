import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import quality_gate


class QualityGateTest(unittest.TestCase):
    def test_metrics(self):
        result = quality_gate.compare(
            {"a": {"ids": [1, 2], "text": "hello"},
             "b": {"ids": [3], "text": "world"}},
            {"a": {"ids": [1, 2], "text": "hello"},
             "b": {"ids": [4], "text": "word"}})
        self.assertEqual(result["token_exact_rate"], 0.5)
        self.assertGreater(result["mean_text_similarity"], 0.8)


if __name__ == "__main__":
    unittest.main()
