import copy
import unittest
from compare_qwen38_lowbit_quality import compare


class QualityGate(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(format="nvfp4", arithmetic=0, has_reference=True,
            tokens=1025, token_hash="0123456789abcdef", reference_token_hash="0123456789abcdef",
            reference_hash="abcdef0123456789",
            reference_format=3, reference_arithmetic=0,
            nll=2., perplexity=7.389, relative_l2=.1, max_abs=1., kl=.01)
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate.update(format="fp6", arithmetic=16, nll=1.9,
                              relative_l2=.09, max_abs=.9, kl=.009)

    def test_improved_quality_passes_only_quality(self):
        result = compare(self.baseline, self.candidate)
        self.assertTrue(result["passed"])
        self.assertFalse(result["greedy_trace_validated"])
        self.assertFalse(result["throughput_validated"])

    def test_each_degraded_metric_fails(self):
        for key in ("nll", "relative_l2", "max_abs", "kl"):
            candidate = dict(self.candidate, **{key: self.baseline[key] + .001})
            self.assertFalse(compare(self.baseline, candidate)["passed"])

    def test_nonfinite_missing_and_wrong_reference_fail(self):
        for key, value in (("nll", float("nan")), ("has_reference", False),
                           ("tokens", 1), ("reference_format", 1),
                           ("token_hash", "different"), ("reference_token_hash", None)):
            self.assertFalse(compare(self.baseline, dict(self.candidate, **{key: value}))["passed"])


if __name__ == "__main__":
    unittest.main()
