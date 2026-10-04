"""Reject altered provenance before a streamed reference array can be pruned."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from ref.minimax_h3_native.verify_streaming import validate_receipt
    from ref.minimax_h3_native import verify
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    validate_receipt = None
    verify = None


@unittest.skipIf(validate_receipt is None, "streaming checks require NumPy")
class StreamingReceiptTests(unittest.TestCase):
    def test_altered_or_noncanonical_receipts(self):
        name, shape = "frame_000", [768, 1344, 3]
        sources = {"reference_source_sha256": "r" * 64, "capture_reader_sha256": "c" * 64}
        receipt = {"storage": name + ".npy.gz", "sha256": "s" * 64,
                   "generation_sha256": "g" * 64, "upstream_revision": verify.UPSTREAM,
                   "shared_input": False, "shape": shape, **sources}
        def check(value):
            validate_receipt(name, value, "s" * 64, shape, "g" * 64, sources)
        check(receipt)
        for key in receipt:
            with self.subTest(key=key):
                altered = dict(receipt)
                altered[key] = None
                with self.assertRaises(ValueError):
                    check(altered)
        for storage in ("../frame_000.npy.gz", "/frame_000.npy.gz", "frame_001.npy.gz"):
            with self.assertRaises(ValueError):
                check({**receipt, "storage": storage})
        with self.assertRaises(ValueError):
            check({**receipt, "shared_input": 0})

    def test_only_noise_may_be_shared(self):
        sources = {"reference_source_sha256": "r", "capture_reader_sha256": "c"}
        for name in ("noise_video", "latent_video_000"):
            receipt = {"storage": name + ".npy", "sha256": "s", "generation_sha256": "g",
                       "upstream_revision": verify.UPSTREAM, "shape": [1], **sources,
                       "shared_input": name.startswith("noise_")}
            validate_receipt(name, receipt, "s", [1], "g", sources)
            receipt["shared_input"] = not receipt["shared_input"]
            with self.assertRaises(ValueError):
                validate_receipt(name, receipt, "s", [1], "g", sources)


if __name__ == "__main__":
    unittest.main()
