import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from glm_reap.checkpoint import Checkpoint


class CheckpointTests(unittest.TestCase):
    def test_fp8_non_aligned_rows_multiply_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = np.full((256, 256), 56, np.uint8)  # E4M3 value 1.
            scales = np.array([[2, 3], [5, 7]], np.float32)
            name = "matrix.weight"
            header = {name: {"dtype": "F8_E4M3", "shape": [256, 256], "data_offsets": [0, weights.nbytes]}, name+"_scale_inv": {"dtype": "F32", "shape": [2, 2], "data_offsets": [weights.nbytes, weights.nbytes+scales.nbytes]}}
            encoded = json.dumps(header).encode()
            (root/"model-1.safetensors").write_bytes(struct.pack("<Q", len(encoded))+encoded+weights.tobytes()+scales.tobytes())
            (root/"config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [128, 128]}}))
            (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model-1.safetensors" for key in header}}))
            value = Checkpoint(root).read(name, 127, 129)
            checkpoint = Checkpoint(root)
            checkpoint.mode = "ram"
            np.testing.assert_array_equal(checkpoint.read(name, 127, 129), value)
            checkpoint.cache[name] = weights.copy()
            checkpoint.cache_bytes = weights.nbytes
            np.testing.assert_array_equal(checkpoint.read(name, 127, 129), value)
            checkpoint.set_cache_limit(0)
            self.assertFalse(checkpoint.cache)
            np.testing.assert_array_equal(value[0, :128], np.full(128, 2))
            np.testing.assert_array_equal(value[0, 128:], np.full(128, 3))
            np.testing.assert_array_equal(value[1, :128], np.full(128, 5))
            np.testing.assert_array_equal(value[1, 128:], np.full(128, 7))

            import torch
            if torch.cuda.is_available():
                checkpoint = Checkpoint(root)
                for dtype in (torch.float32, torch.bfloat16):
                    actual = checkpoint.tensor(name, 127, 129, "cuda", dtype)
                    expected = torch.from_numpy(value).to("cuda", dtype)
                    self.assertTrue(torch.equal(actual, expected))
