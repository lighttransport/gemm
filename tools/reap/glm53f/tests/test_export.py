import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from glm_reap.native import FORMATS, pack


class SerializationTests(unittest.TestCase):
    def test_packed_candidate_survives_gguf(self):
        import gguf
        q = np.tile(np.arange(256)%4, (2, 1)).astype(np.int16)
        raw = pack("Q2_K", q, np.full((2, 16), 3), np.full((2, 1), .25), np.full((2, 16), 2), np.full((2, 1), .125))
        expected = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"test.gguf"
            writer = gguf.GGUFWriter(target, "llama")
            writer.add_tensor_info("test.weight", (2, 256), np.dtype(np.float32), raw.nbytes, raw_dtype=gguf.GGMLQuantizationType.Q2_K)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            writer.write_tensor_data(raw)
            writer.close()
            reader = gguf.GGUFReader(target)
            actual = np.ascontiguousarray(reader.tensors[0].data).view(np.uint8).reshape(-1)
            self.assertEqual(hashlib.sha256(actual).hexdigest(), expected)
