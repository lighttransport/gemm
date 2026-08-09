#!/usr/bin/env python3
import os
import struct
import tempfile
import unittest

from k3_gguf_mla_materialize import bf16, materialize_combined

def q8_plane(cols, rows, base):
    out = bytearray()
    for r in range(rows):
        for c in range(0, cols, 32):
            out += struct.pack("<e", 1.0)
            out += bytes(((base + r + c + j) & 0xff) for j in range(32))
    return bytes(out)

class MlaMaterializeTest(unittest.TestCase):
    def test_native_head_interleave(self):
        with tempfile.TemporaryDirectory(prefix="k3-mla-test-") as d:
            src = os.path.join(d, "q8.bin")
            with open(src, "wb") as f:
                f.write(q8_plane(128, 512, 0))
                v_off = f.tell()
                f.write(q8_plane(512, 128, 100))
            kr = {"type": "Q8_0", "dims": [128, 512, 96],
                  "data_start": 0, "row_bytes": 136, "source": src}
            vr = {"type": "Q8_0", "dims": [512, 128, 96],
                  "data_start": v_off, "row_bytes": 544, "source": src}
            out = os.path.join(d, "kv.bf16")
            materialize_combined(kr, vr, out, head_count=1)
            self.assertEqual(os.path.getsize(out), 256 * 512 * 2)
            with open(out, "rb") as f:
                first_k = struct.unpack("<512H", f.read(512 * 2))
                f.seek(128 * 512 * 2)
                first_v = struct.unpack("<512H", f.read(512 * 2))
            # K output row 0 reads source row i, column 0; V output row 0 is
            # already contiguous in source row 0.
            self.assertEqual(first_k[0], bf16(0))
            self.assertEqual(first_k[1], bf16(1))
            self.assertEqual(first_v[0], bf16(100))
            self.assertEqual(first_v[1], bf16(101))

if __name__ == "__main__":
    unittest.main()
