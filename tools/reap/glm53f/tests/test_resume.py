import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glm_reap.resume import TileStore, write_payload


class ResumeTests(unittest.TestCase):
    def test_tile_reopen_exact_and_reject_changed_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tiles.sqlite"
            initial = np.arange(200, dtype=np.uint8)
            optimized = initial.copy()
            optimized[3] ^= 17
            store = TileStore(path, "identity", reserve=0)
            store.save("tile", initial, optimized, {"improved": True})
            store.close()
            store = TileStore(path, "identity", reserve=0)
            raw, metrics = store.restore("tile", initial)
            np.testing.assert_array_equal(raw, optimized)
            self.assertTrue(metrics["improved"])
            with self.assertRaises(ValueError):
                store.restore("tile", optimized)
            store.close()
            with self.assertRaises(ValueError):
                TileStore(path, "changed", reserve=0)

    def test_export_interrupt_resume_matches_uninterrupted(self):
        import gguf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.arange(512, dtype=np.float32).view(np.uint8)
            tensors = [(name, (2, 256), "F32", [raw]) for name in ("a", "b")]
            def writer():
                result = gguf.GGUFWriter(None, "llama")
                for name, shape, _, chunks in tensors:
                    result.add_tensor_info(name, shape, np.dtype(np.float32), raw.nbytes, raw_dtype=gguf.GGMLQuantizationType.F32)
                return result
            from glm_reap.common import atomic_json
            def interrupt(path, state):
                atomic_json(path, state)
                if len(state["tensors"]) == 1:
                    raise KeyboardInterrupt()
            with patch("glm_reap.resume.atomic_json", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    write_payload(writer(), tensors, root/"resume.gguf", root/"resume.json", "id", 0)
            # An uncommitted tail must be discarded.
            with (root/"resume.gguf").open("ab") as stream:
                stream.write(b"uncommitted")
            write_payload(writer(), tensors, root/"resume.gguf", root/"resume.json", "id", 0)
            write_payload(writer(), tensors, root/"full.gguf", root/"full.json", "id", 0)
            self.assertEqual((root/"resume.gguf").read_bytes(), (root/"full.gguf").read_bytes())
            reader = gguf.GGUFReader(root/"resume.gguf")
            self.assertEqual(len(reader.tensors), 2)
            with (root/"resume.gguf").open("r+b") as stream:
                state = json.loads((root/"resume.json").read_text())
                stream.seek(state["tensors"]["a"]["start"])
                stream.write(b"bad")
            with self.assertRaises(ValueError):
                write_payload(writer(), tensors, root/"resume.gguf", root/"resume.json", "id", 0)
