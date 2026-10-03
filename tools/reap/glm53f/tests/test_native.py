import os
from pathlib import Path
import sys
import unittest

import numpy as np

from glm_reap.native import NativeCodec, FORMATS, decode, pack, unpack, nbytes
from glm_reap.reap import select
from glm_reap.rco import allocate


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("GGML_LIBRARY", "vendor/llama.cpp/build-compression/bin/libggml-base.so")
        if not Path(library).exists():
            raise unittest.SkipTest("Set GGML_LIBRARY or build native ggml first")
        cls.codec = NativeCodec(library)

    def test_native_golden_and_serialization(self):
        import gguf
        rng = np.random.default_rng(42)
        weights = [rng.normal(size=(8, 256)).astype(np.float32), np.zeros((8, 256), np.float32), rng.uniform(-20, 20, (8, 256)).astype(np.float32)]
        for weight in weights:
            for kind, (enum, _, _) in FORMATS.items():
                with self.subTest(kind=kind):
                    raw = self.codec.encode(weight, kind)
                    reference = gguf.dequantize(raw, gguf.GGMLQuantizationType(enum)).reshape(weight.shape)
                    np.testing.assert_array_equal(decode(raw, kind, weight.shape), reference)
                    self.assertEqual(raw.nbytes, nbytes(weight.shape, kind))
                    if kind not in ("F16", "F32"):
                        np.testing.assert_array_equal(pack(kind, *unpack(raw, kind)), raw)

    def test_importance_weights(self):
        weight = np.random.default_rng(10).normal(size=(4, 256)).astype(np.float32)
        for kind in ("Q2_K", "Q3_K", "Q4_K", "Q6_K", "Q8_0"):
            raw = self.codec.encode(weight, kind, np.linspace(0.1, 2, 256).astype(np.float32))
            self.assertTrue(np.isfinite(decode(raw, kind, weight.shape)).all())


class SelectionTests(unittest.TestCase):
    def test_reap_tie_and_original_order(self):
        self.assertEqual(select([1, 3, 3, 0, 2], 3), [1, 2, 4])
        with self.assertRaises(ValueError):
            select([float("nan")], 1)

    def test_strict_budget(self):
        costs = {"a": np.array([84, 110]), "b": np.array([84, 110]), "fixed": np.array([10])}
        scores = {"a": [0, 3], "b": [0, 1], "fixed": [0]}
        selected, size = allocate(costs, scores, 204)
        self.assertLessEqual(size, 204)
        self.assertEqual(selected["a"], 1)
        self.assertEqual(selected["b"], 0)
        with self.assertRaises(ValueError):
            allocate(costs, scores, 177)
