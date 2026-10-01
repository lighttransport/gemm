"""Registration and shape smoothing must honor the configured execution device."""
from importlib.util import find_spec
import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import gpu
from .rig import expressions, register, template


@unittest.skipUnless(find_spec("torch") and find_spec("scipy"), "optional torch/scipy unavailable")
class RigBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        self.positions = np.array([[-.01, -.01, .05], [.01, -.01, .05],
                                   [.01, .01, .05], [-.01, .01, .05]])
        tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        self.template = SimpleNamespace(n=4, tris=tris, tri_mat=np.zeros(2, np.int32),
                                        chart=template.chart_of_points(self.positions))
        self.fields = SimpleNamespace(tmpl=self.template, eye_k=np.ones(4),
                                      eye_side=np.full(4, -1), lip_k=np.array([0, 1, 1, 1]))
        self.shapes = {"smile": np.arange(12).reshape(4, 3).astype(np.float64) * .001}
        points = np.array([[x, y, .051] for x in (-.01, 0, .01) for y in (-.01, 0, .01)])
        self.subject = SimpleNamespace(positions=points, normals=np.tile([0., 0., 1.], (9, 1)))

    def refine(self):
        return register.refine(self.template, self.subject, self.positions,
                               np.array([True, False, False, False]), np.ones(4, bool), iters=3)

    def test_cpu_selection_ignores_available_gpu(self):
        with gpu.execution("cpu"):
            # Adam itself queries CUDA availability even for CPU parameters.
            with patch.object(self.torch.cuda, "set_device",
                              side_effect=AssertionError("unexpected GPU selection")):
                out, stats = self.refine()
            with patch.object(self.torch.cuda, "is_available",
                              side_effect=AssertionError("unexpected GPU selection")):
                shapes = expressions.smooth_shapes(self.fields, self.shapes, iters=2)
        self.assertEqual(stats["backend"], "cpu")
        self.assertEqual(stats["device"], "cpu")
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out[0], self.positions[0])
        np.testing.assert_array_equal(shapes["smile"][0], self.shapes["smile"][0])

    def test_rocm_matches_cpu_and_preserves_fixed_vertices(self):
        if not self.torch.version.hip or not self.torch.cuda.is_available():
            self.skipTest("ROCm PyTorch device unavailable")
        with gpu.execution("cpu"):
            cpu, _ = self.refine()
            cpu_shapes = expressions.smooth_shapes(self.fields, self.shapes, iters=2)
        with gpu.execution("rocm", 0):
            actual, stats = self.refine()
            actual_shapes = expressions.smooth_shapes(self.fields, self.shapes, iters=2)
        self.assertEqual(stats["backend"], "rocm")
        self.assertEqual(stats["device"], "cuda:0")
        np.testing.assert_allclose(actual, cpu, atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual_shapes["smile"], cpu_shapes["smile"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(actual[0], self.positions[0])
        np.testing.assert_array_equal(actual_shapes["smile"][0], self.shapes["smile"][0])

    def test_invalid_smoothing_indices_fail_before_sparse_dispatch(self):
        self.template.tris = np.array([[0, 1, 4]], np.int32)
        with gpu.execution("cpu"), patch.object(self.torch.sparse, "mm",
                side_effect=AssertionError("invalid adjacency reached sparse dispatch")):
            with self.assertRaises(RuntimeError):
                expressions.smooth_shapes(self.fields, self.shapes)

    def test_missing_torch_does_not_silently_fall_back_from_rocm(self):
        original_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("test: torch unavailable")
            return original_import(name, *args, **kwargs)

        with gpu.execution("cpu"):
            reference = expressions.smooth_shapes(self.fields, self.shapes, iters=2)
        with gpu.execution("rocm"), patch.object(builtins, "__import__", side_effect=without_torch):
            with self.assertRaisesRegex(RuntimeError, "PyTorch unavailable for rocm smoothing"):
                expressions.smooth_shapes(self.fields, self.shapes)
            actual = expressions.smooth_shapes(self.fields, self.shapes, iters=2, device="cpu")
        np.testing.assert_allclose(actual["smile"], reference["smile"], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
