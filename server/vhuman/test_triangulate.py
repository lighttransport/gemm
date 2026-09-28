"""Geometry invariants for the experimental annulus triangulator."""
import unittest
import importlib.util
import numpy as np
from .head.triangulate import triangulate_annulus


requires_earcut = unittest.skipUnless(importlib.util.find_spec("mapbox_earcut"), "optional remesh dependency not installed")


class TriangulateTests(unittest.TestCase):
    @requires_earcut
    def test_concave_annulus_and_reversed_input(self):
        outer = np.array([[-4., -3.], [4., -3.], [4., 3.], [2., 3.], [2., 2.], [-4., 2.]])
        inner = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        for a, b in ((outer, inner), (outer[::-1], inner), (outer, inner[::-1])):
            tri = triangulate_annulus(a, b)
            self.assertEqual(len(tri), len(a) + len(b))
            p = np.concatenate([a, b])[tri]
            u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
            area = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
            self.assertTrue((area > 0).all())
            self.assertAlmostEqual(area.sum(), 38.)
            # Affine translation/scaling must preserve coverage.
            shifted = triangulate_annulus(a * .001 + 1e3, b * .001 + 1e3)
            self.assertEqual(len(shifted), len(tri))

    def test_rejects_invalid_boundaries(self):
        outer = np.array([[-4., -4.], [4., -4.], [4., 4.], [-4., 4.]])
        inner = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        cases = [(outer[[0, 2, 1, 3]], inner), (outer, inner + 8),
                 (outer, inner + [3, 0]), (outer, inner * 6),
                 (outer, inner[[0, 1, 1, 3]]), (outer, np.full((3, 2), np.nan)),
                 (outer, inner[:2])]
        for a, b in cases:
            with self.assertRaises(ValueError):
                triangulate_annulus(a, b)

    @requires_earcut
    def test_irregular_dense_boundary(self):
        theta = np.sort(np.random.default_rng(11).uniform(-np.pi, np.pi, 240))
        radius = 3 + .2 * np.cos(theta * 5)
        outer = np.stack([radius * np.cos(theta), radius * np.sin(theta)], 1)
        inner_angle = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        inner = np.stack([1.6 * np.cos(inner_angle), .7 * np.sin(inner_angle)], 1)
        tri = triangulate_annulus(outer, inner)
        self.assertEqual(len(tri), len(outer) + len(inner))
        self.assertEqual(len(np.unique(tri)), len(outer) + len(inner))


if __name__ == '__main__':
    unittest.main()
