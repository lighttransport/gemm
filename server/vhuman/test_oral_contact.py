"""Lip/arch contact deformer: clearance, penetration push-out, caps and inactive vertices."""
import unittest

import numpy as np

from .reconstruction import contact_runtime as cr


def plane_arch():
    # Labial "arch" surface: two triangles in the z=0 plane, normal +z.
    native = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0.]])*.02
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return native, tris


class OralContactTests(unittest.TestCase):
    def spec(self, n, active, weight=1., clearance=5e-4, max_move=.008, neighbours=None):
        native, tris = plane_arch()
        cand = np.repeat(tris[None], len(active), 0)
        return native, dict(active=np.asarray(active), weight=np.full(len(active), weight), candidates=cand,
                            clearance=clearance, max_move=max_move, smoothing=[.5, 2], neighbours=neighbours)

    def test_gap_and_penetration_resolve_to_clearance(self):
        p = np.array([[0, 0, .004], [.005, 0, -.002], [0, .005, .02]])
        native, spec = self.spec(3, [0, 1])
        out = cr.contact_deform(p, native, spec)
        np.testing.assert_allclose(out[:2, 2], 5e-4, atol=1e-9)
        np.testing.assert_allclose(out[:2, :2], p[:2, :2], atol=1e-12)
        np.testing.assert_array_equal(out[2], p[2])

    def test_cap_and_weight(self):
        p = np.array([[0, 0, .05]])
        native, spec = self.spec(1, [0], max_move=.008)
        self.assertAlmostEqual(cr.contact_deform(p, native, spec)[0, 2], .05-.008, places=9)
        native, spec = self.spec(1, [0], weight=.5)
        self.assertAlmostEqual(cr.contact_deform(np.array([[0, 0, .0045]]), native, spec)[0, 2], .0025, places=9)

    def test_smoothing_spreads_without_penetration(self):
        p = np.array([[0, 0, .004], [.002, 0, .004]])
        rows, cols = np.array([0, 1]), np.array([1, 0])
        native, spec = self.spec(2, [0], neighbours=(rows, cols))
        out = cr.contact_deform(p, native, spec)
        self.assertLess(out[1, 2], .004)
        self.assertGreaterEqual(out[0, 2], 5e-4-1e-12)


if __name__ == '__main__':
    unittest.main()
