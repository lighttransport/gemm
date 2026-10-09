"""Lip-aperture occlusion: analytic form factors, horizon handling and per-vertex fits."""
import unittest

import numpy as np

from .reconstruction import oral_occlusion as oo


def disk(r, z, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack((r*np.cos(t), r*np.sin(t), np.full(n, z)), 1)


class OralOcclusionTests(unittest.TestCase):
    def test_disk_form_factor_matches_closed_form(self):
        for r, h in ((1, 2), (.5, 1), (2, .5)):
            ff = oo.polygon_form_factor([[0, 0, 0]], [[0, 0, 1]], disk(r, h))[0]
            self.assertAlmostEqual(ff, r*r/(r*r+h*h), places=3)

    def test_behind_and_bounds(self):
        self.assertEqual(oo.polygon_form_factor([[0, 0, 0]], [[0, 0, -1]], disk(1, 2))[0], 0)
        self.assertGreater(oo.polygon_form_factor([[0, 0, 0]], [[0, 0, 1]], disk(1e3, 1))[0], .99)
        tilted = oo.polygon_form_factor([[0, 0, 0]], [[1, 0, 0]], disk(1, 2))[0]
        self.assertTrue(0 <= tilted < .2)

    def test_closed_aperture_gives_no_visibility(self):
        flat = disk(1e-7, 1)
        self.assertLess(oo.polygon_form_factor([[0, 0, 0]], [[0, 0, 1]], flat)[0], 1e-9)

    def test_contour_attachment(self):
        v = np.arange(12, dtype=float).reshape(4, 3)
        c = oo.contour(v, np.array([[0, 1, 2]]), np.array([[.5, .25, .25]]))
        np.testing.assert_allclose(c, [[.5*v[0]+.25*v[1]+.25*v[2]][0]])

    def test_fits_recover_linear_models(self):
        rng = np.random.default_rng(0)
        f = rng.random((40, 5))
        s = np.array([.2, .5, 1., .8, .1])
        np.testing.assert_allclose(oo.fit_static_scale(f*s, f), s, atol=1e-9)
        feats = rng.random((40, 2))
        w_true = rng.random((3, 6))*.1
        v = np.column_stack((np.ones(40), feats))@w_true
        np.testing.assert_allclose(oo.fit_affine(v, feats, ridge=1e-10), w_true, atol=1e-6)
        self.assertTrue((oo.predict_affine(w_true*20, feats) <= 1).all())


if __name__ == '__main__':
    unittest.main()
