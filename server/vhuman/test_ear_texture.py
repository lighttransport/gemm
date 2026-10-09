"""Ear appearance helpers: atlas rasterization, surface filters, exemplar detail and priors."""
import unittest

import numpy as np

from .reconstruction import ear_texture as et


def plane(n=24):
    x, y = np.meshgrid(np.linspace(0, .03, n), np.linspace(0, .03, n))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tris = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            tris += [(a, a+1, a+n+1), (a, a+n+1, a+n)]
    tris = np.asarray(tris)
    uv = (points[:, :2]/.03*.5+.25)[tris]
    return points, tris, uv


class EarTextureTests(unittest.TestCase):
    def test_atlas_surface_interpolates_positions(self):
        points, tris, uv = plane()
        ids, valid, p, n, bary = et.atlas_surface(points, tris, uv, 128)
        self.assertGreater(valid.sum(), 1000)
        rows, cols = np.nonzero(valid)
        expected_x = ((cols+.5)/128-.25)/.5*.03
        np.testing.assert_allclose(p[:, 0], expected_x, atol=2e-4)
        np.testing.assert_allclose(np.abs(n[:, 2]), 1, atol=1e-9)
        np.testing.assert_allclose(bary.sum(1), 1, atol=1e-5)

    def test_lowpass_preserves_constants_and_reduces_noise(self):
        points, _, _ = plane()
        rng = np.random.default_rng(0)
        const = np.tile([.2, .3, .4], (len(points), 1))
        np.testing.assert_allclose(et.surface_lowpass(points, const, .003), const, atol=1e-12)
        noise = rng.normal(size=(len(points), 3))
        self.assertLess(et.surface_lowpass(points, noise, .003).std(), .6*noise.std())

    def test_exemplar_residual_is_bounded_detail(self):
        points, _, _ = plane()
        rng = np.random.default_rng(1)
        source = np.log(.3+.05*rng.random((len(points), 3)))
        target = points+[.1, 0, 0]
        normals = np.tile([0, 0, 1.], (len(points), 1))
        out = et.exemplar_residual(target, normals, points, source, seed=3)
        self.assertEqual(out.shape, (len(points), 3))
        self.assertLess(np.abs(out.mean(0)).max(), .02)
        self.assertLess(np.abs(out).max(), np.ptp(source))
        np.testing.assert_allclose(et.exemplar_residual(target, normals, points, source, seed=3), out)

    def test_tint_prior_is_capped_and_regional(self):
        xy = np.array([[.1, .5], [.5, .1], [.6, .5]])
        outline = np.array([0., .3, .3])
        tint = et.anatomical_tint(xy, outline)
        self.assertGreater(tint[0, 0], tint[2, 0])
        self.assertGreater(tint[1, 0], tint[2, 0])
        self.assertLessEqual(tint.max(), .06+.05+1e-9)
        self.assertTrue((tint[:, 0] >= tint[:, 1]).all())

    def test_half_illumination_is_bounded(self):
        normals = np.array([[0, 0, 1.], [0, 0, -1.], [1, 0, 0]])
        h = et.half_illumination(normals, dict(log_direction=[0, 0, 5.], irradiance_median=1.))
        self.assertTrue(((h >= np.sqrt(.5)-1e-12) & (h <= np.sqrt(2)+1e-12)).all())


if __name__ == '__main__':
    unittest.main()
