"""Local skin cleanup protection, edge cases and color-outlier reduction."""
import unittest

import numpy as np

from .reconstruction.local_skin_color import correct_regions


class LocalSkinColorTests(unittest.TestCase):
    def fixture(self):
        x, y = np.meshgrid(np.linspace(-.03, .03, 31), np.linspace(-.03, .03, 31))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        colors = np.tile([.35, .20, .13], (len(points), 1))
        colors *= (1 + .05 * np.sin(points[:, :1] * 200))
        patch = np.linalg.norm(points[:, :2], axis=1) < .004
        colors[patch] = [.30, .32, .35]
        region = [dict(name='test', center=[0, 0, 0], radii=[.02, .02, .02])]
        return points, colors, patch, region

    def test_cyan_patch_improves_with_exact_exterior(self):
        p, c, patch, regions = self.fixture()
        out, weights, _ = correct_regions(p, c, regions)
        self.assertLess(np.mean(out[patch, 2] / out[patch, 0]),
                        np.mean(c[patch, 2] / c[patch, 0]))
        np.testing.assert_array_equal(out[weights == 0], c[weights == 0])
        self.assertTrue(np.all(weights[np.linalg.norm(p, axis=1) >= .02] == 0))
        self.assertTrue(np.isfinite(out).all())

    def test_observed_protection_and_zero_strength(self):
        p, c, patch, regions = self.fixture()
        out, weights, _ = correct_regions(p, c, regions, protected=patch)
        np.testing.assert_array_equal(out[patch], c[patch])
        self.assertTrue(np.all(weights[patch] == 0))
        out, weights, _ = correct_regions(p, c, regions, strength=0)
        np.testing.assert_array_equal(out, c)
        self.assertFalse(weights.any())

    def test_uniform_and_empty_region_are_exact(self):
        p, c, _, regions = self.fixture()
        c[:] = [.35, .2, .13]
        out, weights, _ = correct_regions(p, c, regions)
        np.testing.assert_array_equal(out, c)
        self.assertFalse(weights.any())
        regions[0]['center'] = [10, 0, 0]
        out, weights, rows = correct_regions(p, c, regions)
        self.assertEqual(rows[0]['region_texels'], 0)
        np.testing.assert_array_equal(out, c)

    def test_single_trusted_reference_has_valid_shape(self):
        p = np.array([[0, 0, 0], [.003, 0, 0]])
        c = np.array([[.3, .18, .1], [.8, .8, .8]])
        region = [dict(name='tiny', center=[0, 0, 0], radii=[.01] * 3)]
        out, weights, rows = correct_regions(p, c, region)
        self.assertEqual(rows[0]['trusted_samples'], 1)
        self.assertEqual(out.shape, c.shape)
        self.assertGreater(weights[1], 0)
        self.assertLess(out[1, 2], c[1, 2])

    def test_invalid_or_overlapping_regions_fail(self):
        p, c, _, region = self.fixture()
        with self.assertRaises(ValueError):
            correct_regions(p, c, region + [dict(region[0], name='overlap')])
        for radii in ([0, .1, .1], [float('nan')] * 3):
            with self.assertRaises(ValueError):
                correct_regions(p, c, [dict(region[0], radii=radii)])
        with self.assertRaises(ValueError):
            correct_regions(p, c, region, strength=float('nan'))
        with self.assertRaises(ValueError):
            correct_regions(p, c, region, protected=[True])


if __name__ == '__main__':
    unittest.main()
