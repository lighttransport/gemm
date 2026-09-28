"""Known-albedo, changed-light and unsupported-fit checks for illumination removal."""
import unittest
import numpy as np

from .head import relight


class RelightTests(unittest.TestCase):
    def normals(self):
        rng = np.random.default_rng(19)
        n = rng.normal(size=(4000, 3))
        n[:, 2] = np.abs(n[:, 2])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def test_known_albedo_under_changed_light(self):
        n = self.normals()
        rng = np.random.default_rng(23)
        detail = np.where(rng.random(len(n)) < .08, .65, 1.)
        albedo = detail[:, None] * [.36, .21, .14]
        for light in ([-.6,.2,.77], [.6,.2,.77], [0.,.6,.8]):
            source_light = .65 + .35 * (n @ np.asarray(light))
            baked = albedo * source_light[:, None]
            fit = relight.estimate(n, baked, np.ones(len(n)))
            self.assertEqual(fit['status'], 'estimated')
            corrected = baked * relight.gain(n, fit)[:, None]
            # A new key illuminates the corrected albedo. Normalize exposure
            # only; compare the known spatial albedo under that new light.
            new_light = .65 + .35 * (n @ np.array([.4,-.3,.85]))
            truth = albedo * new_light[:, None]
            before = baked * new_light[:, None]
            after = corrected * new_light[:, None]
            before *= np.median(truth / before)
            after *= np.median(truth / after)
            self.assertLess(np.sqrt(np.mean((after-truth)**2)),
                            .3 * np.sqrt(np.mean((before-truth)**2)))
            np.testing.assert_allclose(corrected[:,0]/corrected[:,1], albedo[:,0]/albedo[:,1], atol=1e-14)

    def test_detail_ratios_and_gain_bound(self):
        n = self.normals()
        rgb = np.exp(n @ [.4,-.3,.2])[:, None] * np.array([[.3,.2,.1]])
        fit = relight.estimate(n, rgb, np.ones(len(n)))
        pairs = np.repeat(n[:100], 2, axis=0)
        colour = np.tile([[.3,.2,.1],[.15,.1,.05]], (100,1))
        corrected = colour * relight.gain(pairs, fit)[:, None]
        np.testing.assert_allclose(corrected[::2]/corrected[1::2], 2., atol=1e-14)
        exaggerated = dict(fit, direction=[100.,-100.,100.])
        gains = relight.gain(n, exaggerated)
        self.assertTrue(((gains >= .5) & (gains <= 2.)).all())
        np.testing.assert_array_equal(relight.gain(n, fit, 0), 1.)

    def test_unsupported_and_unrelated_colour_are_noops(self):
        n = self.normals()
        rgb = np.full_like(n, [.3,.2,.1])
        flat = relight.estimate(np.tile([0.,0.,1.], (len(n),1)), rgb, np.ones(len(n)))
        self.assertEqual(flat['status'], 'insufficient normal variation')
        np.testing.assert_array_equal(relight.gain(n, flat), 1.)
        rng = np.random.default_rng(31)
        rgb *= np.exp(rng.normal(0,.1,len(n)))[:,None]
        fit = relight.estimate(n, rgb, np.ones(len(n)))
        self.assertEqual(fit['status'], 'no reliable directional component')
        empty = relight.estimate(np.empty((0,3)), np.empty((0,3)), np.empty(0))
        self.assertEqual(empty['status'], 'insufficient support')
        with self.assertRaises(ValueError):
            relight.estimate(n, rgb, np.full(len(n), -1))


if __name__ == '__main__':
    unittest.main()
