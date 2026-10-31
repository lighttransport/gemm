"""color_chart: exposure and white balance recovered from a synthetic chart render."""
import json
import unittest

import numpy as np

from server.vhuman.reconstruction import color_chart as cc


class ColorChartTest(unittest.TestCase):
    def test_recovers_exposure_and_white_balance(self):
        light = np.array([1.3, 1.0, .8]) * 2.0 ** 1.25          # tinted, 1.25 EV too bright
        rendered = cc.reference_linear() * light
        r = cc.calibrate(rendered)
        self.assertAlmostEqual(r['exposure_ev'], -1.25 - np.log2(1.0), places=3)
        np.testing.assert_allclose(r['wb_gains'], [1 / 1.3, 1, 1 / .8], rtol=1e-6)
        self.assertLess(r['delta_e_mean'], .6)                   # 8-bit rounding only

    def test_layout_and_sampling(self):
        centres, size = cc.layout()
        self.assertEqual(centres.shape, (24, 2))
        self.assertAlmostEqual(centres[:, 0].mean(), 0, places=9)
        img = np.zeros((400, 600, 3))
        patches = []
        for i, (x, y) in enumerate(centres):
            px, py = 300 + x * 2000, 200 - y * 2000
            img[int(py) - 15:int(py) + 16, int(px) - 15:int(px) + 16] = cc.reference_linear()[i]
            patches.append(dict(name=cc.PATCHES[i][0], pixel_center=[px, py], pixel_half_size=size / 2 * 2000))
        np.testing.assert_allclose(cc.sample(img, patches), cc.reference_linear(), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
