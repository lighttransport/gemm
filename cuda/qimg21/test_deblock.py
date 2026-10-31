"""qimg21_i23d.deblock: removes a texture-scaled 16-px pattern, leaves pattern-free images nearly alone."""
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from qimg21_i23d.deblock import cell_pattern_std, deblock


def _texture(seed, size=512):
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.normal(0, 30, (size, size)), 40) + 128     # smooth shading
    fine = gaussian_filter(rng.normal(0, 6, (size, size)), 1.2)          # skin-like fine texture
    amp = np.clip(gaussian_filter(rng.random((size, size)), 60) * 4, 0.2, 2)
    return base, fine * amp, amp


class DeblockTest(unittest.TestCase):
    def test_removes_injected_cell_pattern(self):
        base, fine, amp = _texture(0)
        cell = np.random.default_rng(1).normal(0, 1, (16, 16)); cell -= cell.mean()
        pattern = np.tile(cell, (32, 32)) * amp * 0.8
        img = np.stack([base + fine + pattern] * 3)
        out = deblock(img)
        before = cell_pattern_std(img[0]); after = cell_pattern_std(out[0])
        self.assertLess(after, 0.5 * before)
        self.assertLess(np.sqrt(((out - img - (-pattern)) ** 2).mean()), np.sqrt((pattern ** 2).mean()))

    def test_pattern_free_image_barely_changes(self):
        base, fine, _ = _texture(2)
        img = np.stack([base + fine] * 3)
        out = deblock(img)
        self.assertLess(np.abs(out - img).max(), 3.0 + 1e-9)
        self.assertLess(np.sqrt(((out - img) ** 2).mean()), 0.3)

    def test_scale_and_shape(self):
        img = np.zeros((3, 20, 20))
        np.testing.assert_array_equal(deblock(img), img)          # too small for two cells: unchanged
        base, fine, _ = _texture(3, 128)
        x = np.stack([base + fine] * 3)
        a = deblock(x); b = deblock(x / 127.5 - 1, scale=2.0)
        np.testing.assert_allclose((b + 1) * 127.5, a, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
