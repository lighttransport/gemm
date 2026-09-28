"""Layer-constrained upper-lid cleanup on synthetic folded surfaces."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from .head import camera, cleanup, landmarks


class UpperLidCleanupTest(unittest.TestCase):
    def fixture(self, bump=True, back=True, thin=False, duplicate=False):
        cam = camera.PixalCamera(math.radians(20), 1., 0., 0., 512)
        k = 6.
        yy, xx = np.mgrid[:512, :512]
        eye = landmarks.Eye('right', 256., 256., 18., 5.,
                            opening=((xx - 256) / 42) ** 2 + ((yy - 256) / 13) ** 2 < 1)
        x, y = np.meshgrid(np.arange(206., 307., 2), np.arange(212., 273., 2))
        _, rays = cam.rays(x.ravel(), y.ravel())
        n = x.size
        height = .003 * k * np.exp(-((x.ravel() - 256) / 3.) ** 2 - ((y.ravel() - 236) / 3.) ** 2) if bump else np.zeros(n)
        distance = np.full(n, cam.distance) - height
        positions = cam.origin + rays * distance[:, None]
        row, col = np.mgrid[:x.shape[0] - 1, :x.shape[1] - 1]
        a = (row * x.shape[1] + col).ravel(); b = a + 1; c = a + x.shape[1]; d = c + 1
        tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
        normals = np.tile([0., 0., -1.], (n, 1))
        if back:
            rear = distance + .0001 * k if thin else np.full(n, cam.distance + .0008 * k)
            positions = np.concatenate([positions, cam.origin + rays * rear[:, None]])
            normals = np.concatenate([normals, -normals])
            tri = np.concatenate([tri, tri[:, ::-1] + n])
        if duplicate:
            total = len(positions)
            positions = np.concatenate([positions, positions])
            normals = np.concatenate([normals, normals])
            tri[::2] += total
        mesh = dict(positions=positions, normals=normals, triangles=tri)
        pose = SimpleNamespace(center=np.array([0., 0., .06]), units_per_m=k)
        return mesh, np.ones(len(tri), bool), cam, [eye], [pose]

    def test_reduces_bulge_preserves_projection_and_rear_layer(self):
        args = self.fixture()
        original = args[0]['positions'].copy()
        out, info = cleanup.relax_upper_lids(*args)
        self.assertGreater(info['right']['moved_vertices'], 0)
        self.assertLessEqual(info['right']['max_move_mm'], 3.5)
        cam = args[2]; n = len(original) // 2
        np.testing.assert_allclose(cam.project(original), cam.project(out['positions']), atol=1e-10)
        np.testing.assert_array_equal(out['positions'][n:], original[n:])
        front = np.linalg.norm(out['positions'][:n] - cam.origin, axis=1)
        rear = np.linalg.norm(out['positions'][n:] - cam.origin, axis=1)
        self.assertTrue((rear - front >= .0002 * 6 - 1e-9).all())
        self.assertGreater(front.min(), np.linalg.norm(original[:n] - cam.origin, axis=1).min())
        np.testing.assert_array_equal(args[0]['positions'], original)  # input is immutable
        np.testing.assert_allclose(np.linalg.norm(out['normals'], axis=1), 1., atol=1e-8)

    def test_no_support_no_motion(self):
        for options in ({'back': False}, {'bump': False}, {'thin': True}):
            args = self.fixture(**options)
            out, info = cleanup.relax_upper_lids(*args)
            self.assertEqual(info['right']['moved_vertices'], 0, options)
            np.testing.assert_array_equal(out['positions'], args[0]['positions'])
            np.testing.assert_array_equal(out['normals'], args[0]['normals'])

    def test_uv_seams_stay_welded(self):
        args = self.fixture(duplicate=True)
        out, info = cleanup.relax_upper_lids(*args)
        n = len(out['positions']) // 2
        self.assertGreater(info['right']['moved_vertices'], 0)
        np.testing.assert_array_equal(out['positions'][:n], out['positions'][n:])
        np.testing.assert_array_equal(out['normals'][:n], out['normals'][n:])

    def test_missing_bilinear_support(self):
        f = np.array([[1., 2.], [3., np.inf]])
        self.assertTrue(np.isnan(cleanup._sample_finite(f, np.array([[.5, .5]]))[0]))
        np.testing.assert_allclose(cleanup._sample_finite(np.array([[1., 2.], [3., 4.]]), np.array([[.5, .5]])), [2.5])


if __name__ == '__main__':
    unittest.main()
