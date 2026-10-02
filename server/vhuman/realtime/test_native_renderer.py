"""Native CUDA tests; Torch/gsplat are optional independent reference oracles."""
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from .src.avatar.bundle import bind, deform
from .src.avatar.cuda_runtime import CudaRuntime
from .src.renderer.native import NativeGaussianRenderer


def fixture(policy='trace-v1', count=700):
    rng = np.random.default_rng(716)
    vertices = np.array([[-.13, -.12, 1], [.15, -.1, 1.04], [.02, .16, .98],
                         [-.04, -.1, .84], [.11, .04, .9], [-.08, .14, .87]], np.float32)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    avatar = bind(vertices, triangles, ['jawOpen', 'smile'], count=count)
    avatar.metadata['covariance_policy'] = policy
    a = avatar.arrays
    a['rgb'][:] = rng.uniform(.02, .9, (count, 3))
    a['opacity'][:] = rng.uniform(.05, .8, count)
    a['color_basis'][:] = rng.normal(0, .1, (count, 8, 3))
    a['expression_matrix'][:] = rng.normal(0, .2, (2, 8))
    a['covariance_local'] *= 30
    view = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[230, 0, 48], [0, 220, 40], [0, 0, 1]], np.float32)
    return avatar, vertices, triangles, view, intrinsic


class NativeRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            runtime = CudaRuntime()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        runtime.close()

    def test_geometry_matches_numpy_and_frames_own_their_storage(self):
        for policy in ('trace-v1', 'eigen-v1'):
            with self.subTest(policy=policy):
                avatar, vertices, triangles, view, intrinsic = fixture(policy)
                renderer = NativeGaussianRenderer(avatar, triangles)
                frames = []
                try:
                    controls = np.array([.8, -.4], np.float32)
                    actual = renderer.deform(vertices, controls)
                    expected = deform(avatar, vertices, triangles, controls)
                    for got, want in zip(actual, expected):
                        np.testing.assert_allclose(got, want, atol=3e-7, rtol=3e-5)
                    first = renderer.render(vertices, view, intrinsic, (96, 80), controls)
                    frames.append(first)
                    saved = first.rgba.numpy()
                    self.assertGreater(saved[..., 3].max(), .9)
                    moved = vertices.copy(); moved[:, 0] += .1
                    frames.append(renderer.render(moved, view, intrinsic, (96, 80), controls))
                    np.testing.assert_array_equal(first.rgba.numpy(), saved)
                    self.assertGreater(abs(frames[1].rgba.numpy()-saved).mean(), .01)
                    with self.assertRaisesRegex(RuntimeError, 'release frames'):
                        renderer.close()
                    self.assertEqual(first.pixels().shape, (80, 96, 3))
                    self.assertEqual(first.pixels(straight_alpha=True).shape, (80, 96, 4))
                    with self.assertRaisesRegex(ValueError, 'camera'):
                        renderer.render(vertices, np.zeros((2, 2)), intrinsic)
                    # Fully clipped scene and degenerate triangles produce transparent pixels.
                    vertices[:, 2] = -.1
                    frames.append(renderer.render(vertices, view, intrinsic, (96, 80)))
                    np.testing.assert_array_equal(frames[-1].rgba.numpy(), 0)
                    vertices[:] = [0, 0, 1]
                    frames.append(renderer.render(vertices, view, intrinsic, (96, 80)))
                    np.testing.assert_array_equal(frames[-1].rgba.numpy(), 0)
                finally:
                    for frame in frames:
                        frame.ready.close(); frame.rgba.close()
                    renderer.close()

    def test_failed_native_open_propagates_and_releases_runtime_borrow(self):
        avatar, _, triangles, _, _ = fixture(count=4)
        runtime = CudaRuntime()
        try:
            with patch('server.vhuman.realtime.src.renderer.native.library') as loader:
                loader.return_value.vhs_open.return_value = 0
                loader.return_value.vhs_error.return_value = b'fixture upload failure'
                with self.assertRaisesRegex(RuntimeError, 'fixture upload failure'):
                    NativeGaussianRenderer(avatar, triangles, runtime=runtime)
            self.assertEqual(runtime.borrowers, 0)
            runtime.synchronize()
        finally:
            runtime.close()

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('gsplat'), 'optional gsplat oracle')
    def test_pixels_match_gsplat(self):
        from .src.renderer.gsplat_reference import GaussianRenderer as ReferenceRenderer
        for policy in ('trace-v1', 'eigen-v1'):
            avatar, vertices, triangles, view, intrinsic = fixture(policy)
            native, reference = NativeGaussianRenderer(avatar, triangles), ReferenceRenderer(avatar, triangles)
            actual = expected = None
            try:
                controls = np.array([.8, -.4], np.float32)
                actual = native.render(vertices, view, intrinsic, (96, 80), controls)
                expected = reference.render(vertices, view, intrinsic, (96, 80), controls)
                expected.ready.synchronize()
                a, b = actual.rgba.numpy(), expected.rgba.cpu().numpy()
                error = abs(a-b)
                print(f'{policy}: native/gsplat RGBA max={error.max():.8g} mean={error.mean():.8g}')
                self.assertLess(error.max(), 2e-4)
                self.assertLess(error.mean(), 2e-6)
            finally:
                if actual:
                    actual.ready.close(); actual.rgba.close()
                if expected:
                    expected.ready.close(); expected = None
                native.close(); reference.close()


if __name__ == '__main__':
    unittest.main()
