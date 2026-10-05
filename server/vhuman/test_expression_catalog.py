"""Camera recovery invariants for the GNM expression catalog."""
import unittest
import numpy as np
from .rig.expression_catalog import similarity_camera, fit_pca_residual


class CameraTests(unittest.TestCase):
    def test_known_scale_roll_and_translation(self):
        rng = np.random.default_rng(12)
        base = rng.normal(size=(3, 8, 2))
        expected_scale = np.array([.8, 2.1, 1.3])
        expected_angle = np.array([-.7, .4, .1])
        center = np.array([[.2, .4], [-1, .5], [.4, -.3]])
        c, s = np.cos(expected_angle), np.sin(expected_angle)
        rotation = np.stack([c, s, -s, c], -1).reshape(3, 2, 2)
        target = (base @ rotation) * expected_scale[:, None, None] + center[:, None]
        scale, angle, fitted_center = similarity_camera(base, target)
        np.testing.assert_allclose(scale, expected_scale, atol=1e-12)
        np.testing.assert_allclose(angle, expected_angle, atol=1e-12)
        np.testing.assert_allclose(fitted_center, center, atol=1e-12)

    def test_degenerate_and_nonfinite_observations_rejected(self):
        with self.assertRaises(ValueError):
            similarity_camera(np.zeros((1, 8, 2)), np.zeros((1, 8, 2)))
        with self.assertRaises(ValueError):
            similarity_camera(np.full((1, 8, 2), np.nan), np.ones((1, 8, 2)))

    def test_mirrored_observation_does_not_invert_mesh(self):
        base = np.random.default_rng(2).normal(size=(1, 8, 2))
        target = base * [-1, 1]
        scale, angle, center = similarity_camera(base, target)
        c, s = np.cos(angle[0]), np.sin(angle[0])
        projected = base @ np.array([[c, s], [-s, c]]) * scale[:, None, None] + center[:, None]
        self.assertGreater(np.linalg.norm(projected - target), .1)
        self.assertGreater(scale[0], 0)


class ExpressionPcaTests(unittest.TestCase):
    def test_recover_observable_motion_with_unobservable_coefficients(self):
        matrix = np.zeros((2, 4, 7))
        matrix[:, :, :4] = np.eye(4)[None] * .01
        known = np.array([[.2, -.4, .6, .8], [-.1, .7, .3, -.5]])
        residual = np.einsum('fde,fe->fd', matrix[:, :, :4], known)
        recovered = fit_pca_residual(matrix, residual, ridge=1e-10)
        np.testing.assert_allclose(recovered[:, :4], known, atol=1e-6)
        np.testing.assert_allclose(recovered[:, 4:], 0)

    def test_unrepresentable_motion_is_not_invented(self):
        matrix = np.zeros((1, 4, 8)); matrix[0, 0, 0] = .01
        residual = np.array([[0, .1, 0, 0]])
        coefficients = fit_pca_residual(matrix, residual)
        np.testing.assert_array_equal(coefficients, 0)

    def test_bound_large_motion_and_reject_nonfinite_input(self):
        matrix = np.ones((1, 1, 1)) * .01
        self.assertEqual(fit_pca_residual(matrix, np.ones((1, 1)))[0, 0], 3)
        with self.assertRaises(ValueError):
            fit_pca_residual(matrix, np.full((1, 1), np.nan))

    def test_saturated_mode_redirects_motion_to_available_mode(self):
        matrix = np.ones((1, 1, 2)) * .01
        prior = np.array([[3., 0.]])
        coefficients = fit_pca_residual(matrix, np.array([[.02]]), prior=prior)
        # The production ridge slightly shrinks the recovered motion.
        expected = .01 * .02 / (.01 ** 2 + 1e-7)
        np.testing.assert_allclose(coefficients, [[0, expected]], atol=1e-4)
        self.assertLessEqual(np.max(prior + coefficients), 3)
