import unittest
import numpy as np
from .reconstruction.blink_prior import CONTOURS, propose_blink


class BlinkPriorTests(unittest.TestCase):
    def setUp(self):
        self.centers = {'right': np.array([-.032, 0., 0.]),
                        'left': np.array([.032, 0., 0.])}
        self.points = np.zeros((468, 3))
        vertices = []
        for side, contour in CONTOURS.items():
            center = self.centers[side]
            for key, height in (('lower', -.003), ('upper', .004)):
                x = np.linspace(-.012, .012, len(contour[key]))
                y = height * np.sin(np.linspace(0, np.pi, len(x)))
                z = np.sqrt(.015**2-x*x-y*y)
                self.points[contour[key]] = center + np.c_[x, y, z]
            self.points[contour['brow']] = center + np.c_[
                np.linspace(-.012, .012, 5), np.full(5, .012), np.full(5, .016)]
            vertices.extend(self.points[[contour['upper'][4], contour['lower'][4]]])
        self.vertices = np.r_[np.array(vertices), [[0., -.07, .04], [.032, .02, .016]]]
        self.movable = np.ones(len(self.vertices), bool)

    def run_prior(self, strength, **kwargs):
        return propose_blink(self.vertices, self.movable, self.points,
                             self.centers, strength, **kwargs)

    def test_neutral_controls_and_inputs_remain_exact(self):
        before = self.vertices.copy()
        actual, _ = self.run_prior(0)
        np.testing.assert_array_equal(actual, before)
        actual, _ = self.run_prior(1, remaining_aperture=1)
        np.testing.assert_array_equal(actual, before)
        self.run_prior(1)
        np.testing.assert_array_equal(self.vertices, before)

    def test_closure_reduces_aperture_without_moving_remote_skin(self):
        closed, report = self.run_prior(1)
        half, _ = self.run_prior(.5)
        for upper, lower in ((0, 1), (2, 3)):
            start = np.linalg.norm(self.vertices[upper]-self.vertices[lower])
            self.assertLess(np.linalg.norm(closed[upper]-closed[lower]), start*.1)
            self.assertGreater(np.linalg.norm(half[upper]-half[lower]), start*.4)
        np.testing.assert_array_equal(closed[4:], self.vertices[4:])
        self.assertFalse(report['observed_motion'])
        self.assertTrue(report['requires_geometry_and_contact_validation'])

    def test_mask_is_a_hard_constraint(self):
        self.movable[:2] = False
        actual, _ = self.run_prior(1)
        np.testing.assert_array_equal(actual[:2], self.vertices[:2])
        self.assertGreater(np.linalg.norm(actual[2:4]-self.vertices[2:4]), 0)

    def test_coordinate_frame_and_translation_equivariance(self):
        c, s = np.cos(.4), np.sin(.4)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        translation = np.array([.12, -.08, .27])
        expected, _ = self.run_prior(.8)
        actual, _ = propose_blink(self.vertices@rotation.T+translation,
            self.movable, self.points@rotation.T+translation,
            {k: v@rotation.T+translation for k, v in self.centers.items()},
            .8, head_rotation=rotation)
        np.testing.assert_allclose(actual, expected@rotation.T+translation, atol=1e-12)

    def test_invalid_geometry_and_controls_are_rejected(self):
        for strength in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                self.run_prior(strength)
        with self.assertRaises(ValueError):
            self.run_prior(.5, head_rotation=np.diag([2., 1., 1.]))
        with self.assertRaises(ValueError):
            self.run_prior(.5, remaining_aperture=0)
        self.vertices[0, 1] = np.nan
        with self.assertRaises(ValueError):
            self.run_prior(.5)


if __name__ == '__main__':
    unittest.main()
