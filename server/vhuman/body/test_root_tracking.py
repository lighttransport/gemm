"""The root objective preserves clock and measured translation under SQP steps."""
import unittest
import numpy as np

from .terra_root import SourceRootTerm


class RootTrackingTest(unittest.TestCase):
    def test_root_rows_match_finite_differences_with_selected_coordinates(self):
        targets = np.array([[.2, -.1, 1.], [.3, -.2, 1.05]])
        term = SourceRootTerm(targets, 1000.)
        q = np.array([.1, .2, .9, 1., 0., 0., 0., .4])
        indices = np.array([0, 2, 4, 7])
        rows, residual = term.linearize(q, indices, 1)
        np.testing.assert_allclose(residual, targets[1]-q[:3])
        for column, index in enumerate(indices):
            moved = q.copy(); moved[index] += 1e-6
            _, changed = term.linearize(moved, indices, 1)
            np.testing.assert_allclose((residual-changed)/1e-6, rows[:, column], atol=1e-9)

    def test_warmup_frames_do_not_shift_the_source_targets(self):
        target = np.array([[0., 0., 1.], [.01, 0., 1.], [.03, 0., 1.]])
        padded = np.r_[np.repeat(target[:1], 30, axis=0), target]
        term = SourceRootTerm(padded, 1000.)
        padded[:] = 99  # the term owns its prepared source, not mutable solver scratch
        for frame in (0, 29, 30):
            _, residual = term.linearize(np.r_[target[0], 1., 0., 0., 0.], np.arange(7), frame)
            np.testing.assert_array_equal(residual, np.zeros(3))
        _, residual = term.linearize(np.r_[target[0], 1., 0., 0., 0.], np.arange(7), 32)
        np.testing.assert_allclose(residual, [.03, 0., 0.])
        for frame in (-1, 33):
            with self.assertRaises(ValueError):term.linearize(np.zeros(7), np.arange(7), frame)

    def test_invalid_objective_rejected(self):
        for target, weight in (([], 1), ([[0, 0]], 1), ([[0, 0, np.nan]], 1),
                               ([[0, 0, 0]], 0), ([[0, 0, 0]], np.inf)):
            with self.subTest(target=target, weight=weight), self.assertRaises(ValueError):
                SourceRootTerm(target, weight)


if __name__ == '__main__':
    unittest.main()
