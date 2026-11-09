import unittest

import numpy as np

from . import model
from .assemble import _face_alignment, _mhr_rig, _state_matrix


class BodyModelTest(unittest.TestCase):
    def test_default_is_mhr(self):
        bm = model.get()
        self.assertIs(bm, model.MHR)
        self.assertEqual((bm.pose_dim, bm.identity_dim, bm.unit_to_m), (204, 45, .01))
        self.assertEqual(bm.head_joint, "mhr_c_head")

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown body model"):
            model.get("nope")

    def test_check_params(self):
        model.MHR.check_params(np.zeros(204), np.zeros(45))
        with self.assertRaisesRegex(ValueError, "MHR pose or identity"):
            model.MHR.check_params(np.zeros(136), np.zeros(45))

    def test_unit_scaling_is_explicit(self):
        s = np.array([100., 200., 300., 0, 0, 0, 1, 1])
        np.testing.assert_allclose(_state_matrix(s)[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(_state_matrix(s, 1.0)[:3, 3], [100, 200, 300])

    def test_rig_unit_threading(self):
        rig = {"names": ["a", "b"], "parents": np.array([-1, 0])}
        state = np.array([[0, 0, 0, 0, 0, 0, 1, 1.], [0, 100, 0, 0, 0, 0, 1, 1.]])
        _, _, bind = _mhr_rig(rig, state)
        self.assertAlmostEqual(bind[1, 1, 3], 1.0)
        _, _, bind = _mhr_rig(rig, state, 1.0)
        self.assertAlmostEqual(bind[1, 1, 3], 100.0)


if __name__ == "__main__":
    unittest.main()
