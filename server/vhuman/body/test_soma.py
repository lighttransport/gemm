import os
import unittest

import numpy as np

from . import model, soma_backend, soma_runner


class SomaUnitTest(unittest.TestCase):
    def test_registered(self):
        bm = model.get("soma")
        self.assertEqual((bm.unit_to_m, bm.head_joint, bm.neck_joint), (1.0, "soma_Head", "soma_Neck1"))

    def test_landmark_mapping_one_hot(self):
        m = model.get("soma").landmark_mapping()
        self.assertEqual(m.shape, (5, 18056 + 78))
        np.testing.assert_array_equal(m.sum(1), np.ones(5))
        self.assertEqual(len(set(m.argmax(1))), 5)

    def test_face_coefficients_rejected(self):
        with self.assertRaisesRegex(ValueError, "no facial coefficients"):
            soma_backend.decode(None, np.zeros((1, 204)), np.zeros(45), face=np.zeros((1, 72)))

    def test_bad_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, "SOMA pose"):
            soma_backend.decode(None, np.zeros((1, 136)), np.zeros(45))
        with self.assertRaisesRegex(ValueError, "SOMA identity"):
            soma_backend.decode(None, np.zeros((1, 204)), np.zeros(44))

    def test_mhr_assets_dir_is_not_soma_assets(self):
        old = os.environ.pop("VHUMAN_SOMA_ASSETS", None)
        try:
            got = soma_backend.resolve_assets("/models/sam3d-body/safetensors")
            self.assertNotEqual(str(got), "/models/sam3d-body/safetensors")
            os.environ["VHUMAN_SOMA_ASSETS"] = "/x/soma"
            self.assertEqual(str(soma_backend.resolve_assets("/m")), "/x/soma")
        finally:
            os.environ.pop("VHUMAN_SOMA_ASSETS", None)
            if old is not None:
                os.environ["VHUMAN_SOMA_ASSETS"] = old

    def test_from_avatar(self):
        self.assertIs(model.from_avatar({"body": {"model": "SOMA-X"}}), model.SOMA)
        self.assertIs(model.from_avatar({}), model.MHR)
        with self.assertRaisesRegex(ValueError, "unknown body model"):
            model.from_avatar({"body": {"model": "X"}})

    def test_missing_interpreter_message(self):
        old = os.environ.pop("VHUMAN_SOMA_PYTHON", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "VHUMAN_SOMA_PYTHON"):
                soma_backend.python()
        finally:
            if old is not None:
                os.environ["VHUMAN_SOMA_PYTHON"] = old

    def test_quaternion_and_state(self):
        th = .7
        r = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        t = np.eye(4)
        t[:3, :3] = r * 1.0
        t[:3, 3] = [1, 2, 3]
        s = soma_runner._state(t[None, None])[0, 0]
        np.testing.assert_allclose(s[:3], [1, 2, 3])
        np.testing.assert_allclose(s[3:7], [0, np.sin(th / 2), 0, np.cos(th / 2)], atol=1e-9)
        self.assertAlmostEqual(s[7], 1.0)


@unittest.skipUnless(os.environ.get("VHUMAN_SOMA_PYTHON"), "set VHUMAN_SOMA_PYTHON for the real SOMA-X runner")
class SomaRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        p = np.zeros((2, 204), np.float32)
        p[1, 6:130] = .1 * rng.standard_normal(124)
        cls.out = model.get("soma").decode(None, p, np.zeros(45, np.float32))

    def test_shapes_and_hierarchy(self):
        v, state, rig, report = self.out
        self.assertEqual((v.shape, state.shape), ((2, 18056, 3), (2, 78, 8)))
        self.assertTrue(np.all(rig["parents"][1:] < np.arange(1, 78)))
        self.assertEqual(rig["parents"][0], -1)
        self.assertEqual(rig["faces"].shape, (36108, 3))

    def test_units_and_height(self):
        v = self.out[0][0]
        self.assertTrue(1.4 < v[:, 1].max() - v[:, 1].min() < 2.1)

    def test_pose_inversion_residual(self):
        err = self.out[3]["mhr_to_soma_error_m"]
        self.assertLess(err["mean"], .01)

    def test_head_above_neck(self):
        _, state, rig, _ = self.out
        names = rig["names"]
        self.assertGreater(state[0, names.index("soma_Head"), 1], state[0, names.index("soma_Neck1"), 1])


if __name__ == "__main__":
    unittest.main()
