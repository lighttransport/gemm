"""Opt-in CPU tests using registered local assets, never downloading data.

Set MUSCLEMIMIC_TEST_ASSETS, MHR_TEST_ASSETS and MHR_TEST_AVATAR. The generated
standing trajectory checks mechanics and rejection handling, not motion quality.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..service import ROOT


@unittest.skipUnless(all(os.environ.get(k) for k in
    ("MUSCLEMIMIC_TEST_ASSETS", "MHR_TEST_ASSETS", "MHR_TEST_AVATAR")), "local physical assets not selected")
class PhysicalIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .musclemimic_assets import read_installation, sha256
        from .musclemimic_worker import configure_process, names
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name).resolve()
        configure_process(cls.work, "cpu")
        cls.installation = read_installation(os.environ["MUSCLEMIMIC_TEST_ASSETS"], verify=True)
        import sys
        sys.path[:0] = [cls.installation["upstream"], cls.installation["gmr"]]
        import mujoco
        from loco_mujoco.task_factories import TaskFactory
        config = json.loads((Path(cls.installation["checkpoint"]) / "config/metadata").read_text())
        params = config["experiment"]["env_params"]
        params.update(env_name="MyoFullBody", num_envs=1, headless=True)
        params["goal_params"]["visualize_goal"] = False
        cls.env = TaskFactory.get_factory_cls("ImitationFactory").make(**params)
        model = cls.env.model
        cls.fixture = cls.work / "fixture.npz"
        cls.fields = dict(qpos=np.tile(model.qpos0, (31, 1)), frequency=30.,
            joint_names=np.array(names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)),
            jnt_type=model.jnt_type, split_points=[0, 31])
        np.savez(cls.fixture, **cls.fields)
        cls.installation["references"] = {"fixture": {"path": str(cls.fixture), "sha256": sha256(cls.fixture),
            "source_url": "local:generated-qpos0", "terms_url": "local:generated-test"}}
        body = Path(os.environ["MHR_TEST_AVATAR"]).resolve()
        cls.request = {"reference_motion": "fixture", "avatar": str(body / "avatar.glb"),
            "avatar_json": str(body / "avatar.json"), "body_meta": str(body / "body_mhr.glb.json"),
            "mhr_assets": str(Path(os.environ["MHR_TEST_ASSETS"]).resolve())}

    def test_reference_mapping_and_invalid_quaternion(self):
        from .musclemimic_worker import reference_qpos
        q, hz = reference_qpos(self.fixture, self.env.model)
        self.assertEqual((q.shape, hz), ((31, 89), 30.))
        bad = dict(self.fields)
        bad["qpos"] = bad["qpos"].copy()
        bad["qpos"][:, 3] = 2
        path = self.work / "invalid.npz"
        np.savez(path, **bad)
        with self.assertRaisesRegex(ValueError, "not normalized"):
            reference_qpos(path, self.env.model)

    def test_native_mhr_to_gmr(self):
        from .musclemimic_worker import calibrated_gmr
        from .musclemimic_kinematics import MHRKinematics
        fk = MHRKinematics(json.loads(Path(self.request["avatar_json"]).read_text()),
            Path(self.request["mhr_assets"]), json.loads(Path(self.request["body_meta"]).read_text())["model_params"])
        p, r, _ = fk.forward(fk.bind_params)
        source = {"positions": np.tile(p[fk.sites], (3, 1, 1)),
            "matrices": np.tile(r[fk.sites], (3, 1, 1, 1)),
            "rest_positions": p[fk.sites], "rest_matrices": r[fk.sites]}
        q, hz, report = calibrated_gmr(self.env, source, self.work)
        self.assertEqual(q.shape, (3, 89))
        self.assertLess(report["gmr_max_site_error_m"], .15)

    def test_real_checkpoint_episode_and_portable_exports(self):
        from .musclemimic_worker import run, export_tracks
        report = run(self.request, self.installation, self.work)
        self.assertEqual(report["model"], {"nq": 89, "nv": 88, "actuators": 354})
        self.assertGreater(report["frames"], 3)
        with np.load(self.work / "rollout.npz", allow_pickle=False) as data:
            for key in ("qpos", "qvel", "commands", "activations", "positions"):
                self.assertTrue(np.isfinite(data[key]).all(), key)
                self.assertEqual(len(data[key]), report["frames"])
            self.assertGreater(np.max(data["activations"]), 0.)
            np.testing.assert_allclose(np.diff(data["times"]), .01)
        output = export_tracks(self.request, self.work, report)
        print("physical smoke:", json.dumps({"termination": report["termination"], "frames": report["frames"],
              "root_rms_m": report["root_rms_m"], "avatar": output}, sort_keys=True), flush=True)
        self.assertTrue((self.work / "baseline.glb").is_file())
        track = json.loads((self.work / "baseline.json").read_text())
        self.assertEqual(track["version"], 2)
        self.assertEqual(len(track["joints"]), 127)
        self.assertIn("skin_ground_penetration_max_m", output["motion"])
        if not report["completed"]:
            self.assertFalse(output["accepted"])
            self.assertFalse((self.work / "motion.json").exists())


if __name__ == "__main__":
    unittest.main()
