"""Portable unit tests; real CPU simulator tests require the isolated environment."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..eye.glb import GLB, GLBBuilder
from ..service import ROOT, EyeService
from .motion import _export_glb
from .motion_format import document, validate_tracks
from .musclemimic import validate_request
from .musclemimic_assets import reference, sha256
from .musclemimic_kinematics import (change_basis, Y_TO_Z, proportions, contact_metrics,
                                    MHRKinematics, SITE_PARENTS)
from .musclemimic_worker import record_episode
from .musclemimic_video import estimate_root


class BodyPhysicalMotionTest(unittest.TestCase):
    def test_basis_roundtrip_and_quaternion_order(self):
        p = np.array([[1., 2., 3.], [-2., 1., 4.]])
        r = Rotation.from_euler("xyz", [[.2, -.6, .9], [-.3, .4, .5]]).as_matrix()
        out_p, out_r = change_basis(p, r)
        np.testing.assert_allclose(out_p[0], [1, -3, 2], atol=1e-12)
        actual_p, actual_r = change_basis(out_p, out_r, Y_TO_Z.T)
        np.testing.assert_allclose(actual_p, p, atol=1e-12)
        np.testing.assert_allclose(actual_r, r, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(out_r), 1, atol=1e-12)

    def test_v2_root_export_and_shortest_arc(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as d:
            source, out = Path(d)/"avatar.glb", Path(d)/"motion.glb"
            b = GLBBuilder()
            b.node("mhr_root", translation=[0, 1, 0], scale=[1.1]*3)
            b.write(source)
            q = np.array([[[0., 0, 0, 1]], [[0., 0, 0, -1]]])
            root = np.array([[0., 1, 0], [2., 1.2, 0]])
            track = document(["mhr_root"], q, [0., 1.], root)
            self.assertEqual(track["version"], 2)
            self.assertEqual(track["frames"][1]["rotations"][0][-1], 1)
            _export_glb(source, out, ["mhr_root"], q, np.array([0., 1.]), root_translation=root)
            glb = GLB.load(out)
            animation = glb.doc["animations"][0]
            channel = next(c for c in animation["channels"] if c["target"]["path"] == "translation")
            sampler = animation["samplers"][channel["sampler"]]
            np.testing.assert_allclose(glb.accessor(sampler["output"]), root)
            self.assertEqual(glb.doc["nodes"][0]["scale"], [1.1]*3)

    def test_invalid_track_fails_before_export(self):
        q = np.tile([0., 0, 0, 1], (2, 1, 1))
        for times in ([0., 0.], [1., 2.], [0., np.nan]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate_tracks(["mhr_root"], q, times)
        with self.assertRaises(ValueError):
            validate_tracks(["mhr_root"], q*2, [0., 1.])
        with self.assertRaises(ValueError):
            document(["mhr_root"], q, [0., 1.], [[0, 0, 0], [0, np.inf, 0]])

    def test_param_coupling_and_nonunit_parent_scale(self):
        fk = MHRKinematics.__new__(MHRKinematics)
        fk.transform = np.zeros((3, 7, 1))
        fk.transform[1, 5, 0] = 1
        fk.transform[2, 5, 0] = .5
        fk.prerot = Rotation.identity(3)
        fk.parents = np.array([-1, 0, 1])
        fk.root = 0
        fk.translations = np.array([[0., 0, 0], [1., 0, 0], [1., 0, 0]])
        fk.scales = np.array([2., .5, 1.])
        xyz, matrices, q = fk.forward([np.pi/2], root=[0, 1, 0])
        np.testing.assert_allclose(xyz, [[0, 1, 0], [2, 1, 0], [2, 2, 0]], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q[2]).as_rotvec(), [0, 0, np.pi/4], atol=1e-12)
        np.testing.assert_allclose(matrices[2], Rotation.from_euler("z", 135, degrees=True).as_matrix(), atol=1e-12)

    def test_segment_scaling_preserves_lengths(self):
        rest = np.zeros((17, 3))
        for j, parent in enumerate(SITE_PARENTS):
            if parent >= 0:
                rest[j] = rest[parent] + [1, .1*j, .2]
        target = rest * 1.3
        result = proportions(rest[None], rest, target)[0]
        np.testing.assert_allclose(result, target)

    def test_stance_drift_resets_on_swing(self):
        feet = np.zeros((5, 1, 3))
        feet[:, 0, 0] = [0, .01, .5, 1., 1.01]
        report = contact_metrics(feet, [[1], [1], [0], [1], [1]])
        self.assertAlmostEqual(report["stance_drift_p95_m"], .01)
        self.assertIsNone(contact_metrics(feet, np.zeros((5, 1)))["stance_drift_p95_m"])
        feet[2, 0, 1] = -.03
        self.assertAlmostEqual(contact_metrics(feet, np.ones((5, 1)))["ground_penetration_max_m"], .03)

    def test_episode_never_resets_after_termination(self):
        class Env:
            resets = 0
            step_no = 0
            def reset(self):
                self.resets += 1
                return 0
            def step(self, action):
                self.step_no += 1
                return 0, 0, False, self.step_no == 2, {"subtraj_step_no": self.step_no, "traj_len": 8}
        e = Env()
        frames, reason, _ = record_episode(e, lambda _: [0], 10, snapshot=lambda: e.step_no)
        self.assertEqual(frames, [0, 1, 2])
        self.assertEqual(reason, "tracking_failure")
        self.assertEqual(e.resets, 1)

    def test_episode_reference_end_and_absorption(self):
        for absorbing, expected in ((False, "reference_end"), (True, "tracking_failure")):
            env = SimpleNamespace(reset=lambda: 0, step=lambda _: (0, 0, absorbing, True,
                                  {"subtraj_step_no": 4, "traj_len": 5}))
            _, reason, _ = record_episode(env, lambda _: [0], 5, snapshot=lambda: 0)
            self.assertEqual(reason, expected)

    def test_input_constraints(self):
        validate_request({"reference_motion": "walk"})
        validate_request({"upload_id": "a", "fixed_camera": True})
        for req in ({}, {"upload_id": "a"}, {"upload_id": "a", "reference_motion": "walk"}):
            with self.assertRaises(ValueError):
                validate_request(req)

    def test_reference_hash_and_id(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as d:
            path = Path(d)/"reference.npz"; path.write_bytes(b"fixture")
            data = {"references": {"walk": {"path": str(path), "sha256": sha256(path)}}}
            reference(data, "walk")
            with self.assertRaises(ValueError):
                reference(data, "../walk")
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                reference(data, "walk")

    def test_rejected_motion_lists_only_available_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as d:
            service = EyeService(Path(d))
            body = Path(d)/"heads/abc/body"; body.mkdir(parents=True)
            (body/"avatar.json").write_text("{}")
            take = body/"motions/0123456789ab"; take.mkdir(parents=True)
            (take/"manifest.json").write_text(json.dumps({"id": take.name, "accepted": False}))
            (take/"baseline.json").write_text("{}")
            (take/"evaluation.json").write_text("{}")
            listed = service.list_motions("abc")[0]
            self.assertNotIn("motion", listed["urls"])
            self.assertIn("baseline", listed["urls"])

    def test_stationary_root_estimate(self):
        p = np.zeros((8, 17, 3))
        p[:, 2, 1] = .7
        p[:, [11, 12, 15, 16], 1] = -1
        out, rotation, contact, report = estimate_root(p, np.zeros((8, 3)), np.arange(8)/30)
        np.testing.assert_allclose(out[:, 0, 1], 1, atol=1e-5)
        self.assertTrue(contact.all())
        self.assertLess(report["camera_contact_rms_m"], 1e-5)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
    def test_video_keeps_variable_timestamps(self):
        from .musclemimic_video import sample_video
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            work = Path(directory)
            source = work / "variable.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x128:r=25:d=0.4",
                "-vf", "setpts='if(lt(N,5),N/(25*TB),(0.2+(N-5)/10)/TB)'",
                "-vsync", "vfr", "-c:v", "mpeg4", str(source)], check=True)
            frames, times = sample_video(source, work, threading.Event())
            self.assertEqual(len(frames), len(times))
            self.assertEqual(times[0], 0)
            self.assertGreater(np.ptp(np.diff(times)), .02)
            self.assertTrue(np.all(np.diff(times) > 0))

    def test_camera_gate_rejects_accumulated_pan_and_blank_background(self):
        import cv2
        from .musclemimic_video import check_camera
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            work = Path(directory)
            image = np.random.default_rng(3).integers(0, 256, (240, 320), dtype=np.uint8)
            image = cv2.GaussianBlur(image, (3, 3), 0)
            paths = [work / f"{i}.png" for i in range(16)]
            boxes = [(130, 50, 190, 220)] * len(paths)
            for path in paths:
                cv2.imwrite(str(path), image)
            self.assertLess(check_camera(paths, boxes)["background_motion_max_fraction"], 1e-4)
            for i, path in enumerate(paths):
                cv2.imwrite(str(path), cv2.warpAffine(image, np.float32([[1, 0, i*.4], [0, 1, 0]]), (320, 240)))
            with self.assertRaisesRegex(ValueError, "background moves"):
                check_camera(paths, boxes)
            for path in paths:
                cv2.imwrite(str(path), np.zeros_like(image))
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                check_camera(paths, boxes)

    def test_camera_gate_requires_late_tracking_and_the_final_frame(self):
        import cv2
        from .musclemimic_video import check_camera
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            rng = np.random.default_rng(42)
            initial = rng.integers(0, 256, (240, 320), dtype=np.uint8)
            for count, invalid_from in ((16, 3), (240, 239)):
                paths = [Path(directory) / f"{i}.png" for i in range(count)]
                boxes = [(130, 50, 190, 220)] * len(paths)
                with self.subTest(count=count, invalid_from=invalid_from):
                    for i, path in enumerate(paths):
                        image = initial.copy() if i < invalid_from else rng.integers(0, 256, initial.shape, dtype=np.uint8)
                        image[50:220, 130:190] = 120
                        cv2.imwrite(str(path), image)
                    with self.assertRaisesRegex(ValueError, "cannot be verified"):
                        check_camera(paths, boxes)
            for path in paths:
                cv2.imwrite(str(path), initial)
            report = check_camera(paths, boxes)
            self.assertEqual(report["camera_checks"], 12)
            self.assertEqual(report["camera_check_frames"][-1], len(paths)-1)

    def test_cuda_process_restricts_visibility_to_the_locked_gpu(self):
        from .musclemimic_worker import configure_process
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7,3", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}):
                configure_process(Path(directory), "cuda", 2)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
                self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda")
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7,3"}):
                configure_process(Path(directory), "cpu", 2)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7,3")
                self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
            with self.assertRaises(ValueError):
                configure_process(Path(directory), "cuda", -1)

    def test_rejected_job_publishes_diagnostics_and_cancel_publishes_nothing(self):
        from . import musclemimic, mhr
        from .. import gpu
        for cancelled, backend in ((False, "cpu"), (True, "cpu"), (False, "cuda")):
            with self.subTest(cancelled=cancelled, backend=backend), tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
                work = Path(directory)
                body = work / "heads/abc/body"; body.mkdir(parents=True)
                for name in ("avatar.glb", "avatar.json", "body_mhr.glb.json"):
                    (body / name).write_text("{}")
                model = work / "model/safetensors"; model.mkdir(parents=True)
                for name in mhr.ASSET_FILES:
                    (model / name).touch()
                event = threading.Event()
                def worker(command, cancel, **kwargs):
                    self.assertIn("--policy-device", command)
                    self.assertEqual(command[command.index("--policy-device") + 1], backend)
                    self.assertEqual(command[command.index("--policy-device-index") + 1], "2")
                    self.assertEqual(gpu.device_index(), 2)
                    stage = Path(command[command.index("--work") + 1])
                    if cancelled:
                        event.set()
                        raise gpu.Cancelled("cancelled")
                    (stage / "simulation.json").write_text(json.dumps({"completed": False, "termination": "tracking_failure"}))
                    (stage / "baseline.json").write_text(json.dumps({"frames": [{}, {}], "duration": .1}))
                    raise RuntimeError("worker rejected the motion")
                def device_session(*args, **kwargs):
                    self.assertEqual(gpu.backend(), "cuda")
                    self.assertEqual(gpu.device_index(), 2)
                    return nullcontext()
                with gpu.execution(backend, 2), \
                     patch.object(musclemimic.gpu, "device_session", side_effect=device_session) as lock, \
                     patch.object(musclemimic, "read_installation", return_value={}), \
                     patch.object(musclemimic, "reference", return_value={}), \
                     patch.object(musclemimic.job, "_run", side_effect=worker):
                    args = (EyeService(work), {"head_id": "abc", "reference_motion": "test"}, lambda *a: None, event)
                    opts = dict(model_dir=model.parent, python=sys.executable, assets=work, device=backend)
                    if cancelled:
                        with self.assertRaises(gpu.Cancelled):
                            musclemimic.fit(*args, **opts)
                        self.assertEqual(list((body / "motions").iterdir()), [])
                    else:
                        result = musclemimic.fit(*args, **opts)
                        self.assertFalse(result["accepted"])
                        self.assertIsNone(result["motion_url"])
                        take = body / "motions" / result["id"]
                        self.assertTrue((take / "evaluation.json").exists())
                        self.assertFalse((take / "motion.json").exists())
                    self.assertEqual(lock.call_count, int(backend == "cuda"))


@unittest.skipUnless(os.environ.get("MHR_TEST_ASSETS") and os.environ.get("MHR_TEST_AVATAR"), "local MHR assets/avatar not selected")
class NativeMHRParityTest(unittest.TestCase):
    def test_exported_skin_preserves_bind_and_root_translation(self):
        from .musclemimic_kinematics import AvatarFeet
        body = Path(os.environ["MHR_TEST_AVATAR"])
        meta = json.loads((body / "body_mhr.glb.json").read_text())
        fk = MHRKinematics(json.loads((body / "avatar.json").read_text()),
                           Path(os.environ["MHR_TEST_ASSETS"]), meta["model_params"])
        skin = AvatarFeet(body / "avatar.glb", fk)
        p, r, _ = fk.forward(fk.bind_params)
        before = skin.vertices(p, r)
        after = skin.vertices(p + [1., 0, 0], r)
        np.testing.assert_allclose(after-before, np.tile([1., 0, 0], (len(before), 1)), atol=1e-6)

    def test_fk_matches_native_joint_states(self):
        from . import mhr
        body = Path(os.environ["MHR_TEST_AVATAR"])
        assets = Path(os.environ["MHR_TEST_ASSETS"])
        meta = json.loads((body/"body_mhr.glb.json").read_text())
        fk = MHRKinematics(json.loads((body/"avatar.json").read_text()), assets, meta["model_params"])
        params = np.tile(fk.bind_params, (3, 1))
        params[1, fk.active] += .02
        params[2, fk.active] -= .03
        _, states, _, _ = mhr.decode(assets, params.astype(np.float32), np.array(meta["shape"], np.float32),
                                     skeleton_only=True, backend="cpu")
        for i in range(len(params)):
            p, r, _ = fk.forward(params[i])
            np.testing.assert_allclose(p, states[i, :, :3]*.01, atol=2e-5)
            np.testing.assert_allclose(r, Rotation.from_quat(states[i, :, 3:7]).as_matrix(), atol=2e-5)


if __name__ == "__main__":
    unittest.main()
