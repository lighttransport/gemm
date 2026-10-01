"""Checks for upload validation and interchange motion on the MHR rig."""
from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from ..eye.glb import GLB, GLBBuilder
from ..service import ROOT, EyeService, ServiceError
from .motion import _continuous, _export_glb, upload
from . import motion


class BodyMotionTest(unittest.TestCase):
    def test_cpu_motion_decode_does_not_require_gpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            avatar = root / "avatar.glb"
            avatar.touch()
            (root / "avatar.json").write_text("{}")
            (root / "body_mhr.glb.json").write_text(json.dumps({"shape": [0.] * 45}))
            model = root / "model/dinov3/assets/mhr_model.pt"
            model.parent.mkdir(parents=True)
            model.touch()
            python = root / "python"
            python.touch()
            service = SimpleNamespace(body_file=lambda hid, name: root / name)

            def decode(command, cancel, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"duration": 0.}))

            with motion.gpu.execution("cpu"), \
                 mock.patch.object(motion, "source_file", return_value=(avatar, {"kind": "image", "sha256": "test"})), \
                 mock.patch.object(motion, "_frames", return_value=[avatar]), \
                 mock.patch.object(motion, "_bbox", return_value=(0, 0, 1, 1)), \
                 mock.patch.object(motion, "_sam_frame", return_value=root / "pose.json"), \
                 mock.patch.object(motion.body_job, "_run", side_effect=decode) as runner, \
                 mock.patch.object(motion.gpu, "device_session", side_effect=AssertionError("CPU needs no GPU")):
                result = motion.fit(service, {"head_id": "head", "upload_id": "upload"},
                                    lambda *_: None, threading.Event(), model_dir=root / "model", rig_python=python)
            runner.assert_called_once()
            self.assertEqual(result["frames"], 1)
            self.assertEqual(result["duration"], 0.)

    def test_uploaded_image_is_verified_and_bounded(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            service = EyeService(Path(folder))
            image = io.BytesIO()
            Image.new("RGB", (16, 24), "red").save(image, format="PNG")
            item = upload(service, io.BytesIO(image.getvalue()), len(image.getvalue()), "image/png")
            self.assertEqual(item["kind"], "image")
            with self.assertRaises(ServiceError):
                upload(service, io.BytesIO(b"bad"), 3, "video/mp4")
            with self.assertRaises(ServiceError):
                upload(service, io.BytesIO(b""), 0, "image/png")

    def test_glb_contains_animated_local_rotations(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            source = Path(folder) / "avatar.glb"
            out = Path(folder) / "motion.glb"
            b = GLBBuilder()
            b.node("mhr_root", rotation=[0, 0, 0, 1])
            b.node("mhr_arm", rotation=[0, 0, 0, 1])
            b.write(source)
            q = np.array([[[0., 0., 0., 1.], [0., 0., 0., 1.]],
                          [[0., 0., 0., 1.], [0., 0., .5, .8660254]]], np.float64)
            _export_glb(source, out, ["mhr_root", "mhr_arm"], q, np.array([0., .25], np.float32))
            glb = GLB.load(out)
            anim = glb.doc["animations"][0]
            self.assertEqual(len(anim["channels"]), 2)
            arm = anim["samplers"][1]
            np.testing.assert_allclose(glb.accessor(arm["input"]), [0., .25])
            np.testing.assert_allclose(glb.accessor(arm["output"])[-1], q[-1, 1], atol=1e-6)

    def test_quaternion_track_keeps_shortest_arc(self):
        q = np.array([[[0., 0., 0., 1.]], [[0., 0., 0., -1.]]])
        out = _continuous(q)
        self.assertGreater(float(np.sum(out[0, 0] * out[1, 0])), .99)


if __name__ == "__main__":
    unittest.main()
