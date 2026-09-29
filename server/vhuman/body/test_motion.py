"""Checks for upload validation and interchange motion on the MHR rig."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye.glb import GLB, GLBBuilder
from ..service import ROOT, EyeService, ServiceError
from .motion import _continuous, _export_glb, upload


class BodyMotionTest(unittest.TestCase):
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
