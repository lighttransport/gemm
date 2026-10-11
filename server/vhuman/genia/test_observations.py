import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
from PIL import Image

from ..service import ROOT, ServiceError
from . import FORMAT
from . import job, observations, upload


class ObservationsTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp/genia-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = SimpleNamespace(work=self.root / "work")
        Image.new("RGB", (16, 12), (80, 100, 120)).save(self.root / "image.png")
        Image.new("L", (16, 12), 255).save(self.root / "mask.png")
        np.save(self.root / "depth.npy", np.full((12, 16), 2, np.float32))
        self.document = {"format": FORMAT, "units": "m", "views": [{"image": "image.png",
            "mask": "mask.png", "depth": "depth.npy", "K": [[10,0,8],[0,10,6],[0,0,1]],
            "c2w": np.eye(4).tolist()}]}
        self.write()

    def write(self):
        (self.root / "scene.json").write_text(json.dumps(self.document))

    def bundle(self, extra=None):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("scene.json", "image.png", "mask.png", "depth.npy"):
                archive.write(self.root / name, name)
            if extra:
                archive.writestr(*extra)
        return data.getvalue()

    def test_integer_pixel_unprojection_and_unit_conversion(self):
        view = observations.load(self.root / "scene.json").views[0]
        np.testing.assert_array_equal(view.pointmap()[6, 8], [0,0,2])
        np.testing.assert_allclose(view.pointmap()[0, 0], [-1.6,-1.2,2])
        self.document["units"] = "cm"
        self.document["views"][0]["c2w"][0][3] = 100
        np.save(self.root / "depth.npy", np.full((12, 16), 200, np.float32)); self.write()
        view = observations.load(self.root / "scene.json").views[0]
        np.testing.assert_array_equal(view.c2w[:3, 3], [1,0,0])
        np.testing.assert_array_equal(view.pointmap()[6,8], [0,0,2])

    def test_invalid_depth_mask_intrinsics_and_camera(self):
        tests = (
            ("negative depth", lambda: np.save(self.root / "depth.npy", np.full((12,16), -1, np.float32))),
            ("float64 depth", lambda: np.save(self.root / "depth.npy", np.full((12,16), 2, np.float64))),
            ("empty mask", lambda: Image.new("L", (16,12), 0).save(self.root / "mask.png")),
            ("skew", lambda: self.document["views"][0]["K"][0].__setitem__(1, .1)),
            ("reflection", lambda: self.document["views"][0]["c2w"][0].__setitem__(0, -1)),
        )
        for name, mutate in tests:
            with self.subTest(name=name):
                np.save(self.root / "depth.npy", np.full((12,16), 2, np.float32))
                Image.new("L", (16,12), 255).save(self.root / "mask.png")
                self.document["views"][0]["K"][0][1] = 0
                self.document["views"][0]["c2w"][0][0] = 1
                mutate(); self.write()
                with self.assertRaises(ValueError): observations.load(self.root / "scene.json")

    def test_camera_float32_overflow(self):
        for field, row, column in (("K", 0, 0), ("c2w", 0, 3)):
            with self.subTest(field=field):
                self.document["views"][0]["K"] = [[10,0,8],[0,10,6],[0,0,1]]
                self.document["views"][0]["c2w"] = np.eye(4).tolist()
                self.document["views"][0][field][row][column] = 1e300
                self.write()
                with self.assertRaisesRegex(ValueError, "float32"):
                    observations.load(self.root / "scene.json")

        self.document["views"][0]["c2w"] = np.eye(4).tolist()
        self.document["views"][0]["K"][0][0] = 1e-300
        self.write()
        with self.assertRaisesRegex(ValueError, "underflow"):
            observations.load(self.root / "scene.json")

    def test_manifest_bounds_checksums_and_containment(self):
        self.document["views"][0]["image"] = "../image.png"; self.write()
        with self.assertRaises(ValueError): observations.load(self.root / "scene.json")
        self.document["views"][0]["image"] = "image.png"
        self.document["views"][0]["image_sha256"] = "bad"; self.write()
        with self.assertRaisesRegex(ValueError, "checksum"): observations.load(self.root / "scene.json")
        self.document["views"][0].pop("image_sha256")
        self.document["views"] *= 17; self.write()
        with self.assertRaisesRegex(ValueError, "1..16"): observations.load(self.root / "scene.json")
        self.document["views"] = self.document["views"][:1]
        self.document["units"] = []; self.write()
        with self.assertRaisesRegex(ValueError, "units"): observations.load(self.root / "scene.json")

    def test_valid_upload_reuse_and_job_validation(self):
        raw = self.bundle()
        first = upload.upload(self.service, io.BytesIO(raw), len(raw), "application/zip")
        second = upload.upload(self.service, io.BytesIO(raw), len(raw), "application/zip")
        self.assertEqual(first, second)
        self.assertEqual(first["views"], 1)
        result = job.validate_job(self.service, {"upload_id": first["upload_id"]}, lambda *_: None, threading.Event())
        self.assertEqual(result["observations"]["camera_convention"], "R3")
        self.assertFalse(result["genia"]["available"])
        with self.assertRaisesRegex(ServiceError, "not implemented"):
            job.reconstruct_job(self.service, {"upload_id": first["upload_id"]}, lambda *_: None, threading.Event())
        self.assertFalse((self.service.work / "genia").exists())

    def test_unsafe_upload_does_not_publish(self):
        for name in ("../escape.txt", "/escape.txt", "extra.txt", "image.png"):
            with self.subTest(name=name):
                raw = self.bundle((name, "bad"))
                with self.assertRaises(ServiceError):
                    upload.upload(self.service, io.BytesIO(raw), len(raw), "application/zip")
        self.assertEqual(list((self.service.work / "genia_uploads").iterdir()), [])

    def test_remote_paths_and_truncated_upload_rejected(self):
        with self.assertRaises(ServiceError):
            job.validate_job(self.service, {"observations": "/etc/passwd"}, lambda *_: None, threading.Event())
        raw = self.bundle()
        with self.assertRaisesRegex(ServiceError, "truncated"):
            upload.upload(self.service, io.BytesIO(raw[:-1]), len(raw), "application/zip")


if __name__ == "__main__":
    unittest.main()
