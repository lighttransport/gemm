"""Image-free synthetic identities (reconstruction/synthetic.py)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .reconstruction import synthetic

ROOT = Path(__file__).resolve().parents[2]


def _model():
    from .face_assets import asset_path
    try:
        asset_path("gnm")
    except Exception as exc:                                   # noqa: BLE001
        raise unittest.SkipTest(f"GNM model asset unavailable: {exc}")
    from .rig.gnm_model import GNMModel
    return GNMModel()


class SyntheticIdentityTest(unittest.TestCase):
    def test_sampling_is_bounded_and_seeded(self):
        a, b = synthetic.sample_identity(3), synthetic.sample_identity(3)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, synthetic.sample_identity(4)))
        self.assertLessEqual(np.abs(a).max(), 2.0)
        self.assertTrue((a[synthetic.IDENTITY_MODES:] == 0).all())
        self.assertTrue((synthetic.sample_identity(3, 0.0) == 0).all())
        with self.assertRaises(ValueError):
            synthetic.sample_identity(1, 3.0)

    def test_head_inputs(self):
        model = _model()
        from .head import landmarks
        from .head.camera import mesh_from_glb
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            rec = synthetic.head_inputs(tmp, seed=5, texture_res=512, size=512, iris="blue", model=model)
            out = Path(tmp)
            self.assertEqual(rec["source"], "synthetic")
            self.assertEqual(json.loads((out / "synthetic.json").read_text())["seed"], 5)
            mesh = mesh_from_glb(out / "pixal3d.glb")
            self.assertEqual(len(mesh["triangles"]), rec["triangles"])
            self.assertLess(np.abs(mesh["positions"]).max(), 0.6)                  # Pixal3D-like framing
            # the Pixal3D crop of the rendered portrait is the whole image (within rounding)
            self.assertLess(abs(rec["framing"]["side"] - 512) / 512, 0.08, rec["framing"])
            eyes = landmarks.find_eyes(out / "portrait.png")
            self.assertEqual([e.side for e in eyes], ["right", "left"])
            self.assertGreater(eyes[1].cx - eyes[0].cx, 40)                         # a plausible IPD in pixels
            # explicit coefficients and strength 0 give the template head
            flat = synthetic.head_inputs(out / "flat", seed=5, strength=0.0, texture_res=256, size=256, model=model)
            self.assertTrue(all(v == 0 for v in flat["identity"]))
            self.assertNotEqual(flat["head_extent_m"], rec["head_extent_m"])


if __name__ == "__main__":
    unittest.main()
