"""Small checks for the body/face attachment and sparse morph import."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..eye.glb import GLB
from ..rig.gltf import RigGLB
from ..service import ROOT
from .assemble import _joint_entry, _similarity
from .garments import _fit_extents
from . import job


class BodyGeometryTest(unittest.TestCase):
    def test_similarity_preserves_landmarks(self):
        source = np.array([[0., 0., 0.], [.04, 0., 0.], [-.04, 0., 0.],
                           [0., -.05, .03], [0., .08, -.02]])
        turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = source @ turn.T * 1.14 + [0., 1.58, .05]
        matrix, err = _similarity(source, target)
        self.assertLess(err, 1e-10)
        np.testing.assert_allclose(source @ matrix[:3, :3].T + matrix[:3, 3], target, atol=1e-10)

    def test_scaled_joint_hierarchy_reproduces_bind(self):
        parent = np.eye(4)
        parent[:3, 3] = [0., .94, 0.]
        child = parent.copy()
        child[:3, :3] *= 1.08
        child[:3, 3] = [0., 1.48, .02]
        entry = _joint_entry("mhr_c_head", "mhr_root", child, parent)
        local = np.eye(4)
        local[:3, :3] = np.asarray(entry["rest_rotation"]) * entry["rest_scale"]
        local[:3, 3] = entry["rest_translation"]
        np.testing.assert_allclose(parent @ local, child, atol=1e-10)

    def test_sparse_face_target_roundtrip(self):
        writer = RigGLB("sparse test")
        ids = np.array([1, 4], np.int64)
        values = np.array([[.1, .2, .3], [-.1, 0., .05]], np.float32)
        acc = writer.sparse_vec3(6, ids, values)
        reader = GLB(writer.to_bytes())
        got_ids, got_values = reader.sparse_accessor(acc)
        np.testing.assert_array_equal(got_ids, ids)
        np.testing.assert_allclose(got_values, values)
        dense = reader.accessor(acc)
        np.testing.assert_allclose(dense[ids], values)
        self.assertEqual(int(np.count_nonzero(dense)), 5)

    def test_garment_fit_is_not_biased_by_waist_vertex_density(self):
        y = np.linspace(-.5, .5, 200)
        source = np.column_stack([np.linspace(-.3, .3, 200), y, np.zeros(200)])
        source = np.concatenate([source, np.tile([0., .35, 0.], (100, 1))])
        target = np.column_stack([np.linspace(-.3, .3, 200), y + .7, np.zeros(200)])
        scale, src_center, target_center = _fit_extents(source, target)
        fitted = (source - src_center) * scale + target_center
        median_fit = (source - np.median(source, axis=0)) * scale + np.median(target, axis=0)
        target_top = np.percentile(target[:, 1], 95)
        self.assertLess(abs(np.percentile(fitted[:, 1], 95) - target_top),
                        abs(np.percentile(median_fit[:, 1], 95) - target_top))


class GarmentJobTest(unittest.TestCase):
    def test_distinct_garment_outputs_and_configured_assets(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            out = Path(directory)
            Image.new("RGBA", (64, 64), (100, 80, 60, 255)).save(out / "body_image.png")
            model = out / "assets" / "sam3.safetensors"
            model.parent.mkdir()
            model.touch()
            bpe = out / "tokenizer"
            bpe.mkdir()
            (bpe / "vocab.json").touch()
            (bpe / "merges.txt").touch()
            commands = []

            def segment(cmd, cancel):
                commands.append(cmd)
                np.save(cmd[cmd.index("-o") + 1], np.ones((1, 64, 64), dtype=bool))

            def reconstruct(folder, image, name, quality, mock, cancel):
                (folder / name).touch()
                return {"status": "ok"}

            names = ["コート", "シャツ", "shirt", "shirt"]
            with patch.object(job.gpu, "gpu_status", return_value=None), \
                 patch.object(job, "_binary", return_value=Path("sam3")), \
                 patch.object(job, "_run", side_effect=segment), \
                 patch.object(job, "_pixal", side_effect=reconstruct):
                records = job._garments(out, names, False, threading.Event(), lambda *_: None,
                                        model, bpe)

            self.assertEqual([entry["prompt"] for entry in records], names)
            self.assertTrue(all(entry["status"] == "reconstructed" for entry in records))
            self.assertEqual(len({entry["name"] for entry in records}), len(names))
            self.assertEqual(len({entry["glb"] for entry in records}), len(names))
            self.assertEqual(len({entry["mask"] for entry in records}), len(names))
            self.assertEqual(len({cmd[cmd.index("-o") + 1] for cmd in commands}), len(names))
            for cmd, entry in zip(commands, records):
                self.assertEqual(cmd[1], str(model))
                self.assertEqual(cmd[cmd.index("--vocab") + 1], str(bpe / "vocab.json"))
                self.assertEqual(cmd[cmd.index("--merges") + 1], str(bpe / "merges.txt"))
                self.assertTrue((out / entry["glb"]).is_file())
                self.assertTrue((out / entry["mask"]).is_file())
                self.assertTrue((out / f"garment_{entry['name']}_input.png").is_file())


if __name__ == "__main__":
    unittest.main()
