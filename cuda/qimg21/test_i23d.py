#!/usr/bin/env python3
"""Tests for the Image-to-3D preprocessing package (qimg21_i23d) and runner.py.

Everything here runs on MockBackend and tiny images: no GPU, no weights.
    tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_i23d.py
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import math
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from qimg21_i23d import imageops, ops, views as viewlib  # noqa: E402
from qimg21_i23d.backends import BackendError, GenRequest, MockBackend  # noqa: E402
from qimg21_i23d.validate import validate_dataset  # noqa: E402
import runner  # noqa: E402

PIXAL3D_EXAMPLE = ROOT / "ref/pixal3d/upstream/assets/mv_images/example/transforms.json"


def object_image(width=320, height=240, color=(200, 40, 30)) -> np.ndarray:
    """A red antialiased disc, off-centre, on transparency."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.sqrt((xx + 0.5 - width * 0.3) ** 2 + (yy + 0.5 - height * 0.4) ** 2)
    alpha = np.clip(40.0 - dist + 0.5, 0.0, 1.0)
    out = np.zeros((height, width, 4), np.uint8)
    out[..., :3] = color
    out[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    return out


class Tmp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="qimg21-i23d-test-")
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def write(self, name, array, mode="RGBA"):
        return imageops.save_png(array, self.dir / name, mode=mode)


class ViewSpecTest(unittest.TestCase):
    def test_default_ring_is_every_30_degrees(self):
        specs = viewlib.ring(12)
        self.assertEqual([s.azimuth_deg for s in specs], [i * 30.0 for i in range(12)])
        self.assertTrue(all(s.elevation_deg == 0 for s in specs))

    def test_arbitrary_counts_and_rings(self):
        for count in (4, 6, 8, 12, 16, 7):
            self.assertEqual(len(viewlib.ring(count)), count)
        specs = viewlib.rings(12, [-20, 0, 20, 45])
        self.assertEqual(len(specs), 48)
        self.assertEqual([s.elevation_deg for s in specs[::12]], [-20, 0, 20, 45])
        self.assertEqual({viewlib.ring_relpath(s).split("/")[0] for s in specs}, {"e-20", "e000", "e020", "e045"})
        self.assertEqual(viewlib.ring_relpath(specs[13]), "e000/a030.png")

    def test_invalid_specs_say_why(self):
        bad = [({"azimuth_deg": float("nan")}, "finite"), ({"azimuth_deg": 0, "elevation_deg": 95}, "elevation"),
               ({"azimuth_deg": 0, "distance": 0}, "distance"), ({"azimuth_deg": 0, "fov_deg": 0.1}, "fov"),
               ({"azimuth_deg": 0, "width": 500}, "multiple of 32"),
               ({"azimuth_deg": 0, "projection": "fisheye"}, "projection")]
        for fields, reason in bad:
            with self.subTest(fields=fields), self.assertRaisesRegex(viewlib.ViewSpecError, reason):
                viewlib.ViewSpec(**fields).validated()
        with self.assertRaisesRegex(viewlib.ViewSpecError, "positive"):
            viewlib.ring(0)
        with self.assertRaisesRegex(viewlib.ViewSpecError, "repeat"):
            viewlib.rings(4, [0, 0])
        with self.assertRaisesRegex(viewlib.ViewSpecError, "numbers"):
            viewlib.parse_elevations("up,down")

    def test_prompt_names_the_view_and_is_deterministic(self):
        spec = viewlib.ViewSpec(90, 20).validated()
        text = viewlib.view_prompt(spec)
        self.assertIn("Camera azimuth: 90 degrees", text)
        self.assertIn("Camera elevation: 20 degrees", text)
        self.assertIn("left side view", text)
        self.assertIn("background is transparent", text)
        self.assertEqual(text, viewlib.view_prompt(viewlib.ViewSpec(90, 20).validated()))
        self.assertIn("rear view", viewlib.view_prompt(viewlib.ViewSpec(180).validated()))
        self.assertIn("orthographic", viewlib.view_prompt(viewlib.ViewSpec(0, projection="orthographic").validated()))

    def test_seeds_depend_on_the_view_not_its_position(self):
        a, b = viewlib.ViewSpec(30).validated(), viewlib.ViewSpec(60).validated()
        self.assertEqual(viewlib.derive_seed(7, a), 7)
        per_a, per_b = viewlib.derive_seed(7, a, "per_view"), viewlib.derive_seed(7, b, "per_view")
        self.assertNotEqual(per_a, per_b)
        self.assertEqual(per_a, viewlib.derive_seed(7, viewlib.ViewSpec(30).validated(), "per_view"))

    def test_matrices_match_pixal3d_and_are_rigid(self):
        for spec in viewlib.rings(8, [-60, 0, 30]) + viewlib.top_bottom():
            m = np.asarray(viewlib.transform_matrix(spec))
            rot = m[:3, :3]
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(rot), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(m[:3, 3])), spec.distance, places=6)
            # The camera looks at the origin: -back points at the object.
            np.testing.assert_allclose(-rot[:, 2] * spec.distance, -m[:3, 3], atol=1e-6)
        if PIXAL3D_EXAMPLE.is_file():
            frames = json.loads(PIXAL3D_EXAMPLE.read_text())["frames"]
            for frame, az in zip(frames, (0, 90, 180, 270)):
                ours = viewlib.transform_matrix(viewlib.ViewSpec(az, distance=3.119).validated())
                np.testing.assert_allclose(ours, frame["transform_matrix"], atol=1e-3)


class ImageOpsTest(Tmp):
    def test_normalize_keeps_aspect_fills_and_centers(self):
        rgba = object_image()
        out, transform = imageops.normalize_object(rgba, size=(256, 256), fill=0.5)
        bbox = imageops.alpha_bbox(out)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.assertAlmostEqual(max(width, height) / 256, 0.5, delta=0.03)
        self.assertAlmostEqual(width / height, 1.0, delta=0.05)
        self.assertAlmostEqual((bbox[0] + bbox[2]) / 2, 128, delta=2)
        self.assertAlmostEqual((bbox[1] + bbox[3]) / 2, 128, delta=2)
        self.assertEqual(transform.output_size, (256, 256))

    def test_antialiased_edges_keep_their_colour(self):
        out, _ = imageops.normalize_object(object_image(), size=(512, 512), fill=0.9)
        edge = (out[..., 3] > 10) & (out[..., 3] < 245)
        self.assertTrue(edge.any())
        # Premultiplied resampling: no black fringe from the transparent RGB.
        self.assertGreater(out[edge][:, 0].min(), 150)

    def test_crop_and_empty_alpha(self):
        out, transform = imageops.normalize_object(object_image(), crop=True, pad=4)
        self.assertEqual(out.shape[:2], (88, 88))
        with self.assertRaisesRegex(imageops.MaskError, "no foreground"):
            imageops.normalize_object(np.zeros((64, 64, 4), np.uint8))

    def test_masks_from_every_source(self):
        rect = imageops.make_mask((64, 48), rect=(8, 8, 16, 10))
        self.assertEqual(rect.shape, (48, 64))
        self.assertEqual(int((rect == 255).sum()), 160)
        circle = imageops.make_mask((64, 64), circle=(32, 32, 10))
        self.assertAlmostEqual((circle / 255).sum(), math.pi * 100, delta=8)
        rgba = object_image(64, 64)
        from_alpha = imageops.make_mask((64, 64), image=self.write("m.png", rgba))
        np.testing.assert_array_equal(from_alpha, rgba[..., 3])
        gray = self.write("g.png", np.full((32, 32), 200, np.uint8), mode="L")
        self.assertEqual(int(imageops.make_mask((32, 32), image=gray).min()), 200)

    def test_bad_masks_say_why(self):
        small = self.write("small.png", np.full((10, 10), 255, np.uint8), mode="L")
        cases = [({"image": small}, "10x10 but the image is 64x64"),
                 ({"image": self.write("zero.png", np.zeros((64, 64), np.uint8), mode="L")}, "selects nothing"),
                 ({"rect": (100, 100, 5, 5)}, "does not overlap"), ({"rect": (1, 2, 3)}, "x,y,w,h"),
                 ({"circle": (5, 5, -1)}, "positive"), ({}, "exactly one"),
                 ({"image": self.dir / "missing.png"}, "cannot read")]
        for kwargs, reason in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(imageops.MaskError, reason):
                imageops.make_mask((64, 64), **kwargs)
        self.assertEqual(imageops.make_mask((64, 64), image=small, resize=True).shape, (64, 64))

    def test_latent_mask_and_paste_back(self):
        mask = imageops.make_mask((64, 64), rect=(16, 16, 16, 16))
        weights = imageops.latent_mask(mask, 4, 4, dilate=0)
        self.assertEqual(weights.shape, (16,))
        self.assertEqual(weights.reshape(4, 4)[1, 1], 1.0)
        self.assertEqual(weights.sum(), 1.0)
        grown = imageops.latent_mask(mask, 4, 4, dilate=1)
        self.assertEqual(int((grown > 0).sum()), 9)
        original = object_image(64, 64)
        generated = np.full_like(original, 99)
        pasted = imageops.paste_outside(original, generated, mask)
        np.testing.assert_array_equal(pasted[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(pasted[mask == 255], generated[mask == 255])


class RequestTest(Tmp):
    def test_requests_are_checked(self):
        ref = self.write("r.png", object_image(64, 64))
        base = dict(prompt="x", out=self.dir / "o.png", references=(ref,))
        good = GenRequest(**base).validate(1)
        self.assertEqual(good.restart_step(), 0)
        self.assertEqual(GenRequest(**base, init_image=ref, strength=0.5).validate(1).restart_step(), 10)
        cases = [({"width": 300}, "multiple of 32"), ({"steps": 0}, "steps"), ({"strength": 0.5}, "init image"),
                 ({"mask": ref}, "init image"), ({"prompt": " "}, "empty"),
                 ({"references": (ref, ref)}, "at most 1"), ({"references": (self.dir / "no.png",)}, "missing")]
        for fields, reason in cases:
            with self.subTest(fields=fields), self.assertRaisesRegex(BackendError, reason):
                GenRequest(**{**base, **fields}).validate(1)


class OpsTest(Tmp):
    def setUp(self):
        super().setUp()
        self.backend = MockBackend()
        self.photo = self.write("photo.png", object_image())

    def test_object_preprocess_with_existing_alpha(self):
        out = self.dir / "object.png"
        info = ops.preprocess_object(self.photo, out, method="alpha", size=(256, 256))
        result = imageops.load_rgba(out)
        self.assertEqual(result.shape, (256, 256, 4))
        self.assertIn("transform", info)
        opaque = self.write("opaque.png", np.full((64, 64, 4), 255, np.uint8))
        with self.assertRaisesRegex(ValueError, "opaque"):
            ops.preprocess_object(opaque, out, method="alpha")

    def test_qwen_extraction_keeps_original_pixels_only_when_aligned(self):
        info = ops.preprocess_object(self.photo, self.dir / "o.png", self.backend, method="qwen",
                                     normalize_scale=False, center=False)
        # The mock draws a different object, so the originals must not be used.
        self.assertTrue(info["warnings"])
        self.assertLess(info["alignment"], 0.85)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(len(self.backend.calls[0].references), 1)
        self.assertIn("transparent", self.backend.calls[0].prompt)

    def test_masked_edit_changes_nothing_outside_the_region(self):
        out = self.dir / "edited.png"
        info = ops.edit_object(self.photo, "remove the hand", out, self.backend, rect=(0, 0, 100, 100))
        edited = imageops.load_rgba(out)
        source = imageops.resize_rgba(imageops.load_rgba(self.photo), info["width"], info["height"])
        mask = imageops.make_mask((info["width"], info["height"]),
                                  rect=(0, 0, 100 * info["width"] / 320, 100 * info["height"] / 240))
        np.testing.assert_array_equal(edited[mask == 0], source[mask == 0])
        request = self.backend.calls[-1]
        self.assertIsNotNone(request.mask)
        self.assertIsNotNone(request.init_image)
        self.assertIn("Keep the main object's identity", request.prompt)

    def test_occlusion_and_texture_options_are_checked(self):
        with self.assertRaisesRegex(ValueError, "region"):
            ops.complete_occlusion(self.photo, self.dir / "o.png", self.backend)
        with self.assertRaisesRegex(ValueError, "unknown texture"):
            ops.texture_preprocess(self.photo, self.dir / "o.png", self.backend, operations=("glow",))
        info = ops.texture_preprocess(self.photo, self.dir / "t.png", self.backend,
                                      operations=("neutralize-lighting", "remove-background"))
        self.assertIn("do not beautify", self.backend.calls[-1].prompt)
        self.assertIn("transparent", self.backend.calls[-1].prompt)
        self.assertEqual(info["operations"], ["neutralize-lighting", "remove-background"])

    def test_multiview_dataset(self):
        root = self.dir / "views"
        specs = viewlib.rings(4, [-20, 0, 45], width=256, height=256)
        summary = ops.generate_multiview([self.photo], specs, root, self.backend, ops.ViewParams(seed=5))
        self.assertEqual(summary["views"], 12)
        self.assertTrue((root / "views/e-20/a090.png").is_file())
        self.assertTrue((root / "views/e045/a270.png").is_file())
        meta = json.loads((root / "metadata.json").read_text())
        self.assertIn("not calibration", meta["camera_parameters"])
        self.assertEqual(meta["seed"], 5)
        self.assertEqual([v["azimuth_deg"] for v in meta["views"][:4]], [0, 90, 180, 270])
        self.assertEqual(meta["views"][4]["elevation_deg"], 0)
        self.assertEqual(len(meta["references"][0]["sha256"]), 64)
        transforms = json.loads((root / "transforms.json").read_text())
        self.assertTrue(transforms["generated_views"])
        self.assertEqual(len(transforms["frames"]), 13)
        self.assertFalse(transforms["frames"][0]["generated"])
        self.assertEqual(transforms["frames"][0]["file_path"], "reference/ref_00.png")
        validation = json.loads((root / "validation.json").read_text())
        self.assertEqual(validation["summary"]["images"], 12)
        # Every view is conditioned on the same reference, never on a generated view.
        for request in self.backend.calls:
            self.assertEqual([Path(r).name for r in request.references], ["ref_00.png"])

    def test_turntable_flat_layout(self):
        summary = ops.generate_turntable([self.photo], self.dir / "tt", self.backend, views=6, width=256, height=256)
        self.assertEqual(summary["layout"], "flat")
        self.assertEqual(sorted(p.name for p in (self.dir / "tt/images").iterdir()),
                         [f"{i:03d}.png" for i in range(6)])

    def test_same_inputs_same_outputs(self):
        specs = viewlib.ring(3, width=256, height=256)
        for name in ("a", "b"):
            ops.generate_multiview([self.photo], specs, self.dir / name, MockBackend(), ops.ViewParams(seed=3))
        for i in range(3):
            self.assertEqual((self.dir / f"a/images/{i:03d}.png").read_bytes(),
                             (self.dir / f"b/images/{i:03d}.png").read_bytes())
        ops.generate_multiview([self.photo], specs, self.dir / "c", MockBackend(), ops.ViewParams(seed=4))
        self.assertNotEqual((self.dir / "a/images/001.png").read_bytes(), (self.dir / "c/images/001.png").read_bytes())

    def test_batch_size_does_not_change_the_requests(self):
        class Batching(MockBackend):
            def __init__(self):
                super().__init__()
                self.batches = []

            def generate_batch(self, requests):
                self.batches.append(len(requests))
                return [self.generate(r) for r in requests]

        specs = viewlib.ring(5, width=256, height=256)
        one, many = MockBackend(), Batching()
        ops.generate_multiview([self.photo], specs, self.dir / "one", one, ops.ViewParams(seed=9, seed_mode="per_view"))
        ops.generate_multiview([self.photo], specs, self.dir / "many", many, ops.ViewParams(seed=9, seed_mode="per_view"),
                               batch_size=2)
        self.assertEqual(many.batches, [2, 2])
        self.assertEqual([(r.prompt, r.seed) for r in one.calls], [(r.prompt, r.seed) for r in many.calls])
        for i in range(5):
            self.assertEqual((self.dir / f"one/images/{i:03d}.png").read_bytes(),
                             (self.dir / f"many/images/{i:03d}.png").read_bytes())

    def test_a_long_turntable_does_not_accumulate_memory(self):
        backend = MockBackend()
        gc.collect()
        tracemalloc.start()
        ops.generate_turntable([self.photo], self.dir / "long8", backend, views=8, width=256, height=256)
        _, peak_small = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        ops.generate_turntable([self.photo], self.dir / "long64", backend, views=64, width=256, height=256)
        _, peak_large = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Images stream to disk: 8x the views must not mean 8x the memory.
        self.assertLess(peak_large, peak_small * 2 + (4 << 20))

    def test_image_to_3d_chain(self):
        root = self.dir / "ds"
        summary = ops.generate_image_to_3d_dataset(self.photo, root, self.backend, azimuth_views=4,
                                                   elevations=(0.0, 20.0), width=256, height=256, extract="alpha")
        self.assertEqual(summary["views"], 8)
        self.assertTrue((root / "preprocess/object_rgba.png").is_file())
        self.assertTrue((root / "pipeline.json").is_file())
        self.assertEqual(summary["layout"], "rings")


class ValidationTest(Tmp):
    def test_broken_images_are_reported(self):
        good = object_image(256, 256)
        self.write("good.png", good)
        self.write("blank.png", np.zeros((256, 256, 4), np.uint8))
        self.write("full.png", np.full((256, 256, 4), 255, np.uint8))
        self.write("small.png", object_image(128, 128))
        (self.dir / "junk.png").write_bytes(b"not a png")
        report = validate_dataset(self.dir, ["good.png", "blank.png", "full.png", "small.png", "junk.png",
                                             "missing.png"], width=256, height=256)
        by_file = {Path(r["file"]).name: r for r in report["images"]}
        self.assertTrue(by_file["good.png"]["ok"])
        self.assertIn("foreground covers", by_file["blank.png"]["errors"][0])
        self.assertTrue(any("not removed" in e for e in by_file["full.png"]["errors"]))
        self.assertTrue(any("size 128x128" in e for e in by_file["small.png"]["errors"]))
        self.assertTrue(any("unreadable" in e for e in by_file["junk.png"]["errors"]))
        self.assertEqual(by_file["missing.png"]["errors"], ["missing"])
        self.assertFalse(report["summary"]["consistent_size"])
        self.assertTrue((self.dir / "validation.json").is_file())


class CliTest(Tmp):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = runner.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_every_command_has_help(self):
        for command in ("object-preprocess", "edit", "texture-preprocess", "multiview", "turntable",
                        "image-to-3d", "validate"):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit) as exit_:
                    self.run_cli(command, "--help")
                self.assertEqual(exit_.exception.code, 0)

    def test_multiview_and_validate_through_the_cli(self):
        photo = self.write("p.png", object_image())
        code, out, _ = self.run_cli("multiview", "--input", str(photo), "--azimuth-views", "4", "--elevations",
                                    "-20,0,20", "--output", str(self.dir / "mv"), "--width", "256", "--height",
                                    "256", "--backend", "mock", "--seed", "11")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["views"], 12)
        code, out, _ = self.run_cli("validate", str(self.dir / "mv"))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["images"], 12)

    def test_useful_errors(self):
        photo = self.write("p.png", object_image())
        code, _, err = self.run_cli("turntable", "--input", str(photo), "--views", "0", "--output",
                                    str(self.dir / "x"), "--backend", "mock")
        self.assertEqual(code, 2)
        self.assertIn("positive", err)
        code, _, err = self.run_cli("edit", "--input", str(photo), "--output", str(self.dir / "e.png"),
                                    "--instruction", "fix", "--mask-rect", "999,999,5,5", "--backend", "mock")
        self.assertEqual(code, 2)
        self.assertIn("does not overlap", err)

    def test_object_preprocess_alias_and_alpha_method(self):
        photo = self.write("p.png", object_image())
        code, out, _ = self.run_cli("image-to-3d-preprocess", "--input", str(photo), "--output",
                                    str(self.dir / "o.png"), "--method", "alpha", "--size", "256")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["output_size"], [256, 256])


if __name__ == "__main__":
    unittest.main()
