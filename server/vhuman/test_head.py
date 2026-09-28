"""Tests for the head pipeline (mock Qwen and Pixal3D, no GPU).

    python3 -m unittest server.vhuman.test_head -v
"""
from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from .eye.glb import GLB
from .eye import optics
from .head import camera, cleanup, eyeedge, eyeshell, fit, iris_match, landmarks, lids, pipeline, texture
from .service import ROOT, EyeService

TMP = ROOT / "tmp" / "vhuman" / "test-head"


class HeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="work-", dir=TMP))
        cls.portrait = pipeline.mock_portrait(cls.work / "portrait.png", seed=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_camera_round_trip(self):
        cam = camera.PixalCamera.from_portrait(self.portrait, math.radians(20))
        pts = np.array([[0.1, 0.05, -0.2], [-0.2, -0.1, 0.1], [0.0, 0.3, 0.0]])
        pix = cam.project(pts)
        o, d = cam.rays(pix[:, 0], pix[:, 1])
        # the ray through a point's projection passes through the point
        v = pts - o
        cross = np.linalg.norm(np.cross(v, d), axis=1)
        self.assertLess(float(cross.max()), 1e-9)
        # grid-frame check: the camera sits at -Z in the GLB frame
        self.assertLess(cam.origin[2], 0)

    def test_ray_mesh(self):
        v0 = np.array([[-1.0, -1.0, 0.0]])
        e1 = np.array([[2.0, 0.0, 0.0]])
        e2 = np.array([[0.0, 2.0, 0.0]])
        t, k = camera.ray_mesh(np.array([-0.2, -0.2, -3.0]), np.array([0.0, 0.0, 1.0]), v0, e1, e2)
        self.assertAlmostEqual(t, 3.0)
        self.assertEqual(k, 0)
        t, k = camera.ray_mesh(np.array([5.0, 5.0, -3.0]), np.array([0.0, 0.0, 1.0]), v0, e1, e2)
        self.assertEqual(k, -1)

    def test_eyes_found_in_mock_portrait(self):
        eyes = landmarks.find_eyes(self.portrait)
        self.assertEqual([e.side for e in eyes], ["right", "left"])
        self.assertLess(eyes[0].cx, eyes[1].cx)                  # the subject's right is image left
        self.assertAlmostEqual(eyes[0].r, eyes[1].r, delta=0.15 * eyes[0].r)
        for e in eyes:
            self.assertGreater(e.opening.sum(), math.pi * e.r * e.r * 0.5)
            self.assertEqual(len(e.corners), 2)
            # the fissure model contains the opening and spans the anatomical width
            self.assertTrue(np.all(e.fissure[e.opening]))
            cols = np.flatnonzero(e.fissure.any(0))
            self.assertGreater(cols[-1] - cols[0], 2 * landmarks.FISSURE_HALF_WIDTH_R * e.r - 3)

    def test_pipeline_mock(self):
        svc = EyeService(self.work / "svc")
        r = pipeline.head_job(svc, {"subject": "test person", "seed": 4, "quality": "preview"},
                              lambda f, m: None, threading.Event(), mock=True)
        folder = svc.work / "heads" / r["id"]
        rec = json.loads((folder / "head.json").read_text())
        self.assertGreater(rec["export"]["triangles_removed"], 0)
        self.assertGreater(rec["fit"]["fit"]["units_per_m"], 0)
        g = GLB.load(folder / "head_eyes.glb")
        names = {n["name"] for n in g.doc["nodes"]}
        self.assertTrue({"head", "eye_right", "eye_left", "eye_right_shell", "eye_left_iris"} <= names)
        eye = next(n for n in g.doc["nodes"] if n["name"] == "eye_left")
        self.assertEqual(len(eye["rotation"]), 4)
        self.assertAlmostEqual(float(np.linalg.norm(eye["rotation"])), 1.0, places=6)
        self.assertIn("KHR_materials_transmission", g.doc["extensionsUsed"])
        # the eyeshells (occlusion) and the lid cut
        self.assertTrue({"eye_right_eyeshell", "eye_left_eyeshell"} <= names)
        occ = next(m for m in g.doc["materials"] if m["name"] == "eye_left_occlusion")
        self.assertEqual(occ["alphaMode"], "BLEND")
        for side in ("right", "left"):
            self.assertIn(f"eye_{side}_tearline", names)
            self.assertIn(f"eye_{side}_caruncle", names)
            wet = next(m for m in g.doc["materials"] if m["name"] == f"eye_{side}_tearline")
            self.assertEqual(wet["alphaMode"], "BLEND")
            self.assertGreater(wet["pbrMetallicRoughness"]["baseColorFactor"][3], 0)
            self.assertLess(wet["pbrMetallicRoughness"]["baseColorFactor"][3], 1)
        lid = rec["fit"]["fit"]["lids"]
        for side in ("right", "left"):
            self.assertGreater(lid[side]["cut_triangles"], 0)
            self.assertGreater(lid[side]["new_triangles"], 0)
        self.assertEqual(rec["iris"]["requested"], "portrait")
        if rec["iris"]["source"] == "portrait":
            self.assertIn(rec["fit"]["plate"], rec["iris"]["candidates"])
            self.assertEqual(svc.plate(rec["fit"]["plate"])["source_head"], r["id"])
        listing = svc.list_heads()
        self.assertEqual(listing[0]["id"], r["id"])
        self.assertEqual(svc.head_file(r["id"], "head_eyes.glb"), folder / "head_eyes.glb")
        with self.assertRaises(Exception):
            svc.head_file(r["id"], "../head.json")
        self.assertGreater(rec["skin"]["covered_texels"], 0)
        self.assertTrue(svc.head_file(r["id"], "skin_normal.png").is_file())
        source_bytes = (folder / "head.json").read_bytes()
        # A CPU skin variant preserves the chosen eye plate and source.
        with patch("server.vhuman.qwen.make_backend", side_effect=AssertionError("CPU-only variant")):
            variant = pipeline.skin_job(svc, {"head_id": r["id"], "skin": {"enabled": False}},
                                       lambda f, m: None, threading.Event())
        updated = json.loads(svc.head_file(variant["id"], "head.json").read_text())
        self.assertEqual(updated["source_head"], r["id"])
        self.assertEqual(updated["fit"]["plate"], rec["fit"]["plate"])
        self.assertIsNone(updated["skin"])
        self.assertEqual((folder / "head.json").read_bytes(), source_bytes)
        with self.assertRaises(ValueError):
            pipeline.skin_job(svc, {"head_id": r["id"], "skin": {"roughness": -1}},
                              lambda f, m: None, threading.Event())

    def test_iris_color_controls(self):
        target = np.array([.1119, .0648, .0242])
        source = np.array([.2174, .0604, .0003])
        p = iris_match.color_controls(target, source)
        matched = iris_match.controlled_color(source, p)
        self.assertLess(iris_match.color_distance(target, matched), .001)
        self.assertLess(p["global_saturation"], 1.)
        np.testing.assert_allclose(iris_match.controlled_color(target, iris_match.color_controls(target, target)), target)
        # A bounded tint never pretends to brighten a too-dark neutral plate.
        dark = np.array([.01, .01, .01])
        matched = iris_match.controlled_color(dark, iris_match.color_controls(target, dark))
        self.assertGreater(iris_match.color_distance(target, matched), 1.)

    def test_texture_padding(self):
        size = 64
        uvs = np.array([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
        tris = np.array([[0, 1, 2]])
        img = np.zeros((size, size, 3), np.uint8)
        cov = texture.coverage(uvs, tris, (size, size))
        img[cov] = (200, 120, 100)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        png, info = texture.pad_png(buf.getvalue(), uvs, tris, steps=4, border=1)
        out = np.asarray(Image.open(io.BytesIO(png)))
        # texels just outside the chart now carry the chart's colour
        grown = texture.pad(np.zeros((size, size, 1), np.uint8), cov, 2)
        ring = (~cov) & np.asarray(Image.fromarray(cov.astype(np.uint8) * 255).resize((size, size))).astype(bool)
        edge = (~cov) & (np.roll(cov, 1, 0) | np.roll(cov, -1, 0) | np.roll(cov, 1, 1) | np.roll(cov, -1, 1))
        self.assertTrue(np.all(out[edge] == (200, 120, 100)))
        self.assertGreater(info["covered"], 0.0)
        del grown, ring

    def test_clip(self):
        # a unit square (two triangles) cut at x = 0.3 (f = x - 0.3): the
        # part x < 0.3 goes, the rest keeps its area and winding
        P = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        N = np.tile([0.0, 0, 1], (4, 1))
        UV = P[:, :2].copy()
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        f = P[:, 0] - 0.3
        new_p, new_n, new_uv, new_t, cut_ids, dropped = lids._clip(tris, np.arange(2), f, P, N, UV, len(P))
        self.assertTrue(dropped.all())
        allp = np.concatenate([P, new_p])
        e1 = allp[new_t[:, 1]] - allp[new_t[:, 0]]
        e2 = allp[new_t[:, 2]] - allp[new_t[:, 0]]
        z = np.cross(e1, e2)[:, 2]
        self.assertTrue(np.all(z >= -1e-12))                         # winding kept (+Z)
        self.assertAlmostEqual(float(z.sum() / 2), 0.7, places=9)    # area of x >= 0.3
        np.testing.assert_allclose(new_p[:, 0], 0.3)
        np.testing.assert_allclose(new_uv, new_p[:, :2])
        self.assertEqual(len(cut_ids), 3)                            # the edges x=0.3 crosses, shared once

    def test_band_crosses_uv_seams(self):
        # Two texture charts share a geometric edge but no vertex indices.
        pos = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [.01, .01, 0.],
                        [.01, 0., 0.], [.02, 0., 0.], [.01, .01, 0.], [.02, .01, 0.],
                        [.01, 0., .001], [.02, 0., .001], [.01, .01, .001]])
        tris = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6], [8, 9, 10]])
        original = pos.copy()
        band = lids._band(tris, pos, {0: np.array([0., 0., -.003])}, .03)
        for a, b in ((1, 4), (3, 6)):
            np.testing.assert_allclose(band[a][0] * band[a][1], band[b][0] * band[b][1])
        self.assertIn(5, band)  # falloff continues across the chart boundary
        self.assertLess(band[5][1], band[4][1])
        self.assertFalse(any(i in band for i in (8, 9, 10)))  # nearby separate layer stays separate
        np.testing.assert_array_equal(pos, original)
        self.assertEqual(lids._band(tris, pos, {}, .03), {})

    def test_drape_preserves_projection(self):
        origin = np.array([0., 0., -3.])
        centre = np.zeros(3)
        pos = np.array([[.3, .1, -1.5], [-.2, .2, -1.3], [.1, -.2, -1.8], [.5, .5, -1.5]])
        before = pos.copy()
        # A seed's displacement has a lateral component relative to these
        # neighbouring rays; the move and sphere push-out must ignore it.
        band = {0: (np.array([.1, -.1, 1.]), 1.),
                1: (np.array([.1, -.1, 1.]), 1.),
                2: (np.array([.1, -.1, .2]), .5)}
        self.assertEqual(lids._drape(pos, band, origin, centre, 1.), 2)
        np.testing.assert_allclose(np.cross(before - origin, pos - origin), 0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(pos[:2], axis=1), 1., atol=1e-12)
        self.assertTrue((np.linalg.norm(pos, axis=1) >= 1. - 1e-12).all())
        np.testing.assert_array_equal(pos[3], before[3])
        self.assertEqual(lids._drape(pos, {}, origin, centre, 1.), 0)

    def test_normals_cross_uv_seams(self):
        # Two triangles meet at a right angle across duplicated chart vertices.
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        normals = lids.vertex_normals(pos, np.array([[0, 1, 2], [3, 4, 5]]))
        np.testing.assert_allclose(normals[0], normals[3])
        np.testing.assert_allclose(normals[2], normals[4])
        np.testing.assert_allclose(normals[0], np.array([1., 0., 1.]) / math.sqrt(2))
        np.testing.assert_allclose(normals[1], [0., 0., 1.])
        np.testing.assert_allclose(normals[5], [1., 0., 0.])

    def test_depth_map(self):
        cam = camera.PixalCamera(math.radians(20), 1., 0., 0., 64.)
        xy = np.array([[10., 10.], [50., 10.], [50., 50.], [10., 50.]])
        o, ray = cam.rays(xy[:, 0], xy[:, 1])
        normal = np.array([.2, .15, 1.])
        pos = o + ray * (2.5 / (ray @ normal))[:, None]
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        depth = cleanup.depth_map(pos, tris, cam, np.zeros(2), (64, 64))
        yy, xx = np.mgrid[11:50, 11:50]
        _, d = cam.rays(xx, yy)
        np.testing.assert_allclose(depth[11:50, 11:50], 2.5 / (d @ normal), atol=1e-10)
        self.assertTrue(np.isinf(depth[:9]).all())
        # Depth ordering must ignore draw order and winding.
        front = o + ray * (2. / ray[:, 2])[:, None]
        joined = np.concatenate([pos, front])
        combined = np.concatenate([tris, tris[:, ::-1] + 4])
        near = cleanup.depth_map(joined, combined, cam, np.zeros(2), (64, 64))
        np.testing.assert_allclose(near[11:50, 11:50], 2. / d[..., 2], atol=1e-10)
        np.testing.assert_array_equal(near, cleanup.depth_map(joined, combined[::-1], cam, np.zeros(2), (64, 64)))

    def test_raised_surface(self):
        yy, xx = np.mgrid[:81, :81]
        for depth in (np.full((81, 81), 2.),
                      2. - .004 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / 225.),
                      2. - .004 * (xx >= 40)):
            _, raised, _ = cleanup.raised_pixels(depth, 4, .0008, .008)
            self.assertFalse(raised.any())  # flat skin, broad mound, and a broad step
        depth = np.full((81, 81), 2.)
        depth[20:60, 39:42] -= .004
        _, raised, _ = cleanup.raised_pixels(depth, 4, .0008, .008)
        self.assertTrue(raised[25:55, 39:42].all())
        self.assertFalse(raised[:, :35].any())
        depth[:, :40] = np.inf
        _, raised, _ = cleanup.raised_pixels(depth, 4, .0008, .008)
        self.assertFalse(raised[:, 40:42].any())  # insufficient skin support

    def test_sliver_cleanup(self):
        cam = camera.PixalCamera(math.radians(20), 1., 0., 0., 640.)
        # A sparse cheek plane plus a thin foreground strip beneath the lid.
        yy, xx = np.mgrid[265:376:5, 265:376:5]
        _, rays = cam.rays(xx.ravel(), yy.ravel())
        pos = cam.origin + rays * ((cam.distance - .25) / rays[:, 2])[:, None]
        n = xx.shape[1]
        faces = []
        for y in range(n - 1):
            for x in range(n - 1):
                a = y * n + x
                faces.extend([(a, a + 1, a + n), (a + 1, a + n + 1, a + n)])
        y0, x0 = np.mgrid[:640, :640]
        opening = ((x0 - 320) / 30) ** 2 + ((y0 - 320) / 10) ** 2 < 1
        eye = landmarks.Eye("right", 320, 320, 12, 3, opening=opening)
        pose = fit.EyePose("right", np.array([0., 0., -.25]), np.array([0., 0., -.22]), np.eye(3), 2.5)
        for half_width, shift, removed in ((.35, 0., True), (.05, .45, True), (10., 0., False)):
            xy = np.array([[320 + shift - half_width, 345], [320 + shift + half_width, 345],
                           [320 + shift + half_width, 355], [320 + shift - half_width, 355]])
            _, ray = cam.rays(xy[:, 0], xy[:, 1])
            strip = cam.origin + ray * ((cam.distance - .2625) / ray[:, 2])[:, None]
            p = np.concatenate([pos, strip])
            t = np.concatenate([faces, np.array([[0, 1, 2], [0, 2, 3]]) + len(pos)])
            mask = np.ones(len(t), bool)
            kept, info = cleanup.remove_slivers({"positions": p, "triangles": t}, mask, cam, [eye], [pose])
            self.assertTrue(kept[:-2].all())
            self.assertEqual(bool((~kept[-2:]).all()), removed)
            self.assertTrue(mask.all())  # input masks/geometry are not changed

    def test_portrait_iris_matching(self):
        grey = {"median_linear": [.1144, .1195, .1221],
                "suggested": {"primary_color_u": .59, "primary_color_v": .32}}
        self.assertEqual(iris_match.describe_color(grey), "grey")
        self.assertIsNone(iris_match.describe_color({}))
        self.assertEqual(iris_match.color_distance([.1, .2, .3], [.1, .2, .3]), 0.)
        self.assertTrue(math.isinf(iris_match.color_distance([1, 2], [.1, .2, .3])))
        plates = [{"id": "far", "colors": {"primary_linear": [.3, .15, .02]}},
                  {"id": "near", "colors": {"primary_linear": grey["median_linear"]}}]
        _, chosen = fit.choose_iris(grey, plates)
        self.assertEqual(chosen, "near")
        svc = EyeService(self.work / "iris-fallback")
        eyes = landmarks.find_eyes(self.portrait)
        with patch.object(iris_match.qwen, "generate_plates", return_value={"kept": [], "rejected": [{}, {}]}):
            candidates, info = iris_match.generate(svc, eyes, None, head_id="test", seed=1,
                                                   progress=lambda *_: None, cancel=threading.Event())
        self.assertEqual(candidates, [])
        self.assertEqual(info["source"], "library")
        self.assertEqual(info["rejected"], 2)
        self.assertIn("quality gate", info["fallback"])
        with patch.object(iris_match.qwen, "make_backend", side_effect=AssertionError("unexpected GPU load")):
            candidates, info = iris_match.prepare(svc, eyes, head_id="test", seed=1, source="library",
                                                  progress=lambda *_: None, cancel=threading.Event())
        self.assertEqual(info["requested"], "library")
        cancel = threading.Event()
        cancel.set()
        with patch.object(iris_match.qwen, "generate_plates", return_value={"kept": [], "rejected": []}):
            with self.assertRaises(iris_match.gpu.Cancelled):
                iris_match.generate(svc, eyes, None, head_id="test", seed=1, progress=lambda *_: None, cancel=cancel)

    def test_tint(self):
        size = 32
        uvs = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
        img = np.full((size, size, 3), 250, np.uint8)
        out, n = texture.tint(img, uvs, np.array([[0, 1, 2]]), (150, 100, 80), strength=1.0, grow=0)
        cov = texture.coverage(uvs, np.array([[0, 1, 2]]), (size, size))
        self.assertGreater(n, 0)
        self.assertLess(abs(int(out[cov][:, 1].mean()) - 100), 25)
        self.assertTrue(np.all(out[~cov] == 250))

    def test_eyeshell_texture(self):
        eyes = landmarks.find_eyes(self.portrait)
        cam = camera.PixalCamera.from_portrait(self.portrait, math.radians(20))
        pose = fit.EyePose("right", np.zeros(3), np.zeros(3), np.diag([-1.0, 1.0, -1.0]), 3.0)   # gazing at the camera (-Z)
        mesh, rgba, info = eyeshell.build(eyes[0], pose, cam)
        a = rgba[..., 3]
        # clear in the middle of the fissure, strongest under the lids
        self.assertEqual(int(a[eyeshell.TEX // 2, eyeshell.TEX // 2]), 0)
        self.assertEqual(int(a.max()), round(eyeshell.STRENGTH * 255))
        self.assertEqual(mesh.uvs.shape, (len(mesh.positions), 2))
        # Exported unlit alpha blending must never brighten even a black iris.
        rgb = optics.srgb_to_linear(rgba[..., :3].astype(float) / 255)
        alpha = a[..., None].astype(float) / 255
        for background in (0.0, 0.01, 0.1, 0.8):
            composed = rgb * alpha + background * (1 - alpha)
            self.assertTrue(np.all(composed <= background + 1e-12))
        self.assertEqual(info['strength'], eyeshell.STRENGTH)
        self.assertEqual(info['tint_linear'], [0.0, 0.0, 0.0])

    def test_wet_margin_geometry(self):
        cam = camera.PixalCamera(math.radians(20), 1, 0, 0, 1024)
        yy, xx = np.mgrid[:1024, :1024]
        opening = ((xx - 512) / 35) ** 2 + ((yy - 512) / 15) ** 2 < 1
        eye = landmarks.Eye("right", 512, 512, 15, 4, opening=opening)
        contour = lids.eye_contour(eye)
        theta = (np.arange(180) + 0.5) * (2 * math.pi / 180) - math.pi
        r = contour(theta)
        _, rays = cam.rays(512 + r * np.cos(theta), 512 + r * np.sin(theta))
        depth = np.full(180, 2.7)
        depth[20] += 0.02  # isolated socket-layer spike
        points = cam.origin + rays * depth[:, None]
        edge = eyeedge.margin(points, cam, eye, contour)
        np.testing.assert_allclose(np.linalg.norm(edge - cam.origin, axis=1), 2.7, atol=1e-9)
        rp, rt, uv = lids._ribbon(np.arange(len(points)), points, cam.origin, cam, eye,
                                  np.zeros(3), 0.12, 0.1)
        self.assertGreater(len(rt), 0)
        self.assertTrue(np.isfinite(rp).all())
        self.assertEqual(set(uv[:, 1]), {0., 1.})
        # Bottom lies exactly on the sphere; front lies on the smoothed depth.
        np.testing.assert_allclose(np.linalg.norm(rp[uv[:, 1] == 1], axis=1), 0.12, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(rp[uv[:, 1] == 0] - cam.origin, axis=1), 2.7, atol=1e-9)
        pose = fit.EyePose("right", np.zeros(3), np.zeros(3), np.diag([-1., 1., -1.]), 3.)
        film = eyeedge.tearline(edge, pose, cam)
        self.assertTrue(np.isfinite(film.positions).all())
        np.testing.assert_allclose(np.linalg.norm(film.normals, axis=1), 1, atol=1e-6)
        p = film.positions[film.indices]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        # Every face has area and outward winding (single-sided transparency).
        self.assertTrue((np.linalg.norm(cross, axis=1) > 1e-12).all())
        self.assertTrue((np.einsum("ij,ij->i", cross, film.normals[film.indices].mean(1)) > 0).all())
        plane = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        for side in ("right", "left"):
            eye.side = pose.side = side
            tissue = eyeedge.caruncle(eye, pose, cam, contour, plane, tri)
            self.assertIsNotNone(tissue)
            self.assertTrue(np.isfinite(tissue.positions).all())
            np.testing.assert_allclose(np.linalg.norm(tissue.normals, axis=1), 1, atol=1e-6)
            p = tissue.positions[tissue.indices]
            cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            self.assertTrue((np.einsum("ij,ij->i", cross, tissue.normals[tissue.indices].mean(1)) > 0).all())
            # Subject-right is image-left; its medial corner is image-right.
            projected = cam.project(tissue.positions.mean(0))
            self.assertGreater((projected[0] - eye.cx) * (1 if side == "right" else -1), 0)
        self.assertIsNone(eyeedge.margin(np.zeros((0, 3)), cam, eye, contour))
        self.assertIsNone(eyeedge.tearline(None, pose, cam))
        self.assertIsNone(eyeedge.caruncle(eye, pose, cam, contour, np.zeros((0, 3)), np.zeros((0, 3), int)))

    def test_tearline_frames_stop_at_layer_gaps(self):
        cam = camera.PixalCamera(math.radians(20), 1, 0, 0, 1024)
        pose = fit.EyePose("right", np.zeros(3), np.zeros(3), np.eye(3), 1.)
        theta = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        points = np.stack([.01 * np.cos(theta), .005 * np.sin(theta), np.zeros(180)], -1)
        points[40:90, 2] = .01
        film = eyeedge.tearline(points, pose, cam)
        changed = points.copy()
        changed[40:90, 2] += .02
        other = eyeedge.tearline(changed, pose, cam)
        # Moving a disconnected layer must not rotate either end of this one.
        for end in (39, 90):
            ring = slice(end * 7, (end + 1) * 7)
            np.testing.assert_array_equal(film.positions[ring], other.positions[ring])
            np.testing.assert_array_equal(film.normals[ring], other.normals[ring])
        self.assertEqual(len(film.indices), (180 - 2) * 12)
        np.testing.assert_allclose(np.linalg.norm(film.normals[np.unique(film.indices)], axis=1), 1, atol=1e-6)
        p = film.positions[film.indices]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        self.assertTrue((np.einsum("ij,ij->i", cross, film.normals[film.indices].mean(1)) > 0).all())
        # No strip for isolated or duplicate points.
        self.assertIsNone(eyeedge.tearline(np.zeros((3, 3)), pose, cam))
        self.assertIsNone(eyeedge.tearline(points[::60], pose, cam))

    def test_lining_does_not_bridge_socket_layers(self):
        cam = camera.PixalCamera(math.radians(20), 1, 0, 0, 1024)
        yy, xx = np.mgrid[:1024, :1024]
        eye = landmarks.Eye("right", 512, 512, 15, 4,
                            opening=((xx - 512) / 35) ** 2 + ((yy - 512) / 15) ** 2 < 1)
        contour = lids.eye_contour(eye)
        theta = (np.arange(180) + .5) * (2 * math.pi / 180) - math.pi
        r = contour(theta)
        _, rays = cam.rays(eye.cx + r * np.cos(theta), eye.cy + r * np.sin(theta))
        depth = np.full(180, 2.7)
        # A sustained layer change survives the isolated-spike median.
        depth[40:90] -= .035
        points = cam.origin + rays * depth[:, None]
        radius = .12
        rp, rt, uv = lids._ribbon(np.arange(len(points)), points, cam.origin, cam, eye,
                                  np.zeros(3), radius, .14)
        self.assertGreater(len(rt), 0)
        scale = radius / (optics.ANATOMICAL.sclera_radius + lids.LID_WRAP_MARGIN_M)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            top = (uv[rt[:, a], 1] == 0) & (uv[rt[:, b], 1] == 0)
            length = np.linalg.norm(rp[rt[:, a]] - rp[rt[:, b]], axis=1)
            self.assertTrue((length[top] <= eyeedge.MAX_MARGIN_STEP_M * scale + 1e-9).all())
        # The steady source layers remain represented; no broad flattening.
        represented = np.linalg.norm(rp[uv[:, 1] == 0] - cam.origin, axis=1)
        self.assertLess(represented.min(), 2.67)
        self.assertGreater(represented.max(), 2.69)

    def test_quaternion(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            x, y, z, w = fit._quat(q)
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            np.testing.assert_allclose(r, q, atol=1e-9)

    def test_portrait_prompt(self):
        self.assertIn("looking directly into the camera", pipeline.portrait_prompt("a person"))
        with self.assertRaises(ValueError):
            pipeline.portrait_prompt("   ")


if __name__ == "__main__":
    unittest.main()
