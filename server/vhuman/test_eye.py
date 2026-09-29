"""Unit tests for the procedural eye (no GPU).

    python3 -m unittest server.vhuman.test_eye -v
"""
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from .eye import assets, chart, extract, geometry, iris, noise, optics, render, sclera
from .eye import params as P
from .eye.glb import GLB

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp" / "vhuman" / "test-eye"
def setUpModule():
    TMP.mkdir(parents=True, exist_ok=True)


class ParamsTest(unittest.TestCase):
    def test_default_geometry_and_invalid_measurements(self):
        p = P.defaults()
        profile = optics.profile_from_params(p)
        self.assertLess(profile.limbus_radius, profile.cornea_radius)
        self.assertLess(profile.cornea_radius, profile.sclera_radius)
        self.assertEqual(optics.uniforms(p)["uvMapping"], "angular-two-segment-v1")
        for override in ({"sclera_radius": float("nan")}, {"ior": float("inf")},
                         {"limbus_radius": .009, "cornea_radius": .005}):
            with self.assertRaises(P.ParamError):
                P.validate({"optics": override})

    def test_clamp_and_errors(self):
        p = P.validate({"cornea": {"size": 0.5}, "pupil": {"dilation": 0.1}})
        self.assertEqual(p["cornea"]["size"], 0.3)
        self.assertEqual(p["pupil"]["dilation"], 0.25)
        with self.assertRaises(P.ParamError):
            P.validate({"iris": {"nope": 1}})
        with self.assertRaises(P.ParamError):
            P.validate({"iris": {"blend_method": "Swirl"}})
        with self.assertRaises(P.ParamError):
            P.validate({"iris": {"limbal_ring_color": [1, 2]}})

    def test_normalized_mapping_inverse(self):
        for spec in P.FIELDS:
            if spec.kind == "float":
                for v in (spec.lo, (spec.lo + spec.hi) / 2, spec.hi):
                    self.assertAlmostEqual(P.from_normalized(spec, P.to_normalized(spec, v)), v, places=9)

    def test_user_measurements_propagate_to_geometry_and_uniforms(self):
        p = P.validate({"optics": {"sclera_radius": .014, "limbus_radius": .0065,
                                  "cornea_radius": .009, "limbus_blend": .0003}})
        profile = optics.profile_from_params(p)
        mesh = geometry.shell(profile, rings=24, segments=32)
        self.assertAlmostEqual(float(mesh.positions[:, 2].max()), profile.apex_z, places=7)
        u = optics.uniforms(p)
        self.assertEqual(u["scleraRadius"], .014)
        self.assertAlmostEqual(u["f0"], ((p["optics"]["ior"] - 1) / (p["optics"]["ior"] + 1)) ** 2)
        self.assertNotEqual(P.structure_key(p, 512), P.structure_key(P.defaults(), 512))
        angle = np.linspace(0, math.pi, 101)
        np.testing.assert_allclose(optics.alpha_from_uv(optics.uv_radius(angle, profile), profile), angle, atol=1e-12)

    def test_presets_and_schema(self):
        self.assertEqual(len(P.PRESETS), 12)
        for name in P.PRESETS:
            P.preset(name)
        schema = json.loads(json.dumps(P.schema()))
        self.assertEqual(len(schema["presets"]), 12)
        live = {f["group"] + "." + f["key"] for f in schema["fields"] if f["live"]}
        self.assertIn("pupil.dilation", live)
        self.assertNotIn("structure.seed", live)

    def test_sclera_tint_and_pupil(self):
        p = P.validate({"sclera": {"skin_u": 1.0}})
        np.testing.assert_allclose(P.sclera_tint(p), [0.9, 0.85, 0.8])
        p = P.validate({"sclera": {"use_custom_tint": True, "tint": [0.9, 0.8, 0.7]}})
        np.testing.assert_allclose(P.sclera_tint(p), [0.9, 0.8, 0.7])
        # Pupil radius scales directly from the synthetic reference ratio.
        self.assertAlmostEqual(P.pupil_ratio(P.defaults()), .5)
        self.assertAlmostEqual(P.pupil_ratio(P.validate({"pupil": {"dilation": 1.0}})), P.P_REF)
        self.assertAlmostEqual(P.pupil_ratio(P.validate({"pupil": {"scale": 2.2, "dilation": 1.2}})),
                               P.P_REF * 2.0 * 1.2)

    def test_gnm_anatomy_and_saved_legacy_profile(self):
        fresh = P.defaults()
        self.assertEqual(fresh["optics"]["profile"], P.GNM_PROFILE)
        self.assertEqual(fresh["optics"]["sclera_radius"], .0146)
        self.assertEqual(fresh["optics"]["cornea_radius"], .0085)
        self.assertAlmostEqual(P.pupil_ratio(fresh), .5)
        old = P.defaults()
        old["optics"].pop("profile")
        old["optics"].update(sclera_radius=.012, cornea_radius=.008)
        old["pupil"]["dilation"] = 1.0
        recovered = P.validate(old)
        self.assertEqual(recovered["optics"]["profile"], P.LEGACY_PROFILE)
        self.assertAlmostEqual(P.pupil_ratio(recovered), .3)


class ChartTest(unittest.TestCase):
    def test_value_axis_and_lookup(self):
        for u in (0.05, 0.36, 0.6, 0.85):
            lum = [float(chart.chart_color(u, v) @ optics.LUMA) for v in np.linspace(0.0, 1.0, 6)]
            self.assertTrue(all(a < b for a, b in zip(lum, lum[1:])), (u, lum))
        brown, blue = chart.chart_color(0.1, 0.5), chart.chart_color(0.85, 0.5)
        self.assertGreater(brown[0], brown[2])      # U: brown ... blue, on the generated palette
        self.assertGreater(blue[2], blue[0])
        img = chart.chart_image(256)
        for u, v in ((0.1, 0.2), (0.5, 0.5), (0.9, 0.8)):
            px = chart.chart_lookup(img, u, v) / 255.0
            uq, vq = round(255 * u) / 255, round(255 * v) / 255
            want = optics.linear_to_srgb(chart.chart_color(uq, vq))
            np.testing.assert_allclose(px, want, atol=1.0 / 255 + 1e-6)


class NoiseTest(unittest.TestCase):
    def test_deterministic_periodic_unit(self):
        a = noise.spectral((64, 512), 7, beta=1.5, aniso=(1, 6), kmin=2)
        b = noise.spectral((64, 512), 7, beta=1.5, aniso=(1, 6), kmin=2)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertAlmostEqual(float(a.std()), 1.0, places=4)
        seam = np.abs(a[:, 0] - a[:, -1]).mean()
        neighbour = np.abs(np.diff(a, axis=1)).mean()
        self.assertLess(seam, 1.5 * neighbour)

    def test_fbm_speed(self):
        t = time.perf_counter()
        noise.spectral((384, 4096), 1, beta=2.0)
        self.assertLess(time.perf_counter() - t, 0.5)


class OpticsTest(unittest.TestCase):
    def test_spheres_meet_at_limbus(self):
        pr = optics.ANATOMICAL
        self.assertAlmostEqual(math.hypot(pr.limbus_radius, pr.z_limbus), pr.sclera_radius)
        self.assertAlmostEqual(math.hypot(pr.limbus_radius, pr.z_limbus - pr.cornea_center_z), pr.cornea_radius)
        self.assertGreater(pr.apex_z, pr.sclera_radius)

    def test_uv_layout(self):
        a_l = optics.ANATOMICAL.alpha_limbus
        self.assertAlmostEqual(float(optics.uv_radius(a_l)), optics.CORNEA_SIZE_REF, places=9)
        for r in (0.05, optics.CORNEA_SIZE_REF, 0.3, 0.6):
            self.assertAlmostEqual(float(optics.uv_radius(optics.alpha_from_uv(r))), r, places=6)
        self.assertAlmostEqual(float(optics.uv_radius(math.pi)), optics.BACK_UV_RADIUS, places=9)

    def test_refract_snell(self):
        n = np.array([[0.0, 0.0, 1.0]])
        d = optics.normalize(np.array([[0.4, 0.0, -1.0]]))
        t = optics.refract(d, n, 1 / 1.336)
        sin_i = np.linalg.norm(np.cross(d, n))
        sin_t = np.linalg.norm(np.cross(t, n))
        self.assertAlmostEqual(float(sin_i), float(1.336 * sin_t), places=6)

    def test_pupil_magnification(self):
        """Paraxial rays through the cornea: the pupil looks ~13% larger
        (the textbook entrance-pupil magnification)."""
        p = P.defaults()
        prof = optics.ANATOMICAL
        h = np.array([2e-4, 4e-4])
        o = np.stack([h, np.zeros(2), np.full(2, 0.05)], -1)
        d = np.tile([0.0, 0.0, -1.0], (2, 1))
        t = (0.05 - (prof.cornea_center_z + prof.cornea_radius))
        x = o + t * d
        n = optics.normalize(x - np.array([0, 0, prof.cornea_center_z]))
        q, r, _ = render.iris_hit(x, optics.refract(d, n, 1 / 1.336), p)
        m = h / r
        self.assertTrue(np.all((m > 1.10) & (m < 1.16)), m)


class IrisTest(unittest.TestCase):
    def test_layout_and_colour(self):
        p = P.preset("blue")
        tex = iris.build(p, 512)
        c = 256
        np.testing.assert_allclose(tex.masks[c, c], iris.PUPIL_MASKS, atol=1e-3)
        np.testing.assert_allclose(tex.masks[4, 4], iris.OUTSIDE, atol=1e-3)
        col = iris.bake_color(tex, p, 512)
        lum = col @ optics.LUMA
        # pupil edge at t = pupil ratio: dark inside, iris outside (+-2 px)
        r_px = P.pupil_ratio(p) * optics.IRIS_TEX_SCALE * 512
        row = lum[c, c:]
        self.assertLess(row[int(r_px - 3)], 0.02)
        self.assertGreater(row[int(r_px + 4)], 0.03)

    def test_pupil_scale(self):
        """Radial warp: identity at 1, the iris edge fixed, the visible
        pupil landing on the texture pupil."""
        t = np.linspace(0, 1, 11)
        np.testing.assert_allclose(optics.pupil_scale(t, 1.0), t)
        self.assertEqual(float(optics.pupil_scale(1.0, 1.7)), 1.0)
        for scale in (0.85, 1.2, 2.0):
            p = P.validate({"pupil": {"dilation": min(scale, 1.2), "scale": scale / min(scale, 1.2)}})
            self.assertAlmostEqual(float(optics.pupil_scale(P.pupil_ratio(p), P.pupil_scale(p))), P.P_REF)

    def test_circ_mask(self):
        self.assertAlmostEqual(float(optics.circ_mask(0.3, 0.3, 0.5, 0.1)), 0.5)
        self.assertEqual(float(optics.circ_mask(0.0, 0.3, 0.5, 0.1)), 1.0)
        self.assertEqual(float(optics.circ_mask(0.36, 0.3, 0.5, 0.1)), 0.0)
        self.assertEqual(float(optics.circ_mask(0.3, 0.3, 0.0, 0.1)), 1.0)     # centre 0: fades beyond size

    def test_custom_iris_blends(self):
        p = P.validate({"iris": {"primary_color_u": 0.85, "secondary_color_u": 0.2, "color_blend": 0.5,
                                 "color_blend_softness": 0.05, "blend_method": "Radial",
                                 "global_saturation": 1.0, "shadow_details": 0.0}})
        m = np.array([[0.5, 1.0, 1.0, 0.0], [0.5, 1.0, 0.0, 0.0]])   # B = 1: secondary region
        c = iris.custom_iris(m, p)
        np.testing.assert_allclose(c[0], chart.pick(0.2, 0.5), atol=1e-9)
        np.testing.assert_allclose(c[1], chart.pick(0.85, 0.5), atol=1e-9)
        p["iris"]["blend_method"] = "Structural"
        c = iris.custom_iris(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), p)
        np.testing.assert_allclose(c[0], 1.35 * chart.pick(0.2, 0.5), atol=1e-9)
        np.testing.assert_allclose(c[1], .65 * chart.pick(0.85, 0.5), atol=1e-9)
        p["iris"]["global_saturation"] = 0.0
        grey = iris.custom_iris(np.array([[0.5, 1.0, 0.0, 0.0]]), p)[0]
        self.assertAlmostEqual(float(np.ptp(grey)), 0.0, places=9)
        np.testing.assert_allclose(iris.desaturate(np.array([0.2, 0.4, 0.6]), 0.0), [0.2, 0.4, 0.6])

    def test_corneal_composite(self):
        p = P.defaults()
        self.assertGreater(float(iris.cornea_mask(0.0, p)), 0.99)
        self.assertLess(float(iris.cornea_mask(p["cornea"]["size"] + 0.06, p)), 0.01)

    def test_speed(self):
        p = P.validate({"structure": {"seed": int(time.time()) % 10000 + 11}})
        t = time.perf_counter()
        iris.build(p, 1024)
        self.assertLess(time.perf_counter() - t, 2.0)


class ScleraTest(unittest.TestCase):
    def test_irritation_area(self):
        """Vessels show outside the selected coverage radius
        radius, so a larger 'coverage' keeps the veins further out."""
        p = P.preset("hazel")
        p["sclera"]["vascularity_intensity"] = 1.5
        tex = sclera.build(p, 512)
        c = (np.arange(512) + 0.5) / 512 - 0.5
        r = np.hypot(c[None, :], c[:, None])
        visible = (r > 0.17) & (r < 0.3)
        reds = []
        for cov in (0.1, 0.25, 0.4):
            p["sclera"]["vascularity_coverage"] = cov
            col = sclera.colorize(tex.masks, r, p)
            redness = col[..., 0] / np.maximum(col[..., 1], 1e-3)
            reds.append(float(redness[visible].mean()))
        self.assertTrue(reds[0] > reds[1] > reds[2], reds)


class GeometryTest(unittest.TestCase):
    def test_shell_closed_with_consistent_normals(self):
        sh = geometry.shell()
        self.assertTrue(np.all(geometry.edge_use_counts(sh.indices) == 2))
        np.testing.assert_allclose(np.linalg.norm(sh.normals, axis=1), 1.0, atol=1e-5)
        tri = sh.indices.astype(int)
        p = sh.positions.astype(float)
        fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        self.assertTrue(np.all(np.sum(fn * sh.normals[tri].mean(1), 1) > 0))
        # UV continuity away from the back pole
        front = np.all(np.linalg.norm(sh.uvs[tri] - 0.5, axis=2) < 0.5, axis=1)
        jump = np.max(np.linalg.norm(sh.uvs[tri[front]] - sh.uvs[tri[front]][:, [1, 2, 0]], axis=2))
        ring_step = 2 * math.pi * 0.5 / 160       # the angular step at r_uv = 0.5 dominates
        self.assertLess(jump, 1.2 * ring_step)

    def test_iris_faces_forward(self):
        m = geometry.iris_disk(P.defaults())
        tri = m.indices.astype(int)
        p = m.positions.astype(float)
        fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        disk = np.abs(m.normals[tri].mean(1)[:, 2]) > 0.9
        self.assertTrue(np.all(fn[disk][:, 2] > 0))


class ExportTest(unittest.TestCase):
    def test_glb_round_trip(self):
        out = TMP / "eye.glb"
        info = assets.export_glb(P.preset("green"), out, 512)
        g = GLB.load(out)
        self.assertEqual(info["bytes"], out.stat().st_size)
        self.assertEqual({m["name"] for m in g.doc["meshes"]}, {"eyeball_shell", "iris"})
        self.assertIn("KHR_materials_transmission", g.doc["extensionsUsed"])
        shell = g.mesh_arrays("eyeball_shell")
        self.assertEqual(shell["POSITION"].shape[1], 3)
        self.assertEqual(shell["TANGENT"].shape[1], 4)
        self.assertEqual(g.image(0).shape[:2], (512, 512))

    def test_engine_neutral_texture_export(self):
        info = assets.export_textures(P.preset("brown"), TMP / "textures-independent", 512)
        self.assertIn("measurements.json", info["files"])
        report = json.loads((TMP / "textures-independent" / "measurements.json").read_text())
        self.assertEqual(report["units"], "metres")
        self.assertEqual(report["uv_mapping"], "angular-two-segment-v1")
        self.assertEqual(report["normal_convention"], "+Y / OpenGL")


class RenderTest(unittest.TestCase):
    def test_render_is_finite_and_bounded(self):
        img = render.render(P.preset("amber"), 96, render.Camera(yaw_deg=20), spp=1)
        self.assertTrue(np.isfinite(img).all())
        self.assertTrue((img >= 0).all() and (img <= 1).all())
        # the pupil is dark, the sclera bright
        front = render.render(P.preset("blue"), 96, spp=1)
        lum = front[..., :3] @ optics.LUMA
        self.assertLess(lum[48, 48], 0.1)
        self.assertGreater(lum[48, 15], 0.5)


class ExtractTest(unittest.TestCase):
    def test_synthetic_accuracy(self):
        names = list(P.PRESETS)
        good = 0
        for k in range(8):
            p = P.preset(names[(k * 5) % 12])
            p["cornea"]["limbus_softness"] = 0.03
            p["iris"]["limbal_ring_softness"] = 0.02
            img, lim, pup = extract.synthetic_eye(p, 640, 2000 + k)
            plate = extract.extract(img, res=256)
            ce = math.hypot(plate.limbus.cx - lim.cx, plate.limbus.cy - lim.cy) / lim.r
            # The edge found sits ~4% inside the composite's 50% point, at the
            # dark limbal ring; the dark ruff widens the visible pupil.
            re = abs(plate.limbus.r - lim.r) / lim.r
            pe = abs(plate.pupil.r - pup.r) / pup.r
            good += ce < 0.01 and re < 0.06 and pe < 0.07 and plate.quality["ok"]
        self.assertGreaterEqual(good, 7)

    def test_kaleidoscope_and_highlights(self):
        g = np.random.default_rng(0)
        tile = g.random((64, 64, 3)).astype(np.float32)
        strip = np.tile(tile, (1, 8, 1))                   # the same pattern 8 times around
        self.assertGreater(extract.kaleidoscope_score(strip), 0.6)
        natural = g.random((64, 512, 3)).astype(np.float32)
        self.assertLess(extract.kaleidoscope_score(natural), 0.3)
        s = np.full((64, 256, 3), 0.2, np.float32)
        s[20:24, 100:104] = 1.0
        self.assertTrue(extract.highlight_mask(s)[21, 101])

    def test_circle_fit_rejects_outliers(self):
        a = np.linspace(0, 2 * math.pi, 100, endpoint=False)
        x, y = 50 + 20 * np.cos(a), 40 + 20 * np.sin(a)
        x[:15] += 8
        c, _ = extract.fit_circle(x, y)
        self.assertAlmostEqual(c.cx, 50, delta=0.2)
        self.assertAlmostEqual(c.r, 20, delta=0.2)


class GpuLockTest(unittest.TestCase):
    def test_shares_the_pixal3d_lock_file(self):
        from . import gpu
        src = (ROOT / "server/pixal3d/app.py").read_text()
        self.assertIn("device-locks", src)
        self.assertIn('f"cuda-{device}.lock"', src)
        self.assertEqual(gpu.LOCK_PATH, ROOT / "tmp/pixal3d/device-locks/cuda-0.lock")

    def test_lock_excludes_and_cancels(self):
        from . import gpu
        path = TMP / "lock-test.lock"
        order = []
        with gpu.file_lock(path, 1.0):
            cancel = threading.Event()

            def other():
                try:
                    with gpu.file_lock(path, 5.0, cancel):
                        order.append("acquired")
                except gpu.Cancelled:
                    order.append("cancelled")
            t = threading.Thread(target=other)
            t.start()
            time.sleep(0.3)
            cancel.set()
            t.join(2)
        self.assertEqual(order, ["cancelled"])


class MockPipelineTest(unittest.TestCase):
    """Plates and the baseline end to end with the mock backends."""

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="mock-", dir=TMP))

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_plates_and_baseline(self):
        from . import baseline, qwen
        from .service import EyeService
        svc = EyeService(self.work)
        r = qwen.plates_job(svc, {"count": 2, "seed": 3, "color": "blue"}, lambda f, m: None, threading.Event(),
                            mock=True)
        self.assertEqual(len(r["kept"]) + r["rejected"], 2)
        self.assertGreaterEqual(len(r["kept"]), 1)
        rec = svc.plate(r["kept"][0])
        self.assertEqual(rec["license"], "qwen-research")
        tex = svc.textures({}, 512, {"source": "plate", "id": rec["id"]})
        self.assertIn("iris_photo", tex["urls"])
        b = baseline.baseline_job(svc, {"source": "analytic"}, lambda f, m: None, threading.Event(), mock=True)
        self.assertLess(b["metrics"]["chamfer_rms_mm"], 1.0)
        self.assertLess(b["metrics"]["sphere_rms_over_radius"], 0.05)
        with self.assertRaises(ValueError):
            qwen.plates_job(svc, {"count": 99}, lambda f, m: None, threading.Event(), mock=True)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{16}", rec["id"]))


if __name__ == "__main__":
    unittest.main()
