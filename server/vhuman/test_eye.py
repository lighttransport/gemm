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
        self.assertEqual(P.validate({'sclera': {'vascularity_intensity': 1.8}})['sclera']['vascularity_intensity'], 1.8)

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
    def test_radial_color_mask_selects_secondary_inside(self):
        p = P.defaults()
        p['iris'].update(blend_method='Radial', color_blend=.5, color_blend_softness=.1,
                         primary_color_u=0., primary_color_v=0., secondary_color_u=1., secondary_color_v=1.,
                         global_saturation=1., shadow_details=0.)
        masks = np.array([[.5, 1., 0., 0.], [.5, 1., 1., 0.]])
        colors = iris.custom_iris(masks, p)
        np.testing.assert_allclose(colors[0], iris.chart.pick(0., 0.), atol=1e-6)
        np.testing.assert_allclose(colors[1], iris.chart.pick(1., 1.), atol=1e-6)

    def test_layered_iris_calibration_does_not_change_sclera(self):
        p = P.defaults(); it, st = assets.structures(p, 128)
        original = assets.eye_basecolor(p, it, st, 128)
        adjusted, inner, outer = assets.eye_basecolor_layered_calibrated(p, it, st, 128, [.1, .08, .06], [.12, .09, .07])
        r = np.linalg.norm(assets.uv_grid(128) - .5, axis=-1)
        outside = r > p['cornea']['size'] + .5 * p['cornea']['limbus_softness'] + .002
        np.testing.assert_allclose(adjusted[outside], original[outside], atol=1e-6)
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertTrue(((inner >= .5) & (inner <= 3.)).all())
        self.assertTrue(((outer >= .5) & (outer <= 2.)).all())
        bands, _, _ = assets.eye_basecolor_layered_calibrated(
            p, it, st, 128, [.1, .08, .06], [.12, .09, .07],
            radial_targets=[[.1, .08, .06], [.08, .09, .12], [.05, .08, .14]])
        np.testing.assert_allclose(bands[outside], original[outside], atol=1e-6)
        self.assertTrue(np.isfinite(bands).all())
        self.assertGreater(float(np.mean(np.abs(bands - adjusted))), .0001)
        targets = [[.1, .08, .04], [.08, .09, .09], [.04, .08, .12]]
        no_fibers, _, _ = assets.eye_basecolor_layered_calibrated(
            p, it, st, 128, targets[0], targets[-1], radial_targets=targets)
        fibers, _, _ = assets.eye_basecolor_layered_calibrated(
            p, it, st, 128, targets[0], targets[-1], radial_targets=targets, fiber_chroma=.6)
        np.testing.assert_allclose(fibers[outside], original[outside], atol=1e-6)
        self.assertTrue(np.isfinite(fibers).all())
        self.assertGreater(float(np.mean(np.abs(fibers - no_fibers))), .0001)
        with self.assertRaises(ValueError):
            assets.eye_basecolor_layered_calibrated(p, it, st, 128, [.1, .08, .06], [.12, .09, .07], radial_targets=[[1, 2, 3]])

    def test_outer_iris_calibration_preserves_inner_and_outer_sclera(self):
        p = P.defaults(); uv = assets.uv_grid(128); r = np.linalg.norm(uv - .5, axis=-1) / p['cornea']['size']
        color = np.empty((128, 128, 3), np.float32); color[:] = [.1, .05, .01]
        adjusted, gains = assets.calibrate_iris_outer(color, p, [.2, .025, .02])
        np.testing.assert_allclose(gains, [2., .5, 2.], atol=1e-6)
        np.testing.assert_allclose(adjusted[r < .5], color[r < .5], atol=1e-6)
        np.testing.assert_allclose(adjusted[r > 1.1], color[r > 1.1], atol=1e-6)
        self.assertGreater(float(adjusted[..., 0][(r > .7) & (r < .85)].mean()), .18)

    def test_baked_iris_calibration_preserves_sclera_and_bounds_dark_channels(self):
        p = P.defaults(); uv = assets.uv_grid(128)
        r = np.linalg.norm(uv - .5, axis=-1)
        color = np.empty((128, 128, 3), np.float32); color[:] = [.1, .05, .001]
        corrected, gains = assets.calibrate_iris_color(color, p, [.2, .025, .3])
        np.testing.assert_allclose(gains, [2., .5, 1.], atol=1e-6)
        np.testing.assert_allclose(corrected[64, 64], [.2, .025, .001], atol=1e-6)
        np.testing.assert_allclose(corrected[r > .3], color[r > .3], atol=1e-6)

    def test_angular_contrast_calibration_preserves_radial_mean_and_sclera(self):
        p = P.defaults()
        uv = assets.uv_grid(128); x, y = uv[..., 0] - .5, uv[..., 1] - .5
        r = np.hypot(x, y); field = .5 + .2 * np.cos(np.arctan2(y, x))
        color = np.repeat(field[..., None], 3, axis=-1).astype(np.float32)
        adjusted = assets.reduce_iris_angular_contrast(color, p, .6)
        ring = (r > .08) & (r < .12)
        self.assertLess(adjusted[..., 0][ring].std(), .5 * color[..., 0][ring].std())
        self.assertAlmostEqual(float(adjusted[..., 0][ring].mean()), float(color[..., 0][ring].mean()), places=5)
        np.testing.assert_array_equal(adjusted[r > .3], color[r > .3])

    def test_angular_contrast_ignores_radial_brightness(self):
        uv = assets.uv_grid(256); x, y = uv[..., 0] - .5, uv[..., 1] - .5
        r = np.hypot(x, y) * 2.
        sector = np.ones_like(r, dtype=bool)
        radial = .2 + r * .3
        patterned = radial * (1. + .3 * np.cos(8. * np.arctan2(y, x)))
        self.assertLess(iris.angular_contrast(radial, r, sector), .02)
        self.assertGreater(iris.angular_contrast(patterned, r, sector), .1)
        self.assertIsNone(iris.angular_contrast(radial, r, np.zeros_like(sector)))

    def test_chromatic_fiber_signal_rejects_radial_gradient(self):
        uv = assets.uv_grid(128); x, y = uv[..., 0] - .5, uv[..., 1] - .5
        r = np.hypot(x, y) * 2.; sector = np.ones_like(r, dtype=bool)
        lum = .2 + .1 * r
        base = np.repeat(lum[..., None], 3, axis=-1)
        pattern = .15 * np.cos(9. * np.arctan2(y, x))
        varied = base * (1. + pattern[..., None] * np.array([1., -.2, -1.]))
        inner, outer = np.array([.3, .2, .1]), np.array([.1, .2, .3])
        self.assertLess(iris.chromatic_fiber_signal(base, r, sector, inner, outer), .001)
        self.assertGreater(iris.chromatic_fiber_signal(varied, r, sector, inner, outer), .05)
        self.assertIsNone(iris.chromatic_fiber_signal(base, r, sector, inner, inner))

    def test_baked_normal_rotates_with_iris_and_flattens_pupil(self):
        p = P.validate({'iris': {'rotation': .25}, 'pupil': {'dilation': 1., 'scale': 1.}})
        n = np.broadcast_to(np.array([.6, 0., .8], np.float32), (64, 64, 3)).copy()
        tex = iris.IrisTextures(64, np.zeros((64, 64, 4), np.float32), n, None)
        baked = iris.bake_normal(tex, p, 64)
        np.testing.assert_allclose(baked[32, 43], [0., .6, .8], atol=1e-5)
        np.testing.assert_allclose(baked[32, 32], [0., 0., 1.], atol=1e-5)
        np.testing.assert_allclose(baked[0, 0], [0., 0., 1.], atol=1e-5)

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
    def test_sclera_microtexture_is_deterministic_and_excludes_iris(self):
        p = P.defaults()
        f = assets.eye_sclera_microfactor(p, 128, .2)
        np.testing.assert_array_equal(f, assets.eye_sclera_microfactor(p, 128, .2))
        np.testing.assert_allclose(f[64, 64], [.5, .5, .5], atol=1e-6)
        self.assertGreater(float(f[10:30, 10:30, 0].std()), .005)
        with self.assertRaises(ValueError):
            assets.eye_sclera_microfactor(p, 128, .5)
        surface = assets.eye_sclera_surface_microfactor(p, 128, .1)
        np.testing.assert_array_equal(surface, assets.eye_sclera_surface_microfactor(p, 128, .1))
        np.testing.assert_allclose(surface[64, 64], [.5, .5, .5], atol=1e-6)
        self.assertGreater(float(surface[10:30, 10:30, 0].std()), .005)
        fine = assets.eye_sclera_surface_microfactor(p, 128, .05, fine=True)
        np.testing.assert_array_equal(fine, assets.eye_sclera_surface_microfactor(p, 128, .05, fine=True))
        np.testing.assert_allclose(fine[64, 64], [.5, .5, .5], atol=1e-6)
        self.assertGreater(float(np.mean(np.abs(fine - surface))), .005)

    def test_tissue_factor_is_reproducible_and_preserves_iris(self):
        p = P.validate({'sclera': {'vascularity_intensity': 1.8}})
        f = assets.eye_tissue_factor(p, 128)
        np.testing.assert_array_equal(f, assets.eye_tissue_factor(p, 128))
        np.testing.assert_allclose(f[64, 64], [.5, .5, .5], atol=1e-6)
        self.assertGreater(float(f[64, 113, 0]), float(f[64, 113, 1]))
        low = assets.eye_tissue_factor(P.validate({'sclera': {'vascularity_intensity': 0.}}), 128)
        self.assertLess(float(f[64, 113, 1]), float(low[64, 113, 1]))

    def test_whole_globe_vessels_are_reproducible_and_cover_all_quadrants(self):
        p = P.defaults()
        mask = sclera.isotropic_vessels(p, 256)
        np.testing.assert_array_equal(mask, sclera.isotropic_vessels(p, 256))
        y, x = np.mgrid[:256, :256] - 127.5
        r = np.hypot(x, y)
        for region in (x > 2 * np.abs(y), x < -2 * np.abs(y), y > 2 * np.abs(x), y < -2 * np.abs(x)):
            self.assertGreater(float(mask[region & (r > 60) & (r < 105)].mean()), .001)
        self.assertLess(float(mask[r < 40].max()), .01)
        curved = sclera.isotropic_vessels(p, 256, curvy=True)
        np.testing.assert_array_equal(curved, sclera.isotropic_vessels(p, 256, curvy=True))
        self.assertGreater(float(np.mean(np.abs(curved - mask))), .002)
        irregular = sclera.irregular_vessels(p, 256)
        np.testing.assert_array_equal(irregular, sclera.irregular_vessels(p, 256))
        self.assertTrue(np.isfinite(irregular).all())
        self.assertGreater(float(irregular.mean()), .001)
        self.assertLess(float(irregular[r < 40].max()), .01)
        dense = sclera.irregular_vessels(p, 256, dense=True)
        np.testing.assert_array_equal(dense, sclera.irregular_vessels(p, 256, dense=True))
        self.assertGreater(float(dense.mean()), float(irregular.mean()))
        self.assertLess(float(dense[r < 40].max()), .01)
        lines = sclera.irregular_vessels(p, 256, dense=True, mottling=False)
        self.assertTrue(np.all(lines <= dense + 1e-6))
        self.assertGreater(float(np.mean(dense - lines)), .005)
        sparse = sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0.)
        np.testing.assert_array_equal(sparse, sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0.))
        self.assertLess(float(sparse.mean()), float(lines.mean()))
        organic = sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0., organic=True)
        np.testing.assert_array_equal(organic, sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0., organic=True))
        self.assertGreater(float(organic.mean()), 0.)
        self.assertLess(float(organic[r < 40].max()), .01)
        self.assertGreater(float(np.mean(np.abs(organic - sparse))), .0001)
        organic_dense = sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0., organic=True, organic_dense=True)
        np.testing.assert_array_equal(organic_dense, sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=0., organic=True, organic_dense=True))
        self.assertGreater(float(np.mean(organic_dense > .01)), float(np.mean(organic > .01)))
        self.assertLess(float(organic_dense.max()), float(organic.max()))
        with self.assertRaises(ValueError):
            sclera.irregular_vessels(p, 256, organic_dense=True)
        np.testing.assert_array_equal(lines, sclera.irregular_vessels(p, 256, dense=True, mottling=False, density=1.))
        with self.assertRaises(ValueError):
            sclera.irregular_vessels(p, 256, density=1.1)

    def test_guided_vessel_paths_reduce_looping(self):
        def paths(pull):
            return sclera._grow(noise.rng(123, 55), 1024, .2, 60, (1., 3.), .004, .04, 0.,
                                start_r=(.3, .5), reach=(.3, 1.), inward_pull=.01, turn_noise=.32,
                                initial_turn=1.15, max_steps=100, width_decay=.975, heading_pull=pull)
        def loop_fraction(branches):
            ratios = [np.linalg.norm(points[-1] - points[0]) / np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                      for points, _, _, _ in branches if len(points) > 20]
            self.assertGreater(len(ratios), 50)
            return float(np.mean(np.asarray(ratios) < .3))
        self.assertLess(loop_fraction(paths(.05)), loop_fraction(paths(0.)))
        with self.assertRaises(ValueError):
            sclera.irregular_vessels(P.defaults(), 256, guided=True)

    def test_vessel_crossing_is_trimmed_after_join_region(self):
        t = np.linspace(-.4, .4, 17)
        horizontal = np.stack((t, np.zeros_like(t)), axis=-1)
        vertical = np.stack((np.zeros_like(t), t), axis=-1)
        branch = lambda points: (points, np.ones(len(points) - 1), .8, np.zeros(len(points)))
        result = sclera._stop_at_crossings([branch(horizontal), branch(vertical)], 128)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0][0]), len(horizontal))
        self.assertLess(len(result[1][0]), len(vertical))
        self.assertLess(result[1][0][-1, 1], 0.)
        with self.assertRaises(ValueError):
            sclera.irregular_vessels(P.defaults(), 256, avoid_crossings=True)

    def test_self_crossing_is_trimmed_without_trimming_other_branches(self):
        loop = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.2, .1], [.2, .2],
                         [.1, .2], [0., .2], [0., .1], [0., 0.], [-.1, 0.]])
        line = np.array([[-.3, .05], [.3, .05]])
        branch = lambda points: (points, np.ones(len(points) - 1), .8, np.zeros(len(points)))
        result = sclera._stop_at_crossings([branch(loop), branch(line)], 128,
                                           inter_branch=False, self_crossing=True)
        self.assertEqual(len(result), 2)
        self.assertLess(len(result[0][0]), len(loop))
        self.assertEqual(len(result[1][0]), len(line))
        with self.assertRaises(ValueError):
            sclera.irregular_vessels(P.defaults(), 256, avoid_self=True)

    def test_vessel_overlay_is_deterministic_and_laterally_biased(self):
        p = P.defaults()
        tex = sclera.build(p, 256)
        mask = assets.sclera_vessel_texture(tex, 256)
        self.assertEqual(mask.shape, (256, 256, 3))
        np.testing.assert_array_equal(mask, assets.sclera_vessel_texture(sclera.build(p, 256), 256))
        y, x = np.mgrid[:256, :256] - 127.5
        r = np.hypot(x, y)
        annulus = (r > 55) & (r < 105)
        horizontal = annulus & (np.abs(x) > 2 * np.abs(y))
        vertical = annulus & (np.abs(y) > 2 * np.abs(x))
        self.assertGreater(float(mask[..., 0][horizontal].mean()), float(mask[..., 0][vertical].mean()))

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
        self.assertIn('f"{backend}-{device}.lock"', src)
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
