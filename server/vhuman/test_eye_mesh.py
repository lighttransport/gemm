"""The runtime (low-polygon) eyeball against the analytic ray tracer.

The mesh eye shades cornea and sclera analytically from the rasterised
surface point, so a ~1.2k-triangle shell must match the tracer closely."""
from __future__ import annotations

import math
import unittest

import numpy as np

from .eye import geometry, optics, render
from .eye import params as P

GATE_MEAN, GATE_IRIS = 1.0, 2.5          # /255, 384 px CPU camera
SHELL_BUDGET, EYE_BUDGET = 1200, 1500    # triangles: shell alone, shell + iris + pupil cup


def _diff(a, b, p, size):
    """Mean |a-b| (/255) where both cover, in the iris, the peak, and the coverage mismatch."""
    both = (a[..., 3] > 0.5) & (b[..., 3] > 0.5)
    d = np.abs(a[..., :3] - b[..., :3]).mean(-1) * 255
    g = optics.profile_from_params(p)
    r_iris = optics.iris_radius(p, g) / render.Camera().half_width * size / 2
    yy, xx = np.mgrid[:size, :size]
    iris = (np.hypot(xx + 0.5 - size / 2, yy + 0.5 - size / 2) < 0.9 * r_iris) & both
    mismatch = ((a[..., 3] > 0.5) != (b[..., 3] > 0.5)).sum() / max((a[..., 3] > 0.5).sum(), 1)
    return float(d[both].mean()), float(d[iris].mean()), float(d[both].max()), float(mismatch)


class RuntimeMeshTest(unittest.TestCase):
    def test_budget_and_topology(self):
        p = P.preset("hazel")
        prof = optics.profile_from_params(p)
        shell = geometry.runtime_shell(prof)
        self.assertLessEqual(len(shell.indices), SHELL_BUDGET)
        self.assertTrue((geometry.edge_use_counts(shell.indices) == 2).all())
        iris = geometry.runtime_iris(p, prof)
        self.assertLessEqual(geometry.triangle_budget([shell, iris]) + 24, EYE_BUDGET)
        # Ring spacing on the visible sclera keeps the facet sag well under the lid clearance.
        alphas = geometry.runtime_alphas(prof, 16)
        step = np.diff(alphas[alphas <= math.radians(80.0)]).max()
        self.assertLess(prof.sclera_radius * (1 - math.cos(step / 2)) * 1000, 0.06)
        # LOD variants shrink monotonically.
        counts = [len(geometry.runtime_shell(prof, r, s).indices) for r, s in ((16, 36), (12, 24), (8, 16))]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_mesh_matches_tracer(self):
        p = P.preset("hazel")
        prof = optics.profile_from_params(p)
        shell = geometry.runtime_shell(prof)
        for yaw, pitch in ((0.0, 0.0), (20.0, -10.0)):
            cam = render.Camera(yaw_deg=yaw, pitch_deg=pitch)
            ref = render.render(p, 384, cam, spp=1, res=512, jitter=False, transparent=True)
            mesh = render.render_mesh(p, shell, 384, cam, res=512)
            mean, iris, peak, mismatch = _diff(ref, mesh, p, 384)
            self.assertLess(mean, GATE_MEAN, (yaw, pitch, mean, iris, peak))
            self.assertLess(iris, GATE_IRIS, (yaw, pitch, mean, iris, peak))
            self.assertLess(mismatch, 0.02, mismatch)       # the 36-gon silhouette only

    def test_caustic_and_soft_wrap(self):
        n = optics.normalize(np.array([[0.3, 0.0, 0.95]]))
        light = optics.normalize(np.array([-0.5, 0.2, 0.8]))
        cn = optics.caustic_normal(n, np.ones(1), np.zeros(1))
        self.assertLess(cn[0, 2], 0)                       # at the pupil the lobe faces away from the viewer
        far = optics.caustic_normal(n, np.ones(1), np.ones(1))
        self.assertGreater(far[0, 2], cn[0, 2])            # relaxes toward the iris plane at the limbus
        lobe = optics.caustic(cn, np.array([[0, 0, 1.0]]), light, 0.5, 12.0)
        self.assertTrue(np.isfinite(lobe).all() and (lobe >= 0).all() and (lobe <= 0.5).all())
        self.assertEqual(float(optics.caustic(cn, np.array([[0, 0, 1.0]]), light, 0.0, 12.0)[0]), 0.0)
        ndl = np.linspace(-1, 1, 9)
        soft = optics.soft_wrap(ndl, 0.3)
        self.assertTrue((np.diff(soft) > 0).all())
        np.testing.assert_allclose(soft[-1], 1.0, atol=0.02)
        self.assertGreater(soft[0], 0.0)


if __name__ == "__main__":
    unittest.main()


class RuntimeEyePartsTest(unittest.TestCase):
    def test_cap_occlusion_follows_the_fissure(self):
        from .head import eyeshell
        # An almond: wide open fissure, 9 mm tall, 28 mm wide, centred on the eye.
        t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        polygon = np.stack([0.014 * np.cos(t), 0.0045 * np.sin(t)], -1)
        inside = eyeshell.polygon_inside_distance(np.array([[0.0, 0.0], [0.0, 0.004], [0.0, 0.02], [0.03, 0.0]]), polygon)
        self.assertGreater(inside[0], 0.004)
        self.assertGreater(inside[1], 0.0)
        self.assertLess(inside[1], 0.001)
        self.assertLess(inside[2], 0.0)
        self.assertLess(inside[3], 0.0)
        cap = eyeshell.shell_mesh(rings=6, segments=24)
        alpha = eyeshell.cap_occlusion(cap.positions[:, :2].astype(np.float64), polygon, 1.0)
        self.assertTrue(((alpha >= 0) & (alpha <= eyeshell.STRENGTH + 1e-9)).all())
        centre = np.linalg.norm(cap.positions[:, :2], axis=1) < 0.002
        self.assertLess(alpha[centre].max(), 0.01)                   # open centre: no shadow
        under_lid = cap.positions[:, 1] > 0.006                      # beyond the upper margin: full strength
        self.assertGreater(alpha[under_lid].min(), eyeshell.STRENGTH * 0.95)
        # the top blur is wider than the bottom one (BLUR top .35 vs bottom .15)
        xy = np.array([[0.0, 0.0030], [0.0, -0.0030]])
        top, bottom = eyeshell.cap_occlusion(xy, polygon, 1.0)
        self.assertGreater(top, bottom)

    def test_runtime_parts_budget_and_bindings(self):
        from .reconstruction import offline_assets as oa
        p = P.preset("brown")
        prof = optics.profile_from_params(p)
        joint = np.array([0.03, 0.0, 0.01])
        records = {}
        def part(name, positions, triangles, texcoords, material, *, joint=None, runtime=False, **kw):
            self.assertTrue(runtime)
            self.assertEqual(texcoords.shape, (len(triangles), 3, 2))
            self.assertTrue(np.isfinite(positions).all())
            records[name] = dict(material=material, joint=joint, triangles=len(triangles), uv=texcoords)
        t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        polygon = np.stack([0.013 * np.cos(t), 0.004 * np.sin(t), np.zeros(48)], -1)
        report = oa.runtime_eye_parts(part, "left", 2, p, prof, lambda x: x + joint, polygon, -1.0)
        self.assertEqual(set(records), {f"left_{n}" for n in oa.RUNTIME_EYE_PARTS})
        eye = sum(r["triangles"] for n, r in records.items() if "_runtime_" in n)
        self.assertLessEqual(eye, oa.RUNTIME_EYE_TRIANGLES)
        self.assertEqual(records["left_runtime_eye"]["joint"], 2)       # the eyeball turns with the eye joint
        self.assertEqual(records["left_eyeshell"]["joint"], 1)          # the shadow cap stays with the head (lids)
        self.assertEqual(records["left_eyeshell"]["material"], "eyeshell")
        u = records["left_eyeshell"]["uv"][..., 0]
        self.assertTrue(((u >= 0) & (u <= 0.46)).all())
        self.assertLess(report["cap_alpha_min"], 0.01)
        self.assertGreater(report["cap_alpha_max"], 0.4)
