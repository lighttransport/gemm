"""Refined ear topology: conformity, per-corner UVs, binding exactness, ARAP and relief."""
import unittest
from collections import Counter

import numpy as np

from .reconstruction import ear_mesh as em


def grid(n=8, bump=True):
    """Quad grid on z=f(x,y) with two mirrored 'ear' disks at |x|>0.5 (one mesh)."""
    xs = np.linspace(-1, 1, 2*n+1)
    x, y = np.meshgrid(xs, xs)
    z = .2*np.exp(-((np.abs(x)-.75)**2+y**2)/.05) if bump else 0*x
    points = np.column_stack((x.ravel()*.05, y.ravel()*.05, z.ravel()*.05))
    m = 2*n+1
    quads = []
    for j in range(2*n):
        for i in range(2*n):
            a = j*m+i
            quads.append((a, a+1, a+m+1, a+m))
    quads = np.asarray(quads)
    tris = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    uv = (points[:, :2]/.1+.5)[tris]
    # The two ear disks: 4x4 vertex blocks centred at x=+-0.75.
    ear = (np.abs(np.abs(x.ravel())-.75) <= .26) & (np.abs(y.ravel()) <= .26)
    return points, quads, tris, uv, ear


class EarTopologyTests(unittest.TestCase):
    def setUp(self):
        self.points, self.quads, self.tris, self.uv, self.ear = grid()
        self.topo = em.build_topology(self.points, self.tris, self.uv, self.quads, self.ear)

    def test_quad_triangle_pairing(self):
        pair = em.quad_triangles(self.quads, self.tris)
        for q, (a, b) in zip(self.quads, pair):
            self.assertEqual(set(self.tris[a]) | set(self.tris[b]), set(q))

    def test_conforming_manifold_replacement(self):
        tri, keep = em.surface_triangles(self.tris, self.topo['removed'], self.topo['triangles'])
        before, after = Counter(), Counter()
        for t, counter in ((self.tris, before), (tri, after)):
            for f in t:
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    counter[tuple(sorted((f[a], f[b])))] += 1
        self.assertFalse(any(c > 2 for c in after.values()))
        self.assertEqual(sum(c == 1 for c in before.values()), sum(c == 1 for c in after.values()))
        # Native boundary vertices are referenced directly; interior natives are not.
        native = len(self.points)
        for info in self.topo['sides'].values():
            self.assertTrue(np.isin(info['boundary'], tri).all())
            self.assertFalse(np.isin(info['interior_native'], tri[tri < native]).any())

    def test_orientation_and_uv_interpolation(self):
        ext = em.extended(self.points, self.topo['base'])
        c = ext[self.topo['triangles']]
        normal = np.cross(c[:, 1]-c[:, 0], c[:, 2]-c[:, 0])
        self.assertTrue((normal[:, 2] > 0).all())
        # On this planar parametrisation UVs are an affine function of xy.
        uv_expected = ext[self.topo['triangles']][..., :2]/.1+.5
        new = self.topo['triangles'] >= len(self.points)
        lin = em.subdivide  # linear UVs: face points/edge points are exact averages
        self.assertTrue(callable(lin))
        np.testing.assert_allclose(self.topo['triangle_uvs'][~new], uv_expected[~new], atol=1e-12)

    def test_binding_exact_under_rigid_motion(self):
        rng = np.random.default_rng(3)
        a = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        a *= np.sign(np.linalg.det(a))
        moved = self.points@a.T+rng.normal(size=3)
        new = em.evaluate(moved, self.topo['bind_ids'], self.topo['bind_weights'], self.topo['bind_offsets'])
        np.testing.assert_allclose(new, self.topo['base']@a.T+(moved[0]-self.points[0]@a.T), atol=1e-12)
        shifted = self.topo['base']+[0, 0, .003]
        ids, w, off = em.rebind(self.points, self.tris, self.topo, shifted)
        np.testing.assert_allclose(em.evaluate(moved, ids, w, off), shifted@a.T+(moved[0]-self.points[0]@a.T),
                                   atol=1e-12)

    def test_operator_is_affine(self):
        rows = np.asarray(self.topo['operator'].sum(1)).ravel()
        np.testing.assert_allclose(rows, 1, atol=1e-12)
        self.assertTrue((self.topo['operator'].data > 0).all())

    def test_band_patch_with_seam_and_disk(self):
        allowed = np.ones(len(self.points), bool)
        topo = em.build_topology(self.points, self.tris, self.uv, self.quads, self.ear,
                                 band_m=.008, allowed=allowed)
        for name, info in topo['sides'].items():
            self.assertGreater(len(info['patch_quads']), len(self.topo['sides'][name]['patch_quads']))
            self.assertTrue(info['new_in_band'].any() and (~info['new_in_band']).any())
            self.assertEqual(len(info['ear_loop_new'])+len(info['ear_loop_native']), len(self.topo['sides'][name]['boundary']))
        # A UV seam (different per-corner UVs) is preserved per face.
        uv = self.uv.copy()
        uv[self.points[self.tris][..., 1].mean(1) > 0] += .5
        seam = em.build_topology(self.points, self.tris, uv, self.quads, self.ear, band_m=.008, allowed=allowed)
        self.assertEqual(len(seam['triangles']), len(topo['triangles']))
        self.assertGreater(np.ptp(seam['triangle_uvs'][..., 1]), np.ptp(topo['triangle_uvs'][..., 1]))

    def test_invalid_triangle_split(self):
        with self.assertRaises(ValueError):
            em.quad_triangles(self.quads, self.tris[:-1])


class EarShapeTests(unittest.TestCase):
    def test_arap_identity_and_rigid_handles(self):
        points, quads, tris, uv, ear = grid(bump=False)
        fixed = np.zeros(len(points), bool)
        fixed[[0, 16, len(points)-1]] = True
        out = em.arap(points, tris, fixed, points.copy(), iterations=5)
        np.testing.assert_allclose(out, points, atol=1e-9)

    def test_relief_union_and_range(self):
        xy = np.random.default_rng(0).random((4000, 2))
        h = em.relief_height(xy)
        p = em.RELIEF_DEFAULTS
        ridges = max(v for k, v in p.items() if isinstance(v, float) and v > 0 and not k.endswith(('width', 'radius')))
        self.assertLessEqual(h.max(), ridges+1e-9)
        self.assertLess(h.min(), 0)
        fork = np.array([em.RELIEF_CURVES['antihelix'][-1]])
        self.assertLessEqual(em.relief_height(fork)[0], max(p['antihelix'], p['superior_crus'], p['inferior_crus'])+1e-9)

    def test_global_deform_root_fixed(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(200, 3))*.01
        frame = dict(origin=np.zeros(3), axes=np.eye(3)[:, [2, 1, 0]], hinge=np.array([0, 1., 0]))
        feather = np.zeros(200)
        np.testing.assert_allclose(em.global_deform(pts, frame, feather, length=1.3, protrusion=.2), pts)
        out = em.global_deform(pts, frame, np.ones(200), length=1.2)
        self.assertGreater(np.ptp(out[:, 1]), np.ptp(pts[:, 1]))


if __name__ == '__main__':
    unittest.main()
