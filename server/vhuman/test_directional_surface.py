import unittest
import numpy as np
from .reconstruction.directional_surface import area_constraints, constrain_offsets


class DirectionalSurfaceTests(unittest.TestCase):
    def test_affine_area_matches_direct_cross_products(self):
        rng = np.random.default_rng(17)
        frames = rng.normal(size=(2, 12, 3))*.01
        triangles = np.arange(12).reshape(-1, 3)
        direction = np.array([.6, .8, 0])
        offsets = rng.normal(size=12)
        matrix = area_constraints(frames, triangles, direction)
        expected = []
        for frame in frames:
            def normals(v):
                p = v[triangles]
                return np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
            n0 = normals(frame)
            n1 = normals(frame+offsets[:,None]*direction*.001)
            expected.extend((n0*n1).sum(1)/np.square(n0).sum(1))
        np.testing.assert_allclose(1+matrix@offsets, expected, atol=1e-12)

    def test_projection_prevents_flip_and_preserves_feasible_offsets(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        triangles = np.array([[0,1,2]])
        desired = np.array([0.,0.,-2.])
        offsets, report = constrain_offsets(frame, triangles, [0,1,0], desired)
        self.assertTrue(report['converged'])
        self.assertGreaterEqual(report['minimum_area_ratio'], .2-1e-8)
        self.assertLessEqual(abs(offsets).max(), 2)
        np.testing.assert_allclose(offsets, [-.6, 0, -1.4], atol=1e-8)
        safe = np.array([.1,.1,.1])
        actual, report = constrain_offsets(frame, triangles, [0,1,0], safe)
        np.testing.assert_array_equal(actual, safe)

    def test_incomplete_projection_is_not_reported_as_converged(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        _, report = constrain_offsets(frame, np.array([[0,1,2]]), [0,1,0],
                                      [0,0,-2], max_sweeps=1)
        self.assertFalse(report['converged'])

    def test_bad_direction_rejected(self):
        with self.assertRaises(ValueError):
            area_constraints(np.zeros((1,3,3)), np.array([[0,1,2]]), [0,2,0])

    def test_multiple_poses_enforce_stricter_triangle(self):
        frames = np.array([[[0,0,0], [.001,0,0], [0,.001,0]],
                           [[0,0,0], [.001,0,0], [0,.0002,0]]])
        offsets, report = constrain_offsets(frames, np.array([[0,1,2]]), [0,1,0], [0,0,-2])
        self.assertTrue(report['converged'])
        np.testing.assert_allclose(offsets, [-.92,0,-1.08], atol=1e-8)

    def test_degenerate_only_reference_rejected(self):
        with self.assertRaises(ValueError):
            area_constraints(np.zeros((1,3,3)), np.array([[0,1,2]]), [0,1,0])

    def test_barycentric_attachment_stays_fixed(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        ids = np.array([[0,1,2]])
        weights = np.array([[.25,.25,.5]])
        offsets, report = constrain_offsets(frame, ids, [0,1,0], [0,0,-2],
                                            fixed_attachments=(ids, weights))
        self.assertTrue(report['converged'])
        self.assertLess(abs((weights@offsets).item()), 1e-8)
        self.assertGreaterEqual(report['minimum_area_ratio'], .2-1e-8)

    def test_invalid_attachment_rejected(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        with self.assertRaises(ValueError):
            constrain_offsets(frame, np.array([[0,1,2]]), [0,1,0], [0,0,-2],
                              fixed_attachments=(np.array([[0,1,3]]), np.array([[0,0,1]])))

    def test_edge_stretch_bound_across_poses(self):
        frames = np.array([[[0,0,0], [.001,0,0], [0,.001,0]],
                           [[0,0,0], [.001,0,0], [0,.0008,0]]])
        offsets, report = constrain_offsets(frames, np.array([[0,1,2]]), [0,1,0],
                                            [0,0,2], maximum_edge_ratio=1.25)
        self.assertTrue(report['converged'])
        self.assertLessEqual(report['maximum_edge_ratio'], 1.25+1e-7)
        self.assertGreaterEqual(report['minimum_area_ratio'], .2-1e-8)
        for frame in frames:
            changed = frame+offsets[:,None]*np.array([0,.001,0])
            for a,b in ((0,1),(1,2),(2,0)):
                self.assertLessEqual(np.linalg.norm(changed[a]-changed[b]),
                                     1.25*np.linalg.norm(frame[a]-frame[b])+1e-10)

    def test_invalid_edge_bound_rejected(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        with self.assertRaises(ValueError):
            constrain_offsets(frame, np.array([[0,1,2]]), [0,1,0], [0,0,2], maximum_edge_ratio=1)

    def test_dependent_attachments_project_as_one_subspace(self):
        frame = np.array([[[0,0,0], [.001,0,0], [0,.001,0]]])
        ids = np.array([[0,1,2]])
        weights = np.array([[.25,.25,.5]])
        one, _ = constrain_offsets(frame, ids, [0,1,0], [0,0,2],
                                   fixed_attachments=(ids,weights), maximum_edge_ratio=1.25)
        duplicate, report = constrain_offsets(frame, ids, [0,1,0], [0,0,2],
            fixed_attachments=(np.repeat(ids,2,axis=0),np.repeat(weights,2,axis=0)), maximum_edge_ratio=1.25)
        self.assertTrue(report['converged'])
        np.testing.assert_allclose(duplicate, one, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
