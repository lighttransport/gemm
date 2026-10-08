import unittest
import numpy as np
from .reconstruction.mesh_crossings import strict_crossings, crossing_pairs, crossing_points


class MeshCrossingTests(unittest.TestCase):
    def setUp(self):
        self.flat = np.array([[0.,0,0], [2,0,0], [0,2,0]])*.001
        self.through = np.array([[.5,.5,-1], [.5,.5,1], [1.5,.5,0]])*.001

    def test_strict_crossing_and_symmetry(self):
        self.assertTrue(strict_crossings(self.flat[None],self.through[None])[0])
        self.assertTrue(strict_crossings(self.through[None],self.flat[None])[0])

    def test_separated_coplanar_and_touch_are_excluded(self):
        separated = self.through+np.array([.01,0,0])
        touch = self.through.copy();touch[:,2]=np.maximum(touch[:,2],0)
        for other in (separated,self.flat,touch):
            self.assertFalse(strict_crossings(self.flat[None],other[None])[0])

    def test_broadphase_matches_brute_force(self):
        rng = np.random.default_rng(33)
        points = rng.normal(size=(30,3,3))*.002
        vertices = points.reshape(-1,3)
        triangles = np.arange(len(vertices)).reshape(-1,3)
        expected = {(a,b) for a in range(len(points)) for b in range(a+1,len(points))
                    if strict_crossings(points[a:a+1],points[b:b+1])[0]}
        actual = set(map(tuple,crossing_pairs(vertices,triangles,batch_size=7)))
        self.assertEqual(actual,expected)
        self.assertTrue(actual)

    def test_empty_and_invalid_mesh(self):
        self.assertEqual(crossing_pairs(self.flat,np.empty((0,3),int)).shape,(0,2))
        with self.assertRaises(ValueError):
            crossing_pairs(self.flat,np.array([[0,1,3]]))

    def test_degenerate_segment_is_not_a_surface_crossing(self):
        line = np.array([[.5,.5,-1], [.5,.5,0], [.5,.5,1]])*.001
        self.assertFalse(strict_crossings(self.flat[None],line[None])[0])

    def test_intersection_points_lie_on_both_planes(self):
        through=self.through.copy();through[2]=[.0012,.0005,.0003]
        points,mask=crossing_points(self.flat[None],through[None])
        selected=points[mask]
        self.assertGreaterEqual(len(selected),2)
        np.testing.assert_allclose(selected[:,2],0,atol=1e-12)
        np.testing.assert_allclose(selected[:,1],.0005,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
