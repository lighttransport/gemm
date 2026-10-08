import unittest
import numpy as np
from .reconstruction.component_spacing import separate_components,extend_offsets,repair_free_areas
from .reconstruction.mesh_crossings import crossing_pairs


class ComponentSpacingTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[0,0,0],[2,0,0],[0,2,0],
                                [.5,.5,-1],[.5,.5,1],[1.2,.5,.3]],float)*.001
        self.triangles=np.arange(6).reshape(2,3)

    def test_spacing_preserves_component_shape(self):
        original=self.vertices.copy()
        delta,report=separate_components(original,self.triangles,maximum_shift_mm=5)
        self.assertTrue(report['accepted'])
        self.assertEqual(len(crossing_pairs(original+delta,self.triangles)),0)
        for tri in self.triangles:
            np.testing.assert_allclose(delta[tri],np.repeat(delta[tri[:1]],3,axis=0),atol=1e-12)
        np.testing.assert_array_equal(original,self.vertices)

    def test_limit_rejects_proposal(self):
        _,report=separate_components(self.vertices,self.triangles,maximum_shift_mm=.001)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['reason'],'translation limit exceeded')

    def test_separated_components_stay_unchanged(self):
        vertices=self.vertices.copy();vertices[3:]+=np.array([.01,0,0])
        delta,report=separate_components(vertices,self.triangles)
        self.assertTrue(report['accepted'])
        np.testing.assert_array_equal(delta,np.zeros_like(delta))

    def test_extension_preserves_fixed_offsets(self):
        vertices=self.vertices[:3]
        offsets=np.zeros_like(vertices);offsets[0]=[.0001,0,0]
        result=extend_offsets(vertices,np.array([[0,1,2]]),np.array([0]),offsets)
        np.testing.assert_array_equal(result[0],offsets[0])
        self.assertTrue((result[1:,0]>0).all())
        self.assertTrue((result[1:,0]<offsets[0,0]).all())

    def test_local_area_repair_keeps_fixed_vertices(self):
        frame=np.array([[[0,0,0],[.001,0,0],[0,.001,0]]])
        offsets=np.array([[0,0,0],[0,0,0],[0,-.00095,0]])
        result,report=repair_free_areas(frame,np.array([[0,1,2]]),offsets,
            np.array([False,False,True]),minimum_ratio=.2,maximum_repair_mm=.2)
        self.assertTrue(report['accepted'])
        self.assertGreaterEqual(report['minimum_area_ratio'],.2-1e-8)
        np.testing.assert_array_equal(result[:2],offsets[:2])
        _,limited=repair_free_areas(frame,np.array([[0,1,2]]),offsets,
            np.array([False,False,True]),minimum_ratio=.2,maximum_repair_mm=.01)
        self.assertFalse(limited['accepted'])


if __name__=='__main__':unittest.main()
