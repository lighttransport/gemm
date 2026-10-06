"""Synthetic completion protects observations and rejects unstable video evidence."""
import unittest
import json
import numpy as np
from .reconstruction.generated_skin import (band_detail,unseen_weight,triplanar_detail,
    temporal_consistency,render_plate,orbit_camera,apply_detail)


class GeneratedSkinTests(unittest.TestCase):
    def test_final_bake_keeps_photographed_bytes_exactly(self):
        base=np.random.default_rng(12).integers(0,256,(8,8,3),dtype=np.uint8)
        valid=np.ones((8,8),bool);valid[0]=False
        observed=np.zeros((8,8),bool);observed[:,2:4]=True
        result=apply_detail(base,valid,np.full((valid.sum(),3),.02),observed)
        np.testing.assert_array_equal(result[observed],base[observed])
        np.testing.assert_array_equal(result[~valid],base[~valid])
        self.assertTrue(np.any(result[valid&~observed]!=base[valid&~observed]))

    def test_world_space_feather_protects_observed_and_nearby_skin(self):
        points=np.array([[0.,0,0],[.001,0,0],[.003,0,0],[.007,0,0]])
        blend=unseen_weight(points,[True,False,False,False])
        self.assertEqual(blend[0],0);self.assertEqual(blend[-1],1)
        self.assertTrue((np.diff(blend)>0).all())
        with self.assertRaises(ValueError):unseen_weight(points,[False]*4)

    def test_detail_removes_constant_light_and_bounds_contrast(self):
        np.testing.assert_allclose(band_detail(np.full((64,64,3),.6)),0,atol=1e-12)
        image=np.random.default_rng(5).random((64,64,3))
        self.assertLessEqual(abs(band_detail(image)).max(),.025)

    def test_duplicate_uv_vertices_receive_same_spatial_detail(self):
        plate=np.random.default_rng(7).random((64,64,3))
        points=np.array([[.04,.03,-.02],[.04,.03,-.02]])
        normals=np.array([[0.,0,1],[0,0,1]])
        result=triplanar_detail(points,normals,plate)
        np.testing.assert_array_equal(result[0],result[1])
        self.assertLessEqual(abs(result).max(),.012+1e-12)

    def test_i2v_agreement_does_not_accept_photometric_drift(self):
        rng=np.random.default_rng(9)
        image=rng.integers(40,120,(64,64,3),dtype=np.uint8)
        stable,report=temporal_consistency(image,[image]*3)
        _,_,consensus=temporal_consistency(image,[image]*3,return_consensus=True)
        # Dense flow interpolation can round a few static edge pixels by 1 LSB.
        self.assertLessEqual(abs(consensus.astype(int)-image.astype(int)).max(),1)
        self.assertGreater((stable>0).mean(),.9)
        drift,_=temporal_consistency(image,[image+100]*3)
        self.assertLess((drift>0).mean(),.01)
        self.assertFalse(report['proves_observed_accuracy'])
        with self.assertRaises(ValueError):temporal_consistency(image,[image])

    def test_render_mask_excludes_photographed_surface_and_silhouette(self):
        points=np.array([[-.1,-.1,0],[.1,-.1,0],[0,.1,0]])
        geometry=dict(captured=points[None],triangles=np.array([[0,1,2]]),
                      triangle_uvs=np.array([[[0,0],[1,0],[.5,1]]]))
        camera=orbit_camera(points,0,64)
        json.dumps(orbit_camera(points.astype(np.float32),0,64).as_dict())
        _,mask,depth=render_plate(geometry,np.full((32,32,3),.3),np.zeros((32,32)),camera,64)
        self.assertFalse(mask.any());self.assertTrue(np.isfinite(depth).any())
        _,mask,depth=render_plate(geometry,np.full((32,32,3),.3),np.ones((32,32)),camera,64)
        self.assertTrue(mask.any());self.assertFalse(mask[~np.isfinite(depth)].any())


if __name__=='__main__':unittest.main()
