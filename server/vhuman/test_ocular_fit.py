"""Bounded, fixed-centre optical eye reprojection fitting."""
import unittest
import numpy as np
from .reconstruction.ocular_fit import fit_eye
from .reconstruction.reference import Camera


class OcularFitTests(unittest.TestCase):
    def setUp(self):
        self.camera=Camera(1200.,180.,200.,np.array([0.,0.,1.]),np.eye(3))

    def test_offset_iris_fits_without_moving_globe(self):
        centre=np.array([.03,0.,0.]);pixel,_=self.camera.project((centre+[0,0,.012])[None])
        observation=dict(center=(pixel[0]+[.4,2.4]).tolist(),radius=1200*.006/.988)
        result=fit_eye(self.camera,centre,.012,.006,observation)
        self.assertTrue(result['accepted']);self.assertTrue(result['globe_center_fixed'])
        self.assertLess(result['center_error_after_px'],.01)
        r=np.asarray(result['rotation']);np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-10)
        self.assertLessEqual(np.linalg.norm(result['rotation_degrees']),20)
        self.assertTrue(.9<=result['scale']<=1.1)

    def test_unreachable_detection_is_not_promoted(self):
        result=fit_eye(self.camera,np.zeros(3),.012,.006,dict(center=[300,300],radius=8))
        self.assertFalse(result['accepted'])

    def test_bad_eye_observations_fail(self):
        for obs in (dict(center=[float('nan'),2],radius=8),dict(center=[180,200],radius=-1)):
            with self.assertRaises(ValueError):fit_eye(self.camera,np.zeros(3),.012,.006,obs)


if __name__=='__main__':unittest.main()
