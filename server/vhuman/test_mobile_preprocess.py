"""Metric/detail training checks independent of portrait and GPU assets."""
import unittest
import numpy as np
from .mobile.preprocess import regional_strain,tangent_slopes,fit_driver,features


class StrainTests(unittest.TestCase):
    def test_rigid_translation_and_known_uniform_area_change(self):
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        triangles=np.array([[0,1,2]]);regions=np.ones((1,3))
        basis=np.stack((vertices,np.ones_like(vertices)))
        result=regional_strain(vertices,basis,triangles,regions,np.array([[.01,0],[0,.2]]))
        np.testing.assert_allclose(result[:,0],[6*(1-1.01**2),0],atol=1e-10)

    def test_metric_tangent_slopes_account_for_uv_scale_and_shear(self):
        res=128;y,x=np.mgrid[:res,:res];u=(x+.5)/res;v=(y+.5)/res
        geometry=dict(neutral=np.array([[0.,0,0],[.2,0,0],[.05,.1,0]]),
            triangles=np.array([[0,1,2]]),triangle_uvs=np.array([[[0.,0],[1,0],[0,1]]]))
        field=.02*(.2*u+.05*v)
        slopes=tangent_slopes(geometry,field[None],res)
        np.testing.assert_allclose(slopes[0,20:40,20:40,0],.02,atol=2e-6)
        np.testing.assert_allclose(slopes[0,20:40,20:40,1],0,atol=2e-6)
        flat=tangent_slopes(geometry,np.zeros((1,res,res)),res)
        np.testing.assert_array_equal(flat,0)

    def test_heldout_driver_fit_and_exact_zero_reference(self):
        rng=np.random.default_rng(24);prior=np.array([[.5,.2,-.1],[.1,-.3,.7]])
        train=rng.normal(0,.15,(512,3));test=rng.normal(0,.15,(128,3))
        truth=np.array([[.7,.2],[.1,-.4],[-.3,.5],[.1,.2],[.2,-.1]])
        target=features(train,prior)@truth;expected=features(test,prior)@truth
        weights,scores=fit_driver(train,target,test,expected,prior)
        self.assertTrue(scores['accepted']);self.assertLess(scores['trained']['p95'],.005)
        np.testing.assert_array_equal(features(np.zeros((1,3)),prior)@weights,0)


if __name__=='__main__':unittest.main()
