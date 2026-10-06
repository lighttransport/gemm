"""Native portrait refinement: withheld-evidence and anatomy safeguards."""
import unittest
import numpy as np
from .reconstruction.refine_fit import acceptance_gate,landmark_metrics,surface_basis,surface_displacement


class NativeFitTests(unittest.TestCase):
    def test_gate_rejects_training_only_gain_folds_and_nonmouth_regression(self):
        before=dict(all_px=4.,heldout_px=4.,mouth_px=6.,heldout_mouth_px=7.,nonmouth_px=3.)
        after=dict(all_px=2.,heldout_px=2.6,mouth_px=1.9,heldout_mouth_px=2.5,nonmouth_px=2.3)
        self.assertTrue(acceptance_gate(before,after,.16))
        for bad in (dict(heldout_px=4.1),dict(heldout_mouth_px=7.1),dict(nonmouth_px=3.4),dict(mouth_px=float('nan'))):
            self.assertFalse(acceptance_gate(before,dict(after,**bad),.16))
        self.assertFalse(acceptance_gate(before,after,.01))
        self.assertFalse(acceptance_gate(before,after,.16,target_px=1.5))
        self.assertTrue(acceptance_gate(before,dict(after,heldout_px=1.49),.16,target_px=1.5))
        self.assertFalse(acceptance_gate(before,dict(after,heldout_px=1.5),.16,target_px=1.5))

    def test_surface_correction_bound_depth_and_gradient(self):
        import torch
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(24)
        vertices=rng.normal(0,.02,(80,3));points=vertices[:50]
        basis=surface_basis(vertices,points,count=24)
        # Changing the coordinate origin cannot alter a metric fit field.
        np.testing.assert_allclose(basis,surface_basis(vertices+2,points+2,count=24),atol=1e-7)
        far=surface_basis(np.array([[1.,1.,1.]]),points,count=24)
        self.assertLess(float(far.sum()),1e-6)
        latent=torch.tensor(rng.normal(0,100,(24,2)),dtype=torch.float64,requires_grad=True)
        rotation=torch.tensor(Rotation.from_rotvec([.4,-.2,.1]).as_matrix())
        correction=surface_displacement(torch.tensor(basis,dtype=torch.float64),latent,rotation,5.)
        self.assertLessEqual(float(correction.detach().norm(dim=-1).max()),.005)
        np.testing.assert_allclose((correction@rotation.T)[:,2].detach(),0,atol=1e-12)
        correction.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(latent.grad).all()))
        self.assertGreater(float(latent.grad.norm()),0)

    def test_surface_field_rejects_invalid_metric_inputs(self):
        points=np.zeros((4,3))
        for width in (0,-1,float('nan')):
            with self.assertRaises(ValueError):surface_basis(points,points,width=width)
        with self.assertRaises(ValueError):surface_basis(points,np.empty((0,3)))

    def test_heldout_metric_does_not_count_training_fit(self):
        target=np.zeros((468,2));predicted=np.zeros_like(target)
        weights=np.ones(468);heldout=np.arange(468)%10==0
        predicted[heldout,0]=3.;predicted[~heldout,0]=100.
        report=landmark_metrics(predicted,target,weights,heldout)
        self.assertAlmostEqual(report['heldout_px'],3.)
        self.assertAlmostEqual(report['heldout_mouth_px'],3.)
        self.assertGreater(report['all_px'],80.)


if __name__=='__main__':unittest.main()
