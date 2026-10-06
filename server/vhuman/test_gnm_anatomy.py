"""Complete anatomy, differentiability and fixed attachment regression gates."""
import unittest
import numpy as np
from .face_assets import asset_path
from .rig.gnm_model import GNMModel
from .reconstruction.fitting import attached_point


class AttachmentTests(unittest.TestCase):
    def test_barycentric_attachment_is_translation_equivariant(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        anchor={'p':dict(barycentric=[.1,.3,.6])}
        row=('p',np.arange(3),np.zeros(2),1.)
        np.testing.assert_allclose(attached_point(p,row,anchor),[.3,.6,0])
        np.testing.assert_allclose(attached_point(p+[4,5,6],row,anchor),[4.3,5.6,6])
        with self.assertRaises(ValueError):
            attached_point(p,row,{'p':dict(barycentric=[1,1,1])})


class GNMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not asset_path('gnm').is_file():
            raise unittest.SkipTest('GNM assets not installed')
        cls.model=GNMModel()

    def test_complete_neutral_and_translation(self):
        m=self.model;p,j=m.evaluate()
        self.assertEqual(p.shape,(17821,3))
        np.testing.assert_allclose(p,m.data['template_vertex_positions'],atol=1e-7)
        moved,k=m.evaluate(translation=[.1,-.2,.3])
        np.testing.assert_allclose(moved-p,np.tile([.1,-.2,.3],(len(p),1)),atol=1e-7)
        np.testing.assert_allclose(k-j,np.tile([.1,-.2,.3],(len(j),1)),atol=1e-7)
        for group in ('skin','left_eye','right_eye','upper_teeth_and_gums','lower_teeth_and_gums','tongue'):
            self.assertGreater(m.group(group).sum(),0)

    def test_native_expression_refinement_matches_complete_forward(self):
        m=self.model;rng=np.random.default_rng(31)
        beta=rng.normal(0,.1,m.identity_dim);expression=rng.normal(0,.05,m.expression_dim)
        rest,_=m.evaluate(beta);actual,_=m.evaluate(beta,expression)
        linear=rest+np.einsum('e,evc->vc',expression,m.data['expression_basis'])
        np.testing.assert_allclose(actual,linear,atol=1e-7)
        # The upper dental arch stays fixed to the head; the lower arch and
        # tongue have native expression deformation.
        np.testing.assert_allclose(actual[m.group('upper_teeth_and_gums')],rest[m.group('upper_teeth_and_gums')],atol=1e-7)
        for group in ('lower_teeth_and_gums','tongue'):
            self.assertGreater(np.linalg.norm((actual-rest)[m.group(group)]),0)

    def test_numpy_torch_parity_and_gradient(self):
        import torch
        m=self.model;t=GNMModel(device='cpu');rng=np.random.default_rng(19)
        beta=rng.normal(0,.2,m.identity_dim);expr=rng.normal(0,.1,m.expression_dim)
        rotation=rng.normal(0,.15,(4,3))
        p,j=m.evaluate(beta,expr,rotation,[.03,0,0])
        coefficients=torch.tensor(beta,dtype=torch.float32,requires_grad=True)
        q,k=t.evaluate(coefficients,expr,rotation,[.03,0,0])
        self.assertLess(np.sqrt(np.mean((q.detach().numpy()-p)**2)),1e-5)
        np.testing.assert_allclose(k.detach().numpy(),j,atol=1e-6)
        q.square().mean().backward()
        self.assertTrue(torch.isfinite(coefficients.grad).all())
        self.assertGreater(float(coefficients.grad.abs().sum()),0)

    def test_fixed_dense_mapping_remains_anatomical(self):
        from .reconstruction.dense_landmarks import attachments,MP68
        ids,weights,confidence=attachments()
        self.assertEqual(ids.shape,(468,3))
        np.testing.assert_allclose(weights.sum(1),1)
        np.testing.assert_allclose(confidence[MP68],1)
        self.assertTrue((weights>=0).all())

    def test_batched_sampled_anatomy_matches_full_equations(self):
        import torch
        rng=np.random.default_rng(7);m=GNMModel(device='cpu')
        ids=np.arange(0,17821,97);beta=rng.normal(0,.2,253)
        expr=torch.tensor(rng.normal(0,.1,(2,383)),dtype=torch.float32,requires_grad=True)
        rotations=torch.tensor(rng.normal(0,.1,(2,4,3)),dtype=torch.float32,requires_grad=True)
        translation=torch.tensor([[.01,0,0],[0,.02,0]],dtype=torch.float32)
        p,j=m.frame_evaluator(beta,ids)(expr,rotations,translation)
        for i in range(2):
            ref,k=self.model.evaluate(beta,expr[i].detach().numpy(),rotations[i].detach().numpy(),translation[i].numpy())
            np.testing.assert_allclose(p[i].detach().numpy(),ref[ids],atol=1e-6)
            np.testing.assert_allclose(j[i].detach().numpy(),k,atol=1e-6)
        p.square().mean().backward()
        self.assertTrue(torch.isfinite(expr.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())

    def test_bad_coefficients_are_rejected(self):
        with self.assertRaises(ValueError):self.model.evaluate(identity=np.zeros(170))
        with self.assertRaises(ValueError):self.model.evaluate(expression=np.full(383,np.nan))
