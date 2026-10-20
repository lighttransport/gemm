"""CPU source oral geometry, fixed anatomy and bind-vector transport."""
import importlib.util
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .rig import oral_fit,geometry_fit,biomechanics_mapping
from .rig.biomechanics_native import LIBRARY
from .test_geometry_fit import ocular_fixture

HAS_SCIPY=importlib.util.find_spec('scipy') is not None


def lip_fixture():
    vertices=np.array([[-.002,-.0001,0],[.002,-.0001,0],[0,.001,0],
                       [-.001,-.0002,-.0001],[.001,.0002,.0001],[0,.0009,.0001]])
    faces=np.array([[0,1,2],[3,4,5]],np.int32)
    upper=np.arange(6)<3
    return vertices,faces,upper,~upper


class SkinVectorTests(unittest.TestCase):
    @unittest.skipUnless(LIBRARY.is_file() and HAS_SCIPY,'native CPU goal fitting')
    def test_closure_target_uses_surface_distance_for_staggered_and_tilted_lips(self):
        from .rig import articulation
        from .rig.biomechanics import lip_surface_aperture
        upper=np.array([[0,.0003,0],[-.01,.0003,-.005],[.01,.0003,-.005],
                        [.01,.0003,.005],[-.01,.0003,.005]])
        lower=np.array([[0,0,.002],[-.012,0,-.006],[.012,0,-.006],
                        [.012,0,.006],[-.012,0,.006]])
        rest=np.concatenate((upper,lower));mask=np.arange(10)<5
        f=np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],np.int32);faces=np.concatenate((f,f+5))
        model=SimpleNamespace(group=lambda name:mask if name=='upper_lip' else ~mask if name=='lower_lip' else np.zeros(10,bool))
        values=np.zeros((1,len(articulation.CHANNELS)));values[0,articulation.CHANNELS.index('seal')]=1
        for rotation in (np.eye(3),np.array([[1,0,0],[0,0,-1],[0,1,0]])):
            posed=rest@rotation.T+[.02,.04,-.03]
            arrays=dict(source_ids=np.arange(10),surface_rest=posed,faces=faces)
            target=articulation.geometric_targets(posed,arrays,model,dict(values=values),0)
            gap=lip_surface_aperture(posed,faces,mask,~mask,[target])[0]
            self.assertGreater(gap,.09);self.assertLessEqual(gap,.2)
            partial=values.copy();partial[0,articulation.CHANNELS.index('seal')]=.4
            partial_target=articulation.geometric_targets(posed,arrays,model,dict(values=partial),0)
            partial_gap=lip_surface_aperture(posed,faces,mask,~mask,[partial_target])[0]
            self.assertGreater(partial_gap,.2);self.assertLess(partial_gap,.3)

    def test_surface_aperture_handles_staggered_vertices_and_open_center(self):
        from .rig.biomechanics import lip_surface_aperture,speech_contact_metrics
        upper=np.array([[0,.0001,0],[-.01,.0001,-.005],[.01,.0001,-.005],
                        [.01,.0001,.005],[-.01,.0001,.005]])
        lower=np.array([[0,0,.002],[-.012,0,-.006],[.012,0,-.006],
                        [.012,0,.006],[-.012,0,.006]])
        rest=np.concatenate((upper,lower));mask=np.arange(10)<5
        faces=np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],np.int32)
        faces=np.concatenate((faces,faces+5))
        # No opposing vertices coincide horizontally, while the surfaces are
        # parallel with a known 0.1 mm gap.
        nearest=np.linalg.norm(upper[:,None]-lower[None],axis=2).min(1)
        self.assertGreater(nearest.min()*1000,1.)
        rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        world=rest@rotation.T+[1,2,3]
        gaps=lip_surface_aperture(rest,faces,mask,~mask,[rest,world])
        np.testing.assert_allclose(gaps,[.1,.1],atol=1e-10,rtol=0)
        opened=rest.copy();opened[:5,1]+=.001
        self.assertGreater(lip_surface_aperture(rest,faces,mask,~mask,[opened])[0],1.)
        center_open=rest.copy();center_open[0,1]+=.002
        self.assertGreater(lip_surface_aperture(rest,faces,mask,~mask,[center_open])[0],.2)
        model=SimpleNamespace(group=lambda name:mask if name=='upper_lip' else ~mask)
        arrays=dict(source_ids=np.arange(10),surface_rest=rest,faces=faces)
        profile=dict(times=np.array([0.]),intervals=[dict(phone='m',start=0.,end=.03)])
        with patch('server.vhuman.rig.gnm_model.GNMModel',return_value=model):
            closed=speech_contact_metrics(arrays,profile,[rest],np.eye(4)[None])
            open_report=speech_contact_metrics(arrays,profile,[opened],np.eye(4)[None])
            collapsed=rest.copy();collapsed[mask]=collapsed[0]
            collapsed_report=speech_contact_metrics(arrays,profile,[collapsed],np.eye(4)[None])
        self.assertTrue(closed['passed']);self.assertFalse(open_report['passed'])
        self.assertFalse(collapsed_report['passed'])
        self.assertGreater(closed['events'][0]['vertex_pair_aperture_mm'],.2)
        self.assertAlmostEqual(closed['events'][0]['aperture_mm'],.1)

    def test_weighted_rotation_ignores_joint_translation(self):
        transforms=np.repeat(np.eye(4)[None],2,axis=0)
        transforms[1,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        transforms[:,0,3]=[100,-100]
        joints=np.array([[0,1,0,0]],np.int32);weights=np.array([[.25,.75,0,0]])
        vector=np.array([[1.,0,0]])
        result=biomechanics_mapping.skin_vectors(vector,(joints,weights),transforms)
        np.testing.assert_allclose(result,[[.25,.75,0]],rtol=0,atol=1e-14)
        with self.assertRaisesRegex(ValueError,'correspondence'):
            biomechanics_mapping.skin_vectors(vector,(joints+2,weights),transforms)


@unittest.skipUnless(LIBRARY.is_file() and HAS_SCIPY,'native CPU BVH and SciPy')
class OralFitTests(unittest.TestCase):
    def test_root_fit_preserves_clear_anatomy_and_rejects_a_too_small_bound(self):
        vertices,faces,tongue,arch=lip_fixture()
        model=SimpleNamespace(data={'triangles':faces},group=lambda name:tongue if name=='tongue' else arch)
        parts=[('tongue',tongue,1),('gums_lower',arch,2)]
        clear=vertices.copy();clear[arch,0]+=.02
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=parts):
            result,report=oral_fit.fit_tongue_root(model,clear,iterations=2)
            np.testing.assert_array_equal(result,clear);self.assertFalse(report['applied'])
            self.assertEqual(report['after_root_crossings'],0)
            original=vertices.copy()
            with self.assertRaisesRegex(ValueError,'geometry gates'):
                oral_fit.fit_tongue_root(model,vertices,limit=1e-9,iterations=2)
            np.testing.assert_array_equal(vertices,original)
            cancelled=threading.Event();cancelled.set()
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                oral_fit.fit_tongue_root(model,clear,iterations=2,cancel=cancelled)
            for kwargs in (dict(limit=float('nan')),dict(scale=0),dict(iterations=True)):
                with self.assertRaisesRegex(ValueError,'inputs'):
                    oral_fit.fit_tongue_root(model,clear,**kwargs)

    def test_tongue_fairing_is_bounded_and_keeps_tip_and_other_parts_fixed(self):
        vertices,faces,_,_=lip_fixture()
        vertices=vertices*.1
        vertices[3:,0]+=.00015
        tip=np.array([[0,0,.01],[.0001,0,.0099],[0,.0001,.01]])
        fixed=tip+[.02,0,0]
        vertices=np.concatenate((vertices,tip,fixed))
        triangles=np.concatenate((faces,[[6,7,8],[9,10,11]])).astype(np.int32)
        tongue=np.arange(12)<9
        model=SimpleNamespace(data={'triangles':triangles},group=lambda _:tongue)
        parts=[('tongue',tongue,1),('skin',~tongue,0)]
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=parts):
            fitted,report=oral_fit.fit_tongue(model,vertices)
            repeated,again=oral_fit.fit_tongue(model,fitted)
            with self.assertRaisesRegex(ValueError,'displacement bound'):
                oral_fit.fit_tongue(model,vertices,limit=1e-8)
        self.assertGreater(report['before_tongue_self_crossings'],0)
        self.assertEqual(report['after_tongue_self_crossings'],0)
        self.assertGreaterEqual(report['min_oriented_area_ratio'],.1)
        np.testing.assert_array_equal(fitted[6:],vertices[6:])
        np.testing.assert_array_equal(repeated,fitted)
        self.assertFalse(again['applied'])

    def test_dentition_moves_whole_crowns_and_preserves_other_geometry(self):
        # Two independently indexed, initially intersecting tetrahedral crowns.
        tooth=np.array([[0,0,0],[.002,0,0],[0,.002,0],[0,0,.002]])
        vertices=np.concatenate((tooth,tooth+[.0018,.00005,.00005],tooth+[.01,0,0]))
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        triangles=np.concatenate((faces,faces+4,faces+8))
        mask=np.arange(12)<8
        model=SimpleNamespace(data={'triangles':triangles},group=lambda _:mask)
        parts=[('teeth_upper',mask,-1),('gums_upper',~mask,2)]
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=parts):
            result,report=oral_fit.fit_dentition(model,vertices)
            repeated,again=oral_fit.fit_dentition(model,result)
            with self.assertRaisesRegex(ValueError,'displacement bound'):
                oral_fit.fit_dentition(model,vertices,limit=.000001)
        self.assertGreater(report['before_teeth_crossings'],0)
        self.assertEqual(report['after_teeth_crossings'],0)
        self.assertTrue(report['crown_shapes_preserved'])
        for indices in (np.arange(4),np.arange(4,8)):
            np.testing.assert_allclose(result[indices]-result[indices[0]],
                                       vertices[indices]-vertices[indices[0]],rtol=0,atol=1e-17)
        np.testing.assert_array_equal(result[~mask],vertices[~mask])
        np.testing.assert_array_equal(repeated,result)
        self.assertFalse(again['applied'])
        cancel=threading.Event();cancel.set()
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=parts):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                oral_fit.fit_dentition(model,vertices,cancel=cancel)
        # A thin gingival seam attached to one crown must not flip when the
        # tooth translates, even if it becomes contained without a crossing.
        attached=np.concatenate((vertices,[[-.000001,.000001,0],[-.000001,0,.000001]]))
        attached_faces=np.concatenate((triangles,[[0,12,13]])).astype(np.int32)
        dental=np.r_[mask,False,False];gingiva=~dental;gingiva[0]=True
        attached_model=SimpleNamespace(data={'triangles':attached_faces},group=lambda _:dental)
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=[('teeth',dental,-1),('gums',gingiva,2)]):
            with self.assertRaisesRegex(ValueError,'orientation'):
                oral_fit.fit_dentition(attached_model,attached)

    def test_lip_separation_removes_crossings_without_mutating_source(self):
        vertices,faces,upper,lower=lip_fixture();original=vertices.copy()
        with geometry_fit.CrossingCounter(vertices,faces) as counter:
            self.assertGreater(counter.count(vertices),0)
            fitted,report=oral_fit.separate_lips(vertices,faces,upper,lower)
            self.assertEqual(counter.count(fitted),0)
        self.assertGreater(report['before_crossings'],0)
        self.assertLessEqual(report['max_displacement_mm'],1.5)
        np.testing.assert_array_equal(fitted[:,[0,2]],vertices[:,[0,2]])
        self.assertTrue((fitted[upper,1]>=vertices[upper,1]).all())
        self.assertTrue((fitted[lower,1]<=vertices[lower,1]).all())
        np.testing.assert_array_equal(vertices,original)
        with self.assertRaisesRegex(ValueError,'displacement bound'):
            oral_fit.separate_lips(vertices,faces,upper,lower,limit=.000001)

    def test_crossing_coordinates_reconstruct_both_primitives(self):
        vertices,faces,_,_=lip_fixture()
        with geometry_fit.CrossingCounter(vertices,faces) as counter:
            records=counter.intersections(vertices)
        point,weights=oral_fit.crossing_weights(vertices,records)
        np.testing.assert_allclose((vertices[records[:,:2]]*weights[:,:2,None]).sum(1),point,atol=1e-14,rtol=0)
        np.testing.assert_allclose((vertices[records[:,2:]]*weights[:,2:,None]).sum(1),point,atol=1e-14,rtol=0)

    def test_cancellation_and_bad_masks_stop_before_fit(self):
        vertices,faces,upper,lower=lip_fixture();cancel=threading.Event();cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            oral_fit.separate_lips(vertices,faces,upper,lower,cancel=cancel)
        with self.assertRaisesRegex(ValueError,'inputs'):
            oral_fit.separate_lips(vertices,faces,upper,upper)

    def test_full_fit_preserves_fixed_envelope_and_is_idempotent(self):
        cup,faces,hidden,visible,envelope,envelope_faces=ocular_fixture()
        # A synthetic lining initially crosses a fixed oral envelope. Separate
        # crossing lip triangles exercise the visible stage independently.
        lip,lip_faces,upper,lower=lip_fixture();lip=lip+[.04,0,0]
        vertices=np.concatenate((cup,envelope,lip));nc,ne=len(cup),len(envelope)
        triangles=np.concatenate((faces,envelope_faces+nc,lip_faces+nc+ne)).astype(np.int32)
        skin=np.r_[np.ones(nc,bool),np.zeros(ne,bool),np.ones(6,bool)]
        hidden=np.r_[hidden,np.zeros(ne+6,bool)]
        exterior=skin&~hidden
        groups=dict(skin=skin,mouth_sock=hidden,skin_exterior=exterior,
            upper_lip=np.r_[np.zeros(nc+ne,bool),upper],lower_lip=np.r_[np.zeros(nc+ne,bool),lower],
            tongue=np.r_[np.zeros(nc,bool),np.ones(ne,bool),np.zeros(6,bool)],
            teeth=np.zeros(len(vertices),bool),gums=np.zeros(len(vertices),bool))
        model=SimpleNamespace(data={'triangles':triangles},group=lambda name:groups[name])
        parts=[('skin',skin,0),('tongue',groups['tongue'],1)]
        with patch('server.vhuman.rig.anatomy.part_masks',return_value=parts):
            fitted,report=oral_fit.fit_mouth(model,vertices)
            repeated,again=oral_fit.fit_mouth(model,fitted)
        self.assertTrue(report['accepted']);self.assertTrue(report['applied'])
        self.assertEqual(report['after_lining_oral_crossings'],0)
        self.assertEqual(report['after_skin_self_crossings'],0)
        np.testing.assert_array_equal(fitted[nc:nc+ne],vertices[nc:nc+ne])
        np.testing.assert_array_equal(repeated,fitted)
        self.assertFalse(again['applied'])


if __name__=='__main__':
    unittest.main()
