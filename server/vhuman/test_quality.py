"""Independent numerical gates for scan, motion, materials and optional cues."""
import json
from pathlib import Path
import tempfile
import unittest
from importlib.util import find_spec
import numpy as np
from .reconstruction.reference import Camera
from .reconstruction.scan import closest_surface,similarity,ray_hit
from .reconstruction.sequence import predict
from .reconstruction.materials import fuse_samples
from .reconstruction.detail import field

ROOT=Path(__file__).resolve().parents[2]/'tmp/test-quality'
ROOT.mkdir(parents=True,exist_ok=True)


class QualityTests(unittest.TestCase):
    def test_area_sampled_bidirectional_detects_missing_geometry(self):
        from .reconstruction.benchmark import area_samples,bidirectional
        v=np.array([[0.,0.,0.],[.03,0,0],[0,.03,0],[.03,.03,0]])
        t=np.array([[0,1,2],[1,3,2]])
        np.testing.assert_allclose(area_samples(v,t,50),area_samples(v,t,50))
        ref=v+[0,0,.002]
        # ROI is deliberately wider than both surfaces, fixed independently of target.
        roi=np.array([[-.01,-.01,-.01],[.04,.04,.01]])
        row=bidirectional(v,t,ref,t,roi,100)
        self.assertAlmostEqual(row['symmetric_rms_mm'],2.,places=8)
        with self.assertRaises(ValueError):area_samples(v,np.array([[0,0,0]]))

    def test_effective_scattering_profile_uses_spatial_measurements(self):
        from .reconstruction.calibration import scattering_profile
        distance=np.linspace(-.006,.006,121);radii=np.array([.002,.001,.0007])
        rgb=np.exp(-.5*(distance[:,None]/radii)**2)*[.8,.6,.5]
        result=scattering_profile(distance,rgb)
        np.testing.assert_allclose(result['radii_m'],radii,rtol=1e-5)
        self.assertFalse(result['enabled']);self.assertTrue(all(c['accepted'] for c in result['channels']))
        with self.assertRaises(ValueError):scattering_profile(distance[::-1],rgb)

    def test_gray_card_recovers_radiance_and_rejects_contamination(self):
        from .reconstruction.calibration import gray_card
        rho=np.array([.18,.19,.17]);light=np.array([2.,1.5,1.2]);ambient=np.array([.03,.03,.03])
        direction=np.array([.3,0,np.sqrt(.91)]);exposure=1.3
        pixels=np.tile(rho/np.pi*(light*direction[2]+ambient)*exposure,(100,1))
        result=gray_card(pixels,rho,[0,0,1],direction,exposure,ambient)
        np.testing.assert_allclose(result['radiance'],light,atol=1e-12)
        noisy=pixels.copy();noisy[:50]*=2
        with self.assertRaises(ValueError):gray_card(noisy,rho,[0,0,1],direction,exposure,ambient)
        with self.assertRaises(ValueError):gray_card(pixels,rho,[0,0,1],[1,0,0])

    def test_completion_confidence_rejects_opposite_skin_sheet(self):
        from .reconstruction.texture_completion import harmonic
        points=np.array([[0.,0.,0.],[.001,0,0],[.0005,0,.0001]])
        normals=np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,-1.]])
        colors=np.array([[.2,.3,.4],[.4,.5,.6],[0.,0.,0.]])
        result,confidence,_=harmonic(points,normals,colors,[True,True,False])
        self.assertEqual(confidence[-1],0);np.testing.assert_allclose(result[-1],[.3,.4,.5])

    def test_harmonic_completion_preserves_evidence_and_far_fallback(self):
        from .reconstruction.texture_completion import harmonic
        p=np.array([[i*.001,0,0] for i in range(7)]+[[1.,0,0]])
        n=np.tile([0.,0,1.],(8,1));color=np.zeros((8,3));color[0]=[.2,.3,.4];color[6]=[.4,.5,.6]
        measured=np.array([True,False,False,False,False,False,True,False])
        completed,confidence,report=harmonic(p,n,color,measured)
        np.testing.assert_array_equal(completed[measured],color[measured])
        np.testing.assert_allclose(completed[-1],[.3,.4,.5]);self.assertEqual(confidence[-1],0)
        self.assertFalse(report['completion_is_observation']);self.assertEqual(report['harmonically_completed_samples'],5)
        self.assertTrue((completed[1:6]>=.2).all());self.assertTrue((completed[1:6]<=.6).all())

    def test_occlusion_heuristic_preserves_lips_and_labels_masks(self):
        from .reconstruction.occlusion import estimate
        image=np.tile(np.array([160,100,80],np.uint8),(128,128,1))
        anchors={name:dict(xy=xy) for name,xy in [('eye_right',[36,36]),('eye_left',[92,36]),
                  ('nose_tip',[64,58]),('upper_lip',[64,79]),('lower_lip',[64,85])]}
        image[42:49,30:41]=[0,0,0]
        image[78:86,58:70]=[200,30,40]
        mask,report=estimate(image,anchors,np.ones((128,128),bool))
        self.assertTrue(mask[45,35]);self.assertFalse(mask[82,64])
        self.assertFalse(report['independent_ground_truth'])

    def test_fixed_calibration_and_shared_multiview_expression(self):
        from types import SimpleNamespace
        from .reconstruction.fitting import fit
        vertices=np.array([[-.02,-.02,0],[.02,-.02,0],[.02,.02,0],[-.02,.02,0]])
        triangles=np.array([[0,1,2],[0,2,3]])
        delta=np.zeros((1,4,3));delta[:,:,2]=.001
        source=SimpleNamespace(vertices=vertices,triangles=triangles,identity_basis=None,
                               expression_basis=delta,eye_centers=None)
        views=[];cameras=[]
        for cx in [32.,40.]:
            camera=Camera(1000,cx,32.,np.array([0.,0.,.5]),np.eye(3));cameras.append(camera)
            xy,_=camera.project(vertices)
            views.append(dict(camera=camera.as_dict(),anchors={str(i):dict(vertex=i,xy=p.tolist()) for i,p in enumerate(xy)}))
        _,captured,fitted,report=fit(source,vertices,views,freeze_pose=True,expression_groups=['frame','frame'],iterations=10)
        self.assertEqual(report['pose_calibration'],'fixed')
        np.testing.assert_allclose(captured[0],captured[1])
        for a,b in zip(cameras,fitted):self.assertEqual(a.as_dict(),b.as_dict())

    def test_artifacts_cannot_be_written_into_tracked_source(self):
        from .reconstruction.artifacts import artifact_path
        with self.assertRaises(ValueError):artifact_path(Path(__file__).parent/'learned_weights.safetensors')
        self.assertEqual(artifact_path(ROOT/'fixture.npz'),ROOT/'fixture.npz')

    def test_exact_triangle_distance_not_vertex_distance(self):
        v=np.array([[0.,0.,0.],[10.,0.,0.],[0.,10.,0.]])
        q,ids,bary,distance=closest_surface(np.array([[2.,2.,1.],[-1.,0.,0.],[6.,6.,0.]]),v,np.array([[0,1,2]]))
        np.testing.assert_allclose(q,[[2,2,0],[0,0,0],[5,5,0]])
        np.testing.assert_allclose(distance,[1,1,np.sqrt(2)])
        np.testing.assert_allclose(bary.sum(1),1.)
        np.testing.assert_allclose((v[[[0,1,2]]][ids]*bary[...,None]).sum(1),q)

    def test_similarity_recovers_metric_rotation_and_translation(self):
        from scipy.spatial.transform import Rotation
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        r=Rotation.from_euler('xyz',[.1,.4,-.2]).as_matrix();target=1.2*points@r.T+[.02,.03,.04]
        scale,rotation,translation=similarity(points,target)
        np.testing.assert_allclose(scale*points@rotation.T+translation,target,atol=1e-12)
        with self.assertRaises(ValueError):similarity(np.ones((4,3)),np.ones((4,3)))

    def test_ray_intersection_frontmost(self):
        v=np.array([[-1,-1,0],[1,-1,0],[0,1,0],[-1,-1,-1],[1,-1,-1],[0,1,-1]],float)
        camera=Camera(100,50,50,np.array([0.,0.,1.]),np.eye(3))
        point,tid,bary=ray_hit(v,np.array([[0,1,2],[3,4,5]]),camera,[50,50])
        np.testing.assert_allclose(point,[0,0,0]);self.assertEqual(tid,0);self.assertAlmostEqual(bary.sum(),1.)

    def test_motion_prediction_reads_training_only_and_rejects_extrapolation(self):
        from .reconstruction.observations import sha256
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);v=np.array([[-.1,-.1,0],[.1,-.1,0],[0,.1,0]],np.float32);tri=np.array([[0,1,2]])
            capture=np.array([v,v+np.array([0,0,.02])])
            np.savez(root/'geometry.npz',neutral=v,triangles=tri,captured=capture)
            camera=Camera(100,50,50,np.array([0.,0.,1.]),np.eye(3))
            fitted=Camera(100,50,50,np.array([-.02,0.,1.]),np.eye(3))
            (root/'observations.json').write_text(json.dumps(dict(views=[dict(timestamp_s=t,camera=camera.as_dict()) for t in [0,1]])))
            (root/'manifest.json').write_text(json.dumps(dict(geometry_sha256=sha256(root/'geometry.npz'),geometry=dict(fitted_cameras=[camera.as_dict(),fitted.as_dict()]))))
            report=predict(root,[.5],root/'prediction.npz')
            with np.load(root/'prediction.npz') as z:np.testing.assert_allclose(z['positions'][0],v+[.01,0,.01],atol=1e-8)
            self.assertFalse(report['target_annotations_used'])
            with self.assertRaisesRegex(ValueError,'outside'):predict(root,[2.],root/'bad.npz')

    def test_color_outlier_and_zero_confidence(self):
        colors=np.array([[[.3,.2,.1]],[[.31,.21,.11]],[[1.,1.,1.]]]);confidence=np.ones((3,1))
        color,weight,report=fuse_samples(colors,confidence)
        self.assertLess(np.linalg.norm(color[0]-[.3,.2,.1]),.08)
        self.assertGreater(report['downweighted_observations'],0)
        color,weight,_=fuse_samples(colors,np.zeros_like(confidence))
        np.testing.assert_allclose(color,0);np.testing.assert_allclose(weight,0)

    def test_metric_detail_invariant_to_uv_tangent_basis(self):
        p=np.array([[.01,.02,.03]]);n=np.array([[0.,0.,1.]])
        t=np.array([[1.,0.,0.]]);b=np.array([[0.,1.,0.]])
        a=field(p,n,t,b,np.array([.003]),5.)
        c=field(p,n,b,-t,np.array([.003]),5.)
        np.testing.assert_allclose(a[:,0,None]*t+a[:,1,None]*b+a[:,2,None]*n,
                                   c[:,0,None]*b-c[:,1,None]*t+c[:,2,None]*n,atol=1e-12)
        np.testing.assert_allclose(field(p,n,t,b,np.array([.003]),0),[[0,0,1]])
        self.assertAlmostEqual(np.linalg.norm(a),1.)

    def test_relative_obj_indices_resolve_when_each_face_is_read(self):
        from .rig.face_models import _obj
        from .reconstruction.emily import skin_mesh
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p=Path(directory)/'relative.obj'
            p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl Skin_Blend_01\nf -3/-3 -2/-2 -1/-1\nv 99 99 99\nvt 99 99\n')
            vertices,tri,uv=_obj(p);np.testing.assert_array_equal(tri,[[0,1,2]])
            np.testing.assert_allclose(uv,[[[0,0],[1,0],[0,1]]])
            skin,tri,uv=skin_mesh(p);self.assertEqual(len(skin),3);self.assertLess(skin.max(),.011)

    def test_spatial_calibrated_regions_recover_independent_materials(self):
        from .reconstruction.emily import fit_regions
        from .reconstruction.reflectance import coefficients
        rng=np.random.default_rng(11);normal=rng.normal(0,.3,(240,3));normal[:,2]=1;normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        normals=np.repeat(normal[None],4,0);view=np.tile([0,0,1.],normals.shape[:-1]+(1,))
        lights=[dict(direction_h=d,radiance=[1,1,1]) for d in ([.4,.2,1],[-.4,.1,1],[.1,-.5,1],[-.3,-.2,1])]
        region=np.arange(240)%2;rgb=np.zeros((4,240,3))
        for label,r in [(0,.32),(1,.7)]:
            ids=region==label;d,s=coefficients(normals[:,ids],view[:,ids],lights,r,.02)
            rgb[:,ids]=d*np.array([.3,.2,.15])+s
        albedo,material,reports=fit_regions(rgb,normals,view,np.ones((4,240)),lights,region)
        for label,r in [(0,.32),(1,.7)]:
            self.assertEqual(reports[str(label)]['status'],'fitted global scalar')
            self.assertAlmostEqual(material[region==label,0].mean(),r,delta=.03)

    @unittest.skipUnless(find_spec('torch') and find_spec('OpenEXR'),'optional torch/OpenEXR unavailable')
    def test_cue_network_and_rgb_exr_roundtrip(self):
        import torch
        import OpenEXR
        from .reconstruction.learned_cues import network
        from .reconstruction.emily import read_linear
        model=network();normal,logits=model(torch.zeros((1,3,64,64)))
        self.assertEqual(normal.shape,(1,3,64,64));self.assertEqual(logits.shape,(1,1,64,64))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p=Path(directory)/'linear.exr';rgb=np.full((4,4,3),2.,np.float32);rgb[0,0]=-.05
            with OpenEXR.File(dict(type=OpenEXR.scanlineimage),dict(RGB=rgb)) as f:f.write(str(p))
            decoded,_,report=read_linear(p)
            np.testing.assert_allclose(decoded,rgb);self.assertGreater(report['negative_fraction'],0)


if __name__=='__main__':unittest.main()
