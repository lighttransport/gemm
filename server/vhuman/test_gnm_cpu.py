"""Full-equation GNM fitting/replay and strict CPU execution regressions."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

from . import gpu
from .face_assets import asset_path
from .rig.gnm_model import GNMModel,rotation_matrices
from .reconstruction import gnm_fit
from .reconstruction.reference import Camera

ROOT=Path(__file__).resolve().parents[2]


class CpuUtilityTests(unittest.TestCase):
    def test_all_fitting_environments_supply_threadpool_control(self):
        folder=ROOT/'server/vhuman'
        for name in ('requirements-rig.txt','requirements-cuda-dev.txt','requirements-cuda-dev.lock',
                     'requirements-cpu-fit.txt','requirements-cpu-fit.lock','setup_rocm.sh'):
            with self.subTest(name=name):
                self.assertIn('threadpoolctl==3.6.0',(folder/name).read_text())

    def test_window_coverage_and_overlap(self):
        for count in (1,64,65,129,900):
            spans=list(gnm_fit.windows(count));covered=set()
            for a,b in spans:
                self.assertLessEqual(b-a,64);covered.update(range(a,b))
            self.assertEqual(covered,set(range(count)))
            for (_,end),(start,_) in zip(spans,spans[1:]): self.assertEqual(end-start,16)

    def test_quality_gate_protects_regions_and_geometry(self):
        before=dict(heldout_px=4.,mouth_px=5.,eyes_px=2.,nonmouth_px=3.)
        after=dict(heldout_px=3.,mouth_px=4.,eyes_px=1.,nonmouth_px=2.)
        self.assertTrue(gnm_fit.accepts(before,after,.5))
        for bad in (dict(heldout_px=3.99),dict(eyes_px=2.2),dict(mouth_px=5.3),dict(nonmouth_px=np.nan)):
            self.assertFalse(gnm_fit.accepts(before,dict(after,**bad),.5))
        self.assertFalse(gnm_fit.accepts(before,after,.01))
        self.assertFalse(gnm_fit.accepts(dict(before,heldout_px=None),after,.5))

    def test_spatial_holdouts_keep_duplicate_anatomical_points_together(self):
        names=[f'point_{i}' for i in range(36)];region=np.array(['face']*36)
        positions=np.column_stack((np.arange(36),np.zeros((36,2))))
        positions[7]=positions[0]
        held=gnm_fit.holdout_mask(names,region,positions)
        self.assertTrue(held[0]);self.assertTrue(held[7]);self.assertTrue(held[-1])
        self.assertFalse(held[18])

    def test_rigid_rotation_does_not_make_an_oriented_fold(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        r=rotation_matrices(np.array([0,2.,0]));reference=p@r.T
        self.assertAlmostEqual(gnm_fit.area_ratio(reference,reference,np.array([[0,1,2]])),1.)
        self.assertLess(gnm_fit.area_ratio(reference,reference[[0,2,1]],np.array([[0,1,2]])),0.)

    def test_cpu_subprocess_configuration_skips_discovery(self):
        from .runtime import python_command,torch_device
        with gpu.execution('cpu'),patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU probe')):
            command=python_command(['python','-m','server.vhuman.reconstruction.temporal','input'])
            self.assertEqual(command[command.index('--backend')+1],'cpu')
            self.assertEqual(torch_device(None),'cpu')

    def test_cpu_environment_default_is_cpu_wheel_environment(self):
        from argparse import Namespace
        from .runtime import configure_args
        args=Namespace(inference_backend='cpu',device=0,models_root=str(gpu.model_path('')),
                       rig_python=None,qwen_python=None)
        with gpu.execution('cpu'),patch.object(gpu,'gpu_status',side_effect=AssertionError('probe')):
            configure_args(args)
            self.assertEqual(Path(args.rig_python).resolve(),(ROOT/'tmp/vhuman-cpu-venv/bin/python').resolve())

    def test_cpu_factory_context_overrides_and_restores_ambient_device(self):
        import torch
        with torch.device('meta'):
            with gnm_fit.cpu_threads(2):
                self.assertEqual(torch.zeros(1).device.type,'cpu')
            self.assertEqual(torch.zeros(1).device.type,'meta')


@unittest.skipUnless(asset_path('gnm').is_file(),'GNM assets required')
class GnmCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.model=GNMModel(device='cpu');cls.numpy=GNMModel()
        from .reconstruction.dense_landmarks import attachments
        cls.ids,cls.bary,cls.confidence=attachments()
        cls.exterior=np.flatnonzero(cls.numpy.group('skin_exterior'))
        cls.camera=Camera(1200,256,256,np.array([0.,0,1.]),np.eye(3))
        scratch=ROOT/'tmp/test-gnm-cpu';scratch.mkdir(parents=True,exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=scratch);cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)

    def view(self,expression=None,identity=None,rotations=None,translation=None):
        points,_=self.numpy.evaluate(identity,expression,rotations,translation)
        attached=(points[self.exterior[self.ids]]*self.bary[:,:,None]).sum(1)
        xy,_=self.camera.project(attached)
        return dict(size=[512,512],camera=self.camera.as_dict(),anchors={f'mp_{i:03d}':dict(xy=xy[i].tolist(),weight=float(self.confidence[i])) for i in range(468)})

    def test_zero_and_small_rotations_have_correct_gradients(self):
        import torch
        x=torch.zeros((4,3),dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(rotation_matrices,(x,),eps=1e-6,atol=1e-6))
        np.testing.assert_allclose(rotation_matrices(np.zeros((4,3))),np.tile(np.eye(3),(4,1,1)))

    def test_sampled_identity_and_joint_gradients_match_full(self):
        import torch
        m=self.model;ids=np.arange(0,17821,347)
        beta=torch.zeros(253,requires_grad=True);beta.data[0]=.2
        expr=torch.zeros((1,383));rot=torch.zeros((1,4,3));rot[0,2,1]=.1;t=torch.zeros((1,3))
        fn=m.frame_evaluator(np.zeros(253),ids,fit_identity=True)
        p,j=fn(expr,rot,t,identity=beta)
        expected,joints=m.evaluate(beta,expr[0],rot[0],t[0])
        np.testing.assert_allclose(p.detach()[0],expected.detach()[ids],atol=1e-6)
        np.testing.assert_allclose(j.detach()[0],joints.detach(),atol=1e-6)
        loss=p.square().sum()+j.square().sum();loss.backward()
        self.assertTrue(bool(torch.isfinite(beta.grad).all()));self.assertGreater(float(beta.grad.abs().sum()),0)

    def test_cpu_solve_cannot_probe_or_initialize_gpu(self):
        import torch
        expression=np.random.default_rng(7).normal(0,.04,383)
        view=self.view(expression)
        with ExitStack() as stack:
            stack.enter_context(gpu.execution('cpu'))
            for target,name in ((gpu,'gpu_status'),(gpu,'device_session'),(torch.cuda,'is_available'),
                                (torch.cuda,'init'),(torch.cuda,'set_device'),(torch.cuda,'device_count')):
                stack.enter_context(patch.object(target,name,side_effect=AssertionError('GPU access: '+name)))
            result=gnm_fit.solve([view],iterations=80,modes=32,threads=4,freeze_pose=True,model=self.model)
        self.assertTrue(result['report']['quality_gate'],result['report'])
        self.assertLess(result['report']['after']['heldout_px'],result['report']['before']['heldout_px']*.9)
        np.testing.assert_array_equal(result['identity'],np.zeros(253))
        vertices,_=gnm_fit.replay(self.model,result['identity'],result['expression'],result['rotations'],result['translation'])
        np.testing.assert_allclose(vertices,result['vertices'],atol=1e-6)

    def test_no_holdout_cannot_approve_training_only_fit(self):
        view=self.view(np.random.default_rng(5).normal(0,.02,383))
        view['anchors']={key:view['anchors'][key] for key in list(view['anchors'])[:12]}
        result=gnm_fit.solve([view],iterations=10,modes=8,threads=4,freeze_pose=True,model=self.model)
        self.assertFalse(result['report']['quality_gate'])

    def test_bad_inputs_and_cancellation(self):
        view=self.view();event=threading.Event();event.set()
        with self.assertRaises(gpu.Cancelled):gnm_fit.solve([view],cancel=event,model=self.model)
        with self.assertRaises(ValueError):gnm_fit.solve([view],identity_modes=64,timestamps=[0],model=self.model)
        with self.assertRaises(ValueError):gnm_fit.solve([view,view],timestamps=[0,0],model=self.model)
        with self.assertRaises(ValueError):gnm_fit.solve([view],identity=np.full(253,np.nan),model=self.model)
        with self.assertRaisesRegex(ValueError,'calibrated multiview'):
            gnm_fit.solve([view]*3,identity_modes=170,model=self.model)

    def test_temporal_identity_is_byte_identical_and_tracks_replay(self):
        from .reconstruction.gnm_motion import fit_views
        beta=np.zeros(253,np.float64);beta[0]=.1234567890123
        neutral,_=self.numpy.evaluate(beta)
        expression=np.random.default_rng(22).normal(0,.015,(3,383))
        views=[self.view(e,identity=beta) for e in expression]
        for anchor in views[1]['anchors'].values(): anchor['weight']=0.
        bind=dict(identity=beta,scale=np.array(1.),alignment=np.eye(3),offset=np.zeros(3),bind_residual=np.zeros_like(neutral),
                  reference_expression=np.zeros(383),reference_rotations=np.zeros((4,3)),reference_translation=np.zeros(3),
                  metadata=dict(candidate_geometry_sha256='fixture'))
        result=fit_views(bind,views,np.array([0.,.035,.09]),out=self.root/'motion',iterations=20,modes=8,threads=4)
        with np.load(self.root/'motion/motion.npz') as z:
            self.assertEqual(z['identity'].tobytes(),beta.tobytes())
            np.testing.assert_array_equal(z['timestamps'],[0.,.035,.09])
            p,_=gnm_fit.replay(self.model,z['identity'],z['expression'],z['rotations'],z['translation'])
            np.testing.assert_allclose(z['vertices'],p,atol=1e-6)
        self.assertEqual(result['track']['backend'],'cpu')
        self.assertEqual(result['track']['per_frame_valid'],[True,False,True])
        self.assertIsNone(result['track']['per_frame_error_ipd'][1])
        self.assertEqual(len(result['track']['skin_min_oriented_area_ratio']),3)
        self.assertGreaterEqual(min(result['track']['skin_min_oriented_area_ratio']),.05)

    def test_frozen_pose_preserves_large_turn_and_native_eye_roll(self):
        rotations=np.zeros((1,4,3),np.float32);rotations[0,0,1]=1.5;rotations[0,2:,2]=.07
        view=self.view(rotations=rotations[0])
        result=gnm_fit.solve([view],rotations=rotations,iterations=10,modes=8,threads=4,freeze_pose=True,model=self.model)
        np.testing.assert_array_equal(result['rotations'],rotations)
        # Temporal initialization must also accept a valid source pose outside
        # the old absolute +/-1.4-radian search box.
        temporal=gnm_fit.solve([view,view],rotations=np.repeat(rotations,2,axis=0),timestamps=[0.,.04],
                               iterations=10,modes=8,threads=4,model=self.model)
        self.assertTrue(np.isfinite(temporal['vertices']).all())

    def test_simultaneous_cameras_share_the_complete_expression(self):
        expression=np.zeros((2,383));expression[0,0]=.02;expression[1,0]=-.01
        views=[self.view(e) for e in expression]
        result=gnm_fit.solve(views,expression=expression,groups=['instant','instant'],iterations=10,modes=8,
                             threads=4,freeze_pose=True,model=self.model)
        np.testing.assert_array_equal(result['expression'][0],result['expression'][1])
        self.assertEqual(result['report']['simultaneous_groups_initialized_by_consensus'],1)

    def test_native_sequence_replay_applies_pose_once(self):
        from .reconstruction.sequence import predict
        from .reconstruction.observations import sha256
        from .reconstruction import gnm_motion
        folder=self.root/'sequence';folder.mkdir()
        identity=np.zeros(253);expression=np.zeros((2,383));rotations=np.zeros((2,4,3));rotations[:,0,1]=[.1,.3]
        translation=np.array([[0.,0,0],[.01,0,0]])
        offset=np.array([.1,-.2,.3]);scale=1.2
        np.savez_compressed(folder/'geometry.npz',gnm_expressions=expression,gnm_rotations=rotations,gnm_translations=translation,triangles=np.zeros((0,3),int))
        (folder/'manifest.json').write_text(json.dumps(dict(geometry_sha256=sha256(folder/'geometry.npz'),
                                                         geometry=dict(coordinate_convention='fixed_camera_posed_surfaces_v2'))))
        (folder/'observations.json').write_text(json.dumps(dict(views=[dict(timestamp_s=0.),dict(timestamp_s=1.)])))
        bind=dict(identity=identity,scale=scale,alignment=np.eye(3),offset=offset,bind_residual=np.zeros((17821,3)))
        target=[0.,.5,1.];out=folder/'prediction.npz'
        with patch.object(gnm_motion,'candidate_bind',return_value=bind),gnm_fit.cpu_threads(4):
            report=predict(folder,target,out)
            r=np.zeros((3,4,3));r[:,0,1]=[.1,.2,.3]
            expected,_=gnm_fit.replay(self.model,identity,np.zeros((3,383)),r,np.array([[0.,0,0],[.005,0,0],[.01,0,0]]),scale=scale,offset=offset)
        with np.load(out) as z:
            np.testing.assert_allclose(z['positions'],expected[:,self.exterior],atol=1e-6)
        self.assertEqual(report['safe_steps'],[1.,1.,1.])
        self.assertFalse(report['target_annotations_used'])

    def test_refinement_saves_used_masks_and_consistent_native_metadata(self):
        from PIL import Image
        from .reconstruction import gnm_refine
        from .reconstruction.observations import FORMAT,load,sha256
        from scipy.spatial.transform import Rotation
        folder=self.root/'refine-source';folder.mkdir();out=self.root/'refine-result'
        Image.new('RGB',(8,8)).save(folder/'portrait.png')
        rest,joints=self.numpy.evaluate();offset=np.array([.01,-.02,.03])
        neutral=rest+offset;rotations=np.zeros((1,4,3));rotations[0,0,1]=.2;translation=np.array([[.001,0,0]])
        posed,_=self.numpy.evaluate(rotations=rotations[0],translation=translation[0]);posed+=offset
        np.savez_compressed(folder/'geometry.npz',scale=1.,rotation=np.eye(3),gnm_identity=np.zeros(253),
                            gnm_expressions=np.zeros((1,383)),full_neutral=neutral,neutral=neutral[self.exterior],
                            gnm_joint_positions=joints+offset,triangles=np.array([[0,1,2]]),triangle_uvs=np.zeros((1,3,2)))
        view=dict(image='portrait.png',sha256=sha256(folder/'portrait.png'),size=[8,8],camera=self.camera.as_dict(),anchors={})
        (folder/'observations.json').write_text(json.dumps(dict(format=FORMAT,views=[view])))
        material=dict(roughness=dict(value=.5),f0=dict(value=.028))
        manifest=dict(format='vhuman.reconstruction.v1',face_model='gnm_v3',geometry_sha256=sha256(folder/'geometry.npz'),
                      portrait_sha256=sha256(folder/'portrait.png'),geometry=dict(fitted_cameras=[view['camera']]),
                      config=dict(texture_res=8),material=material)
        (folder/'manifest.json').write_text(json.dumps(manifest));before=sha256(folder/'geometry.npz')
        result=dict(identity=np.zeros(253),expression=np.zeros((1,383)),rotations=rotations,translation=translation,
                    neutral=neutral,vertices=posed[None],baseline_landmarks=np.zeros((1,0,2)),projected_landmarks=np.zeros((1,0,2)),
                    observed_landmarks=np.zeros((1,0,2)),weights=np.zeros((1,0)),heldout=np.zeros(0,bool),
                    report=dict(quality_gate=True,after=dict(heldout_px=0.),expression_safe_steps=[1.]))
        with patch.object(gnm_refine,'solve',return_value=result),patch.object(gnm_refine,'review_portrait'), \
                patch('server.vhuman.face_parsing.FaceParser') as parser, \
                patch('server.vhuman.reconstruction.materials.bake_portrait',return_value=material):
            parser.return_value.predict.return_value=(np.ones((8,8),np.uint8),np.ones((8,8)))
            gnm_refine.refine(folder,out,parsing_model='explicit parser')
        saved=load(out/'observations.json')['views'][0]
        self.assertTrue(Path(saved['exclusion_mask_path']).is_relative_to(out))
        self.assertTrue(Path(saved['silhouette_mask_path']).is_relative_to(out))
        meta=json.loads((out/'manifest.json').read_text())['geometry']
        self.assertEqual(len(meta['expression_labels']),383)
        world=Rotation.from_rotvec(meta['pose_rotations'][0]).as_matrix()
        np.testing.assert_allclose(neutral@world.T+meta['pose_translations'][0],posed,atol=1e-6)
        self.assertEqual(sha256(folder/'geometry.npz'),before)

    def _legacy_catalog(self):
        from PIL import Image
        from .reconstruction.observations import sha256
        root=self.root/self._testMethodName;root.mkdir()
        candidate=root/'candidate';candidate.mkdir()
        folder=root/'neutral';(folder/'video').mkdir(parents=True)
        Image.new('RGB',(8,8),(80,100,120)).save(candidate/'portrait.png')
        Image.open(candidate/'portrait.png').save(root/'reference.png')
        identity=np.zeros(253);identity[0]=.1
        rest,_=self.numpy.evaluate(identity)
        neutral=rest[self.exterior]+np.array([.01,-.02,.03])
        neutral[:,1]+=.001*np.linspace(-1,1,len(neutral))
        # The original six-array PCA layout deliberately has no full_* or gnm_* keys.
        np.savez_compressed(candidate/'geometry.npz',neutral=neutral,captured=neutral,
                            scale=1.,rotation=np.eye(3),triangles=np.array([[0,1,2]]),triangle_uvs=np.zeros((1,3,2)))
        manifest=dict(format='vhuman.reconstruction.v1',face_model='gnm_v3',
                      geometry_sha256=sha256(candidate/'geometry.npz'),portrait_sha256=sha256(candidate/'portrait.png'),
                      geometry=dict(identity_coefficients=[.1],expression_modes=[7],expression_coefficients=[[.2]],
                                    fitted_cameras=[self.camera.as_dict()]))
        (candidate/'manifest.json').write_text(json.dumps(manifest))
        (candidate/'observations.json').write_text(json.dumps(dict(views=[dict(size=[8,8])])))
        (folder/'video/clip.mp4').write_bytes(b'cached clip')
        (folder/'video/manifest.json').write_text(json.dumps(dict(image_sha256=sha256(root/'reference.png'))))
        landmarks=np.zeros((478,3));landmarks[:,:2]=.5
        rows=[dict(visible=True,landmarks=landmarks.tolist(),t=t) for t in (.125,.2)]
        (folder/'observations.json').write_text(json.dumps(dict(source_sha256=sha256(folder/'video/clip.mp4'),
                                                             width=8,height=8,fps=30,observations=rows)))
        np.savez_compressed(folder/'face_parsing.npz',labels=np.ones((2,8,8),np.uint8))
        return candidate,folder,root/'out',neutral

    def test_cpu_catalog_accepts_original_pca_candidates_and_preserves_bind(self):
        from .reconstruction import temporal,gnm_motion
        candidate,folder,out,neutral=self._legacy_catalog()
        before=(candidate/'geometry.npz').read_bytes()
        with gpu.execution('cpu'), patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU probe')), \
                patch.object(gpu,'device_session',side_effect=AssertionError('GPU lock')), \
                patch.object(gnm_motion,'fit_views',return_value=dict(track={})) as fit:
            result=temporal.fit_clip(candidate,folder,out,iterations=20,threads=2)
        bind,views,timestamps=fit.call_args.args
        self.assertAlmostEqual(bind['identity'][0],.1)
        self.assertAlmostEqual(bind['reference_expression'][7],.2)
        rest,_=self.numpy.evaluate(bind['identity'],bind_residual=bind['bind_residual'])
        np.testing.assert_allclose((rest+bind['offset'])[self.exterior],neutral,atol=1e-7)
        np.testing.assert_allclose(timestamps,[.125,.2])
        self.assertTrue(all(view['synthetic'] for view in views))
        self.assertEqual(result['clip_sha256'],gnm_motion.sha256(folder/'video/clip.mp4'))
        self.assertEqual((candidate/'geometry.npz').read_bytes(),before)
        self.assertTrue((out/'motion.json').is_file())

    def test_cpu_catalog_keeps_clip_portrait_and_generation_provenance_checks(self):
        from .reconstruction import temporal,gnm_motion
        from PIL import Image
        candidate,folder,out,_=self._legacy_catalog()
        changes=[(folder/'video/clip.mp4',b'changed clip','cached landmarks'),
                 (folder/'video/manifest.json',b'{"image_sha256":"wrong"}','reference hash'),
                 (candidate/'geometry.npz',b'changed geometry','geometry hash')]
        other=folder/'other.png';Image.new('RGB',(8,8)).save(other)
        changes.append((folder.parent/'reference.png',other.read_bytes(),'different portrait'))
        with patch.object(gnm_motion,'fit_views',side_effect=AssertionError('unverified fitting')):
            for path,changed,error in changes:
                with self.subTest(path=path.name):
                    original=path.read_bytes();path.write_bytes(changed)
                    try:
                        with self.assertRaisesRegex(ValueError,error):
                            temporal.fit_clip(candidate,folder,out,iterations=20,threads=2)
                    finally:
                        path.write_bytes(original)

    def test_bind_offset_comes_from_joints_even_with_biased_skin_residual(self):
        from .reconstruction.gnm_motion import candidate_bind,export_bind,rig_bind
        from .reconstruction import provenance
        folder=self.root/'candidate';folder.mkdir()
        rest,joints=self.numpy.evaluate();offset=np.array([.1,-.2,.3]);rest=rest+offset
        rest[self.exterior,1]+=.005
        np.savez_compressed(folder/'geometry.npz',gnm_identity=np.zeros(253),full_neutral=rest,scale=1.,rotation=np.eye(3),
                            gnm_expressions=np.zeros((1,383)),gnm_joint_positions=joints+offset)
        (folder/'observations.json').write_text(json.dumps({'views':[{'size':[512,512]}]}))
        manifest={'geometry_sha256':'fixture','geometry':{'fitted_cameras':[self.camera.as_dict()]}}
        with patch.object(provenance,'validate_candidate',return_value=manifest):
            bind=candidate_bind(folder)
            np.testing.assert_allclose(bind['offset'],offset,atol=1e-10)
            actual,_=self.numpy.evaluate(bind['identity'],bind_residual=bind['bind_residual'])
            np.testing.assert_allclose(actual+offset,rest,atol=1e-7)
            rig=self.root/'rig';rig.mkdir();(rig/'rig_deformer.safetensors').write_bytes(b'package hash')
            export_bind(rig,folder)
        loaded=rig_bind(rig);np.testing.assert_allclose(loaded['offset'],offset,atol=1e-10)
        (rig/'rig_deformer.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):rig_bind(rig)

    def test_nonlinear_retarget_validates_native_cpu_and_does_not_write_source_rig(self):
        from .reconstruction.gnm_motion import retarget
        from .rig import native
        from .test_rig import _toy_rig
        folder=self.root/'retarget';folder.mkdir()
        definition=_toy_rig();(folder/'rig.json').write_text(json.dumps(definition))
        neutral,_=self.numpy.evaluate();rest=neutral[self.exterior]
        native.write_package(folder/'rig_deformer.safetensors',definition,rest,
                             {'mouthSmileLeft':np.zeros_like(rest),'c':np.zeros_like(rest)},
                             np.zeros((len(rest),4),np.int32),np.tile([1.,0,0,0],(len(rest),1)))
        bind=dict(identity=np.zeros(253),source_ids=self.exterior,scale=np.array(1.),alignment=np.eye(3),offset=np.zeros(3),bind_residual=np.zeros_like(neutral))
        result=dict(identity=np.zeros(253),expression=np.zeros((2,383)),rotations=np.zeros((2,4,3)),translation=np.zeros((2,3)))
        prior=np.zeros((2,len(definition['controls'])),np.float32)
        controls,report=retarget(folder,bind,result,prior,iterations=2,threads=4)
        self.assertLess(report['rms_mm'],1e-4)
        self.assertLessEqual(report['rms_mm'],report['baseline_rms_mm']+1e-6)
        from .reconstruction.observations import sha256
        self.assertEqual(report['rig_package_sha256'],sha256(folder/'rig_deformer.safetensors'))
        self.assertEqual(controls.shape,prior.shape)
        self.assertEqual({p.name for p in folder.iterdir()},{'rig.json','rig_deformer.safetensors'})


if __name__=='__main__':unittest.main()
