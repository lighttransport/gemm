"""Independent mechanics invariants; all tests use CPU and synthetic geometry."""
import ctypes as ct
import threading
import json
import tempfile
import wave
import importlib.util
import shutil
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .rig import anatomy
from .rig.biomechanics_native import LIBRARY, Session, Snapshot, ContactStats, P, library, ptr, volumes

HAS_TORCH=importlib.util.find_spec('torch') is not None
HAS_SCIPY=importlib.util.find_spec('scipy') is not None
HAS_USD=importlib.util.find_spec('pxr') is not None


def fixture():
    rest = np.array([[0,0,0],[.02,0,0],[0,.02,0],[0,0,.02]], float)
    arrays = dict(rest=rest,tets=np.array([[0,1,2,3]],np.int32),
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32),
        inverse_mass=np.full(4,1000.),bone=np.full(4,-1,np.int32),jaw_hinge=np.array([0,.01,-.02]),
        fibers=np.array([[0,0,1,0,0,1000]],float),jaw_muscles=np.empty((0,9)),
        anchor_ids=np.array([0,2,3],np.int32),anchor_bone=np.zeros(3,np.int32),youngs_moduli=np.array([10000],np.float32),
        surface_rest=rest.copy(),embed_ids=np.repeat(np.arange(4)[:,None],4,axis=1).astype(np.int32),
        embed_weights=np.tile([1.,0,0,0],(4,1)),source_ids=np.arange(4,dtype=np.int32))
    return arrays,dict(schema=anatomy.SCHEMA,muscles=['test'],control_prior=[{'mouthSmileLeft':1.}],
                       parts=[dict(name='skin',surface_start=0,surface_count=4,face_start=0,face_count=4)])


class AnatomyTests(unittest.TestCase):
    def test_condylar_track_fit_holds_out_openings_and_keeps_the_hinge(self):
        from .rig import jaw_model,rigdef
        hinge=np.array([0,.015,-.03]);known=np.array([.001,-.004,.006])
        rest=np.array([[-.01,0,.03],[.01,0,.03],[0,.01,.04],[0,-.01,.04]])
        class ReferenceRig:
            jidx={'teeth_lower':0}
            def evaluate(self,controls):
                angle=.3*controls['jawOpen'];rotation=rigdef.euler_matrix(angle,0,0)
                transform=np.eye(4);transform[:3,:3]=rotation
                transform[:3,3]=hinge-rotation@hinge+angle*known
                return {'skin':np.array([transform])}
        with patch('server.vhuman.rig.rigdef.Rig',return_value=ReferenceRig()):
            track,report=jaw_model.fit_track({},hinge,rest)
        self.assertTrue(report['accepted']);self.assertTrue(report['hinge_fixed'])
        self.assertEqual(set(report['training_openings'])&set(report['withheld_openings']),set())
        self.assertGreater(report['withheld_before_mm'],.1)
        self.assertLess(report['withheld_after_mm'],1e-10)
        np.testing.assert_allclose(track,known,atol=1e-15)
        arrays,meta=fixture();arrays['jaw_track']=np.array([0,np.nan,0])
        with self.assertRaisesRegex(ValueError,'condylar'):anatomy.validate(arrays,meta)

    def test_condylar_calibration_keeps_legacy_track_outside_supported_range(self):
        from .rig import jaw_model,rigdef
        hinge=np.zeros(3);rest=np.eye(4,3)
        class ReferenceRig:
            jidx={'teeth_lower':0}
            maximum=0.
            def evaluate(self,controls):
                transform=np.eye(4)
                transform[:3,:3]=rigdef.euler_matrix(self.maximum*controls['jawOpen'],0,0)
                return {'skin':np.array([transform])}
        reference=ReferenceRig()
        with patch('server.vhuman.rig.rigdef.Rig',return_value=reference):
            for maximum in (0.,-.3,.6):
                reference.maximum=maximum
                track,report=jaw_model.fit_track({},hinge,rest)
                np.testing.assert_array_equal(track,jaw_model.DEFAULT_TRACK)
                self.assertFalse(report['accepted'])
                self.assertEqual(report['reason'],'unsupported_native_rotation_range')
            reference.maximum=np.nan
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                jaw_model.fit_track({},hinge,rest)

    def test_server_full_jobs_force_cpu_even_when_other_jobs_select_a_gpu(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from .app import App
        from . import gpu
        scratch=anatomy.ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory,patch('server.vhuman.app.Jobs') as jobs:
            App(SimpleNamespace(work=directory,mock=False,qwen_python=None,inference_backend='cuda',tts_backend='cuda',device=0,rig_python='cpu-python'))
            handlers=jobs.call_args.args[1]
            cases=(('rig_fit_anatomy',{},'anatomy','anatomy_job','cpu'),
                   ('rig_soft_tissue',dict(mode='full'),'soft_tissue','soft_tissue_job','cpu'),
                   ('rig_speech',dict(motion_model='biomechanics'),'speech','speech_job','cpu'),
                   ('rig_speech',dict(motion_model='rig'),'speech','speech_job','cuda'),
                   ('rig_soft_tissue',dict(mode='patch'),'soft_tissue','soft_tissue_job','cuda'))
            for kind,request,module,function,backend in cases:
                with patch.object(gpu,'execution',return_value=nullcontext(backend)) as execution, \
                     patch.object(gpu,'device_index',return_value=0), \
                     patch('server.vhuman.rig.'+module+'.'+function,return_value={}) as handler:
                    result=handlers[kind](request,lambda *_:None,threading.Event())
                    self.assertEqual(execution.call_args.args[0],backend)
                    self.assertEqual(result['device'],'cpu' if backend=='cpu' else 0)
                    self.assertEqual(handler.call_args.kwargs['python'],'cpu-python')
                    if module=='speech':self.assertEqual(handler.call_args.kwargs['backend'],backend)

    def test_meshing_cache_keys_exact_geometry_settings_and_worker(self):
        arrays,_=fixture();v,f=arrays['rest'],arrays['faces']
        key,recipe=anatomy.mesh_cache_recipe(v,f,.004,4,False)
        same,_=anatomy.mesh_cache_recipe(v.astype('>f8'),f.astype('>i8'),.004,4,False)
        self.assertEqual(key,same)
        moved=v.copy();moved[1,0]=np.nextafter(moved[1,0],np.inf)
        for arguments in ((moved,f,.004,4,False),(v,f[:,::-1],.004,4,False),
                          (v,f,.003,4,False),(v,f,.004,1,False),(v,f,.004,4,True)):
            self.assertNotEqual(key,anatomy.mesh_cache_recipe(*arguments)[0])
        with patch.object(anatomy,'MESH_WORKER_SHA256','changed'):
            self.assertNotEqual(key,anatomy.mesh_cache_recipe(v,f,.004,4,False)[0])
        self.assertEqual(recipe['wildmeshing'],anatomy.MESH_VERSION)
        for arguments in ((v,f.astype(float),.004,4,False),(v,f,.004,True,False),(v,f,float('nan'),4,False)):
            with self.assertRaises(ValueError):anatomy.mesh_cache_recipe(*arguments)

    def test_meshing_cache_reuses_verified_volume_and_rejects_corruption(self):
        arrays,_=fixture();scratch=anatomy.ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory);calls=[]
            def mesh(command,**kwargs):
                calls.append(command);np.savez_compressed(command[6],rest=arrays['rest'],tets=arrays['tets']);return 0
            with patch('server.vhuman.process.run',mesh):
                for name in ('first','second'):
                    points,tets=anatomy.tetrahedralize(arrays['rest'],arrays['faces'],folder,name,
                        edge=.004,threads=1,cache=folder/'cache')
                    np.testing.assert_array_equal(points,arrays['rest']);np.testing.assert_array_equal(tets,arrays['tets'])
                self.assertEqual(len(calls),1)
                receipt=json.loads((folder/'second-meshing-cache.json').read_text());self.assertTrue(receipt['hit'])
                volume=folder/'cache'/receipt['key']/'volume.npz'
                with volume.open('ab') as stream:stream.write(b'changed')
                with self.assertRaisesRegex(ValueError,'invalid tetrahedral cache entry'):
                    anatomy.tetrahedralize(arrays['rest'],arrays['faces'],folder,'third',edge=.004,threads=1,cache=folder/'cache')
                self.assertEqual(len(calls),1)
            invalid=folder/'invalid.npz'
            np.savez_compressed(invalid,rest=arrays['rest'],tets=arrays['tets'][:,[0,2,1,3]])
            with self.assertRaisesRegex(ValueError,'orientation/quality'):anatomy.read_volume(invalid)

    def test_meshing_cache_publication_is_atomic_and_cancelled_requests_do_not_reuse(self):
        from concurrent.futures import ThreadPoolExecutor
        from .rig.meshing_cache import MeshCache
        arrays,_=fixture();scratch=anatomy.ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory);source=folder/'volume.npz'
            np.savez_compressed(source,rest=arrays['rest'],tets=arrays['tets'])
            key,recipe=anatomy.mesh_cache_recipe(arrays['rest'],arrays['faces'],.004,1,False)
            store=MeshCache(folder/'cache')
            with ThreadPoolExecutor(max_workers=4) as workers:
                outputs=list(workers.map(lambda _:store.publish(key,recipe,source),range(8)))
            self.assertTrue(all(output==outputs[0] for output in outputs))
            self.assertEqual(list((folder/'cache').iterdir()),[folder/'cache'/key])
            self.assertEqual(outputs[0].read_bytes(),source.read_bytes())
            stop=threading.Event();stop.set()
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                anatomy.tetrahedralize(arrays['rest'],arrays['faces'],folder,'cancelled',edge=.004,threads=1,cache=folder/'cache',cancel=stop)
            self.assertFalse((folder/'cancelled-meshing-cache.json').exists())

    def test_meshing_timeout_names_the_region_budget_and_retained_log(self):
        arrays,_=fixture();scratch=anatomy.ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory)
            with patch('server.vhuman.process.run',side_effect=subprocess.TimeoutExpired(['mesher'],60)):
                with self.assertRaisesRegex(RuntimeError,'skin tetrahedralization exceeded 60 seconds; increase --mesh-timeout'):
                    anatomy.tetrahedralize(arrays['rest'],arrays['faces'],folder,'skin',edge=.004,threads=1,timeout=60,cache=folder/'cache')
            self.assertTrue((folder/'skin-meshing.log').is_file())
            self.assertTrue((folder/'skin-input.npz').is_file())
            self.assertFalse((folder/'cache').exists())

    def test_ocular_fibers_wrap_the_aperture_and_preserve_other_directions(self):
        from .rig.skin_fibers import _ocular_directions
        globe=np.r_[np.eye(3),-np.eye(3)]
        points=np.array([[0,.5,2],[1.,0,2],[0,-.5,2],[-1,0,2],[0,1,2]])
        muscles=np.array([0,0,0,0,1]);directions=np.tile([0.,1.,0.],(5,1))
        arrays=dict(surface_rest=globe);meta=dict(muscles=['orbicularis_oculi_Left','frontalis'],
            parts=[dict(name='eye_left',surface_start=0,surface_count=6)])
        result,report=_ocular_directions(arrays,meta,points,muscles,directions)
        np.testing.assert_allclose(abs(result[[0,2],0]),1.,atol=1e-14)
        np.testing.assert_allclose(abs(result[[1,3],1]),1.,atol=1e-14)
        np.testing.assert_array_equal(result[4],directions[4])
        radii=np.array(report['orbicularis_oculi_Left']['ellipse_radii'])
        np.testing.assert_allclose((points[:4,:2]/radii**2*result[:4,:2]).sum(1),0.,atol=1e-14)
        moved,_=_ocular_directions(dict(surface_rest=globe*2+[.1,.2,.3]),meta,points*2+[.1,.2,.3],muscles,directions)
        np.testing.assert_allclose(moved,result,atol=1e-14)

    def test_orbital_supports_release_anterior_lids_and_keep_canthi_and_other_bones(self):
        from .rig.skin_attachments import orbital_supports
        center=np.array([0,.01,.01]);radius=.005
        surface=center+radius*np.r_[np.eye(3),-np.eye(3)]
        rest=np.array([[0,.01,.0152],[.0045,.01,.0152],[0,.01,.005],
                       [0,.02,.0152],[0,-.01,.0152],[0,.01,.0152]])
        arrays=dict(rest=rest,surface_rest=surface,tets=np.array([[0,1,2,3],[0,3,4,5]]),regions=np.zeros(2,np.int32),
                    anchor_ids=np.arange(6,dtype=np.int32),anchor_bone=np.array([0,0,0,0,0,1],np.int32))
        meta=dict(parts=[dict(name='eye_left',surface_start=0,surface_count=6)])
        keep,report=orbital_supports(arrays,meta,1.)
        np.testing.assert_array_equal(keep,[False,True,True,True,True,True])
        self.assertEqual(report['released_vertices'],[0])
        moved,_=orbital_supports(dict(arrays,rest=rest*2+[.1,.2,.3],surface_rest=surface*2+[.1,.2,.3]),meta,2.)
        np.testing.assert_array_equal(moved,keep);np.testing.assert_array_equal(arrays['anchor_ids'],np.arange(6))

    def test_eyelid_features_preserve_unrequested_eyelids_during_expression_fitting(self):
        from .rig.biomechanics import inverse_samples,regional_target_metrics
        rest=np.zeros((1000,3));rest[:,0]=np.linspace(-.05,.05,1000)
        mask=np.zeros(1000,bool);mask[[1,3,997,999]]=True
        arrays=dict(surface_rest=rest,fit_importance=np.ones(1000),fit_eye_features=mask)
        meta=dict(parts=[dict(name='skin',surface_start=0,surface_count=1000)])
        selected=inverse_samples(arrays,meta,rest,part_samples=4,feature_samples=4,moving_samples=4)
        self.assertTrue(np.isin(np.flatnonzero(mask),selected).all())
        blink=rest.copy();blink[mask,1]+=.001
        report=regional_target_metrics(arrays,meta,rest,blink)
        self.assertTrue(report['skin']['passed']);self.assertFalse(report['eyelid_features']['passed'])
        a,m=fixture();a['fit_eye_features']=np.ones(4,bool);anatomy.validate(a,m)
        for bad in (np.ones(3,bool),np.ones(4,float),np.full(4,2),np.full(4,-1)):
            with self.assertRaisesRegex(ValueError,'eyelid mask'):anatomy.validate(dict(a,fit_eye_features=bad),m)
    def test_solver_freezing_survives_source_replacement_and_concurrent_publication(self):
        from concurrent.futures import ThreadPoolExecutor
        from .rig.biomechanics_native import freeze_library,ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory);source=folder/'solver.so';source.write_bytes(b'first solver bytes')
            old,digest=freeze_library(source,folder/'cache')
            source.write_bytes(b'rebuilt solver bytes')
            with ThreadPoolExecutor(max_workers=4) as workers:
                results=list(workers.map(lambda _:freeze_library(source,folder/'cache'),range(8)))
            self.assertTrue(all(result==results[0] for result in results))
            self.assertNotEqual(results[0][1],digest)
            self.assertEqual(old.read_bytes(),b'first solver bytes')
            path=results[0][0];path.chmod(0o644);path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'solver hash mismatch'):freeze_library(source,folder/'cache')
            self.assertEqual(path.read_bytes(),b'corrupt')
    @unittest.skipUnless(HAS_SCIPY,'SciPy rigid interpolation')
    def test_kinematic_interpolation_preserves_rigid_shape_and_exact_endpoints(self):
        from types import SimpleNamespace
        from .rig.kinematic_targets import KinematicTargets
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        goal=rest*[-1,-1,1]+[.1,.2,.3]
        state=SimpleNamespace(bone=np.zeros(4),mass=np.zeros(4),positions=rest.copy(),rigid_targets=goal.copy(),rest=rest,
            embed_ids=np.repeat(np.arange(4)[:,None],4,axis=1),
            metadata=dict(parts=[dict(name='eye_left',surface_start=0,surface_count=4)]))
        plan=KinematicTargets(state);plan.apply(0.);np.testing.assert_array_equal(state.rigid_targets,rest)
        reference=np.linalg.norm(rest[:,None]-rest,axis=2)
        for fraction in (.125,.5,.875):
            plan.apply(fraction);points=state.rigid_targets
            np.testing.assert_allclose(np.linalg.norm(points[:,None]-points,axis=2),reference,atol=1e-14)
            np.testing.assert_allclose(points.mean(0),(1-fraction)*rest.mean(0)+fraction*goal.mean(0),atol=1e-14)
        plan.apply(1.);np.testing.assert_array_equal(state.rigid_targets,goal)
        state.rigid_targets[1,0]+=.1
        with self.assertRaisesRegex(ValueError,'changes part shape'):KinematicTargets(state)
    def test_native_fiber_packing_preserves_exact_structure_bytes(self):
        from .rig.biomechanics_native import pack_fibers,Fiber,D
        values=np.array([[0,1,.3,.4,.5,1234.5],[100,2,0,-1,0,3000.]],np.float64)
        reference=(Fiber*2)(*[Fiber(int(v[0]),int(v[1]),(D*3)(*v[2:5]),float(v[5])) for v in values])
        self.assertEqual(bytes(pack_fibers(values)),bytes(reference))
        self.assertEqual(len(pack_fibers(np.empty((0,6)))),0)
        for bad in (values[:,:5],np.full((1,6),np.nan),np.array([[.5,0,1,0,0,1]]),
                    np.array([[-1,0,1,0,0,1]]),np.array([[2**31,0,1,0,0,1]])):
            with self.assertRaisesRegex(ValueError,'fiber array'):pack_fibers(bad)

    def test_surface_area_gate_is_rotation_invariant_and_detects_collapse(self):
        from .rig.biomechanics import surface_area_metrics
        rest=np.array([[0,0,0],[1,0,0],[0,1,0]],float);faces=np.array([[0,1,2]],np.int32)
        rotated=rest*np.array([1,-1,-1])+[3,4,5]
        self.assertTrue(surface_area_metrics(rest,rotated,faces)['passed'])
        collapsed=rest.copy();collapsed[2]=[0,.01,0]
        report=surface_area_metrics(rest,collapsed,faces)
        self.assertFalse(report['passed']);self.assertEqual(report['collapsed_faces'],1)
        self.assertAlmostEqual(report['min_area_ratio'],.01)
        report=surface_area_metrics(collapsed*0,collapsed,faces)
        self.assertFalse(report['passed']);self.assertEqual(report['degenerate_source_faces'],1)
    @staticmethod
    def tongue_grid():
        from itertools import product,permutations
        shape=(5,5,9)
        points=np.array(list(product(np.linspace(-.01,.01,5),np.linspace(0,.015,5),np.linspace(0,.03,9))))
        cells=[]
        for start in product(range(4),range(4),range(8)):
            for order in permutations(range(3)):
                corner=np.array(start);cell=[np.ravel_multi_index(corner,shape)]
                for axis in order:
                    corner[axis]+=1;cell.append(np.ravel_multi_index(corner,shape))
                cells.append(cell)
        return points,np.array(cells,np.int32)

    def test_extrinsic_tongue_paths_preserve_geometry_and_scale_authored_stress(self):
        from .rig import tongue_cables
        points,tets=self.tongue_grid();names=[family+'_'+side for family in ('styloglossus','hyoglossus','palatoglossus') for side in ('Left','Right')]
        count=len(tets);fibers=np.column_stack((np.tile(np.arange(count),6),np.repeat(np.arange(6),count),
                                              np.tile([0,1,0],(count*6,1)),np.full(count*6,100.)))
        arrays=dict(rest=points,tets=tets,regions=np.ones(count,np.int32),inverse_mass=np.ones(len(points)),fibers=fibers)
        original=points.copy();original_fields=fibers.copy();meta=dict(muscles=names)
        rows,report=tongue_cables.fit(arrays,meta)
        self.assertEqual(rows.shape,(6,40));self.assertEqual(len(report['cables']),6)
        self.assertTrue(report['geometry_preserved']);self.assertTrue(report['internal_fibers_preserved'])
        np.testing.assert_array_equal(points,original);np.testing.assert_array_equal(fibers,original_fields)
        self.assertTrue((rows[:,5]>0).all())
        for row in rows:
            selected=row[8:8+int(row[1])].astype(int)
            self.assertEqual(len(selected),len(np.unique(selected)))
            self.assertAlmostEqual(float(row[24:40].sum()),1.)
            self.assertTrue((arrays['inverse_mass'][selected]>0).all())
        repeated,_=tongue_cables.fit(arrays,meta);np.testing.assert_array_equal(rows,repeated)
        scaled,_=tongue_cables.fit(dict(arrays,rest=points*2),meta,scale=2)
        np.testing.assert_array_equal(rows[:,:2],scaled[:,:2])
        np.testing.assert_array_equal(rows[:,8:24],scaled[:,8:24])
        np.testing.assert_allclose(scaled[:,24:],rows[:,24:],atol=1e-15)
        np.testing.assert_allclose(scaled[:,2:5],2*rows[:,2:5],atol=1e-15)
        np.testing.assert_allclose(scaled[:,5],4*rows[:,5],rtol=1e-12)
        np.testing.assert_allclose(scaled[:,6],2*rows[:,6],atol=1e-15)
        cancel=threading.Event();cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):tongue_cables.fit(arrays,meta,cancel=cancel)

    def test_external_tongue_package_rows_reject_invalid_geometry_and_padding(self):
        from .rig.tongue_cables import validate_rows
        arrays,meta=fixture();row=np.zeros((1,40));row[0,:8]=[0,2,0,.05,0,.02,1,.15]
        row[0,8:10]=[1,2];row[0,24:26]=[.4,.6]
        center=row[0,24:26]@arrays['rest'][[1,2]];row[0,6]=np.linalg.norm(center-row[0,2:5])
        arrays['tongue_cables']=row;anatomy.validate(arrays,meta)
        for column,value in ((0,1),(1,17),(8,4),(9,1),(24,-.1),(5,-1),(6,0),(7,1),(10,1),(26,.1),(6,row[0,6]*1.1)):
            bad=row.copy();bad[0,column]=value
            with self.subTest(column=column,value=value),self.assertRaisesRegex(ValueError,'cable'):
                anatomy.validate(dict(arrays,tongue_cables=bad),meta)
        mass=arrays['inverse_mass'].copy();mass[1]=0
        with self.assertRaisesRegex(ValueError,'free tissue'):
            validate_rows(row,4,1,inverse_mass=mass,rest=arrays['rest'])

    def test_external_muscle_path_preflight_rejects_a_segment_through_a_tooth(self):
        from .rig.tongue_cables import dental_path_checks
        row=np.zeros((1,40));row[0,:8]=[0,1,-1,0,0,.1,2,.15];row[0,24]=1
        arrays=dict(rest=np.array([[1.,0,0]]),surface_rest=np.array([[0.,-1,-1],[0,1,-1],[0,0,1]]),
                    faces=np.array([[0,1,2]],np.int32),tongue_cables=row)
        meta=dict(muscles=['test'],parts=[dict(name='teeth_upper',face_start=0,face_count=1)])
        report=dental_path_checks(arrays,meta)
        self.assertTrue(report['evaluated']);self.assertFalse(report['passed']);self.assertEqual(report['proper_crossings'],1)
        self.assertTrue(dental_path_checks(arrays,meta,activation=np.zeros(1))['passed'])
        shifted=dict(arrays,rest=arrays['rest']+[0,0,3],tongue_cables=row.copy())
        shifted['tongue_cables'][0,4]=3
        self.assertTrue(dental_path_checks(shifted,meta)['passed'])

    def test_tongue_laplace_gradients_reproduce_manufactured_affine_fields(self):
        from .rig.tongue_fibers import _laplace_system,_potential
        points,tets=self.tongue_grid();matrix,gradients,_=_laplace_system(points,tets)
        np.testing.assert_allclose(np.asarray(matrix.sum(1)),0,atol=1e-16)
        for axis in range(3):
            coordinate=(points[:,axis]-points[:,axis].min())/np.ptp(points[:,axis])
            fixed=(coordinate==0)|(coordinate==1)
            actual,report=_potential(matrix,fixed,coordinate*fixed)
            np.testing.assert_allclose(actual,coordinate,atol=2e-10)
            gradient=np.einsum('ni,nij->nj',actual[tets],gradients)
            expected=np.zeros_like(gradient);expected[:,axis]=1/np.ptp(points[:,axis])
            np.testing.assert_allclose(gradient,expected,atol=2e-7)
            self.assertLess(report['relative_residual'],1e-7)
        cancel=threading.Event();cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            _potential(matrix,fixed,coordinate,cancel=cancel)

    def test_harmonic_tongue_compartments_are_local_invariant_and_leave_skin_unchanged(self):
        from .rig.tongue_fibers import fit_fields
        points,tets=self.tongue_grid();count=len(tets)
        families=('superior_longitudinal','inferior_longitudinal','transverse','vertical',
                  'genioglossus_anterior','genioglossus_medial','genioglossus_posterior')
        names=[name+'_'+side for name in families for side in ('Left','Right')]+['skin']
        fibers=np.array([[i,m,0,0,1,100.] for m in range(len(names)-1) for i in range(count)]+[[count,14,1,0,0,1000.]])
        arrays=dict(rest=np.r_[points,points[:4]+[.1,.1,.1]],tets=np.r_[tets,np.arange(len(points),len(points)+4)[None]],
                    regions=np.r_[np.ones(count,np.int32),0],fibers=fibers)
        result,report=fit_fields(arrays,dict(muscles=names))
        np.testing.assert_array_equal(arrays['fibers'],fibers)
        np.testing.assert_array_equal(result[-1],fibers[-1])
        np.testing.assert_allclose(np.linalg.norm(result[:,2:5],axis=1),1.,atol=1e-12)
        self.assertTrue(np.isfinite(result).all());self.assertTrue((result[:,5]>0).all())
        occupancy=np.bincount(result[:,0].astype(int),weights=result[:,5])
        self.assertLessEqual(occupancy[:count].max(),3500.+1e-10)
        bands=report['compartments']
        self.assertGreater(bands['superior_longitudinal_Left']['mean_harmonic_height'],.7)
        self.assertLess(bands['inferior_longitudinal_Left']['mean_harmonic_height'],.3)
        self.assertGreater(bands['genioglossus_anterior_Left']['mean_harmonic_front'],
                           bands['genioglossus_posterior_Left']['mean_harmonic_front']+.3)
        self.assertLess(len(result),len(fibers))
        moved,_=fit_fields(dict(arrays,rest=arrays['rest']*2+[.1,.3,-.2]),dict(muscles=names))
        np.testing.assert_allclose(moved,result,rtol=2e-8,atol=2e-8)

    def test_lip_displacement_collar_is_local_bounded_and_keeps_other_components_fixed(self):
        from .rig.articulation import _feather_lip_displacements
        rest=np.array([[0,0,0],[0,.001,0],[.001,0,0],[.001,.001,0],[.002,0,0],[.002,.001,0],
                       [.1,0,0],[.101,0,0],[.1,.001,0]])
        faces=np.array([[0,2,1],[1,2,3],[2,4,3],[3,4,5],[6,7,8]],np.int32)
        seeds=np.arange(9)<2;goal=rest.copy();goal[seeds,2]=.0001
        result=_feather_lip_displacements(rest,goal,faces,seeds,width=.0015)
        np.testing.assert_array_equal(result[seeds],goal[seeds]);np.testing.assert_array_equal(result[4:],rest[4:])
        self.assertTrue((result[2:4,2]>0).all());self.assertTrue((result[2:4,2]<.0001).all())
        np.testing.assert_array_equal(rest[:,2],0)

    def test_oral_area_guard_retains_its_original_reference_over_multiple_steps(self):
        from .rig.articulation import _guard_oral_area
        reference=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        base=reference*.5;goal=reference*.01
        result=_guard_oral_area(base,goal,np.array([[0,1,2]],np.int32),reference=reference)
        area=np.cross(result[1]-result[0],result[2]-result[0])[2]
        self.assertGreaterEqual(area,.15);self.assertFalse(np.array_equal(result,goal))

    def test_bilabial_closure_has_no_intrinsic_opening_but_keeps_vowel_anticipation(self):
        from .rig import articulation
        times=np.arange(7)/30
        isolated=articulation.build(dict(duration=.2,intervals=[dict(s='m',start=0,end=.2)]),times)
        mixed=articulation.build(dict(duration=.2,intervals=[dict(s='m',start=0,end=.08),dict(s='a',start=.08,end=.2)]),times)
        opening=articulation.CHANNELS.index('opening');seal=articulation.CHANNELS.index('seal')
        np.testing.assert_array_equal(isolated['values'][:,opening],0)
        self.assertGreater(mixed['values'][2,opening],0)
        self.assertEqual(mixed['values'][2,seal],1)

    def test_facial_fibers_follow_surface_tangents_without_deep_lining_actuation(self):
        from .rig.skin_fibers import tangent_fields
        surface=np.array([[0,0,0],[.02,0,0],[0,.02,0],[.02,.02,0]])
        tet=np.array([[0,0,-.001],[.004,0,-.001],[0,.004,-.001],[0,0,-.002]])
        rest=np.concatenate((tet,tet-[0,0,.02],tet-[0,0,.03]))
        fibers=np.array([[0,0,0,1,1,100],[0,0,0,0,1,100],[1,0,1,0,0,100],[2,1,0,0,1,100]],float)
        arrays=dict(surface_rest=surface,faces=np.array([[0,1,2],[1,3,2]],np.int32),rest=rest,
                    tets=np.arange(12).reshape(3,4),regions=np.array([0,0,1]),fibers=fibers)
        meta=dict(parts=[dict(name='skin',surface_start=0,surface_count=4,face_start=0,face_count=2)],muscles=['facial','tongue'])
        fitted,report=tangent_fields(arrays,meta,np.ones(4,bool),1.)
        self.assertEqual(len(fitted),2)
        np.testing.assert_allclose(fitted[0,2:5],[0,1,0],atol=1e-14)
        np.testing.assert_array_equal(fitted[1],fibers[3])
        np.testing.assert_array_equal(arrays['fibers'],fibers)
        self.assertEqual(report['original_skin_fibers'],3);self.assertEqual(report['retained_skin_fibers'],1)
        self.assertLessEqual(report['max_normal_component'],1e-14)
        delta=np.array([.03,.04,.05])
        moved,_=tangent_fields(dict(arrays,surface_rest=surface+delta,rest=rest+delta),meta,np.ones(4,bool),1.)
        np.testing.assert_allclose(moved,fitted,atol=1e-13)

    def test_japanese_bilabial_fricative_is_not_an_english_dental_bite(self):
        from .rig import articulation
        aux=dict(format='ja_align.v1',duration=.1,intervals=[dict(s='f',start=0.,end=.1)])
        japanese=articulation.build(aux,[0.,.05,.1])
        english=articulation.build(dict(aux,format='en_align.v1'),[0.,.05,.1])
        seal=articulation.CHANNELS.index('seal');dental=articulation.CHANNELS.index('labiodental')
        self.assertEqual(japanese['intervals'][0]['phone'],'f')
        self.assertEqual(japanese['intervals'][0]['articulator_phone'],'ɸ')
        self.assertGreater(japanese['values'][:,seal].max(),0)
        self.assertLess(japanese['values'][:,seal].max(),1)
        self.assertEqual(japanese['values'][:,dental].max(),0)
        self.assertEqual(english['values'][:,dental].max(),1)

    def test_labiodental_contact_requires_the_lower_lip_to_reach_teeth(self):
        from .rig import biomechanics
        rest=np.array([[x,y,z] for y in (.01,0,.002) for x,z in ((-.01,0),(.01,0),(0,.01))])
        ids=np.arange(9);arrays=dict(surface_rest=rest,source_ids=ids,faces=np.arange(9).reshape(3,3))
        class Model:
            def group(self,name):
                return {'upper_lip':ids<3,'lower_lip':(ids>=3)&(ids<6),
                        'teeth':ids>=6,'upper_teeth_and_gums':ids>=6}[name]
        times=np.array([0.,1/30]);profile=dict(times=times,intervals=[dict(phone='f',start=0.,end=1/30)])
        far=np.tile(rest,(2,1,1));near=far.copy();near[1,3:6,1]+=.0018
        with patch('server.vhuman.rig.gnm_model.GNMModel',return_value=Model()):
            self.assertFalse(biomechanics.speech_contact_metrics(arrays,profile,far,None)['passed'])
            report=biomechanics.speech_contact_metrics(arrays,profile,near,None)
        self.assertTrue(report['passed']);self.assertEqual(report['events'][0]['contact_type'],'labiodental')
        self.assertAlmostEqual(report['events'][0]['gap_mm'],.2)

    def test_labiodental_measurement_selects_the_inner_lip_instead_of_the_outer_seam(self):
        from .rig.biomechanics_contacts import ContactGeometry,labiodental_samples
        from .rig import articulation
        triangle=np.array([[-.01,0,0],[.01,0,0],[0,0,.001]])
        rest=np.concatenate((triangle+[0,.01,.006],triangle+[0,0,.006],triangle,triangle+[0,.002,0]))
        source=np.arange(12);faces=source.reshape(4,3);lower=(source>=3)&(source<9)
        teeth=source>=9
        ids,selected=labiodental_samples(rest,faces,lower,teeth)
        self.assertTrue((ids>=6).all());self.assertTrue((ids<9).all())
        np.testing.assert_array_equal(selected,faces[3:])
        class Model:
            def group(self,name):
                return {'upper_lip':source<3,'lower_lip':lower,'teeth':teeth,'upper_teeth_and_gums':teeth}[name]
        geometry=ContactGeometry(dict(surface_rest=rest,faces=faces,source_ids=source),Model())
        values=np.zeros(len(articulation.CHANNELS));values[articulation.CHANNELS.index('labiodental')]=1
        objective,goals=geometry.objective(values)
        self.assertGreater(objective(rest)[0],1.)
        near=rest.copy();near[6:9,1]+=.0018
        np.testing.assert_array_equal(objective(near),[0.])
        self.assertEqual(goals[0]['target_gap_mm'],.35)

    def test_regional_tracking_rejects_a_stationary_small_articulator(self):
        from .rig.biomechanics import regional_target_metrics
        rest=np.zeros((1004,3));target=rest.copy();target[1000:,1]=.006
        arrays=dict(surface_rest=rest)
        meta=dict(parts=[dict(name='skin',surface_start=0,surface_count=1000),
                         dict(name='tongue',surface_start=1000,surface_count=4)])
        self.assertLess(np.sqrt(np.mean((rest-target)**2))*1000,2)
        report=regional_target_metrics(arrays,meta,target,rest)
        self.assertTrue(report['skin']['passed']);self.assertFalse(report['tongue']['passed'])
        self.assertAlmostEqual(report['tongue']['neutral_rms_mm'],6/np.sqrt(3))
        self.assertTrue(regional_target_metrics(arrays,meta,target,target*.6)['tongue']['passed'])
        target[:10,1]=.001
        report=regional_target_metrics(arrays,meta,target,rest)
        self.assertFalse(report['moving_skin']['passed'])

    def test_skull_tethers_exclude_mandible_and_scale_with_nodal_area(self):
        from .rig import skin_attachments
        arrays,meta=fixture();arrays['regions']=np.array([0],np.int32)
        arrays['anchor_bone'][-1]=1
        meta['landmarks']=dict(mouth=[0,-.01,.02],eye_y=.005)
        exterior=np.ones(4,bool)
        n,t,report=skin_attachments.frames(arrays,meta,exterior,1.)
        self.assertGreater(report['sliding_anchors'],0)
        np.testing.assert_array_equal(n[-1],np.zeros(3));self.assertEqual(t[-1],0)
        _,twice,_=skin_attachments.frames(arrays,meta,exterior,1.,tangent_density=20000.)
        np.testing.assert_allclose(twice,t/2,rtol=0,atol=1e-16)
        none,zero,report=skin_attachments.frames(arrays,meta,np.zeros(4,bool),1.)
        np.testing.assert_array_equal(none,np.zeros_like(n));np.testing.assert_array_equal(zero,np.zeros_like(t))
        self.assertEqual(report['sliding_anchors'],0)
        arrays.update(anchor_normals=n,anchor_tangent_compliance=t)
        anatomy.validate(arrays,meta)
        arrays['anchor_normals'][-1]=[0,0,1]
        with self.assertRaisesRegex(ValueError,'skull frame'):anatomy.validate(arrays,meta)

    def test_oral_target_guard_checks_interior_area_not_only_endpoint(self):
        from .rig.articulation import _guard_oral_area
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[2,2,0]],float)
        goal=rest.copy();goal[1,0]=-1;goal[2,1]=-1
        limited=_guard_oral_area(rest,goal,np.array([[0,1,2]],np.int32))
        np.testing.assert_array_equal(limited,rest+.25*(goal-rest))
        np.testing.assert_array_equal(limited[3],rest[3])

    def test_short_closure_survives_a_coarse_audio_timeline(self):
        from .rig import articulation
        aux=dict(duration=.1,intervals=[dict(s='m',start=.046,end=.054)])
        profile=articulation.build(aux,[0.,.1])
        self.assertEqual(profile['values'][:,articulation.CHANNELS.index('seal')].max(),1.)
        silent=articulation.build(aux,[0.,.1],strength=0)
        np.testing.assert_array_equal(silent['values'],np.zeros_like(silent['values']))

    def test_japanese_closures_follow_consonant_place(self):
        from .rig import articulation
        for next_phone,expected in [('p','m'),('k','ŋ'),('t','n'),('s','n')]:
            np.testing.assert_array_equal(articulation._pose('N',next_phone),articulation._pose(expected))
        for next_phone in ('p','k','t','s'):
            np.testing.assert_array_equal(articulation._pose('cl',next_phone),articulation._pose(next_phone))
        self.assertGreater(articulation._pose('N')[7],0)
        self.assertEqual(articulation._pose('N')[5],0)
        with self.assertRaisesRegex(ValueError,'phoneme symbols'):
            articulation._pose('N',None)

    def test_viseme_articulation_rejects_nonfinite_time_bases_and_empty_curves(self):
        from .rig import articulation
        for fps,values in [(float('nan'),[[1]]),(float('inf'),[[1]]),(30.,[])]:
            with self.assertRaises(ValueError):
                articulation.build(dict(duration=.1,visemes=dict(names=['PP'],frames=values,fps=fps)),[0.,.1])

    def test_articulation_closes_seams_without_flattening_lip_thickness(self):
        from .rig import articulation
        upper=np.array([[x,y,0] for x in (-.01,0,.01) for y in (.001,.004)])
        lower=np.array([[x,y,0] for x in (-.01,0,.01) for y in (-.001,-.004)])
        teeth=np.array([[x,y,z] for x in (-.01,0,.01) for y,z in ((.006,.003),(.005,.004))])
        rest=np.concatenate((upper,lower,teeth));source=np.arange(len(rest))
        class Model:
            def group(self,name):
                return {'upper_lip':source<6,'lower_lip':(source>=6)&(source<12),
                        'tongue':source<0,'teeth':source>=12,'upper_teeth_and_gums':source>=12}[name]
        arrays=dict(source_ids=source,surface_rest=rest)
        for channel in ('seal','labiodental'):
            values=np.zeros((1,len(articulation.CHANNELS)));values[0,articulation.CHANNELS.index(channel)]=1
            result=articulation.geometric_targets(rest,arrays,Model(),dict(values=values),0)
            np.testing.assert_array_equal(result[:,0],rest[:,0])
            np.testing.assert_array_equal(result[12:],rest[12:])
            for ids in (np.arange(6).reshape(3,2),np.arange(6,12).reshape(3,2)):
                np.testing.assert_allclose(result[ids[:,1]]-result[ids[:,0]],rest[ids[:,1]]-rest[ids[:,0]],atol=1e-18)
            if channel=='seal':
                self.assertAlmostEqual(result[2,1]-result[8,1],.0001)
                corners=[0,1,4,5,6,7,10,11]
                np.testing.assert_array_equal(result[corners],rest[corners])
            else:
                self.assertAlmostEqual(result[8,1],.005-.00015)
                self.assertAlmostEqual(result[8,2],.004+.0002)
            delta=np.array([.02,-.05,.03])
            moved=articulation.geometric_targets(rest+delta,dict(arrays,surface_rest=rest+delta),Model(),dict(values=values),0)
            np.testing.assert_allclose(moved,result+delta,rtol=0,atol=1e-16)

    def test_inverse_worker_budget_is_explicit_and_bounded(self):
        from .rig.biomechanics_trials import validate_workers
        validate_workers(1,4);validate_workers(4,4)
        for workers,threads in ((True,4),(0,4),(5,4),(33,64),(1,0),(1,4.),(1,None)):
            with self.assertRaisesRegex(ValueError,'inverse workers'):
                validate_workers(workers,threads)

    def test_inverse_samples_cover_lips_and_moving_exterior_without_hidden_overweighting(self):
        from .rig.biomechanics import inverse_samples
        rest=np.zeros((200,3));rest[:,0]=np.arange(200)*.0001
        importance=np.ones(200);importance[:40]=0;importance[60:100]=4
        target=rest.copy();target[:40,1]+=.02;target[100:120,1]+=.001;target[160:,1]+=.1
        meta=dict(parts=[dict(name='skin',surface_start=0,surface_count=160),
                         dict(name='eye_left',surface_start=160,surface_count=40)])
        ids=inverse_samples(dict(surface_rest=rest,fit_importance=importance),meta,target,
                            part_samples=4,feature_samples=8,moving_samples=8)
        lip=ids[(ids>=60)&(ids<100)];moving=ids[(ids>=100)&(ids<120)]
        self.assertGreaterEqual(len(lip),8);self.assertGreaterEqual(len(moving),8)
        self.assertGreater(np.ptp(rest[lip,0]),.9*np.ptp(rest[60:100,0]))
        self.assertEqual(ids[ids<40].tolist(),[0])
        self.assertEqual(ids[ids>=160].tolist(),[160,170,180,190])
        self.assertEqual(len(ids),len(np.unique(ids)))
        legacy=inverse_samples(dict(surface_rest=rest),meta,target,part_samples=4)
        self.assertEqual(legacy.tolist(),[0,40,80,120,160,170,180,190])

    def test_facial_anchors_select_front_skin_and_follow_translation_and_scale(self):
        # The posterior point exactly matches XY; the frontal point is only
        # 1 mm away. A distant anterior outlier must stay outside the search.
        skin=np.array([[0,0,-.18],[.001,0,.02],[.02,0,.5]])
        point=anatomy.front_surface_anchor(skin,0,0,1.)
        np.testing.assert_array_equal(point,skin[1])
        translation=np.array([.01,.03,-.05]);factor=.7
        moved=anatomy.front_surface_anchor(skin*factor+translation,translation[0],translation[1],factor)
        np.testing.assert_allclose(moved,point*factor+translation,rtol=0,atol=1e-16)
        with self.assertRaisesRegex(ValueError,'anchor'):
            anatomy.front_surface_anchor(np.empty((0,3)),0,0,1.)

    @unittest.skipUnless(shutil.which('git'),'Git required for source-patch validation')
    def test_native_patch_is_reproducible_and_preserves_unrelated_edits(self):
        from .rig import lightgeom_patch
        from .service import ROOT
        original=ROOT/'third_party/LightGeom'
        if not (original/'.git').exists():
            self.skipTest('optional LightGeom checkout missing')
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        receipt=lightgeom_patch.receipt()
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            checkout=Path(directory)/'LightGeom'
            subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(original),str(checkout)],check=True)
            subprocess.run(['git','sparse-checkout','init','--no-cone'],cwd=checkout,check=True)
            patterns='\n'.join('/'+name for name in receipt['files'])+'\n/src/lightphysics/lightphysics.c\n'
            subprocess.run(['git','sparse-checkout','set','--no-cone','--stdin'],input=patterns,text=True,cwd=checkout,check=True)
            subprocess.run(['git','checkout','--quiet','--detach',receipt['base_revision']],cwd=checkout,check=True)
            lightgeom_patch.apply(checkout)
            self.assertTrue(lightgeom_patch.verify(checkout))
            lightgeom_patch.apply(checkout)
            other=checkout/'src/lightphysics/lightphysics.c'
            other.write_text(other.read_text()+'\n/* unrelated user edit */\n')
            before=other.read_bytes()
            self.assertFalse(lightgeom_patch.verify(checkout))
            with self.assertRaises(ValueError):lightgeom_patch.apply(checkout)
            self.assertEqual(other.read_bytes(),before)

    def test_joint_motion_targets_use_evaluated_geometry_once(self):
        from types import SimpleNamespace
        from .rig import biomechanics
        from .reconstruction import gnm_motion  # import before mocking its model dependency
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        rest=np.array([[0,0,0],[0,0,.01],[0,0,.02],[0,0,.03]],float)
        groups={'skin_exterior':np.array([1,1,0,0],bool),
                'lower_teeth_and_gums':np.array([0,0,1,0],bool),
                'tongue':np.array([0,0,0,1],bool),'mouth_sock':np.zeros(4,bool)}
        model=SimpleNamespace(group=lambda name:groups[name],evaluate=lambda *args,**kwargs:(rest.copy(),None))
        bind=dict(identity=np.zeros(1),bind_residual=np.zeros_like(rest),scale=1.,alignment=np.eye(3),offset=np.zeros(3))
        matrix=np.eye(4);matrix[1,3]=-.003
        evaluator=SimpleNamespace(jidx={'teeth_lower':0},evaluate=lambda controls:{'skin':[matrix]})
        posed=rest[:2]+[0,-.004,0]
        teacher=unittest.mock.Mock();teacher.eval.return_value=posed
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory)
            (rig/'rig.json').write_text(json.dumps(dict(controls=[{'name':'jawOpen'},{'name':'headYaw'}],
                face_model_source={'expression_mapping':{'controls':['jawOpen'],'coefficients':[[0]]}})))
            with patch('server.vhuman.rig.gnm_model.GNMModel',return_value=model), \
                 patch('server.vhuman.reconstruction.gnm_motion.rig_bind',return_value=bind), \
                 patch('server.vhuman.rig.rigdef.Rig',return_value=evaluator), \
                 patch('server.vhuman.rig.biomechanics_mapping.load_skinning',return_value=(np.zeros((2,4),np.int32),np.tile([1.,0,0,0],(2,1)))), \
                 patch('server.vhuman.rig.biomechanics_mapping.load_native_rig',return_value=teacher):
                _,target,_=next(biomechanics.target_sequence(rig,dict(source_ids=np.arange(4),surface_rest=rest),
                    {'landmarks':{'mouth':[0,0,0]}},[dict(t=0,v={'jawOpen':.5,'headYaw':1})]))
                evaluate=unittest.mock.Mock(return_value=(rest.copy(),None));model.evaluate=evaluate
                rotations=np.zeros((1,4,3));rotations[0,0,1]=.7;rotations[0,1,0]=.2
                next(biomechanics.target_sequence(rig,dict(source_ids=np.arange(4),surface_rest=rest),
                    {'landmarks':{'mouth':[0,0,0]}},[dict(t=0,v={})],
                    motion=dict(expression=np.zeros((1,1)),rotations=rotations,transforms=[{'head':np.eye(4)}])))
                received=evaluate.call_args.args[2]
                np.testing.assert_array_equal(received,rotations[0])
                self.assertEqual(received[1,0],.2)
                # A fitted ocular bind correction is transported once, while
                # the actual rig teacher keeps its evaluated joint geometry.
                delta=np.zeros_like(rest);delta[:2]=[.0003,-.0002,.0001]
                def fitted_evaluation(*args,**kwargs):
                    return rest+kwargs['bind_residual'],None
                model.evaluate=fitted_evaluation
                matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
                _,fitted_target,_=next(biomechanics.target_sequence(rig,
                    dict(source_ids=np.arange(4),surface_rest=rest,bind_delta=delta),
                    {'landmarks':{'mouth':[0,0,0]}},[dict(t=0,v={'jawOpen':.5,'headYaw':1})]))
                np.testing.assert_allclose(fitted_target[:2],posed+delta[:2]@matrix[:3,:3].T)
                np.testing.assert_allclose(fitted_target[2:],rest[2:]@matrix[:3,:3].T+matrix[:3,3])
                _,captured,_=next(biomechanics.target_sequence(rig,
                    dict(source_ids=np.arange(4),surface_rest=rest,bind_delta=delta),
                    {'landmarks':{'mouth':[0,0,0]}},[dict(t=0,v={})],
                    motion=dict(expression=np.zeros((1,1)),rotations=rotations,transforms=[{'head':np.eye(4)}])))
                np.testing.assert_allclose(captured,rest+delta)
                def expression_evaluation(identity,expression,*args,**kwargs):
                    return rest+kwargs['bind_residual']+[.002*expression[0],0,0],None
                model.evaluate=expression_evaluation
                for fitted,extra in ((.5,0.),(.7,.0002)):
                    _,calibrated,_=next(biomechanics.target_sequence(rig,
                        dict(source_ids=np.arange(4),surface_rest=rest,bind_delta=delta,
                             expression_coefficients=np.array([[fitted]]),
                             expression_seed_coefficients=np.array([[.5]])),
                        {'expression_controls':['jawOpen'],'landmarks':{'mouth':[0,0,0]}},
                        [dict(t=0,v={'jawOpen':.5})]))
                    # The teacher supplies the calibrated response already;
                    # only capture-minus-seed and the bind delta are added.
                    correction=delta[:2]+[extra,0,0]
                    np.testing.assert_allclose(calibrated[:2],posed+correction@matrix[:3,:3].T)
        np.testing.assert_allclose(target[:2],posed)
        np.testing.assert_allclose(target[2:],rest[2:]+[0,-.003,0])
        np.testing.assert_allclose(teacher.eval.call_args.args[0],[.5,0])
        self.assertEqual(teacher.close.call_count,4)

    def test_semantic_gaze_targets_use_exact_rigid_eye_transforms(self):
        from types import SimpleNamespace
        from .rig import biomechanics
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        rest=np.array([[0,0,0],[0,.01,0],[.03,.05,.01],[-.03,.05,.01]])
        source=np.arange(4);groups={'skin_exterior':source<2,'lower_teeth_and_gums':np.zeros(4,bool),
                                  'tongue':np.zeros(4,bool),'mouth_sock':np.zeros(4,bool)}
        model=SimpleNamespace(group=lambda name:groups[name],evaluate=lambda *args,**kwargs:(rest.copy(),None))
        bind=dict(identity=np.zeros(1),bind_residual=np.zeros_like(rest),scale=1.,alignment=np.eye(3),offset=np.zeros(3))
        eye=np.eye(4);eye[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];eye[:3,3]=[.004,.003,.002]
        evaluator=SimpleNamespace(jidx={'teeth_lower':0,'eye_L':1,'eye_R':2},
                                  evaluate=lambda controls:{'skin':[np.eye(4),eye,eye]})
        teacher=unittest.mock.Mock();teacher.eval.return_value=rest[:2]
        meta=dict(landmarks={'mouth':[0,0,0]},parts=[dict(name='eye_left',surface_start=2,surface_count=1),
                                                   dict(name='eye_right',surface_start=3,surface_count=1)])
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory);(rig/'rig.json').write_text(json.dumps(dict(controls=[dict(name='eyeLookUpLeft')],
                face_model_source={'expression_mapping':{'controls':['eyeLookUpLeft'],'coefficients':[[0]]}})))
            with patch('server.vhuman.rig.gnm_model.GNMModel',return_value=model), \
                 patch('server.vhuman.reconstruction.gnm_motion.rig_bind',return_value=bind), \
                 patch('server.vhuman.rig.rigdef.Rig',return_value=evaluator), \
                 patch('server.vhuman.rig.biomechanics_mapping.load_native_rig',return_value=teacher):
                _,target,_=next(biomechanics.target_sequence(rig,dict(source_ids=source,surface_rest=rest),meta,
                                                           [dict(t=0.,v={'eyeLookUpLeft':.5})]))
        np.testing.assert_array_equal(target[:2],rest[:2])
        np.testing.assert_allclose(target[2:],rest[2:]@eye[:3,:3].T+eye[:3,3],atol=1e-15)
        self.assertGreater(np.max(abs(target[2:]-rest[2:])),.01)

    @unittest.skipUnless(HAS_SCIPY,'independent CPU rotation oracle')
    def test_capture_factors_complete_head_motion_and_preserves_local_gaze(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from .rig import biomechanics,biomechanics_mapping
        from .reconstruction import gnm_motion
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        joints=np.array([[0,0,0],[0,.1,0],[.03,.12,.02],[-.03,.12,.02]])
        rest=joints+np.array([[0,.01,.02],[0,.02,.03],[0,0,.01],[0,0,.01]])
        parents=[-1,0,1,1]
        rotations=np.array([[.2,0,0],[0,.3,0],[0,0,.1],[-.1,0,0]])
        translation=np.array([.02,-.01,.03])
        bind=dict(identity=np.zeros(1),bind_residual=np.zeros_like(rest),scale=1.7,
                  alignment=Rotation.from_rotvec([0,0,.15]).as_matrix().astype(np.float32),offset=np.array([.04,-.03,.02]))
        def oracle(rotation,translation):
            local=Rotation.from_rotvec(rotation).as_matrix();world_r=[];world_t=[]
            for i,parent in enumerate(parents):
                world_r.append(local[i] if parent<0 else world_r[parent]@local[i])
                world_t.append(joints[i]+translation if parent<0 else
                               world_r[parent]@(joints[i]-joints[parent])+world_t[parent])
            return np.array(world_r),np.array(world_t)
        def evaluate(identity,expression=None,rotation=None,shift=None,**kwargs):
            r,p=oracle(rotations if rotation is None else rotation,translation if shift is None else shift)
            points=rest+kwargs['bind_residual']
            return np.einsum('vij,vj->vi',r,points-joints)+p,p
        groups={'skin_exterior':np.array([1,1,0,0],bool)}
        model=SimpleNamespace(parents=parents,group=lambda name:groups[name],evaluate=evaluate,
            data=dict(joint_names=np.array(['neck','head','left_eye','right_eye']),
                      template_joint_positions=joints,joint_identity_basis=np.zeros((1,4,3))))
        transforms=biomechanics_mapping.capture_transforms(model,bind,rotations,translation)
        r,p=oracle(rotations,translation)
        for i,name in enumerate(model.data['joint_names']):
            source=bind['scale']*rest[i]@bind['alignment'].T+bind['offset']
            expected=bind['scale']*(r[i]@(rest[i]-joints[i])+p[i])@bind['alignment'].T+bind['offset']
            actual=transforms[name][:3,:3]@source+transforms[name][:3,3]
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory)
            (rig/'rig.json').write_text(json.dumps(dict(controls=[],face_model_source={
                'expression_mapping':{'controls':[],'coefficients':[[]]}})))
            with patch('server.vhuman.rig.gnm_model.GNMModel',return_value=model), \
                 patch.object(gnm_motion,'rig_bind',return_value=bind), \
                 patch('server.vhuman.rig.rigdef.Rig',return_value=SimpleNamespace()), \
                 patch.object(biomechanics_mapping,'load_native_rig',side_effect=AssertionError('capture uses exact geometry')):
                _,target,_=next(biomechanics.target_sequence(rig,dict(source_ids=np.arange(4)),{},
                    [dict(t=0,v={'headYaw':.9})],motion=dict(expression=np.zeros((1,1)),
                    rotations=rotations[None],translation=translation[None])))
        posed,_=evaluate(bind['identity'],rotation=rotations,shift=translation,bind_residual=bind['bind_residual'])
        posed=bind['scale']*posed@bind['alignment'].T+bind['offset']
        head=transforms['head']
        np.testing.assert_allclose(target@head[:3,:3].T+head[:3,3],posed,atol=1e-14,rtol=0)
        # Applying the relative eye transform after the same local solve gives
        # the independently evaluated captured globe, with no semantic gaze.
        for i,name in ((2,'left_eye'),(3,'right_eye')):
            local=np.linalg.inv(head)@transforms[name]
            source=bind['scale']*rest[i]@bind['alignment'].T+bind['offset']
            np.testing.assert_allclose(local[:3,:3]@source+local[:3,3],target[i],atol=1e-14,rtol=0)

    def test_package_rejects_inverted_tetrahedra_and_bad_embeddings(self):
        arrays,meta=fixture();anatomy.validate(arrays,meta)
        bad={**arrays,'tets':arrays['tets'][:,[0,2,1,3]]}
        with self.assertRaisesRegex(ValueError,'volume'):anatomy.validate(bad,meta)
        bad={**arrays,'embed_weights':arrays['embed_weights']+.01}
        with self.assertRaisesRegex(ValueError,'embedding'):anatomy.validate(bad,meta)
        for delta in (np.zeros((4,3)),np.full((17821,3),np.nan)):
            with self.assertRaisesRegex(ValueError,'bind residual'):
                anatomy.validate({**arrays,'bind_delta':delta},meta)
        anatomy.validate({**arrays,'bind_delta':np.zeros((17821,3))},meta)
        expression_meta=dict(meta,expression_controls=['mouthClose'])
        fitted=dict(arrays,expression_coefficients=np.zeros((2,1)),expression_seed_coefficients=np.ones((2,1)))
        anatomy.validate(fitted,expression_meta)
        for seed in (np.zeros((3,1)),np.full((2,1),np.nan),np.zeros((2,2))):
            with self.assertRaisesRegex(ValueError,'expression'):
                anatomy.validate(dict(fitted,expression_seed_coefficients=seed),expression_meta)
        with self.assertRaisesRegex(ValueError,'expression'):
            anatomy.validate(dict(arrays,expression_seed_coefficients=np.zeros((2,1))),expression_meta)

    @unittest.skipUnless(HAS_SCIPY,'CPU fitting SciPy dependency')
    def test_root_cap_is_closed_and_neutral_embedding_is_exact(self):
        arrays,_=fixture();rest=arrays['rest'];faces=arrays['faces'][:3]
        v,f=anatomy.close_root(rest,faces)
        self.assertEqual(len(anatomy.boundary_edges(f)),0)
        points=rest*.2+rest.mean(0)*.8
        ids,weights=anatomy.embed(points,rest,arrays['tets'])
        np.testing.assert_allclose((rest[ids]*weights[...,None]).sum(1),points,atol=1e-14)

    @unittest.skipUnless(HAS_SCIPY,'CPU anatomy SciPy dependency')
    def test_embedding_search_finds_an_enclosing_tet_beyond_near_centroids(self):
        nodes=[];tets=[]
        for i in range(80):
            base=np.array([.002+i*.00004,.002,.002]);start=len(nodes)
            nodes.extend(base+np.array([[0,0,0],[.0001,0,0],[0,.0001,0],[0,0,.0001]]))
            tets.append(np.arange(start,start+4))
        start=len(nodes)
        nodes.extend([[-.02,-.02,-.02],[.02,-.02,-.02],[0,.02,-.02],[0,0,.02]])
        tets.append(np.arange(start,start+4))
        nodes,tets=np.array(nodes),np.array(tets,np.int32)
        point=np.zeros((1,3))
        ids,weights=anatomy.embed(point,nodes,tets)
        np.testing.assert_array_equal(ids,[np.arange(start,start+4)])
        self.assertGreaterEqual(weights.min(),0)
        np.testing.assert_allclose((nodes[ids]*weights[...,None]).sum(1),point,atol=1e-14,rtol=0)
        with self.assertRaisesRegex(ValueError,'support'):
            anatomy.embed(np.array([[1.,1.,1.]]),nodes,tets)

    @unittest.skipUnless(HAS_SCIPY,'CPU anatomy SciPy dependency')
    def test_tongue_prior_encloses_source_folds_without_changing_surface(self):
        from scipy.spatial import ConvexHull
        arrays,_=fixture();original=arrays['rest'].copy()
        nodes,faces=anatomy.tongue_domain(original)
        self.assertEqual(len(anatomy.boundary_edges(faces)),0)
        hull=ConvexHull(nodes)
        signed=original@hull.equations[:,:3].T+hull.equations[:,3]
        self.assertLess(signed.max(),0)
        np.testing.assert_array_equal(arrays['rest'],original)

    @unittest.skipUnless(HAS_SCIPY,'CPU embedding SciPy dependency')
    def test_embedding_bounds_physical_offset_and_amplification(self):
        rest=np.array([[0,0,0],[.02,0,0],[0,.0001,0],[0,0,.02]])
        tets=np.array([[0,1,2,3]],np.int32)
        point=np.array([[.002,-.00005,.002]])
        ids,weights=anatomy.embed(point,rest,tets)
        np.testing.assert_allclose((rest[ids]*weights[...,None]).sum(1),point,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'support'):
            anatomy.embed(np.array([[.002,-.0002,.002]]),rest,tets)

    @unittest.skipUnless(HAS_TORCH,'CPU fitting PyTorch dependency')
    def test_rigid_dental_fit_uses_cpu_and_preserves_distances(self):
        from . import gpu
        rest=np.array([[-.01,0,.02],[.01,0,.02],[-.01,0,.04],[.01,0,.04]])
        hinge=np.array([0,.02,-.02])
        q=.2;c,s=np.cos(q),np.sin(q);r=rest-hinge
        goal=np.column_stack((rest[:,0],hinge[1]+c*r[:,1]-s*r[:,2]-.003333333333333333*q,
                              hinge[2]+s*r[:,1]+c*r[:,2]))
        with patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU probe')):
            fitted,coords,report=anatomy.fit_jaw(rest,goal[None],hinge,iterations=240,threads=1)
        self.assertLess(report['rms_mm'],.2)
        self.assertLess(np.linalg.norm(fitted-hinge),.004)
        self.assertGreater(coords[0,0],.1)

    def test_full_cli_selects_cpu_before_runtime_configuration(self):
        import contextlib,io
        from . import cli,runtime
        def configure(args):
            self.assertEqual(args.inference_backend,'cpu')
        with patch.object(runtime,'configure_args',configure),patch.object(cli,'cmd_rig_fit_anatomy',return_value={'ok':True}) as handler,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['rig-fit-anatomy','--head','fixture','--fit-sockets','--fit-mouth','--fit-dentition','--fit-tongue','--extrinsic-tongue-cables']),0)
            self.assertTrue(handler.call_args.args[0].fit_sockets)
            self.assertTrue(handler.call_args.args[0].fit_mouth)
            self.assertTrue(handler.call_args.args[0].fit_dentition)
            self.assertTrue(handler.call_args.args[0].fit_tongue)
            self.assertTrue(handler.call_args.args[0].extrinsic_tongue_cables)

    def test_http_job_wrappers_select_cpu_and_forward_cpu_interpreter(self):
        import argparse
        from .app import App
        from . import gpu
        from .rig import speech,soft_tissue
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        def handler(*args,**kwargs):
            self.assertEqual(gpu.backend(),'cpu')
            self.assertEqual(kwargs['python'],'/cpu/venv/bin/python')
            return {'ok':True}
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            app=App(argparse.Namespace(work=directory,qwen_python=None,mock=True,
                inference_backend='auto',rig_python='/cpu/venv/bin/python'))
            with patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU probe')), \
                 patch.object(speech,'speech_job',handler),patch.object(anatomy,'anatomy_job',handler), \
                 patch.object(soft_tissue,'soft_tissue_job',handler):
                for kind,request in [('rig_fit_anatomy',{}),('rig_soft_tissue',{'mode':'full'}),
                                     ('rig_speech',{'motion_model':'biomechanics'})]:
                    result=app.jobs.handlers[kind](request,lambda *_:None,threading.Event())
                    self.assertTrue(result['ok']);self.assertEqual(result['device'],'cpu')

    def test_anatomy_job_forwards_socket_fit_to_cpu_worker(self):
        from types import SimpleNamespace
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory);(rig/'anatomy').mkdir()
            (rig/'anatomy/anatomy.json').write_text('{"geometry_only":true}')
            service=SimpleNamespace(rig_file=lambda *_:rig/'rig.json')
            with patch.object(anatomy,'run_worker') as worker:
                result=anatomy.anatomy_job(service,dict(head_id='fixture',fit_sockets=True,fit_mouth=True,fit_dentition=True,fit_tongue=True,
                    fit_tongue_root=True,sliding_skin=True,tangent_fibers=True,harmonic_tongue_fibers=True,extrinsic_tongue_cables=True,
                    mouth_fairing=.25,mesh_timeout=3600),
                    lambda *_:None,threading.Event(),python='/cpu/python')
            self.assertIn('--fit-sockets',worker.call_args.args[1])
            self.assertIn('--fit-mouth',worker.call_args.args[1])
            self.assertIn('--fit-dentition',worker.call_args.args[1])
            self.assertIn('--fit-tongue',worker.call_args.args[1])
            for flag in ('fit-tongue-root','sliding-skin','tangent-fibers','harmonic-tongue-fibers','extrinsic-tongue-cables'):
                self.assertIn('--'+flag,worker.call_args.args[1])
            arguments=worker.call_args.args[1]
            self.assertEqual(arguments[arguments.index('--mouth-fairing')+1],.25)
            self.assertEqual(arguments[arguments.index('--mesh-timeout')+1],3600)
            self.assertEqual(worker.call_args.kwargs['python'],'/cpu/python')
            self.assertTrue(result['report']['geometry_only'])
            with patch.object(anatomy,'run_worker',side_effect=AssertionError('worker started')):
                for flag in ('fit_sockets','fit_mouth','fit_dentition','fit_tongue','fit_tongue_root','sliding_skin','tangent_fibers','harmonic_tongue_fibers','extrinsic_tongue_cables'):
                    with self.assertRaisesRegex(ValueError,'boolean'):
                        anatomy.anatomy_job(service,dict(head_id='fixture',**{flag:'false'}),
                            lambda *_:None,threading.Event())
                for values in (dict(mesh_timeout=True),dict(mesh_timeout=59),dict(mesh_timeout=7201),
                               dict(mouth_fairing=float('nan')),dict(mouth_fairing=.6),dict(mouth_fairing=.25)):
                    with self.assertRaises(ValueError):
                        anatomy.anatomy_job(service,dict(head_id='fixture',**values),lambda *_:None,threading.Event())

    @unittest.skipUnless(importlib.util.find_spec('wildmeshing') is not None,'offline fTetWild dependency')
    def test_actual_cpu_volume_meshing_and_binding(self):
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        arrays,_=fixture()
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rest,tets=anatomy.tetrahedralize(arrays['rest'],arrays['faces'],Path(directory),'tetra',edge=.006,threads=1,cache=Path(directory)/'cache')
            self.assertGreater(len(tets),0)
            self.assertGreater(volumes(rest,tets).min(),0)
            ids,weights=anatomy.embed(arrays['rest'],rest,tets)
            np.testing.assert_allclose((rest[ids]*weights[...,None]).sum(1),arrays['rest'],atol=1e-10)
            with patch('server.vhuman.process.run',side_effect=AssertionError('a cache hit must not launch fTetWild')):
                cached,indices=anatomy.tetrahedralize(arrays['rest'],arrays['faces'],Path(directory),'repeat',
                    edge=.006,threads=1,cache=Path(directory)/'cache')
            np.testing.assert_array_equal(cached,rest);np.testing.assert_array_equal(indices,tets)

    def test_articulation_distinguishes_tongue_contacts_and_short_closures(self):
        from .rig import articulation
        times=np.arange(31)/30
        aux=dict(duration=1.,intervals=[dict(s='m',start=.201,end=.209),dict(s='n',start=.35,end=.5),
                                      dict(s='θ',start=.6,end=.7),dict(s='k',start=.8,end=.9)])
        profile=articulation.build(aux,times)
        values=profile['values'];names=profile['channels']
        self.assertEqual(values[:,names.index('seal')].max(),1.)
        self.assertGreater(values[12,names.index('tip_up')],.8)
        self.assertLess(values[12,names.index('seal')],.01)
        self.assertGreater(values[19,names.index('tip_forward')],.8)
        self.assertGreater(values[25,names.index('dorsum_up')],.9)
        with self.assertRaises(ValueError):articulation.build(aux,[0,.1,.1])
        silent=articulation.build(aux,times,strength=0)
        np.testing.assert_array_equal(silent['values'],np.zeros_like(values))
        controls=articulation.compose_controls({'jawOpen':.8,'mouthSmileLeft':.5,'browInnerUp':.3},silent['values'][0])
        self.assertEqual(controls['jawOpen'],0)
        self.assertNotIn('mouthSmileLeft',controls)
        self.assertEqual(controls['browInnerUp'],.3)
        emotional=articulation.compose_controls({},silent['values'][0],emotions={'joy':1})
        self.assertGreater(emotional['mouthSmileLeft'],0)


@unittest.skipUnless(LIBRARY.is_file(),'build LightGeom lightphysics_vhuman_full')
class NativeBiomechanicsTests(unittest.TestCase):
    def test_local_embedded_contacts_use_shared_node_mass_and_update_unconstrained_samples(self):
        lib=library()
        project=getattr(lib,'lightphysics_surface_contact_project_embedded',None)
        if project is None:self.skipTest('rebuild embedded CPU contact API')
        for fragile in (False,True):
            with self.subTest(volume_guard=fragile):
                initial=np.array([[-.01/.9,-.01/.9,-.001/.9],[.01/.9,-.01/.9,-.001/.9],
                    [0,.01/.9,-.001/.9],[0,0,.01],[0,0,.03],[.02,.02,.02],[-.02,.02,.02]])
                current=initial.copy();current[3:5,2]-=.024
                if fragile:
                    current=np.r_[current,[[.001,0,.007],[0,.001,.007],[0,0,.007]]]
                    initial=np.r_[initial,current[-3:]]
                ids=np.repeat(np.array([0,1,2,3,5,6,3,3],np.int32)[:,None],4,axis=1)
                weights=np.tile([1.,0,0,0],(8,1))
                for j in range(3):ids[j,:2]=[j,3];weights[j,:2]=[.9,.1]
                for j,value in ((3,.5),(6,.2),(7,-.1)):
                    ids[j,:2]=[3,4];weights[j,:2]=[value,1-value]
                inverse=np.zeros(len(current),np.float32);inverse[[3,4]]=[1,3]
                faces=np.array([[0,1,2],[3,4,5]],np.int32)
                rest=(initial[ids]*weights[...,None]).sum(1)
                surface=(current[ids]*weights[...,None]).sum(1)
                effective=(inverse[ids]*weights**2).sum(1)
                tets=np.array([[4,7,8,9]],np.int32) if fragile else np.empty((0,4),np.int32)
                body_rest=current.astype(np.float32);velocity=np.zeros_like(current)
                from .rig.biomechanics_native import Descriptor,Options
                descriptor=Descriptor(len(current),ptr(body_rest),None,ptr(inverse),0,None,len(tets),ptr(tets),0,None)
                options=Options();lib.lightphysics_deformable_options_default(ct.byref(options));options.min_volume_ratio=.06
                body=P();contact=P();dt=.01
                self.assertEqual(lib.lightphysics_deformable_create(ct.byref(descriptor),ct.byref(options),ct.byref(body)),0)
                try:
                    self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(current),ptr(velocity),current.size),0)
                    self.assertEqual(lib.lightphysics_surface_contact_create(ptr(rest),ptr(effective),8,ptr(faces),2,ct.byref(contact)),0)
                    reaction=np.zeros_like(surface);stats=ContactStats();fraction=ct.c_double()
                    self.assertEqual(project(contact,body,ptr(ids),ptr(weights),ptr(rest),ptr(surface),dt,.0002,1,ptr(reaction),ct.byref(stats),ct.byref(fraction)),0)
                    result=np.empty_like(current)
                    self.assertEqual(lib.lightphysics_deformable_read_state_f64(body,ptr(result),None,result.size),0)
                    np.testing.assert_allclose(surface,(result[ids]*weights[...,None]).sum(1),atol=1e-15)
                    self.assertGreater(abs(surface[6,2]-(current[ids[6]]*weights[6,:,None]).sum(0)[2]),0)
                    nodal=np.zeros_like(current)
                    for k in range(4):np.add.at(nodal,ids[:,k],reaction*weights[:,k,None])
                    np.testing.assert_allclose(result-current,inverse[:,None]*nodal*dt,rtol=1e-10,atol=1e-15)
                    np.testing.assert_allclose(reaction.sum(0),0,atol=1e-15)
                    if fragile:
                        self.assertGreaterEqual(float((volumes(result,tets)/volumes(body_rest,tets)).min()),.0599999)
                        self.assertLess(fraction.value,.5)
                    else:
                        lam=.0018/(.4**2+3*.5**2)
                        self.assertEqual(stats.contacts,1)
                        self.assertAlmostEqual(reaction[3,2]*dt,lam,places=14)
                        self.assertAlmostEqual(result[3,2]-current[3,2],.4*lam,places=14)
                        self.assertAlmostEqual(result[4,2]-current[4,2],1.5*lam,places=14)
                    before=result.copy();saved=surface.copy();bad=weights.copy();bad[-1,0]+=.1
                    self.assertEqual(project(contact,body,ptr(ids),ptr(bad),ptr(rest),ptr(surface),dt,.0002,1,ptr(reaction),ct.byref(stats),ct.byref(fraction)),-1)
                    self.assertEqual(lib.lightphysics_deformable_read_state_f64(body,ptr(result),None,result.size),0)
                    np.testing.assert_array_equal(result,before);np.testing.assert_array_equal(surface,saved)
                finally:
                    if contact:lib.lightphysics_surface_contact_destroy(contact)
                    lib.lightphysics_deformable_destroy(body)

    def test_embedded_contact_mode_survives_worker_clones_and_requires_double_state(self):
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        if getattr(library(),'lightphysics_surface_contact_project_embedded',None) is None:
            self.skipTest('rebuild embedded CPU contact API')
        for mode in ('embedded','embedded_volume'):
            with self.subTest(contact_transport=mode):
                if getattr(library(),'lightphysics_surface_contact_project_'+mode,None) is None:
                    self.skipTest('rebuild embedded CPU contact API')
                with self.assertRaisesRegex(RuntimeError,'double state'):
                    Session(arrays,meta,contact_transport=mode,state_precision='float32')
                with Session(arrays,meta,contact_transport=mode) as state,NativeTrialPool(state,2) as pool,Snapshot(state) as initial:
                    self.assertTrue(all(s.contact_transport==mode for s in pool.sessions))
                    probes=[np.array([.1]),np.array([.3])]
                    expected=[]
                    for probe in probes:
                        initial.restore();state.advance(probe,1/240);expected.append(state.surface())
                    initial.restore()
                    actual=pool.surfaces(initial,probes,1/240)
                    np.testing.assert_array_equal(actual,expected)
                    np.testing.assert_array_equal(state.positions,arrays['rest'])

    def test_coupled_jaw_attachment_respects_effective_mass_and_joint_limit(self):
        lib=library()
        if getattr(lib,'vhface_jaw_coupling_create',None) is None:self.skipTest('rebuild coupled CPU jaw API')
        from .rig.biomechanics_native import Descriptor,Options
        for bounded in (False,True):
            with self.subTest(at_joint_limit=bounded):
                rest=np.array([[0.,.02,.03],[.01,0,.04],[0,0,0],[0,0,.01]],np.float32)
                inverse=np.array([2.,0,0,0],np.float32)
                descriptor=Descriptor(4,ptr(rest),None,ptr(inverse),0,None,0,None,0,None)
                options=Options();lib.lightphysics_deformable_options_default(ct.byref(options))
                options.substeps=1;options.iterations=1;options.damping=1;options.gravity[:]=(0,0,0)
                body=P();coupling=P()
                self.assertEqual(lib.lightphysics_deformable_create(ct.byref(descriptor),ct.byref(options),ct.byref(body)),0)
                try:
                    ids=np.array([0],np.int32);rigid=np.array([1],np.int32)
                    anchor=np.ascontiguousarray(rest[[0]],float);rigid_rest=np.ascontiguousarray(rest[[1]],float)
                    hinge=np.zeros(3);compliance=np.zeros(1);q=np.array([0,.004 if bounded else .001,0])
                    mobility=np.array([0,5.,0]);x=rest.astype(float);x[1,0]+=q[1]
                    if bounded:x[0,0]=.005
                    before=x.copy();velocity=np.zeros_like(x)
                    self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(x),ptr(velocity),x.size),0)
                    self.assertEqual(lib.vhface_jaw_coupling_create(body,ptr(hinge),ptr(ids),ptr(anchor),ptr(compliance),1,
                        ptr(rigid),ptr(rigid_rest),1,ct.byref(coupling)),0)
                    self.assertEqual(lib.vhface_jaw_coupling_prepare(coupling,ptr(q),ptr(mobility),1/240),0)
                    output=np.empty_like(rest)
                    self.assertEqual(lib.lightphysics_deformable_step(body,None,1/240,ptr(output)),0)
                    actual=np.empty(3);self.assertEqual(lib.vhface_jaw_coupling_read(coupling,ptr(actual)),0)
                    self.assertEqual(lib.lightphysics_deformable_read_state_f64(body,ptr(x),None,x.size),0)
                    if bounded:
                        np.testing.assert_allclose(actual,q,atol=1e-15)
                        self.assertAlmostEqual(x[0,0],.004,places=14)
                    else:
                        self.assertAlmostEqual(x[0,0],.001*2/7,places=14)
                        self.assertAlmostEqual(actual[1],x[0,0],places=14)
                        self.assertAlmostEqual((x[0,0]-before[0,0])/2+(actual[1]-q[1])/5,0,places=14)
                    np.testing.assert_allclose(x[1],rigid_rest[0]+[actual[1],0,0],atol=1e-15)
                finally:
                    if coupling:lib.vhface_jaw_coupling_destroy(coupling)
                    lib.lightphysics_deformable_destroy(body)

    def test_coupled_jaw_replay_and_workers_preserve_static_configuration(self):
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture();arrays['anchor_bone'][0]=1
        if getattr(library(),'vhface_jaw_coupling_create',None) is None:self.skipTest('rebuild coupled CPU jaw API')
        with self.assertRaisesRegex(RuntimeError,'double state'):
            Session(arrays,meta,jaw_mode='coupled',state_precision='float32')
        with self.assertRaisesRegex(ValueError,'inertia'):
            Session(arrays,meta,jaw_inertia=(0,1,1))
        with self.assertRaisesRegex(ValueError,'damping'):
            Session(arrays,meta,jaw_damping=(1,-1,1))
        for invalid in (-1,33,1.5,True):
            with self.assertRaisesRegex(ValueError,'inertial repair'):
                Session(arrays,meta,inertial_repair_sweeps=invalid)
        for invalid in (-1,4097,1.5,True):
            with self.assertRaisesRegex(ValueError,'volume rescue'):
                Session(arrays,meta,volume_rescue_limit=invalid)
        for mode in ('coupled','coupled_volume','coupled_block'):
            for track in (None,(.001,-.004,.006)):
                with Session(arrays,meta,jaw_mode=mode,contacts=False,jaw_damping=(.1,30,30),jaw_track=track,inertial_repair_sweeps=4,volume_rescue_limit=16) as state, \
                     NativeTrialPool(state,2) as pool,Snapshot(state) as initial:
                    self.assertEqual(len(state.anchor_ids),2);self.assertEqual(len(state.jaw_anchor_ids),1)
                    self.assertTrue(all(s.jaw_mode==mode and tuple(s.jaw_damping)==(.1,30,30) for s in pool.sessions))
                    self.assertTrue(all(np.array_equal(s.jaw_track,state.jaw_track) for s in pool.sessions))
                    self.assertTrue(all(s.inertial_repair_sweeps==4 for s in pool.sessions))
                    self.assertTrue(all(s.volume_rescue_limit==16 for s in pool.sessions))
                    probes=[np.array([.1]),np.array([.5])];expected=[];coordinates=[]
                    for probe in probes:
                        initial.restore();state.advance(probe,1/30);expected.append(state.surface());coordinates.append(state.q.copy())
                    initial.restore()
                    native=np.empty(3);self.assertEqual(state.lib.vhface_jaw_coupling_read(state.coupling,ptr(native)),0)
                    np.testing.assert_array_equal(native,state.q)
                    np.testing.assert_array_equal(pool.surfaces(initial,probes,1/30),expected)
                    np.testing.assert_array_equal(state.positions,arrays['rest'])
                    state.advance(probes[1],1/30)
                    np.testing.assert_array_equal(state.surface(),expected[1]);np.testing.assert_array_equal(state.q,coordinates[1])

    def test_external_cable_package_survives_snapshot_and_worker_clones(self):
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture();arrays['fibers']=np.empty((0,6))
        cable=np.zeros((1,40));cable[0,:8]=[0,1,.02,.04,0,.02,.04,.15];cable[0,8]=1;cable[0,24]=1
        arrays['tongue_cables']=cable
        if getattr(library(),'lightphysics_active_tissue_set_cables',None) is None:self.skipTest('rebuild active cable CPU API')
        with Session(arrays,meta,contacts=False) as state,NativeTrialPool(state,2) as pool,Snapshot(state) as initial:
            self.assertEqual(state.cable_count,1)
            self.assertTrue(all(worker.cable_count==1 for worker in pool.sessions))
            probes=[np.array([.2]),np.array([.7])];expected=[]
            for probe in probes:
                initial.restore();state.advance(probe,1/30);expected.append(state.surface())
                self.assertEqual(state.cable_reactions().shape,(1,3))
                self.assertLess(state.cable_reactions()[0,1],0)
            self.assertGreater(float(np.max(abs(expected[-1]-arrays['surface_rest']))),1e-6)
            initial.restore();np.testing.assert_array_equal(pool.surfaces(initial,probes,1/30),expected)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            state.advance(probes[-1],1/30);np.testing.assert_array_equal(state.surface(),expected[-1])

    def test_external_active_cable_matches_mass_system_and_balances_support_reaction(self):
        from .rig.biomechanics_native import Descriptor,Options,ActiveCable,Fiber,D
        lib=library()
        if getattr(lib,'lightphysics_active_tissue_set_cables',None) is None:self.skipTest('rebuild active cable CPU API')
        rest=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1],[.9,.04,0],[1.1,-.01333333,0]],np.float32)
        original=rest.astype(float);tet=np.array([[0,1,2,3]],np.int32)
        mass=np.array([0,0,0,0,2,3],np.float32);weights=np.array([.25,.75+5e-9]);normalized=weights/weights.sum()
        origin=normalized@original[4:]-[1,0,0];dt=1/240
        cable=ActiveCable();cable.muscle=0;cable.vertex_count=2;cable.vertices[:2]=[4,5];cable.weights[:2]=weights
        cable.origin[:]=origin;cable.max_force=10;cable.rest_length=1;cable.shortening=.15
        for implicit in (False,True):
            desc=Descriptor(6,ptr(rest),None,ptr(mass),0,None,1,ptr(tet),0,None)
            options=Options();lib.lightphysics_deformable_options_default(ct.byref(options))
            options.substeps=1;options.iterations=1;options.damping=1;options.gravity[:]=[0,0,0]
            body=P();field=P();self.assertEqual(lib.lightphysics_deformable_create(ct.byref(desc),ct.byref(options),ct.byref(body)),0)
            try:
                invalid=P()
                self.assertEqual(lib.lightphysics_active_tissue_create(body,ptr(rest),5,ptr(tet),1,None,0,1,ct.byref(invalid)),-1)
                self.assertFalse(invalid)
                extra_rest=np.r_[rest,np.zeros((1,3),np.float32)]
                outside=np.array([[0,1,2,6]],np.int32);fiber=(Fiber*1)(Fiber(0,0,(D*3)(1,0,0),1000))
                self.assertEqual(lib.lightphysics_active_tissue_create(body,ptr(extra_rest),7,ptr(outside),1,fiber,1,1,ct.byref(invalid)),-1)
                self.assertFalse(invalid)
                reference=extra_rest if implicit else rest
                reference_tets=np.r_[tet,tet] if implicit else tet
                self.assertEqual(lib.lightphysics_active_tissue_create(body,ptr(reference),len(reference),ptr(reference_tets),len(reference_tets),None,0,1,ct.byref(field)),0)
                self.assertEqual(lib.lightphysics_active_tissue_set_cables(field,ct.byref(cable),1),0)
                self.assertEqual(lib.lightphysics_active_tissue_enable_implicit(field,int(implicit)),0)
                activation=np.ones(1);output=np.empty(18,np.float32);actual=np.empty_like(original);reaction=np.empty(3)
                def step():
                    self.assertEqual(lib.lightphysics_active_tissue_apply(field,ptr(activation),1,dt),0)
                    self.assertEqual(lib.lightphysics_deformable_step(body,None,dt,ptr(output)),0)
                    self.assertEqual(lib.lightphysics_deformable_read_state_f64(body,ptr(actual),None,18),0)
                    self.assertEqual(lib.lightphysics_active_tissue_read_cable_reactions(field,ptr(reaction),3),0)
                step()
                multiplier=(-.15/(1/(10/.15*dt*dt)+float(mass[4:]@normalized**2)) if implicit else -10*dt*dt)
                expected=original.copy();expected[4:,0]+=mass[4:]*normalized*multiplier
                np.testing.assert_allclose(actual,expected,atol=1e-15)
                np.testing.assert_allclose(reaction,[-multiplier/dt,0,0],atol=2e-13)
                impulse=(actual[4:]-original[4:])/mass[4:,None]/dt
                np.testing.assert_allclose(impulse.sum(0)+reaction,0,atol=2e-13)
                np.testing.assert_allclose(np.cross(original[4:],impulse).sum(0)+np.cross(origin,reaction),0,atol=2e-13)
                expected=actual.copy();bad=(ActiveCable*2)();bad[0]=cable;bad[1]=cable;bad[1].origin[2]=np.nan
                self.assertEqual(lib.lightphysics_active_tissue_set_cables(field,bad,2),-1)
                for kind in ('pinned','duplicate','weight','length'):
                    invalid=ActiveCable.from_buffer_copy(cable)
                    if kind=='pinned':invalid.vertices[1]=0
                    elif kind=='duplicate':invalid.vertices[1]=4
                    elif kind=='weight':invalid.weights[1]=-1
                    else:invalid.rest_length=0
                    self.assertEqual(lib.lightphysics_active_tissue_set_cables(field,ct.byref(invalid),1),-1)
                velocity=np.zeros_like(original)
                self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(original),ptr(velocity),18),0)
                step();np.testing.assert_array_equal(actual,expected)
                reaction[:]=-77
                self.assertEqual(lib.lightphysics_active_tissue_read_cable_reactions(field,ptr(reaction),2),-1)
                np.testing.assert_array_equal(reaction,-77)
                self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(original),ptr(velocity),18),0)
                activation[:]=0;step();np.testing.assert_array_equal(reaction,0)
                slack=original.copy();slack[4:,0]-=.2
                self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(slack),ptr(velocity),18),0)
                activation[:]=1;step();np.testing.assert_array_equal(actual,slack);np.testing.assert_array_equal(reaction,0)
                self.assertEqual(lib.lightphysics_active_tissue_set_cables(field,None,0),0)
                self.assertEqual(lib.lightphysics_active_tissue_read_cable_reactions(field,None,0),0)
            finally:
                if field:lib.lightphysics_active_tissue_destroy(field)
                if body:lib.lightphysics_deformable_destroy(body)

    def test_simultaneous_jaw_attachments_preserve_coherent_small_tet_motion(self):
        from .rig.biomechanics_native import Descriptor,Options
        lib=library()
        if getattr(lib,'vhface_jaw_coupling_enable_block_solve',None) is None:self.skipTest('rebuild block CPU jaw API')
        rest=np.array([[.03,-.05,.04],[.03006,-.05,.04],[.03,-.04994,.04],[.03,-.05,.04006]],np.float32)
        original=rest.astype(np.float64);tet=np.array([[0,1,2,3]],np.int32)
        inverse_mass=np.full(4,1e8,np.float32);anchors=np.arange(3,-1,-1,dtype=np.int32)
        anchor_rest=np.ascontiguousarray(original[anchors]);compliance=np.zeros(4)
        hinge=np.zeros(3);q=np.array([.03,0,.001]);inverse=np.zeros(3);target=np.empty_like(original)
        self.assertEqual(lib.vhface_jaw_transform(ptr(original),4,ptr(hinge),ptr(q),ptr(target)),0)
        errors=[]
        for block in (False,True):
            desc=Descriptor(4,ptr(rest),None,ptr(inverse_mass),0,None,1,ptr(tet),0,None)
            options=Options();lib.lightphysics_deformable_options_default(ct.byref(options))
            options.substeps=1;options.iterations=1;options.material=2;options.damping=1;options.gravity[:]=[0,0,0]
            body=P();coupling=P()
            self.assertEqual(lib.lightphysics_deformable_create(ct.byref(desc),ct.byref(options),ct.byref(body)),0)
            try:
                self.assertEqual(lib.vhface_jaw_coupling_create(body,ptr(hinge),ptr(anchors),ptr(anchor_rest),ptr(compliance),4,None,None,0,ct.byref(coupling)),0)
                self.assertEqual(lib.vhface_jaw_coupling_enable_block_solve(coupling,int(block)),0)
                self.assertEqual(lib.vhface_jaw_coupling_prepare(coupling,ptr(q),ptr(inverse),1/240),0)
                output=np.empty(12,np.float32);actual=np.empty_like(original)
                self.assertEqual(lib.lightphysics_deformable_step(body,None,1/240,ptr(output)),0)
                self.assertEqual(lib.lightphysics_deformable_read_state_f64(body,ptr(actual),None,12),0)
                errors.append(float(np.sqrt(np.mean((actual-target)**2))*1000))
                if block:
                    np.testing.assert_allclose(actual,target,atol=1e-15)
                    self.assertAlmostEqual(float(volumes(actual,tet)[0]/volumes(original,tet)[0]),1.,places=10)
            finally:
                if coupling:lib.vhface_jaw_coupling_destroy(coupling)
                if body:lib.lightphysics_deformable_destroy(body)
        self.assertGreater(errors[0],.1)
        self.assertLess(errors[1],1e-10)

    def test_fitted_jaw_track_geometry_and_virtual_work_match_finite_differences(self):
        from .rig import jaw_model
        arrays,meta=fixture();track=np.array([.001,-.004,.006])
        if getattr(library(),'vhface_jaw_transform_model',None) is None:self.skipTest('rebuild fitted CPU jaw API')
        with Session(arrays,meta,contacts=False,jaw_track=track) as state:
            state.q[:]=[.2,.001,.003]
            posed=state.transform(arrays['rest'])
            np.testing.assert_allclose(posed,jaw_model.points(arrays['rest'],state.hinge,state.q[None],track)[0],atol=1e-15)
            forces=np.array([[.4,-.2,.1],[-.3,.1,.2],[.2,.7,-.6],[-.1,.4,.3]])
            actual=state.generalized(posed,forces);reference=state.q.copy()
            for axis,epsilon in enumerate((1e-6,1e-8,1e-8)):
                state.q[:]=reference;state.q[axis]+=epsilon;plus=state.transform(arrays['rest'])
                state.q[:]=reference;state.q[axis]-=epsilon;minus=state.transform(arrays['rest'])
                self.assertAlmostEqual(actual[axis],float(((plus-minus)*forces).sum()/(2*epsilon)),places=8)
            state.q[:]=reference
            np.testing.assert_array_equal(track,[.001,-.004,.006])

    def test_rigid_contact_updates_jaw_velocity_and_replays_without_double_feedback(self):
        from .rig.biomechanics_trials import NativeTrialPool
        if getattr(library(),'vhface_jaw_contact_project',None) is None:self.skipTest('rebuild reduced CPU contact API')
        arrays,meta=fixture();tet=arrays['rest']+[.2,0,0]
        fixed=np.array([[-.02,-.02,0],[.02,-.02,0],[0,.02,0]])
        teeth=np.array([[-.005,-.005,-.001],[.005,-.005,-.001],[0,.005,-.001]])
        arrays['rest']=np.r_[tet,fixed,teeth]
        arrays['inverse_mass']=np.r_[arrays['inverse_mass'],np.zeros(6)]
        arrays['bone']=np.r_[arrays['bone'],np.zeros(3,np.int32),np.ones(3,np.int32)]
        arrays['jaw_hinge']=np.array([0.,.03,-.03]);arrays['surface_rest']=arrays['rest'].copy()
        arrays['faces']=np.r_[arrays['faces'],[[4,5,6],[7,8,9]]].astype(np.int32)
        arrays['source_ids']=np.arange(10,dtype=np.int32)
        arrays['embed_ids']=np.repeat(np.arange(10,dtype=np.int32)[:,None],4,axis=1)
        arrays['embed_weights']=np.tile([1.,0,0,0],(10,1))
        with Session(arrays,meta,jaw_mode='coupled_volume',contact_transport='embedded_coupled') as state:
            state.q[2]=.002;state.speed[2]=-1;rigid=state.bone==1
            state.positions[rigid]=state.transform(state.rest[rigid]);velocity=np.zeros_like(state.positions)
            self.assertEqual(state.write_state(state.body,ptr(state.positions),ptr(velocity),state.positions.size),0)
            with Snapshot(state) as initial,NativeTrialPool(state,2) as pool:
                report=state.step(np.zeros(1),1/240);first=state.positions.copy();q=state.q.copy();speed=state.speed.copy()
                self.assertAlmostEqual(q[2],.0012,places=10)
                self.assertAlmostEqual(speed[2],(q[2]-.002)*240,places=12)
                self.assertEqual(report['contacts'],1);self.assertEqual(state.intersections(),0)
                np.testing.assert_array_equal(state.reaction,0)
                initial.restore();state.step(np.zeros(1),1/240)
                np.testing.assert_array_equal(state.positions,first);np.testing.assert_array_equal(state.q,q)
                np.testing.assert_array_equal(state.speed,speed)
                initial.restore()
                surfaces=pool.surfaces(initial,[np.zeros(1),np.zeros(1)],1/240)
                np.testing.assert_array_equal(surfaces,[state.surface(first),state.surface(first)])
                np.testing.assert_array_equal(state.q,[0,0,.002])

    def test_contacts_inside_one_rigid_jaw_cannot_drive_its_coordinates(self):
        lib=library()
        if getattr(lib,'vhface_jaw_contact_project',None) is None:self.skipTest('rebuild reduced CPU contact API')
        from .rig.biomechanics_native import Descriptor,Options
        triangle=np.array([[-.01,-.01,0],[.01,-.01,0],[0,.01,0]],np.float32)
        rest=np.r_[triangle,triangle+[0,0,.000001]].astype(np.float32);mass=np.zeros(6,np.float32)
        descriptor=Descriptor(6,ptr(rest),None,ptr(mass),0,None,0,None,0,None)
        options=Options();lib.lightphysics_deformable_options_default(ct.byref(options));options.substeps=1
        body=P();coupling=P();contact=P()
        self.assertEqual(lib.lightphysics_deformable_create(ct.byref(descriptor),ct.byref(options),ct.byref(body)),0)
        try:
            original=rest.astype(float);current=np.empty_like(original);q=np.array([.01,.0002,.001]);expected_q=q.copy()
            hinge=np.array([0,.03,-.03]);self.assertEqual(lib.vhface_jaw_transform(ptr(original),6,ptr(hinge),ptr(q),ptr(current)),0)
            velocity=np.zeros_like(current);self.assertEqual(lib.lightphysics_deformable_write_state_f64(body,ptr(current),ptr(velocity),current.size),0)
            rigid=np.arange(6,dtype=np.int32);empty=np.empty(0,np.int32);empty_rest=np.empty((0,3));compliance=np.empty(0)
            inverse=np.array([500.,5.,5.]);dt=.01
            self.assertEqual(lib.vhface_jaw_coupling_create(body,ptr(hinge),ptr(empty),ptr(empty_rest),ptr(compliance),0,
                ptr(rigid),ptr(original),6,ct.byref(coupling)),0)
            self.assertEqual(lib.vhface_jaw_coupling_prepare(coupling,ptr(q),ptr(inverse),dt),0)
            faces=np.array([[0,1,2],[3,4,5]],np.int32);ids=np.repeat(rigid[:,None],4,axis=1)
            weights=np.tile([1.,0,0,0],(6,1));surface_mass=np.zeros(6)
            self.assertEqual(lib.lightphysics_surface_contact_create(ptr(original),ptr(surface_mass),6,ptr(faces),2,ct.byref(contact)),0)
            expected=current.copy();stats=ContactStats();fraction=ct.c_double();reaction=np.zeros_like(current)
            self.assertEqual(lib.vhface_jaw_contact_project(coupling,contact,ptr(ids),ptr(weights),ptr(original),ptr(current),dt,.0002,1,
                ptr(reaction),ct.byref(stats),ct.byref(fraction)),0)
            self.assertEqual(lib.vhface_jaw_coupling_read(coupling,ptr(q)),0)
            self.assertEqual(stats.contacts,0);np.testing.assert_array_equal(current,expected)
            np.testing.assert_array_equal(q,expected_q);np.testing.assert_array_equal(reaction,0)
        finally:
            if contact:lib.lightphysics_surface_contact_destroy(contact)
            if coupling:lib.vhface_jaw_coupling_destroy(coupling)
            lib.lightphysics_deformable_destroy(body)

    def test_closed_cancelled_session_releases_without_cyclic_collection(self):
        import threading,weakref
        arrays,meta=fixture()
        state=Session(arrays,meta,cancel=threading.Event())
        reference=weakref.ref(state)
        state.close();state.close()
        del state
        self.assertIsNone(reference())

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_warm_control_is_forward_validated_and_duplicate_seeds_are_not_retried(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as reference,Session(arrays,meta,contacts=False) as state:
            reference.advance(np.array([.3]),1/30);target=reference.surface()
            chosen,report=inverse_step(state,target,[.9],1/30,previous=[.3],warm_start=[.3],
                stop=lambda surface:np.max(abs(surface-target))<1e-10)
            np.testing.assert_array_equal(chosen,[.3]);np.testing.assert_array_equal(state.positions,reference.positions)
            self.assertEqual(report['seed_source'],'warm_start');self.assertEqual(report['evaluations'],1)
            self.assertTrue(report['early_stopped']);self.assertEqual(report['native_steps_evaluated'],8)
        with Session(arrays,meta,contacts=False) as state:
            advance=state.advance;calls=[]
            def guarded(excitation,seconds):
                calls.append(excitation.copy())
                if excitation[0]>.5:raise FloatingPointError('unsafe seed')
                return advance(excitation,seconds)
            with patch.object(state,'advance',guarded):
                chosen,report=inverse_step(state,arrays['surface_rest'],[1.],1/30,previous=[0.],warm_start=[1.],iterations=0)
            self.assertEqual(len(calls),2);self.assertEqual(report['unsafe_trials'],1)
            self.assertEqual(report['seed_source'],'previous_drive');np.testing.assert_array_equal(chosen,[0.])
            for invalid in ([float('nan')],[-.1],[1.1],[0.,0.]):
                with self.assertRaisesRegex(ValueError,'history/prior'):inverse_step(state,target,[0.],1/30,warm_start=invalid)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_geometry_filter_fits_a_safe_control_without_clamping_output(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as reference,Session(arrays,meta,contacts=False) as state, \
             Session(arrays,meta,contacts=False) as replay:
            reference.advance(np.array([1.]),1/30);target=reference.surface()
            boundary=(target[1,0]+arrays['rest'][1,0])/2
            self.assertLess(target[1,0],boundary)
            def validate(surface):
                if surface[1,0]<boundary:raise FloatingPointError('synthetic surface crossing')
            chosen,report=inverse_step(state,target,[1.],1/30,previous=[0.],iterations=2,validate_trial=validate)
            self.assertTrue(report['seed_fallback']);self.assertGreater(report['geometry_rejections'],0)
            self.assertLess(report['target_rms_after_mm'],report['target_rms_before_mm'])
            self.assertGreaterEqual(state.surface()[1,0],boundary)
            replay.advance(chosen,1/30);np.testing.assert_array_equal(replay.positions,state.positions)
            before=state.positions.copy();activation=state.activation.copy()
            def fail(_):raise RuntimeError('geometry callback failed')
            with self.assertRaisesRegex(RuntimeError,'geometry callback failed'):
                inverse_step(state,target,[0.],1/30,validate_trial=fail)
            np.testing.assert_array_equal(state.positions,before);np.testing.assert_array_equal(state.activation,activation)

    def test_native_cancellation_interrupts_an_internal_solve_and_rolls_back(self):
        class CancelInsideNative:
            def __init__(self):self.polls=0
            def is_set(self):self.polls+=1;return self.polls>=5
        arrays,meta=fixture();cancel=CancelInsideNative()
        with Session(arrays,meta,contacts=False,cancel=cancel) as state:
            if getattr(state.lib,'lightphysics_deformable_set_cancel_callback',None) is None:
                self.skipTest('rebuild native cancellation checkpoints')
            before=state.positions.copy();native=np.empty_like(before,dtype=state.state_dtype)
            self.assertEqual(state.read_state(state.body,ptr(native),None,native.size),0)
            with self.assertRaisesRegex(RuntimeError,'cancelled'):state.step(np.array([.5]),1/240)
            self.assertEqual(cancel.polls,5);self.assertEqual(state.native_steps,1)
            np.testing.assert_array_equal(state.positions,before);np.testing.assert_array_equal(state.activation,[0.])
            stored=np.empty_like(native);self.assertEqual(state.read_state(state.body,ptr(stored),None,stored.size),0)
            np.testing.assert_array_equal(stored,native)
            state.cancel=None;state.advance(np.array([0.]),1/30)
            np.testing.assert_allclose(state.positions,before,atol=1e-8)

    def test_double_bridge_preserves_sub_float32_motion_and_worker_precision(self):
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers']=np.empty((0,6))
        with Session(arrays,meta,contacts=False) as state:
            if state.state_precision!='float64':self.skipTest('rebuild native double state API')
            precise=arrays['rest'].astype(np.float32).astype(np.float64)+2**-40
            velocity=np.zeros_like(precise)
            self.assertEqual(state.write_state(state.body,ptr(precise),ptr(velocity),precise.size),0)
            state.positions[:]=precise
            state.step(np.array([0.]),1/240)
            stored=np.empty_like(precise)
            self.assertEqual(state.read_state(state.body,ptr(stored),None,stored.size),0)
            np.testing.assert_array_equal(state.positions,stored)
            np.testing.assert_array_equal(stored,precise)
            self.assertFalse(np.array_equal(stored,stored.astype(np.float32).astype(np.float64)))
        with Session(arrays,meta,contacts=False,state_precision='float32') as state,NativeTrialPool(state,2,lazy=True) as pool:
            with Snapshot(state) as initial:pool.surfaces(initial,np.array([[0.]]),1/240)
            self.assertTrue(all(s.state_precision=='float32' for s in pool.sessions))

    def test_failed_contact_evidence_survives_physical_rollback(self):
        from .rig.biomechanics_native import D,GeometryStepError
        arrays,meta=fixture()
        def blocked(handle,before,corrected,seconds,thickness,iterations,reactions,stats_pointer):
            points=np.ctypeslib.as_array(ct.cast(corrected,ct.POINTER(D)),shape=(12,)).reshape(4,3)
            if reactions is None:points[1,2]+=.0001
            stats=ct.cast(stats_pointer,ct.POINTER(ContactStats)).contents
            stats.max_correction=.0001;stats.contacts=1
            return 0
        with Session(arrays,meta) as state:
            original=state.positions.copy();activation=state.activation.copy()
            with patch.object(state.lib,'lightphysics_surface_contact_project',blocked):
                with self.assertRaises(GeometryStepError) as caught:state.step(np.array([.5]),1/240,subdivisions=0)
            detail=caught.exception.diagnostics
            self.assertAlmostEqual(detail['contact_residual_mm'],.1)
            vertex=detail['contact_vertices'][0]
            self.assertEqual(vertex['source_vertex'],1);self.assertEqual(vertex['part'],'skin')
            self.assertAlmostEqual(vertex['remaining_correction_mm'],.1)
            self.assertEqual(len(detail['minimum_tetrahedra']),1)
            self.assertGreater(detail['minimum_tetrahedra'][0]['volume_ratio'],.06)
            json.dumps(detail,allow_nan=False)
            np.testing.assert_array_equal(state.positions,original)
            np.testing.assert_array_equal(state.activation,activation)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_lazy_native_workers_are_allocated_only_when_a_fit_needs_probes(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state,NativeTrialPool(state,2,lazy=True) as pool:
            self.assertEqual(pool.sessions,[])
            _,report=inverse_step(state,arrays['surface_rest'],[0.],1/30,iterations=1,trial_pool=pool,stop=lambda surface:True)
            self.assertTrue(report['early_stopped']);self.assertEqual(report['inverse_active_workers'],0)
            self.assertEqual(report['inverse_workers'],2);self.assertEqual(pool.sessions,[])
            target=arrays['surface_rest'].copy();target[1,0]-=.0005
            _,report=inverse_step(state,target,[.2],1/30,iterations=1,trial_pool=pool)
            self.assertEqual(len(pool.sessions),2)
            self.assertEqual(report['native_steps_evaluated'],8*report['evaluations'])
        self.assertTrue(pool.closed);self.assertEqual(pool.sessions,[])

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_stopping_retains_a_full_native_trial_and_ages_the_cache(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache
        arrays,meta=fixture();cache=SensitivityCache(1)
        with Session(arrays,meta,contacts=False) as state,Session(arrays,meta,contacts=False) as reference:
            cache.refresh(state,1/30,np.zeros((1,4,3)))
            for i in range(2):
                reference.advance(np.array([.4]),1/30);target=reference.surface()
                x,r=inverse_step(state,target,[.4],1/30,iterations=2,sensitivities=cache,
                    stop=lambda surface:np.max(abs(surface-target))<1e-10)
                self.assertTrue(r['early_stopped']);self.assertEqual(r['iterations_completed'],0)
                self.assertEqual(r['evaluations'],1);self.assertEqual(r['native_steps_evaluated'],8)
                self.assertEqual(r['sensitivity_refreshes'],0);self.assertEqual(cache.uses,i+1)
                np.testing.assert_array_equal(x,[.4]);np.testing.assert_array_equal(state.positions,reference.positions)
                np.testing.assert_array_equal(state.activation,reference.activation)
            self.assertFalse(cache.compatible(state,1/30))
            before=state.positions.copy();activation=state.activation.copy()
            def fail(_):raise RuntimeError('invalid tolerance callback')
            with self.assertRaisesRegex(RuntimeError,'tolerance callback'):
                inverse_step(state,target,[.4],1/30,iterations=1,sensitivities=cache,stop=fail)
            np.testing.assert_array_equal(state.positions,before);np.testing.assert_array_equal(state.activation,activation)
            self.assertIsNone(cache.owner)

    def test_embedded_contacts_guard_volume_and_scale_the_paired_jaw_reaction(self):
        from .rig.biomechanics_native import D
        arrays,meta=fixture();arrays['inverse_mass'][0]=0;arrays['bone'][0]=1
        arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers']=np.empty((0,6));calls=[];reaction=np.array([1.,2.,3.]);dt=1/240
        def contact(handle,before,corrected,seconds,thickness,iterations,reactions,stats_pointer):
            points=np.ctypeslib.as_array(ct.cast(corrected,ct.POINTER(D)),shape=(12,)).reshape(4,3)
            stats=ct.cast(stats_pointer,ct.POINTER(ContactStats)).contents
            if not calls:
                proposal=points.copy();proposal[:,:2]*=-1
                stats.contacts=1;stats.max_correction=float(np.linalg.norm(proposal-points,axis=1).max())
                points[:]=proposal
                values=np.ctypeslib.as_array(ct.cast(reactions,ct.POINTER(D)),shape=(12,)).reshape(4,3)
                values[0]=reaction
            calls.append(1)
            return 0
        with Session(arrays,meta) as state:
            with patch.object(state.lib,'lightphysics_surface_contact_project',contact):
                report=state.step(np.array([0.]),dt)
            fraction=report['contact_volume_fraction_min']
            self.assertGreater(fraction,0);self.assertLess(fraction,.5)
            self.assertGreaterEqual(report['min_volume_ratio'],.0599)
            self.assertLess(report['max_contact_correction_mm'],report['max_contact_proposal_mm'])
            expected=fraction*state.generalized(arrays['rest'][:1],reaction[None]/dt)
            np.testing.assert_allclose(state.reaction,expected,rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(HAS_SCIPY,'SciPy rigid interpolation')
    def test_rigid_targets_advance_in_substeps_and_adaptive_retries(self):
        arrays,meta=fixture();arrays['inverse_mass'][:]=0;arrays['bone'][:]=0
        arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        goal=arrays['rest']*[-1,-1,1]+[.03,.02,.01]
        with Session(arrays,meta,contacts=False) as state:
            state.rigid_targets[:]=goal;original=state._step;visited=[]
            def observe(excitation,dt):
                visited.append(state.rigid_targets.copy());return original(excitation,dt)
            with patch.object(state,'_step',observe):report=state.advance(np.array([0.]),1/30)
            self.assertEqual(len(visited),8);self.assertEqual(report['subdivision_retries'],0)
            self.assertGreater(report['min_volume_ratio'],.99999)
            self.assertFalse(np.allclose(visited[0],goal))
            np.testing.assert_array_equal(state.rigid_targets,goal)
            np.testing.assert_allclose(state.positions,goal,atol=2e-9)
        with Session(arrays,meta,contacts=False) as state:
            state.rigid_targets[:]=goal;original=state._step;visited=[]
            def retry(excitation,dt):
                visited.append(state.rigid_targets.copy());report=original(excitation,dt)
                if len(visited)==1:raise FloatingPointError('test rigid retry')
                return report
            with patch.object(state,'_step',retry):state.step(np.array([0.]),1/240)
            self.assertEqual(len(visited),3)
            self.assertFalse(np.allclose(visited[1],goal))
            np.testing.assert_array_equal(visited[2],goal)
            np.testing.assert_array_equal(state.rigid_targets,goal)
            np.testing.assert_allclose(state.positions,goal,atol=2e-9)
    def test_crossing_queries_accept_the_actual_exported_surface_without_mutating_dynamics(self):
        arrays,meta=fixture()
        with Session(arrays,meta) as state:
            original=state.positions.copy()
            self.assertEqual(state.intersections(state.surface().astype(np.float32)+[1,2,3]),0)
            np.testing.assert_array_equal(state.positions,original)
            for bad in (np.zeros((1,3)),np.full((4,3),np.nan)):
                with self.assertRaisesRegex(ValueError,'surface crossing query'):state.intersections(bad)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_short_native_probes_calibrate_and_commit_only_full_duration_trials(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache,NativeTrialPool
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers'][0,5]=10000
        with Session(arrays,meta,contacts=False) as serial,Session(arrays,meta,contacts=False) as parallel, \
                Session(arrays,meta,contacts=False) as reference,Session(arrays,meta,contacts=False) as replay, \
                NativeTrialPool(parallel,2) as pool:
            a,b=SensitivityCache(2),SensitivityCache(2)
            for i in range(2):
                reference.advance(np.array([.8]),1/30);target=reference.surface()
                options=dict(iterations=2,probe_substeps=1)
                x,r=inverse_step(serial,target,[.2],1/30,sensitivities=a,**options)
                y,s=inverse_step(parallel,target,[.2],1/30,sensitivities=b,trial_pool=pool,**options)
                self.assertLess(r['target_rms_after_mm'],r['target_rms_before_mm'])
                if not i:
                    self.assertLess(r['native_steps_evaluated'],r['evaluations']*8)
                    calibration=r['sensitivity_probe_calibrations'][0]
                    self.assertTrue(calibration['directional_calibration_valid'])
                    self.assertEqual(calibration['full_substeps'],8)
                    self.assertAlmostEqual(calibration['seconds'],1/240)
                replay.advance(x,1/30)
                np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(serial.positions,parallel.positions)
                np.testing.assert_array_equal(serial.positions,replay.positions)
                np.testing.assert_array_equal(serial.activation,replay.activation)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_unshortened_probe_mode_matches_the_original_inverse_result(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture();target=arrays['surface_rest'].copy();target[1,0]-=.001
        with Session(arrays,meta,contacts=False) as a,Session(arrays,meta,contacts=False) as b:
            x,r=inverse_step(a,target,[.2],1/30,iterations=1)
            y,s=inverse_step(b,target,[.2],1/30,iterations=1,probe_substeps=8)
            np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a.positions,b.positions)
            self.assertEqual(s['sensitivity_probe_calibrations'],[])
            self.assertEqual(r['evaluations'],s['evaluations'])
            for bad in (-1,49,True,1.,None):
                with self.assertRaisesRegex(ValueError,'probe substeps'):
                    inverse_step(a,target,[.2],1/30,probe_substeps=bad)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_simulated_contact_objective_guides_native_and_cached_trials(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache,NativeTrialPool
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers'][0,5]=10000
        with Session(arrays,meta,contacts=False) as serial,Session(arrays,meta,contacts=False) as parallel, \
                Session(arrays,meta,contacts=False) as reference,NativeTrialPool(parallel,2) as pool:
            a,b=SensitivityCache(2),SensitivityCache(2)
            for count in (1,2):
                reference.advance(np.array([.8]),1/30)
                goal=reference.surface().copy()
                def contact(surface):
                    return ((surface[[1,2],0]-goal[[1,2],0])*1000)[:count]
                options=dict(iterations=1,extra_residual=contact,extra_weight=4.)
                x,r=inverse_step(serial,arrays['surface_rest'],[.1],1/30,sensitivities=a,**options)
                y,s=inverse_step(parallel,arrays['surface_rest'],[.1],1/30,sensitivities=b,trial_pool=pool,**options)
                self.assertLess(r['extra_objective_rms_after'],r['extra_objective_rms_before'])
                self.assertLess(r['objective_after'],r['objective_before'])
                self.assertEqual(r['extra_objective_terms'],count)
                self.assertEqual(r['sensitivity_reused'],count==2)
                np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(serial.positions,parallel.positions)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_invalid_contact_residual_rolls_back_the_native_trial(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            rest=state.positions.copy()
            for bad in (lambda surface:np.array([np.nan]),lambda surface:np.zeros((1,1))):
                with self.assertRaisesRegex(ValueError,'extra geometry residual'):
                    inverse_step(state,arrays['surface_rest'],[.2],1/30,extra_residual=bad)
                np.testing.assert_array_equal(state.positions,rest)
            calls=[]
            def changing(surface):
                calls.append(1)
                return np.zeros(1 if len(calls)==1 else 2)
            with self.assertRaisesRegex(ValueError,'extra geometry residual'):
                inverse_step(state,arrays['surface_rest'],[.2],1/30,extra_residual=changing)
            np.testing.assert_array_equal(state.positions,rest);np.testing.assert_array_equal(state.activation,[0.])
            for weight in (-1.,float('inf'),float('nan')):
                with self.assertRaisesRegex(ValueError,'extra objective weight'):
                    inverse_step(state,arrays['surface_rest'],[.2],1/30,extra_weight=weight)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_cached_sensitivities_refresh_periodically_and_accept_only_native_improvements(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers'][0,5]=10000
        with Session(arrays,meta,contacts=False) as state,Session(arrays,meta,contacts=False) as reference:
            cache=SensitivityCache(1);reports=[]
            for i in range(3):
                reference.advance(np.array([.8]),1/30)
                _,report=inverse_step(state,reference.surface(),[.2],1/30,previous=[.2],iterations=1,
                    sensitivities=cache,sample_ids=np.arange(4) if i!=1 else np.array([0,2,3]))
                self.assertLess(report['target_rms_after_mm'],report['target_rms_before_mm'])
                reports.append(report)
            self.assertEqual([r['sensitivity_reused'] for r in reports],[False,True,False])
            self.assertEqual([r['sensitivity_refreshes'] for r in reports],[1,0,1])
            self.assertLess(reports[1]['evaluations'],reports[0]['evaluations'])
            self.assertEqual(reports[1]['target_sample_count'],3)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_bad_cached_model_falls_back_to_fresh_native_sensitivities(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers'][0,5]=10000
        with Session(arrays,meta,contacts=False) as state,Session(arrays,meta,contacts=False) as reference:
            cache=SensitivityCache(4);reference.advance(np.array([.8]),1/30)
            inverse_step(state,reference.surface(),[.2],1/30,iterations=1,sensitivities=cache)
            reference.advance(np.array([.8]),1/30)
            with patch.object(cache,'jacobian',return_value=np.zeros((12,1))):
                _,report=inverse_step(state,reference.surface(),[.2],1/30,iterations=1,sensitivities=cache)
            self.assertTrue(report['sensitivity_reused']);self.assertEqual(report['sensitivity_fallbacks'],1)
            self.assertEqual(report['sensitivity_refreshes'],1)
            self.assertLess(report['target_rms_after_mm'],report['target_rms_before_mm'])
            self.assertEqual(cache.uses,0)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_parallel_cached_sensitivities_match_serial_controls_and_geometry(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache,NativeTrialPool
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        arrays['fibers'][0,5]=10000
        with Session(arrays,meta,contacts=False) as serial,Session(arrays,meta,contacts=False) as parallel, \
                Session(arrays,meta,contacts=False) as reference,NativeTrialPool(parallel,2) as pool:
            a,b=SensitivityCache(2),SensitivityCache(2)
            for _ in range(3):
                reference.advance(np.array([.8]),1/30);target=reference.surface()
                x,r=inverse_step(serial,target,[.2],1/30,iterations=1,sensitivities=a)
                y,s=inverse_step(parallel,target,[.2],1/30,iterations=1,sensitivities=b,trial_pool=pool)
                np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(serial.positions,parallel.positions)
                np.testing.assert_array_equal(a.derivatives,b.derivatives)
                self.assertEqual(r['evaluations'],s['evaluations'])

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_cached_trial_cancellation_restores_state_and_discards_model(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import SensitivityCache
        arrays,meta=fixture();cache=SensitivityCache(2)
        with Session(arrays,meta,contacts=False) as state:
            target=arrays['surface_rest'].copy();target[1,0]-=.0005
            inverse_step(state,target,[.2],1/30,iterations=1,sensitivities=cache)
            before=state.positions.copy();activation=state.activation.copy()
            def interrupt(fraction,message):
                if 'reusing' in message:raise RuntimeError('cancelled cached trial')
            with self.assertRaisesRegex(RuntimeError,'cancelled cached trial'):
                inverse_step(state,target,[.2],1/30,iterations=1,sensitivities=cache,progress=interrupt)
            np.testing.assert_array_equal(state.positions,before);np.testing.assert_array_equal(state.activation,activation)
            self.assertIsNone(cache.derivatives);self.assertIsNone(cache.owner)

    def test_regular_timestamps_keep_a_constant_physics_time_base(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as uniform,Session(arrays,meta,contacts=False) as timeline:
            with patch.object(uniform,'step',wraps=uniform.step) as a,patch.object(timeline,'step',wraps=timeline.step) as b:
                for i in range(12):
                    excitation=np.array([.2 if i<6 else 0.])
                    uniform.advance(excitation,1/30)
                    timeline.advance(excitation,(i+1)/30-i/30)
                self.assertEqual(a.call_count,96);self.assertEqual(b.call_count,96)
            np.testing.assert_allclose(timeline.positions,uniform.positions,rtol=0,atol=1e-10)
            np.testing.assert_allclose(timeline.activation,uniform.activation,rtol=0,atol=1e-14)

    def test_native_work_counts_include_retries_and_survive_dynamic_rollback(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            original=state._step;attempts=[]
            def retry_once(excitation,dt):
                result=original(excitation,dt);attempts.append(dt)
                if len(attempts)==1:raise FloatingPointError('force one subdivision')
                return result
            with Snapshot(state) as snapshot:
                with patch.object(state,'_step',retry_once):state.step(np.array([.1]),1/240)
                self.assertEqual(state.native_steps,3);self.assertEqual(state.subdivision_retries,1)
                snapshot.restore()
                np.testing.assert_array_equal(state.positions,arrays['rest'])
                self.assertEqual(state.native_steps,3)
            report=state.advance(np.array([0.]),1/30)
            self.assertEqual(report['nominal_substeps'],8);self.assertEqual(report['native_steps'],8)
            self.assertEqual(report['subdivision_retries'],0);self.assertEqual(state.native_steps,11)

    def test_sliding_attachments_survive_snapshot_clone_and_continuation(self):
        if not hasattr(library(),'lightphysics_deformable_set_attachment_frames'):
            self.skipTest('rebuild CPU library with anisotropic attachments')
        arrays,meta=fixture()
        arrays['anchor_normals']=np.tile([0.,0.,1.],(3,1))
        arrays['anchor_tangent_compliance']=np.ones(3)
        with Session(arrays,meta,contacts=False) as source,Session(arrays,meta,contacts=False) as clone:
            source.advance(np.array([.4]),1/30)
            with Snapshot(source) as initial:initial.restore_into(clone)
            source.advance(np.array([.2]),1/30);clone.advance(np.array([.2]),1/30)
            np.testing.assert_array_equal(source.positions,clone.positions)
            np.testing.assert_array_equal(source.activation,clone.activation)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_parallel_inverse_matches_serial_state_and_continuation(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        arrays['fibers']=np.array([[0,0,1,0,0,1000],[0,1,0,1,0,800],[0,2,0,0,1,600]],float)
        meta.update(muscles=['x','y','z'],control_prior=[{}, {}, {}])
        target=arrays['surface_rest'].copy();target[1]+=[-.0005,.0002,.0001]
        with Session(arrays,meta,contacts=False) as serial,Session(arrays,meta,contacts=False) as parallel:
            with NativeTrialPool(parallel,2) as pool:
                for _ in range(2):
                    x,a=inverse_step(serial,target,[.4,.2,.1],1/30,iterations=2,previous=[0.,0.,0.])
                    y,b=inverse_step(parallel,target,[.4,.2,.1],1/30,iterations=2,previous=[0.,0.,0.],trial_pool=pool)
                    np.testing.assert_array_equal(x,y)
                    np.testing.assert_array_equal(serial.positions,parallel.positions)
                    np.testing.assert_array_equal(serial.activation,parallel.activation)
                    self.assertEqual(a['evaluations'],b['evaluations'])
                    self.assertEqual(a['target_rms_after_mm'],b['target_rms_after_mm'])
                    self.assertEqual(b['inverse_workers'],2)
            serial.advance(np.zeros(3),1/30);parallel.advance(np.zeros(3),1/30)
            np.testing.assert_array_equal(serial.positions,parallel.positions)
            np.testing.assert_array_equal(serial.activation,parallel.activation)

    def test_trial_pool_runs_independent_native_bodies_concurrently_in_input_order(self):
        from contextlib import ExitStack
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        ids=np.arange(4);probes=[np.array([.2]),np.array([.8])]
        with Session(arrays,meta,contacts=False) as state,NativeTrialPool(state,2) as pool,Snapshot(state) as initial:
            expected=[]
            for probe in probes:
                initial.restore();state.advance(probe,1/30)
                expected.append((state.surface()-arrays['surface_rest']).reshape(-1)/.001)
            initial.restore()
            barrier=threading.Barrier(2,timeout=5)
            def simultaneous(advance):
                def run(probe,seconds):
                    barrier.wait()
                    return advance(probe,seconds)
                return run
            with ExitStack() as stack:
                for worker in pool.sessions:
                    stack.enter_context(patch.object(worker,'advance',simultaneous(worker.advance)))
                actual=pool.residuals(initial,probes,1/30,ids,arrays['surface_rest'])
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,[0.])

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_parallel_inverse_cancellation_joins_workers_and_restores_owner(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        target=arrays['surface_rest'].copy();target[1,0]-=.0005
        def interrupt(fraction,message):
            if 'sensitivity' in message:raise RuntimeError('cancelled after concurrent trial')
        with Session(arrays,meta,contacts=False) as state,NativeTrialPool(state,2) as pool:
            with self.assertRaisesRegex(RuntimeError,'cancelled after concurrent'):
                inverse_step(state,target,[.4],1/30,trial_pool=pool,progress=interrupt)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,[0.])
            # A cancelled batch can be followed by a fresh batch on this pool.
            _,report=inverse_step(state,target,[.4],1/30,iterations=1,trial_pool=pool)
            self.assertEqual(report['inverse_workers'],2)

    def test_trial_pool_reports_unsafe_samples_and_rejects_other_owner(self):
        from contextlib import ExitStack
        from .rig.biomechanics_trials import NativeTrialPool
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state,Session(arrays,meta,contacts=False) as other, \
                NativeTrialPool(state,2) as pool,Snapshot(state) as initial,Snapshot(other) as wrong:
            with self.assertRaisesRegex(ValueError,'different session'):
                pool.residuals(wrong,[np.zeros(1)],1/30,np.arange(4),arrays['surface_rest'])
            def guarded(advance):
                def run(probe,seconds):
                    if probe[0]>.5:raise FloatingPointError('unsafe')
                    return advance(probe,seconds)
                return run
            with ExitStack() as stack:
                for worker in pool.sessions:
                    stack.enter_context(patch.object(worker,'advance',guarded(worker.advance)))
                results=pool.residuals(initial,[np.zeros(1),np.ones(1)],1/30,np.arange(4),arrays['surface_rest'])
            self.assertIsNotNone(results[0]);self.assertIsNone(results[1])
            np.testing.assert_array_equal(state.positions,arrays['rest'])

    def test_snapshot_clones_dynamics_into_an_identical_session(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as source,Session(arrays,meta,contacts=False) as clone:
            source.advance(np.array([.6]),1/30)
            with Snapshot(source) as snapshot:
                snapshot.restore_into(clone)
                np.testing.assert_array_equal(source.positions,clone.positions)
                np.testing.assert_array_equal(source.activation,clone.activation)
                source.advance(np.array([.2]),1/30);clone.advance(np.array([.2]),1/30)
                np.testing.assert_array_equal(source.positions,clone.positions)
                np.testing.assert_array_equal(source.activation,clone.activation)
                with Session(arrays,meta,contacts=False,iterations=12) as wrong:
                    with self.assertRaisesRegex(ValueError,'same static anatomy'):
                        snapshot.restore_into(wrong)
                    np.testing.assert_array_equal(wrong.positions,arrays['rest'])

    @unittest.skipUnless(HAS_TORCH and HAS_SCIPY and HAS_USD,'CPU bake/validation dependencies')
    def test_first_frame_unsafe_prior_falls_back_to_neutral_drive(self):
        from .rig import biomechanics
        from .test_rig import _toy_rig
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        arrays,meta=fixture()
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory)/'rig';take=Path(directory)/'take';rig.mkdir();take.mkdir()
            definition=_toy_rig()
            definition['joints'].append(dict(name='head',parent='root',bind=np.eye(4).tolist(),
                rest_translation=[0,0,0],rest_rotation=np.eye(3).tolist()))
            (rig/'rig.json').write_text(json.dumps(definition));(rig/'rig_deformer.safetensors').write_bytes(b'fixture')
            package=rig/'anatomy';package.mkdir();np.savez_compressed(package/'anatomy.npz',**arrays)
            meta.update(rig_sha256=anatomy.digest(rig/'rig.json'),rig_package_sha256=anatomy.digest(rig/'rig_deformer.safetensors'),
                        arrays_sha256=anatomy.digest(package/'anatomy.npz'))
            anatomy.write_json(package/'anatomy.json',meta)
            frames=[dict(t=0,v={'mouthSmileLeft':1.})]
            anatomy.write_json(take/'animation.json',dict(format='vhuman.performance.v1',fps=30,frames=frames))
            def targets(*args,**kwargs):
                target=arrays['surface_rest'].copy();target[1,0]-=.0005
                yield frames[0]['v'],target,None
            original=Session.advance
            def guarded(state,excitation,seconds):
                if excitation.max()>.5:
                    raise FloatingPointError('unsafe semantic jump')
                return original(state,excitation,seconds)
            with patch.object(biomechanics,'target_sequence',targets),patch.object(Session,'advance',guarded):
                report=biomechanics.simulate(rig,take,threads=1,iterations=0,review=False)
            self.assertTrue(report['per_frame'][0]['seed_fallback'])
            self.assertEqual(report['per_frame'][0]['unsafe_trials'],1)
            self.assertTrue(report['numerically_valid'])
            with np.load(take/'physics_points_0000.npz') as z:
                np.testing.assert_array_equal(z['excitations'][0],[0.])
                np.testing.assert_array_equal(z['activations'][0],[0.])
            self.assertEqual(report['per_frame'][0]['active_muscles'],0)
            self.assertEqual(report['per_frame'][0]['candidate_muscles'],1)
            # Export the actual dynamic activation separately from its drive.
            # The accepted state advances once from the saved neutral state,
            # even though the inverse controller also evaluates a trial.
            with patch.object(biomechanics,'target_sequence',targets):
                driven=biomechanics.simulate(rig,take,out=Path(directory)/'driven',threads=1,iterations=0,review=False)
            expected=1-np.exp(-(1/30)/.025)
            with np.load(Path(directory)/'driven/physics_points_0000.npz') as z:
                np.testing.assert_array_equal(z['excitations'][0],[1.])
                np.testing.assert_allclose(z['activations'][0],[expected],rtol=0,atol=1e-12)
            with np.load(Path(directory)/'driven/physics_state.npz') as z:
                self.assertEqual(z['muscle_names'].tolist(),['test'])
                np.testing.assert_allclose(z['activations'][0],[expected],rtol=0,atol=1e-12)
            self.assertEqual(driven['per_frame'][0]['active_muscles'],1)
            frames[0]['v']={'mouthSmileLeft':.01}
            anatomy.write_json(take/'animation.json',dict(format='vhuman.performance.v1',fps=30,frames=frames))
            def tiny_targets(*args,**kwargs):
                yield frames[0]['v'],arrays['surface_rest'].copy(),None
            with patch.object(biomechanics,'target_sequence',tiny_targets):
                tiny=biomechanics.simulate(rig,take,out=Path(directory)/'tiny',threads=1,iterations=0,review=False)
            # A small requested expression is not the neutral initialization,
            # even when its geometric target falls below the rest tolerance.
            self.assertEqual(tiny['per_frame'][0]['evaluations'],1)
            self.assertEqual(tiny['per_frame'][0]['active_muscles'],1)
            with np.load(Path(directory)/'tiny/physics_points_0000.npz') as z:
                np.testing.assert_allclose(z['activations'][0],[.01*expected],rtol=0,atol=1e-12)

    @unittest.skipUnless(HAS_TORCH and HAS_SCIPY and HAS_USD,'CPU bake/validation dependencies')
    def test_capture_bake_uses_recorded_head_and_eye_transforms(self):
        from types import SimpleNamespace
        from .rig import biomechanics,biomechanics_mapping
        from .reconstruction import gnm_motion
        from .test_rig import _toy_rig
        from .service import ROOT
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        arrays,meta=fixture()
        eye=np.array([[.05,0,.01],[.06,0,.01],[.05,.01,.01]])
        arrays['rest']=np.concatenate((arrays['rest'],eye))
        arrays['surface_rest']=arrays['rest'].copy()
        arrays['faces']=np.concatenate((arrays['faces'],[[4,5,6]])).astype(np.int32)
        arrays['inverse_mass']=np.r_[arrays['inverse_mass'],np.zeros(3)]
        arrays['bone']=np.r_[arrays['bone'],np.zeros(3,np.int32)]
        arrays['embed_ids']=np.repeat(np.arange(7)[:,None],4,axis=1).astype(np.int32)
        arrays['embed_weights']=np.tile([1.,0,0,0],(7,1));arrays['source_ids']=np.arange(7,dtype=np.int32)
        meta['parts'].append(dict(name='eye_left',surface_start=4,surface_count=3,face_start=4,face_count=1))
        bind=dict(identity=np.zeros(1),scale=1.,alignment=np.eye(3),offset=np.zeros(3))
        model=SimpleNamespace(parents=[-1,0,1,1],data=dict(
            joint_names=np.array(['neck','head','left_eye','right_eye']),joint_identity_basis=np.zeros((1,4,3)),
            template_joint_positions=np.array([[0,0,0],[0,0,0],[.05,0,0],[-.05,0,0]])))
        rotations=np.zeros((2,4,3));rotations[:,0]=[.1,.2,0];rotations[:,1]=[0,.05,0]
        rotations[:,2,2]=[.001,.002]  # sub-0.1 mm gaze still applies in frame zero
        translation=np.array([[.01,.002,0],[.011,.002,0]])
        transforms=[biomechanics_mapping.capture_transforms(model,bind,r,t) for r,t in zip(rotations,translation)]
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory)/'rig';take=Path(directory)/'take';rig.mkdir();take.mkdir()
            definition=_toy_rig()
            definition['joints'].append(dict(name='head',parent='root',bind=np.eye(4).tolist(),
                rest_translation=[0,0,0],rest_rotation=np.eye(3).tolist()))
            (rig/'rig.json').write_text(json.dumps(definition));(rig/'rig_deformer.safetensors').write_bytes(b'fixture')
            package=rig/'anatomy';package.mkdir();np.savez_compressed(package/'anatomy.npz',**arrays)
            meta.update(rig_sha256=anatomy.digest(rig/'rig.json'),rig_package_sha256=anatomy.digest(rig/'rig_deformer.safetensors'),
                        arrays_sha256=anatomy.digest(package/'anatomy.npz'))
            anatomy.write_json(package/'anatomy.json',meta)
            frames=[dict(t=i/30,v={'headYaw':.9,'eyeLookDownLeft':1.}) for i in range(2)]
            anatomy.write_json(take/'animation.json',dict(format='vhuman.performance.v1',fps=30,frames=frames))
            np.savez_compressed(take/'motion.npz',identity=bind['identity'],expression=np.zeros((2,1)),
                rotations=rotations,translation=translation,timestamps=np.arange(2)/30)
            anatomy.write_json(take/'motion.json',dict(quality_gate=True,motion_sha256=anatomy.digest(take/'motion.npz')))
            def targets(*args,**kwargs):
                np.testing.assert_array_equal(kwargs['motion']['translation'],translation)
                for frame,pose in zip(frames,transforms):
                    target=arrays['surface_rest'].copy();local=np.linalg.inv(pose['head'])@pose['left_eye']
                    target[4:]=eye@local[:3,:3].T+local[:3,3]
                    yield frame['v'],target,None
            with patch.object(biomechanics,'target_sequence',targets), \
                 patch.object(gnm_motion,'rig_bind',return_value=bind), \
                 patch('server.vhuman.rig.gnm_model.GNMModel',return_value=model):
                report=biomechanics.simulate(rig,take,threads=1,iterations=0,review=False)
            self.assertEqual(report['cranial_motion_source'],'captured GNM joint transforms')
            self.assertTrue(report['quality_gate'])
            with np.load(take/'physics_points_0000.npz') as z:
                for i,pose in enumerate(transforms):
                    np.testing.assert_allclose(z['head_transforms'][i],pose['head'],atol=1e-14)
                    expected=eye@pose['left_eye'][:3,:3].T+pose['left_eye'][:3,3]
                    np.testing.assert_allclose(z['positions'][i,4:],expected,atol=1e-8)

    @unittest.skipUnless(HAS_TORCH and HAS_SCIPY and HAS_USD and shutil.which('ffmpeg'),'full CPU bake/validation dependencies')
    def test_complete_cpu_bake_and_usd_replay(self):
        from .rig import biomechanics
        from .test_rig import _toy_rig
        from .service import ROOT
        from . import gpu
        scratch=ROOT/'tmp/vhuman-biomechanics-tests';scratch.mkdir(parents=True,exist_ok=True)
        arrays,meta=fixture()
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            rig=Path(directory)/'rig';take=Path(directory)/'take'
            rig.mkdir();take.mkdir()
            definition=_toy_rig()
            definition['joints'].append(dict(name='head',parent='root',bind=np.eye(4).tolist(),
                                            rest_translation=[0,0,0],rest_rotation=np.eye(3).tolist()))
            (rig/'rig.json').write_text(json.dumps(definition));(rig/'rig_deformer.safetensors').write_bytes(b'fixture')
            package=rig/'anatomy';package.mkdir()
            np.savez_compressed(package/'anatomy.npz',**arrays)
            meta.update(rig_sha256=anatomy.digest(rig/'rig.json'),rig_package_sha256=anatomy.digest(rig/'rig_deformer.safetensors'),
                        arrays_sha256=anatomy.digest(package/'anatomy.npz'))
            anatomy.write_json(package/'anatomy.json',meta)
            frames=[dict(t=i/30,v={} if i==0 else {'mouthSmileLeft':.5}) for i in range(3)]
            anatomy.write_json(take/'animation.json',dict(format='vhuman.performance.v1',fps=30,frames=frames))
            with wave.open(str(take/'audio.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(bytes(2400*2))
            # Use a mechanically reachable target. Compressing the sole free
            # vertex along X conflicts with this fixture's hard volume solve.
            # A whole-head RMS used to hide that local tracking failure.
            with Session(arrays,meta) as reference:
                reference.advance(np.zeros(1),1/30)
                goals=[reference.surface().copy()]
                for _ in range(2):
                    reference.advance(np.array([.5]),1/30)
                    goals.append(reference.surface().copy())
            def targets(*args,**kwargs):
                for i,frame in enumerate(frames):
                    yield frame['v'],goals[i],None
            with patch.object(biomechanics,'target_sequence',targets),patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU probe')):
                report=biomechanics.simulate(rig,take,threads=1,iterations=1,review=True)
                legacy=biomechanics.simulate(rig,take,out=take/'legacy',threads=1,iterations=1,
                                             state_precision='float32',review=False)
                coupled=biomechanics.simulate(rig,take,out=take/'coupled',threads=1,iterations=1,
                                              mechanics_solver='coupled',review=False)
                block=biomechanics.simulate(rig,take,out=take/'block',threads=1,iterations=1,
                                            mechanics_solver='block',review=False)
            self.assertTrue(block['quality_gate'])
            self.assertEqual(block['solver_settings']['mechanics_solver'],'block')
            self.assertEqual(block['solver_settings']['jaw_mode'],'coupled_block')
            self.assertTrue(coupled['quality_gate'])
            self.assertEqual(coupled['python_sources_at_start'],biomechanics.python_source_receipt())
            self.assertIn('server/vhuman/rig/biomechanics_native.py',coupled['python_sources_at_start'])
            self.assertEqual(coupled['solver_settings']['mechanics_solver'],'coupled')
            self.assertEqual(coupled['solver_settings']['jaw_mode'],'coupled_volume')
            self.assertEqual(coupled['solver_settings']['contact_transport'],'embedded_coupled')
            with self.assertRaisesRegex(ValueError,'precision'):
                biomechanics.simulate(rig,take,mechanics_solver='coupled',state_precision='float32')
            with self.assertRaisesRegex(ValueError,'precision'):
                biomechanics.simulate(rig,take,mechanics_solver='block',state_precision='float32')
            with self.assertRaisesRegex(ValueError,'mechanics solver'):
                biomechanics.simulate(rig,take,mechanics_solver='gpu')
            self.assertTrue(legacy['quality_gate'])
            self.assertEqual(legacy['solver_settings']['state_transport_precision'],'float32')
            with self.assertRaisesRegex(ValueError,'precision'):
                biomechanics.simulate(rig,take,state_precision='float16')
            self.assertTrue(report['quality_gate']);self.assertEqual(report['frames'],3)
            self.assertTrue((take/'physics_review.mp4').is_file())
            with np.load(take/'physics_points_0000.npz',allow_pickle=False) as z:
                baked=z['positions'];self.assertEqual(baked.shape,(3,4,3))
            # Verify through an independent USD reader when the pinned optional
            # validation environment is installed.
            try:
                from pxr import Usd,UsdGeom,UsdSkel
            except ImportError:
                return
            stage=Usd.Stage.Open(str(take/'physics.usda'))
            mesh=UsdGeom.Mesh.Get(stage,'/Character/skin')
            for i in range(3):
                np.testing.assert_allclose(np.asarray(mesh.GetPointsAttr().Get(i)),baked[i],atol=1e-8)
            self.assertFalse(mesh.GetPrim().HasAPI(UsdSkel.BindingAPI))

    def test_neutral_equilibrium_and_snapshot_replay(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            for _ in range(5):state.advance(np.zeros(1),1/30)
            np.testing.assert_allclose(state.positions,arrays['rest'],atol=1e-8)
            with Snapshot(state) as snapshot:
                state.advance(np.ones(1),1/30);one=state.positions.copy();activity=state.activation.copy()
                snapshot.restore();state.advance(np.ones(1),1/30)
                np.testing.assert_array_equal(state.positions,one)
                np.testing.assert_array_equal(state.activation,activity)
            self.assertLess(state.positions[1,0],arrays['rest'][1,0])
            self.assertGreater(volumes(state.positions,arrays['tets'])[0],0)

    def test_bad_excitation_does_not_advance_state(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            before=state.positions.copy()
            for bad in ([-.1],[1.1],[float('nan')],[0,1]):
                with self.assertRaises(ValueError):state.step(bad,1/240)
                np.testing.assert_array_equal(state.positions,before)

    def test_rejected_physics_step_rolls_back_native_and_muscle_state(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            original=state._step
            def rejected(excitation,dt):
                original(excitation,dt)
                raise FloatingPointError('synthetic acceptance failure')
            with patch.object(state,'_step',rejected):
                with self.assertRaises(FloatingPointError):state.step(np.ones(1),1/240,subdivisions=0)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,np.zeros(1))
            state.advance(np.zeros(1),1/30)
            np.testing.assert_allclose(state.positions,arrays['rest'],atol=1e-8)

    def test_interval_cancellation_stops_between_steps_and_rolls_back(self):
        arrays,meta=fixture();cancel=threading.Event()
        with Session(arrays,meta,contacts=False,cancel=cancel) as state:
            original=state._step;calls=[]
            def stop_after_first(excitation,dt):
                result=original(excitation,dt);calls.append(dt);cancel.set();return result
            with patch.object(state,'_step',stop_after_first):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    state.advance(np.ones(1),1/30)
            self.assertEqual(len(calls),1)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,[0.])
            cancel.clear();state.advance(np.zeros(1),1/30)
            np.testing.assert_allclose(state.positions,arrays['rest'],rtol=0,atol=1e-8)

    def test_inverse_control_recovers_from_unsafe_semantic_prior(self):
        from .rig.biomechanics import inverse_step
        from .rig.biomechanics_native import GeometryStepError
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            advance=state.advance
            def guarded(excitation,seconds):
                if np.max(excitation)>.1:
                    raise GeometryStepError('unsafe trial',dict(contact_residual_mm=.2))
                return advance(excitation,seconds)
            with patch.object(state,'advance',guarded):
                chosen,report=inverse_step(state,arrays['surface_rest'],np.ones(1),1/30,
                    previous=np.zeros(1),iterations=0)
            np.testing.assert_array_equal(chosen,[0.])
            np.testing.assert_allclose(state.positions,arrays['rest'],atol=1e-8)
            self.assertTrue(report['seed_fallback'])
            self.assertEqual(report['unsafe_trials'],1)
            self.assertEqual(report['unsafe_trial_diagnostics'],[
                dict(error='unsafe trial',failed_step=dict(contact_residual_mm=.2))])

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_rejects_bad_samples_and_history_before_advancing(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            for ids in ([],[.5],[-1],[4],[[0]],np.array([2**63],np.uint64)):
                with self.assertRaisesRegex(ValueError,'samples'):
                    inverse_step(state,arrays['surface_rest'],[0.],1/30,sample_ids=ids)
            for prior,previous in (([float('nan')],[0]),([1.1],[0]),([0],[-1]),([0,1],[0])):
                with self.assertRaisesRegex(ValueError,'history/prior'):
                    inverse_step(state,arrays['surface_rest'],prior,1/30,previous=previous)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,[0.])

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_progress_cancellation_restores_complete_state(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        def cancel(fraction,message):
            self.assertGreaterEqual(fraction,0)
            if 'sensitivity' in message:
                raise RuntimeError('cancelled at muscle sensitivity')
        with Session(arrays,meta,contacts=False) as state:
            target=arrays['surface_rest'].copy();target[1,0]-=.0005
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                inverse_step(state,target,[1.],1/30,iterations=1,progress=cancel)
            np.testing.assert_array_equal(state.positions,arrays['rest'])
            np.testing.assert_array_equal(state.activation,[0.])
            # The native projector has private activation too. A raw solver
            # step must remain neutral after cancellation, before another
            # Session.advance has a chance to overwrite that private drive.
            out=np.empty(state.rest.shape,np.float32)
            self.assertEqual(state.lib.lightphysics_deformable_step(state.body,None,1/240,ptr(out)),0)
            np.testing.assert_allclose(out,arrays['rest'],rtol=0,atol=1e-8)

    @unittest.skipUnless(HAS_SCIPY,'SciPy inverse control')
    def test_inverse_commits_validated_trial_without_reintegrating(self):
        from .rig.biomechanics import inverse_step
        arrays,meta=fixture()
        for iterations in (0,1):
            with Session(arrays,meta,contacts=False) as expected,Session(arrays,meta,contacts=False) as fitted:
                target=arrays['surface_rest'].copy();target[1,0]-=.0005
                with patch.object(fitted,'advance',wraps=fitted.advance) as counted:
                    chosen,report=inverse_step(fitted,target,[.8],1/30,iterations=iterations,previous=[0.])
                self.assertEqual(counted.call_count,report['evaluations'])
                if iterations==0:self.assertEqual(counted.call_count,1)
                expected.advance(chosen,1/30)
                np.testing.assert_array_equal(fitted.positions,expected.positions)
                np.testing.assert_array_equal(fitted.activation,expected.activation)
                # Check history-dependent continuation as well as the points
                # exposed immediately after committing the saved trial.
                fitted.advance(np.zeros(1),1/30);expected.advance(np.zeros(1),1/30)
                np.testing.assert_array_equal(fitted.positions,expected.positions)
                np.testing.assert_array_equal(fitted.activation,expected.activation)

    def test_muscle_impulses_conserve_linear_momentum(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False,implicit=False) as state:
            check=state.lib.lightphysics_active_tissue_apply(state.field,ptr(np.ones(1)),1,1/240)
            self.assertEqual(check,0)
            positions=np.empty((4,3),np.float32);velocity=np.empty_like(positions)
            self.assertEqual(state.lib.lightphysics_deformable_read_state(state.body,ptr(positions),ptr(velocity),12),0)
            np.testing.assert_allclose((velocity/arrays['inverse_mass'][:,None]).sum(0),0,atol=1e-10)
            self.assertLess(velocity[1,0],0)

    def test_shared_tet_fiber_geometry_matches_independent_piola_forces(self):
        arrays,meta=fixture();rest=np.r_[arrays['rest'],arrays['rest']+[.04,0,0]]
        arrays.update(rest=rest,surface_rest=rest.copy(),tets=np.array([[0,1,2,3],[4,5,6,7]],np.int32),
            faces=np.r_[arrays['faces'],arrays['faces']+4],inverse_mass=np.full(8,1000.),bone=np.full(8,-1,np.int32),
            anchor_ids=np.empty(0,np.int32),anchor_bone=np.empty(0,np.int32),youngs_moduli=np.full(2,10000,np.float32),
            embed_ids=np.repeat(np.arange(8)[:,None],4,axis=1).astype(np.int32),embed_weights=np.tile([1.,0,0,0],(8,1)),
            source_ids=np.arange(8,dtype=np.int32),
            fibers=np.array([[1,0,1,2,0,1000],[0,1,0,1,1,700],[1,1,1,0,1,1300],[0,0,1,0,0,500.]]))
        meta=dict(meta,muscles=['first','second'],control_prior=[{},{}])
        deformation=np.diag([.9,1.1,1.03]);positions=np.ascontiguousarray(rest@deformation.T,np.float32)
        activation=np.array([.3,.7]);expected=np.zeros((8,3));dt=1/240
        for row in arrays['fibers']:
            tet,muscle=map(int,row[:2]);ids=arrays['tets'][tet];p=rest[ids]
            inverse=np.linalg.inv((p[1:]-p[0]).T)
            gradient=np.r_[-inverse.sum(0)[None],inverse]
            direction=row[2:5]/np.linalg.norm(row[2:5]);current=deformation@direction
            stretch=np.linalg.norm(current);envelope=np.exp(-((stretch-1)/.5)**2)
            volume=abs(np.linalg.det((p[1:]-p[0]).T))/6
            force=-activation[muscle]*row[5]*envelope*volume/stretch*(gradient@direction)[:,None]*current
            np.add.at(expected,ids,force*dt*arrays['inverse_mass'][ids,None])
        with Session(arrays,meta,contacts=False,implicit=False) as state:
            velocity=np.zeros_like(positions)
            self.assertEqual(state.lib.lightphysics_deformable_write_state(state.body,ptr(positions),ptr(velocity),positions.size),0)
            self.assertEqual(state.lib.lightphysics_active_tissue_apply(state.field,ptr(activation),2,dt),0)
            self.assertEqual(state.lib.lightphysics_deformable_read_state(state.body,ptr(positions),ptr(velocity),positions.size),0)
            np.testing.assert_allclose(velocity,expected,rtol=2e-6,atol=2e-9)

    def test_idle_implicit_projector_resumes_after_snapshot_replay(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            state.advance(np.array([.6]),1/30)
            with Snapshot(state) as snapshot:
                state.advance(np.array([0.]),1/30);expected=state.positions.copy()
                # Exact zero deactivation bypasses an idle projector. Restoring
                # the prior active state must reinstall the same projector.
                for muscle in state.muscles:muscle.activation=0
                state.activation[:]=0
                self.assertEqual(state.lib.lightphysics_active_tissue_apply(state.field,ptr(state.activation),1,1/240),0)
                snapshot.restore();state.advance(np.array([0.]),1/30)
                np.testing.assert_array_equal(state.positions,expected)

    def test_quiet_material_cache_does_not_hide_changed_stiffness(self):
        arrays,meta=fixture();arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        tiny=dict(arrays,youngs_moduli=np.array([1e-30],np.float32))
        positions=arrays['rest'].astype(np.float32);positions[:,0]+=.2*positions[:,1]
        with Session(tiny,meta,contacts=False) as cached,Session(arrays,meta,contacts=False) as reference:
            for state in (cached,reference):
                velocity=np.zeros_like(positions)
                self.assertEqual(state.lib.lightphysics_deformable_write_state(state.body,ptr(positions),ptr(velocity),positions.size),0)
            ignored=np.empty_like(positions)
            self.assertEqual(cached.lib.lightphysics_deformable_step(cached.body,None,1/240,ptr(ignored)),0)
            np.testing.assert_array_equal(ignored,positions)
            high=np.array([10000.],np.float32)
            self.assertEqual(cached.lib.lightphysics_deformable_set_youngs_moduli(cached.body,ptr(high),1),0)
            actual=np.empty_like(positions);expected=np.empty_like(positions)
            self.assertEqual(cached.lib.lightphysics_deformable_step(cached.body,None,1/240,ptr(actual)),0)
            self.assertEqual(reference.lib.lightphysics_deformable_step(reference.body,None,1/240,ptr(expected)),0)
            np.testing.assert_array_equal(actual,expected)
            self.assertGreater(float(abs(actual-positions).max()),1e-6)

    def test_implicit_fibers_contract_without_net_linear_impulse(self):
        arrays,meta=fixture()
        arrays['anchor_ids']=np.empty(0,np.int32);arrays['anchor_bone']=np.empty(0,np.int32)
        with Session(arrays,meta,contacts=False) as state:
            report=state.advance(np.ones(1),1/30)
            positions=np.empty_like(state.rest,dtype=np.float32);velocity=np.empty_like(positions)
            self.assertEqual(state.lib.lightphysics_deformable_read_state(state.body,ptr(positions),ptr(velocity),positions.size),0)
            np.testing.assert_allclose((velocity/arrays['inverse_mass'][:,None]).sum(0),0,atol=1e-8)
            self.assertLess(state.positions[1,0],arrays['rest'][1,0])
            self.assertGreater(report['min_volume_ratio'],.05)

    def test_swept_contact_prevents_a_fast_triangle_crossing(self):
        lib=library();handle=P()
        old=np.array([[-.02,-.02,0],[.02,-.02,0],[0,.02,0],[-.01,-.01,.01],[.01,-.01,.01],[0,.01,.01]],float)
        faces=np.array([[0,1,2],[3,4,5]],np.int32);mass=np.array([0,0,0,1,1,1.],float)
        self.assertEqual(lib.lightphysics_surface_contact_create(ptr(old),ptr(mass),6,ptr(faces),2,ct.byref(handle)),0)
        try:
            x=old.copy();x[3:,2]=-.01;reaction=np.zeros_like(x);stats=ContactStats()
            self.assertEqual(lib.lightphysics_surface_contact_project(handle,ptr(old),ptr(x),1/240,.0002,4,ptr(reaction),ct.byref(stats)),0)
            self.assertGreater(stats.contacts,0)
            self.assertGreaterEqual(x[3:,2].min(),.00019)
            self.assertLess(reaction[:3,2].sum(),0)
            np.testing.assert_allclose(reaction.sum(0),0,atol=1e-10)
        finally:lib.lightphysics_surface_contact_destroy(handle)

    def test_grazing_contact_cannot_tunnel_when_advancement_budget_is_exhausted(self):
        lib=library();handle=P()
        old=np.array([[-.02,-.02,0],[.02,-.02,0],[0,.02,0],
            [-.005,-.005,.000012],[.005,-.005,.000012],[0,.005,.000012]])
        faces=np.array([[0,1,2],[3,4,5]],np.int32);mass=np.array([0.,0,0,1,1,1])
        self.assertEqual(lib.lightphysics_surface_contact_create(ptr(old),ptr(mass),6,ptr(faces),2,ct.byref(handle)),0)
        try:
            x=old.copy();x[3:,0]+=.01;x[3:,2]=-.000012
            stats=ContactStats();reaction=np.zeros_like(x)
            self.assertEqual(lib.lightphysics_surface_contact_project(handle,ptr(old),ptr(x),1/240,.0002,4,ptr(reaction),ct.byref(stats)),0)
            self.assertGreater(stats.contacts,0)
            self.assertGreaterEqual(x[3:,2].min(),.0000099)
            np.testing.assert_array_equal(x[:3],old[:3])
            np.testing.assert_allclose(reaction.sum(0),0,atol=1e-12)
        finally:lib.lightphysics_surface_contact_destroy(handle)

    def test_swept_separating_plane_preserves_a_grazing_near_miss(self):
        from .rig import rigdef
        lib=library()
        triangle=np.array([[-.01,-.01,0],[.01,-.01,0],[0,.01,0]])
        initial=np.r_[triangle,triangle+[-.04,0,.000200001]]
        final=initial.copy();final[3:,0]+=.08
        faces=np.array([[0,1,2],[3,4,5]],np.int32);mass=np.array([0.,0,0,1,1,1])
        for rotation in (np.eye(3),rigdef.euler_matrix(.37,-.25,.61)):
            with self.subTest(rotation=rotation.tolist()):
                old=np.ascontiguousarray(initial@rotation.T+[.03,.01,-.02])
                current=np.ascontiguousarray(final@rotation.T+[.03,.01,-.02]);expected=current.copy();contact=P()
                self.assertEqual(lib.lightphysics_surface_contact_create(ptr(old),ptr(mass),6,ptr(faces),2,ct.byref(contact)),0)
                try:
                    stats=ContactStats()
                    self.assertEqual(lib.lightphysics_surface_contact_project(contact,ptr(old),ptr(current),1/240,.0002,4,None,ct.byref(stats)),0)
                    self.assertEqual(stats.contacts,0)
                    np.testing.assert_array_equal(current,expected)
                finally:lib.lightphysics_surface_contact_destroy(contact)

    def test_contact_replay_ignores_previous_bvh_ordering(self):
        lib=library();handle=P();points=[]
        for i in range(12):
            x=-.012+i*.002
            points.extend([[x-.01,-.01,-.0001*i],[x+.01,-.01,.00005*i],[x,.012,0]])
        points.extend([[-.009,-.008,.01],[.009,-.008,.01],[0,.009,.01]])
        old=np.asarray(points,float);faces=np.arange(len(points),dtype=np.int32).reshape(-1,3)
        mass=np.zeros(len(points));mass[-3:]=1
        self.assertEqual(lib.lightphysics_surface_contact_create(ptr(old),ptr(mass),len(old),ptr(faces),len(faces),ct.byref(handle)),0)
        try:
            def project():
                x=old.copy();x[-3:,2]=-.01;stats=ContactStats()
                self.assertEqual(lib.lightphysics_surface_contact_project(handle,ptr(old),ptr(x),1/240,.0002,3,None,ct.byref(stats)),0)
                self.assertGreater(stats.contacts,0)
                return x
            first=project()
            shuffled=old[:,::-1].copy();count=ct.c_uint64()
            self.assertEqual(lib.lightphysics_surface_contact_count_intersections(handle,ptr(shuffled),ct.byref(count)),0)
            np.testing.assert_array_equal(project(),first)
        finally:lib.lightphysics_surface_contact_destroy(handle)

    def test_touching_bind_surfaces_can_separate_but_cannot_cross(self):
        lib=library();handle=P()
        rest=np.array([[-.02,-.02,0],[.02,-.02,0],[0,.02,0],[-.01,-.01,0],[.01,-.01,0],[0,.01,0]],float)
        faces=np.array([[0,1,2],[3,5,4]],np.int32);mass=np.array([0,0,0,1,1,1.],float)
        self.assertEqual(lib.lightphysics_surface_contact_create(ptr(rest),ptr(mass),6,ptr(faces),2,ct.byref(handle)),0)
        try:
            x=rest.copy();x[3:,2]=.001;stats=ContactStats()
            self.assertEqual(lib.lightphysics_surface_contact_project(handle,ptr(rest),ptr(x),1/240,.0002,3,None,ct.byref(stats)),0)
            np.testing.assert_allclose(x[3:,2],.001,atol=1e-12)
            x=rest.copy();x[3:,2]=-.001
            self.assertEqual(lib.lightphysics_surface_contact_project(handle,ptr(rest),ptr(x),1/240,.0002,3,None,ct.byref(stats)),0)
            self.assertGreaterEqual(x[3:,2].min(),-1e-9)
        finally:lib.lightphysics_surface_contact_destroy(handle)

    def test_stiff_tissue_jaw_feedback_settles_without_external_drive(self):
        arrays,meta=fixture()
        arrays['anchor_ids']=np.arange(4,dtype=np.int32)
        arrays['anchor_bone']=np.array([0,1,0,0],np.int32)
        arrays['youngs_moduli'][:]=1e7
        with Session(arrays,meta,contacts=False) as state:
            state.q[0]=.001
            for _ in range(240):
                report=state.step(np.zeros(1),1/240,subdivisions=0)
                self.assertLessEqual(state.q[0],.00101)
                self.assertGreater(report['min_volume_ratio'],.05)
            self.assertLess(state.q[0],.001)

    def test_jaw_attachment_reaction_matches_finite_difference_work(self):
        arrays,meta=fixture()
        with Session(arrays,meta,contacts=False) as state:
            state.q[:]=[.2,.001,.004]
            rest=np.array([[.01,-.02,.04],[-.01,-.01,.05]])
            forces=np.array([[1.,2.,3.],[-4.,2.,1.]])
            point=state.transform(rest)
            analytic=state.generalized(point,forces)
            numerical=[]
            for i in range(3):
                state.q[i]+=1e-7;plus=state.transform(rest)
                state.q[i]-=2e-7;minus=state.transform(rest)
                state.q[i]+=1e-7
                numerical.append(float(((plus-minus)*forces).sum()/2e-7))
            np.testing.assert_allclose(analytic,numerical,atol=1e-8)

    def test_jaw_is_rigid_and_force_driven(self):
        from .rig.biomechanics_native import JawMuscle
        lib=library();q=np.zeros(3);v=np.zeros(3);hinge=np.zeros(3);a=np.ones(1);reaction=np.zeros(3)
        muscles=(JawMuscle*1)(JawMuscle(0,(ct.c_double*3)(0,-.06,.02),(ct.c_double*3)(0,-.025,.06),4.,.053))
        for _ in range(20):self.assertEqual(lib.vhface_jaw_step(ptr(q),ptr(v),ptr(hinge),muscles,1,ptr(a),1,ptr(reaction),1/240),0)
        self.assertGreater(q[0],0);self.assertLessEqual(q[0],.48)
        points=np.array([[0,0,.02],[.01,-.01,.04],[-.01,0,.04]],float);out=np.empty_like(points)
        self.assertEqual(lib.vhface_jaw_transform(ptr(points),3,ptr(hinge),ptr(q),ptr(out)),0)
        np.testing.assert_allclose(np.linalg.norm(out[:,None]-out[None],axis=-1),np.linalg.norm(points[:,None]-points[None],axis=-1),atol=1e-14)


if __name__=='__main__':unittest.main()
