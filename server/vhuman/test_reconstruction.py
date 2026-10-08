"""Synthetic numerical and isolation gates for the original reconstruction path."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .reconstruction import reference as r, fitting, gaussian, depth, observations

ROOT = Path(__file__).resolve().parents[2] / 'tmp/test-reconstruction'
ROOT.mkdir(parents=True,exist_ok=True)


class ReconstructionTests(unittest.TestCase):
    def test_anatomical_attachments_ignore_image_pose(self):
        p=np.array([[-1,0,0],[1,0,0],[0,1,0]],float)
        camera=r.Camera(32,16,16,np.array([0,0,5.]),np.eye(3))
        for xy in ([0,0],[100,100]):
            rows=fitting.anchor_indices(p,camera,{'eye_right':dict(xy=xy)}, {'eye_right':[0,2]})
            np.testing.assert_array_equal(rows[0][1],[0,2])
        override=fitting.anchor_indices(p,camera,{'eye_right':dict(xy=[0,0],vertex=1)}, {'eye_right':[0,2]})
        np.testing.assert_array_equal(override[0][1],[1])

    def test_surface_completion_does_not_cross_normals_or_distance(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('completion needs scipy')
        from .reconstruction.materials import complete_surface
        p=np.array([[0,0,0],[.01,0,0],[.002,0,0],[1,0,0],[.001,0,0]])
        n=np.tile([0,0,1.],(5,1));n[-1]*=-1
        color=np.array([[.8,.4,.2],[.4,.2,.1],[0,0,0],[0,0,0],[0,0,0]])
        out=complete_surface(p,n,color,np.array([1,1,0,0,0],bool))
        np.testing.assert_allclose(out[:2],color[:2])
        np.testing.assert_allclose(out[3:],np.tile([.6,.3,.15],(2,1)))
        self.assertGreater(out[2,0],.6)

    def test_cluster_uv_continuity_and_seams(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('UV chart transfer needs scipy')
        from .rig.source_lod import cluster_uvs
        tri=np.array([[0,1,2],[1,3,2],[0,4,5]])
        uv=np.array([[[0,0],[.4,0],[0,.4]],[[.4,0],[.4,.4],[0,.4]],[[.8,.8],[1,.8],[.8,1]]])
        group=np.array([0,0,1,2,3,4])
        result=cluster_uvs(tri,uv,group,np.arange(3))
        np.testing.assert_allclose(result[0,0],result[0,1])
        np.testing.assert_allclose(result[0,1],result[1,0])
        np.testing.assert_allclose(result[2],uv[2])

    def test_concave_outline_metric_and_temporal_error(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('outline metric needs scipy')
        from .reconstruction.evaluate import outline_metrics,temporal_error
        mask=np.zeros((32,32),bool);mask[4:28,4:28]=1;mask[4:20,12:20]=0
        perfect=outline_metrics(mask,mask,np.ones_like(mask))
        convex=mask.copy();convex[4:20,12:20]=1
        worse=outline_metrics(convex,mask,np.ones_like(mask))
        self.assertEqual(perfect['iou'],1)
        self.assertEqual(perfect['boundary_f1'],1)
        self.assertLess(worse['boundary_f1'],1)
        truth=np.arange(4)[:,None,None]*np.ones((4,2,3))*.001
        zero=temporal_error(truth,truth,np.arange(4)*.1)
        self.assertEqual(zero['acceleration_error_rms'],0)
        predicted=truth.copy();predicted[2]+=.01
        self.assertGreater(temporal_error(predicted,truth,np.arange(4)*.1)['acceleration_error_rms'],0)
        with self.assertRaises(ValueError):temporal_error(truth,truth,[0,0,1,2])

    def test_mask_silhouette_occlusion(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('mask fitting needs scipy')
        from .reconstruction.silhouette import prepare,residual
        from PIL import Image
        p=np.array([[-.3,-.3,0],[.3,-.3,0],[.3,.3,0],[-.3,.3,0]])
        tri=np.array([[0,1,2],[0,2,3]])
        camera=r.Camera(64,32,32,np.array([0,0,1.]),np.eye(3))
        tid,_,_=r.rasterize(p,tri,camera,(64,64))
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d)/'mask.png';Image.fromarray(np.uint8(tid>=0)*255).save(path)
            row=prepare(p,tri,camera,dict(size=[64,64],silhouette_mask_path=str(path)))
            self.assertLess(np.mean(abs(residual(p,camera,row))),np.mean(abs(residual(p+[.04,0,0],camera,row))))

    def test_evaluation_heldout_and_leakage(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('evaluator needs scipy')
        from unittest.mock import patch
        from types import SimpleNamespace
        from PIL import Image
        from .reconstruction.evaluate import evaluate
        p=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])
        tri=np.array([[0,1,2],[0,2,3]])
        camera=r.Camera(64,32,32,np.array([0,0,1.]),np.eye(3))
        xy,_=camera.project(p)
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root=Path(d);candidate=root/'candidate';candidate.mkdir()
            Image.new('RGB',(64,64),(150,100,80)).save(root/'test.png')
            Image.new('RGB',(64,64),(140,100,80)).save(root/'train.png')
            Image.new('RGB',(32,32),(150,100,80)).save(candidate/'skin_basecolor.png')
            view=dict(image='test.png',sha256=observations.sha256(root/'test.png'),size=[64,64],camera=camera.as_dict(),
                      anchors={str(i):dict(vertex=i,xy=xy[i].tolist()) for i in range(4)})
            doc=dict(format=observations.FORMAT,views=[view]);obs=root/'obs.json';obs.write_text(json.dumps(doc))
            (candidate/'manifest.json').write_text(json.dumps(dict(id='test',face_model='fixture',geometry={})))
            (candidate/'observations.json').write_text(json.dumps(dict(views=[dict(sha256=observations.sha256(root/'train.png'))])))
            np.savez(candidate/'geometry.npz',neutral=p,captured=p[None],triangles=tri,triangle_uvs=np.zeros((2,3,2)))
            reference=root/'truth.npz';np.savez(reference,positions=p[None],triangles=tri)
            with patch('server.vhuman.rig.face_models.load',return_value=SimpleNamespace(name='fixture')):
                result=evaluate(candidate,obs,root/'report',reference=reference)
                self.assertEqual(result['views'][0]['split'],'held-out')
                self.assertAlmostEqual(result['views'][0]['landmarks']['rms_px'],0)
                self.assertEqual(result['geometry']['rms_mm'],0)
                (candidate/'observations.json').write_text(json.dumps(dict(views=[view])))
                with self.assertRaisesRegex(ValueError,'overlaps training'):
                    evaluate(candidate,obs,root/'leaked')
                # The worker rewrites RGB input as RGBA PNG; file hashes differ
                # while decoded pixels match, so this must still count as reuse.
                Image.open(root/'test.png').convert('RGBA').save(candidate/'copied.png')
                copied=dict(view,image='copied.png',sha256=observations.sha256(candidate/'copied.png'))
                self.assertNotEqual(copied['sha256'],view['sha256'])
                (candidate/'observations.json').write_text(json.dumps(dict(views=[copied])))
                with self.assertRaisesRegex(ValueError,'overlaps training'):
                    evaluate(candidate,obs,root/'leaked_reencoded')

    def test_profile_pose_and_disjoint_expression_score(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest('pose fitting needs scipy')
        from types import SimpleNamespace
        from .reconstruction.evaluate import align_pose
        p=np.array([[-.03,.02,0],[.03,.02,0],[0,0,.025],[0,-.05,0],[-.02,-.025,.012],[.02,-.025,.012]])
        tri=np.array([[0,2,1],[0,4,2],[1,2,5],[2,4,3],[2,3,5]])
        camera=r.Camera(1000,256,256,np.array([0,0,1.]),np.eye(3))
        rotation=Rotation.from_rotvec([.03,.43,.05]).as_matrix()
        xy,_=camera.project(p@rotation.T+np.array([.003,0,0]))
        names=['eye_right','eye_left','nose_tip','menton','mouth_right','mouth_left']
        anchors={name:dict(vertex=i,xy=xy[i].tolist()) for i,name in enumerate(names)}
        view=dict(camera=camera.as_dict(),anchors=anchors)
        source=SimpleNamespace(triangles=tri,identity_basis=None,expression_basis=None,eye_centers=None)
        _,captured,cameras,report=fitting.fit(source,p,[view],iterations=80)
        self.assertGreater(abs(report['pose_rotations'][0][1]),.35)
        self.assertLess(np.linalg.norm(cameras[0].project(captured[0])[0]-xy),.2)
        aligned,diagnostic=align_pose(p,camera,view,{})
        self.assertEqual(diagnostic['status'],'pose alignment diagnostic')
        self.assertEqual(set(diagnostic['fitted_anchor_names']),set(names[:4]))
        self.assertEqual(diagnostic['expression_landmarks']['count'],2)
        self.assertLess(diagnostic['expression_landmarks']['rms_px'],.1)

    def test_projection_and_pixal_frame(self):
        from .rig.common import Frame
        from .head.camera import PixalCamera
        from PIL import Image
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/'portrait.png'
            Image.new('RGBA',(128,128),'white').save(p)
            frame=Frame(np.array([.01,.02,-.03]),4.)
            subject=SimpleNamespace(fit={'camera':{'fov_deg':20}},portrait=p,frame=frame)
            cam=r.pixal_camera(subject)
            pix=PixalCamera.from_portrait(p,np.deg2rad(20))
            glb=np.array([[.1,.1,-.2],[-.1,.3,.1]])
            xy,z=cam.project(frame.to_h(glb))
            np.testing.assert_allclose(xy,pix.project(glb),atol=1e-9)
            rays=cam.rays(xy)
            true=frame.to_h(glb)-cam.origin
            true/=np.linalg.norm(true,axis=1,keepdims=True)
            np.testing.assert_allclose(rays,true,atol=1e-9)

    def test_perspective_raster(self):
        camera=r.Camera(32,16,16,np.zeros(3),np.eye(3))
        p=np.array([[-.3,-.3,-1],[.3,-.3,-2],[0,.3,-1]])
        tid,bary,z=r.rasterize(p,np.array([[0,1,2]]),camera,(32,32))
        yy,xx=np.nonzero(tid>=0)
        points=(p*bary[yy,xx,:,None]).sum(1)
        projected,zz=camera.project(points)
        np.testing.assert_allclose(projected,np.stack((xx+.5,yy+.5),-1),atol=2e-6)
        np.testing.assert_allclose(zz,z[yy,xx],atol=1e-6)
        np.testing.assert_allclose(bary[yy,xx].sum(1),1,atol=1e-7)

    def test_color_and_brdf(self):
        values=np.linspace(0,1,100)
        np.testing.assert_allclose(r.linear_to_srgb(r.srgb_to_linear(values)),values,atol=1e-12)
        d,s=r.ggx([.5,.3,.2],[[0,0,1]],[[0,0,1]],[[0,0,1]])
        self.assertTrue(np.isfinite(d).all() and np.isfinite(s).all())
        self.assertTrue((d>=0).all() and (s>=0).all())
        d,s=r.ggx([.5,.3,.2],[[0,0,1]],[[0,0,1]],[[0,0,-1]])
        np.testing.assert_allclose(d+s,0)

    def test_torch_brdf_reference_and_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch reference needs rig interpreter')
        from .reconstruction.torch_reference import ggx
        rgb=torch.tensor([[.5,.3,.2]],dtype=torch.float64)
        n=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
        light=torch.tensor([[.3,.1,1.]],dtype=torch.float64)
        rough=torch.tensor(.55,dtype=torch.float64,requires_grad=True)
        f0=torch.tensor(.028,dtype=torch.float64)
        d,s=ggx(rgb,n,n,light,rough,f0)
        nd,ns=r.ggx(rgb.numpy(),n.numpy(),n.numpy(),light.numpy(),.55,.028)
        np.testing.assert_allclose(d.detach().numpy(),nd,atol=1e-12)
        np.testing.assert_allclose(s.detach().numpy(),ns,atol=1e-12)
        s.sum().backward()
        eps=1e-5
        _,lo=r.ggx(rgb.numpy(),n.numpy(),n.numpy(),light.numpy(),.55-eps,.028)
        _,hi=r.ggx(rgb.numpy(),n.numpy(),n.numpy(),light.numpy(),.55+eps,.028)
        self.assertAlmostEqual(rough.grad.item(),float((hi.sum()-lo.sum())/(2*eps)),places=6)

    def test_topology_guard(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        tri=np.array([[0,1,2]])
        self.assertTrue(fitting.safe_geometry(p,p,tri))
        self.assertFalse(fitting.safe_geometry(p,p[[0,2,1]],tri))
        self.assertFalse(fitting.safe_geometry(p,p*0,tri))

    def test_covariance_transport(self):
        p=np.array([[0,0,0],[.01,0,0],[0,.01,0]],float)
        tri=np.array([[0,1,2]])
        b=gaussian.bind(p,tri,count=10)
        c,cov,a=gaussian.deform(b,p,tri)
        c2,cov2,_=gaussian.deform(b,p*2,tri)
        np.testing.assert_allclose(c2,c*2)
        np.testing.assert_allclose(cov2[:,0,0],cov[:,0,0]*4)
        self.assertTrue((np.linalg.eigvalsh(cov)>0).all())
        _,_,alpha=gaussian.deform(b,p*0,tri)
        np.testing.assert_allclose(alpha,0)
        with self.assertRaises(ValueError):gaussian.deform(b,p,tri[:,::-1])
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d)/'binding.json';gaussian.save(b,path);gaussian.load(path)
            bad=json.loads(path.read_text());bad['opacity'][0]=2;path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):gaussian.load(path)

    def test_depth_alignment_gate(self):
        try:
            import scipy
        except ImportError:
            self.skipTest("depth gate needs rig interpreter scipy")
        metric=np.linspace(.3,.6,128).reshape(8,16)
        relative=(1/metric-.5)/2
        aligned,report=depth.align(relative,metric,np.ones_like(metric))
        np.testing.assert_allclose(aligned,metric,atol=1e-8)
        self.assertLess(report['median_error_m'],1e-7)
        with self.assertRaises(ValueError):depth.align(-relative,metric,np.ones_like(metric))
        with self.assertRaises(ValueError):depth.align(relative,metric,np.zeros_like(metric))
        with self.assertRaises(ValueError):depth.align(relative,np.ones_like(metric),np.ones_like(metric))

    def test_observation_hash_and_camera(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d);Image.new('RGB',(16,16)).save(p/'view.png')
            doc=dict(format=observations.FORMAT,views=[dict(image='view.png',size=[16,16],
                     sha256=observations.sha256(p/'view.png'),anchors={'nose_tip':dict(xy=[8,8],weight=1)})])
            path=p/'obs.json';path.write_text(json.dumps(doc));observations.load(path)
            doc['views'][0]['sha256']='bad';path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):observations.load(path)

    def test_candidate_allowlist(self):
        from .service import EyeService,ServiceError
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            service=EyeService(Path(d));root=service.work/'heads'/'abc'/'reconstruction'/'def'
            root.mkdir(parents=True);(root/'manifest.json').write_text('{}')
            self.assertEqual(service.reconstruction_file('abc','def','manifest.json'),root/'manifest.json')
            completion=root/'skin_completion_confidence.png'
            completion.write_bytes(b'completion map')
            self.assertEqual(service.reconstruction_file('abc','def',completion.name),completion)
            for name in ('../head.json','geometry.npz/../manifest.json','rig/../../manifest.json'):
                with self.assertRaises(ServiceError):service.reconstruction_file('abc','def',name)
            with self.assertRaises(ServiceError):service.reconstruction_file('abc','.partial','manifest.json')

    def test_reconstruction_boolean_options_before_subprocess(self):
        from unittest.mock import patch
        from .service import EyeService
        from .reconstruction.job import reconstruction_job
        import threading
        class StopBeforeLaunch(Exception):pass
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            service=EyeService(Path(d));head=service.work/'heads'/'abc'
            head.mkdir(parents=True);(head/'head.json').write_text('{}')
            base=dict(head_id='abc',build_rig=False)
            with patch('server.vhuman.reconstruction.job.subprocess.Popen',side_effect=StopBeforeLaunch) as launch, \
                 patch('server.vhuman.reconstruction.job.gpu.gpu_status',return_value=None):
                for option in ('spatial_materials','auto_exclusions'):
                    for value in ('false','true',0,1,None,[],{}):
                        with self.subTest(option=option,value=value):
                            with self.assertRaisesRegex(ValueError,option+' must be a boolean'):
                                reconstruction_job(service,dict(base,**{option:value}),lambda *a:None,
                                                   threading.Event(),python=Path(__file__),mock=True)
                    launch.assert_not_called()
                for options in ({},{'spatial_materials':False,'auto_exclusions':False},
                                {'spatial_materials':True,'auto_exclusions':True}):
                    with self.assertRaises(StopBeforeLaunch):
                        reconstruction_job(service,dict(base,**options),lambda *a:None,
                                           threading.Event(),python=Path(__file__),mock=True)
                    command=launch.call_args.args[0]
                    for option in ('spatial_materials','auto_exclusions'):
                        self.assertEqual('--'+option.replace('_','-') in command,options.get(option,False))

    def test_eye_frame_initialization(self):
        from types import SimpleNamespace
        eye=np.array([[-.03,.30,.10],[.03,.30,.10]])
        source=SimpleNamespace(eye_centers=eye,vertices=np.array([[0,.30,.12],[0,.25,.10]]))
        subject=SimpleNamespace(eyes=[{'center':np.array([-.032,0,0])},{'center':np.array([.032,0,0])}])
        p,scale,rot,_=fitting.initialize(source,subject)
        np.testing.assert_allclose(p[0],[0,0,.02*scale],atol=1e-12)
        np.testing.assert_allclose(scale, .064/.06)
        self.assertAlmostEqual(np.linalg.det(rot),1)

    def test_synthetic_pose_fit(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('fit needs rig interpreter scipy')
        from types import SimpleNamespace
        p=np.array([[-.03,-.03,.1],[.03,-.03,.1],[.03,.03,.1],[-.03,.03,.1]])
        tri=np.array([[0,1,2],[0,2,3]])
        camera=r.Camera(400,128,128,np.array([0,0,1.]),np.eye(3))
        xy,_=camera.project(p+np.array([.002,0,0]))
        source=SimpleNamespace(triangles=tri,identity_basis=None,expression_basis=None,eye_centers=None)
        anchors={str(i):dict(vertex=i,xy=xy[i].tolist(),weight=1) for i in range(4)}
        neutral,captured,cams,report=fitting.fit(source,p,[dict(camera=camera.as_dict(),anchors=anchors)],iterations=30)
        self.assertLess(report['objective_after'],report['objective_before'])
        np.testing.assert_allclose(neutral,p,atol=1e-8)
        self.assertTrue(fitting.safe_geometry(p,neutral,tri))
        before=np.linalg.norm(camera.project(p)[0]-xy)
        self.assertLess(np.linalg.norm(cams[0].project(captured[0])[0]-xy),before)

    def test_upload_and_local_path_rejection(self):
        import io
        from PIL import Image
        from .reconstruction.upload import upload,server_request
        from .service import EyeService,ServiceError
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            service=EyeService(Path(d));b=io.BytesIO();Image.new('RGB',(64,64)).save(b,format='PNG')
            data=b.getvalue();result=upload(service,io.BytesIO(data),len(data),'image/png')
            req=server_request(service,{'portrait_upload_id':result['upload_id']},direct=True)
            self.assertTrue(Path(req['portrait']).is_file())
            with self.assertRaises(ServiceError):server_request(service,{'portrait':'/etc/passwd'},direct=True)
            with self.assertRaises(ServiceError):upload(service,io.BytesIO(b'bad'),3,'image/png')

    def test_padding_does_not_wrap(self):
        from .rig.bake import dilate
        image=np.zeros((3,8,1));image[1,0]=1
        covered=np.zeros((3,8),bool);covered[1,0]=True
        result=dilate(image,covered,1)
        self.assertEqual(result[1,-1,0],0)
        self.assertEqual(result[1,1,0],1)

    def test_source_lod_boundaries_and_weights(self):
        from .rig.source_lod import simplify,transfer
        x,y=np.meshgrid(np.linspace(-.03,.03,12),np.linspace(-.03,.03,12))
        p=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        triangles=[]
        for row in range(11):
            for col in range(11):
                i=row*12+col;triangles.extend([[i,i+1,i+12],[i+1,i+13,i+12]])
        t=np.array(triangles)
        q,tt,group,keep,report=simplify(p,t,.012)
        self.assertLess(len(tt),len(t))
        boundary=np.unique(np.concatenate([np.arange(12),np.arange(132,144),np.arange(0,144,12),np.arange(11,144,12)]))
        np.testing.assert_allclose(q[group[boundary]],p[boundary],atol=1e-9)
        n=np.cross(q[tt[:,1]]-q[tt[:,0]],q[tt[:,2]]-q[tt[:,0]])
        self.assertTrue((n[:,2]>0).all())
        joints=np.tile([0,0,1,2],(len(p),1));weights=np.tile([.3,.2,.3,.2],(len(p),1))
        shapes,J,W=transfer(group,{'smile':p*.1},joints,weights)
        np.testing.assert_allclose(W.sum(1),1,atol=1e-7)
        self.assertTrue((J[:,0]==0).all())
        self.assertTrue((W[:,0]>.49).all())

    def test_candidate_rig_cancel_isolation(self):
        from unittest.mock import patch
        from .service import EyeService
        from .rig.job import rig_job
        from .gpu import Cancelled
        import threading
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            service=EyeService(Path(d));head=service.work/'heads'/'abc';head.mkdir(parents=True)
            for name in ('head.json','head_eyes.glb','fit.json','portrait.png'):(head/name).write_text('{}')
            candidate=head/'reconstruction'/'def';(candidate/'rig').mkdir(parents=True)
            (candidate/'manifest.json').write_text('{}');old=candidate/'rig'/'rig.glb';old.write_bytes(b'accepted candidate')
            class Process:
                stdout=[]
                def __init__(self,cmd,**kwargs):
                    out=Path(cmd[cmd.index('--out')+1]);out.mkdir(parents=True)
                    (out/'rig_report.json').write_text('{"seconds":1}')
                    (out/'rig.glb').write_bytes(b'partial replacement')
                def wait(self):
                    cancel.set()
                    return 0
                def poll(self):return 0
            cancel=threading.Event()
            with patch('server.vhuman.rig.job.subprocess.Popen',Process), \
                 patch('server.vhuman.rig.job.gpu.gpu_status', return_value=None):
                with self.assertRaises(Cancelled):
                    rig_job(service,dict(head_id='abc',reconstruction_run='def',res=1024,iters=50),lambda *a:None,cancel,
                            python=Path(__file__),mock=True)
            self.assertEqual(old.read_bytes(),b'accepted candidate')
            self.assertFalse(list(candidate.glob('*.partial')))

    def test_calibrated_reflectance_fit(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('reflectance fitting needs rig interpreter scipy')
        from .reconstruction.reflectance import fit,coefficients
        rng=np.random.default_rng(7);xy=rng.uniform(-.45,.45,(128,2))
        normal=np.column_stack((xy,np.sqrt(1-(xy*xy).sum(1))))
        normals=np.repeat(normal[None],3,axis=0)
        views=np.tile([0,0,1.],(3,128,1))
        lights=[dict(direction_h=d,radiance=[.6,.6,.6]) for d in ([.7,0,1],[-.3,.5,1],[0,-.8,1])]
        diffuse,specular=coefficients(normals,views,lights,.4,.02)
        known=rng.uniform(.15,.5,(128,3));rgb=known[None]*diffuse+specular
        recovered,rough,f0,report=fit(rgb,normals,views,np.ones((3,128)),lights)
        self.assertAlmostEqual(rough,.4,places=3);self.assertAlmostEqual(f0,.02,places=4)
        np.testing.assert_allclose(recovered,known,atol=1e-4)
        self.assertLess(report['heldout_after'],report['heldout_before'])
        with self.assertRaises(ValueError):fit(rgb[:1],normals[:1],views[:1],np.ones((1,128)),lights[:1])

    def test_depth_residual_bound_and_eye_protection(self):
        x,y=np.meshgrid(np.linspace(-.03,.03,5),np.linspace(-.03,.03,5))
        p=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.1)))
        t=[]
        for row in range(4):
            for col in range(4):
                i=row*5+col;t.extend([[i,i+1,i+5],[i+1,i+6,i+5]])
        t=np.array(t);cam=r.Camera(400,64,64,np.array([0,0,1.]),np.eye(3))
        eye=cam.project(p[:1])[0][0]
        q,report=depth.refine(p,t,cam,np.full((128,128),.91),np.ones((128,128)),[(eye,1)])
        self.assertLessEqual(report['max_displacement_m'],.000500001)
        np.testing.assert_allclose(q[0],p[0],atol=1e-8)
        self.assertTrue(fitting.safe_geometry(p,q,t))

    def test_gaussian_visible_radiance_fit(self):
        from PIL import Image
        p=np.array([[-.3,-.3,0],[.3,-.3,0],[0,.3,0]])
        t=np.array([[0,1,2]]);cam=r.Camera(64,32,32,np.array([0,0,1.]),np.eye(3))
        binding=gaussian.bind(p,t,count=32)
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d)/'view.png';Image.new('RGB',(64,64),(140,90,80)).save(path)
            gaussian.fit_radiance(binding,[p],t,[{'image_path':str(path)}],[cam])
            visible=binding['opacity']>0
            self.assertGreater(visible.sum(),16)
            expected=np.tile(r.srgb_to_linear(np.array([140,90,80])/255.),(visible.sum(),1))
            np.testing.assert_allclose(binding['rgb'][visible],expected,atol=1e-7)

    def test_transfer_merges_duplicate_joint_slots(self):
        from .rig.face_models import merged_influences
        j,w=merged_influences(np.array([[0,1,0,1,0,1]]),np.array([[.2,.13,.2,.13,.2,.14]]))
        self.assertAlmostEqual(float(w[0,j[0]==0].sum()),.6,places=6)
        self.assertAlmostEqual(float(w[0,j[0]==1].sum()),.4,places=6)
        np.testing.assert_allclose(w.sum(1),1,atol=1e-7)
        self.assertEqual(len(set(j[0,w[0]>0].tolist())),2)

    def test_material_estimate_gauge(self):
        from .reconstruction.materials import estimate
        n=np.tile([0,0,1.],(128,1));rgb=np.tile([150,100,80],(128,1))
        albedo,report=estimate(rgb,n,np.ones(128))
        self.assertTrue(np.isfinite(albedo).all())
        np.testing.assert_allclose(albedo,r.srgb_to_linear(rgb/255.),atol=1e-10)
        self.assertIn('median',report['gauge'])


    def test_material_lighting_receipt_replays_without_source_fit_mask(self):
        from .reconstruction.materials import estimate
        rng=np.random.default_rng(73)
        normals=rng.normal(size=(256,3))
        normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        illumination=np.exp(normals@np.array([.2,.4,.1]))
        rgb=r.linear_to_srgb(np.clip(illumination[:,None]*[.3,.2,.12],0,1))*255
        albedo,report=estimate(rgb,normals,np.ones(256))
        replay=np.exp(np.clip(normals@report['log_direction'],*report['log_irradiance_clip']))
        replay=np.clip(replay/report['irradiance_median'],*report['normalized_irradiance_clip'])
        np.testing.assert_allclose(albedo,np.clip(r.srgb_to_linear(rgb/255)/replay[:,None],0,1),atol=1e-14)
        self.assertGreater(report['irradiance_median'],0)


if __name__=='__main__':unittest.main()
