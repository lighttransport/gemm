"""Framework-free corrective training and independent analytic math checks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from .native_training import ROOT, matmul
from .rig import native_corrective as native


def fixture():
    from .rig import template as t, rigdef
    rng=np.random.default_rng(19);n=t.MOUTH_N*4+10
    rest=rng.normal(size=(n,3)).astype(np.float32)*.025
    triangles=np.column_stack((np.zeros(n-2,int),np.arange(1,n-1),np.arange(2,n))).astype(np.int32)
    tmpl=SimpleNamespace(n=n,group=np.arange(n)%3,ring=np.zeros(n,int),tris=triangles,
        ring_ids=lambda name,k:np.arange(t.MOUTH_N)+(1-k)*t.MOUTH_N)
    feat=SimpleNamespace(eyes=[dict(side=side,center=[x,.06,.04],radius=.012) for side,x in [('right',-.032),('left',.032)]])
    names=('head','jaw','eye_R','eye_L','teeth_upper','teeth_lower')
    skel={'joints':[dict(name=name,parent=None if i==0 else 'head',rest_translation=[0,0,0],
                        rest_rotation=np.eye(3).tolist(),bind=np.eye(4).tolist()) for i,name in enumerate(names)]}
    teeth=[(SimpleNamespace(positions=rng.normal(size=(362,3))*.01),joint) for joint in ('teeth_upper','teeth_lower')]
    definition=dict(controls=rigdef.control_table(),correctives=[dict(name='jawSmile',inputs=['jawOpen','mouthSmileLeft'],weight=.7)],
        joints=skel['joints'],joint_matrix=[dict(joint='jaw',input='jawOpen',attr='rx',value=.2),
        dict(joint='jaw',input='jawOpen',attr='ty',value=-.002)],blendshapes=[dict(name='jawShape',input='jawOpen')])
    shape=np.zeros_like(rest)
    for ring in (1,0,-1,-2):
        ids=tmpl.ring_ids('mouth',ring);shape[ids[1:t.MOUTH_HALF],1]=-.04;shape[ids[t.MOUTH_HALF+1:],1]=.04
    joints=np.tile([0,1],(n,1)).astype(np.int32);weights=np.tile([.6,.4],(n,1)).astype(np.float32)
    rig=native.NativeRig(definition,rest,{'jawShape':shape},joints,weights)
    from .rig.mldeformer_training import Contacts
    row=np.arange(28)[:,None];angle=np.arange(20)[None]*2*np.pi/20
    tongue_positions=np.stack((np.broadcast_to(.006*np.cos(angle),(28,20)),
        np.broadcast_to(-.015+.002*np.sin(angle),(28,20)),np.broadcast_to(.0005*row,(28,20))),axis=-1).reshape(-1,3)
    tongue=SimpleNamespace(positions=tongue_positions,joints=np.tile([0,1,0,0],(560,1)),
                           weights=np.tile([.6,.4,0,0],(560,1)))
    contacts=Contacts(tmpl,feat,skel,teeth,'cpu',rest,tongue=tongue)
    return tmpl,rig,contacts,feat,skel,teeth,{'jawShape':shape},tongue


@unittest.skipUnless((ROOT/'cpu/vhuman/libvhuman_training.so').is_file(),'native training build required')
class CorrectiveMathTests(unittest.TestCase):
    def test_mlp_all_parameter_blocks(self):
        rng=np.random.default_rng(31);net=native.MLP2(5,7,3,seed=9)
        x=rng.normal(size=(6,5)).astype(np.float32);upstream=rng.normal(size=(6,3)).astype(np.float32)
        weights=net.state_dict();expected=np.maximum(x@weights['fc1.weight'].T+weights['fc1.bias'],0)@weights['fc2.weight'].T+weights['fc2.bias']
        np.testing.assert_allclose(net(x),expected,atol=2e-7,rtol=2e-6)
        _,gradient=net.compute(x,upstream);offset=0
        for shape in net.shapes.values():
            count=int(np.prod(shape));best=offset+np.argmax(abs(gradient[offset:offset+count]));old=net.parameters[best];eps=.002
            net.parameters[best]=old+eps;plus=float(np.sum(net(x)*upstream,dtype=np.float64))
            net.parameters[best]=old-eps;minus=float(np.sum(net(x)*upstream,dtype=np.float64));net.parameters[best]=old
            self.assertAlmostEqual(float(gradient[best]),(plus-minus)/(2*eps),delta=1e-4)
            offset+=count

    def test_contact_and_arap_gradients(self):
        x=np.array([[[.3,.2,.1],[-.2,-.4,.3],[.7,.3,-.2]]],np.float32)
        ids=np.array([0,1,0],np.int32);centers=np.array([[[0,0,0],[.1,-.1,.1]]],np.float32)
        def sphere(v):return native.spheres(v,ids,centers,np.full((3,2),.8,np.float32))
        def pair(v):return native.pairs(v,np.array([0,0],np.int32),np.array([1,2],np.int32),np.array([[0,1,0]],np.float32),np.array([1,1],np.float32))
        linear=x*.95;edges=np.array([[0,1],[1,2]],np.int32);target=np.zeros((1,2,3),np.float32)
        def arap(v):return native.arap(v,linear,edges,target,.4)
        for function in (sphere,pair,arap):
            _,gradient,*_=function(x)
            for i in range(x.size):
                plus=x.copy();minus=x.copy();plus.ravel()[i]+=.001;minus.ravel()[i]-=.001
                numeric=float((function(plus)[0].sum()-function(minus)[0].sum())/.002)
                self.assertAlmostEqual(float(gradient.ravel()[i]),numeric,delta=.001)
        with self.assertRaisesRegex(ValueError,'index'):
            native.spheres(x,np.array([3]),centers,np.ones((1,2)))

    def test_rotation_recovers_rigid_transform_and_rest(self):
        from .rig.mldeformer_training import Solver
        rest=np.array([[0,0,0],[.02,0,0],[0,.02,0],[0,0,.02]],np.float32)
        tmpl=SimpleNamespace(tris=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32))
        rig=SimpleNamespace(rest=rest);solver=Solver(rig,tmpl,None,iters=2)
        rotation=native.euler_zyx(np.array([.2,-.3,.4],np.float32))
        positions=(rig.rest*1000)@rotation.T+np.array([1,2,3],np.float32)
        actual=solver.rotations(positions[None])
        np.testing.assert_allclose(actual[0],np.broadcast_to(rotation,actual.shape[1:]),atol=3e-5,rtol=3e-5)

    def test_posed_lip_objective_gradient(self):
        from .rig.mldeformer_training import objective
        rng=np.random.default_rng(18);n,k,p=4,3,5
        pred=rng.normal(size=(n,k)).astype(np.float32);target=rng.normal(size=(n,k)).astype(np.float32)
        cs=np.array([.4,1.4,2],np.float32);sw=np.array([[1],[5],[1],[5]],np.float32)
        lips=(rng.normal(size=(k,p,3)).astype(np.float32),rng.normal(size=(k,p,3)).astype(np.float32),
            rng.normal(size=(p,3)).astype(np.float32),rng.normal(size=(p,3)).astype(np.float32),
            rng.normal(size=(n,p,3)).astype(np.float32),rng.normal(size=(n,p,3)).astype(np.float32),
            rng.normal(size=(n,p)).astype(np.float32),np.full(p,2,np.float32),np.array([.1,.2,.3],np.float32))
        _,gradient=objective(pred,target,cs,sw,np.arange(n),lips,2)
        for i in range(pred.size):
            plus=pred.copy();minus=pred.copy();plus.ravel()[i]+=.002;minus.ravel()[i]-=.002
            numeric=(objective(plus,target,cs,sw,np.arange(n),lips,2)[0]-objective(minus,target,cs,sw,np.arange(n),lips,2)[0])/.004
            self.assertAlmostEqual(float(gradient.ravel()[i]),numeric,delta=.002)

    def test_complete_training_cache_and_load_without_frameworks(self):
        script='''
import importlib.abc,sys,json,numpy as np
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorflow','mediapipe','gsplat','diffusers'}:
            raise AssertionError('unexpected training framework: '+fullname)
sys.meta_path.insert(0,Block())
from server.vhuman.test_native_corrective import fixture
from server.vhuman.rig.mldeformer_training import train,Solver,MLDeformer
from server.vhuman.rig import safetensors as st
tmpl,rig,contacts,*_=fixture();root=Path(sys.argv[1]);cache=root/'solve.npz'
controls=np.zeros((1,len(rig.controls)),np.float32);controls[0,rig.controls.index('jawOpen')]=.9
solved=Solver(rig,tmpl,contacts,iters=3).solve(controls)
assert solved['before']['lips'].sum()>0
assert np.isfinite(solved['residual_mm']).all()
settings=dict(samples=12,k=3,k_mouth=3,hidden=8,batch=4,iters=2,epochs=3,seed=19,log=None,solve_cache=cache)
report=train(rig,tmpl,contacts,root/'first',**settings)
assert report['backend']=='repository_cpu_gemm' and report['training_device']=='cpu'
repeat=train(rig,tmpl,contacts,root/'second',**settings)
for filename in ('deformer.lrm','deformer_basis.safetensors'):
    a,_=st.load(root/'first'/filename);b,_=st.load(root/'second'/filename)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
model=MLDeformer(root/'first');inp=rig.input_vector(controls)[0]
offset=model.offsets(inp)
assert offset.shape==rig.rest.shape and np.isfinite(offset).all()
rig.rest[0,0]+=.001
try:train(rig,tmpl,contacts,root/'third',**settings)
except ValueError as error:assert 'cache rig/settings' in str(error)
else:raise AssertionError('stale solve cache accepted')
'''
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            result=subprocess.run([sys.executable,'-c',script,directory],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
