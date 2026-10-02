"""Native CNN and Gaussian training math; no model framework or GPU required."""
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from .native_training import ROOT
from .reconstruction.native_cue_training import CueNet
from .realtime.src.avatar.bundle import bind,GaussianAvatar
from .realtime.src.avatar.native_training import AppearanceTrainer,initialize_rgb


def cue_fixture():
    rng=np.random.default_rng(32);rgb=rng.uniform(0,1,(2,3,9,11)).astype(np.float32)
    prior=rng.normal(size=rgb.shape).astype(np.float32);prior/=np.linalg.norm(prior,axis=1,keepdims=True)
    truth=rng.normal(size=rgb.shape).astype(np.float32);truth/=np.linalg.norm(truth,axis=1,keepdims=True)
    mask=(rng.random((2,9,11))>.25).astype(np.float32)
    model=CueNet(9,seed=17)
    return model,rgb,prior,truth,mask


def appearance_fixture():
    vertices=np.array([[-.08,-.07,.95],[.09,-.06,1.02],[.01,.1,.98]],np.float32);triangles=np.array([[0,1,2]],np.int32)
    avatar=bind(vertices,triangles,['jawOpen','mouthSmileLeft'],count=7,seed=21)
    model=AppearanceTrainer(avatar,triangles,seed=31)
    rng=np.random.default_rng(17);model.local[:,8:]=rng.normal(0,.02,(7,24))
    model.local[:,7]=rng.normal(0,.1,7);model.local[:,4:6]+=rng.normal(0,.15,(7,2))
    controls=np.array([.4,.6],np.float32);view=np.eye(4,dtype=np.float32)
    intrinsics=np.array([[65,0,7],[0,63,6],[0,0,1]],np.float32)
    target=rng.uniform(.025,.16,(13,15,3)).astype(np.float32);mask=np.full((13,15),.2,np.float32)
    return model,avatar,vertices,triangles,controls,view,intrinsics,target,mask


@unittest.skipUnless((ROOT/'cpu/vhuman/libvhuman_training.so').is_file(),'native training build required')
class NativeImageTrainingTests(unittest.TestCase):
    def test_cnn_reverse_pass_all_ten_parameter_blocks(self):
        model,rgb,prior,truth,mask=cue_fixture();_,_,loss,g=model.compute(rgb,prior,truth,mask)
        self.assertTrue(np.isfinite(loss));offset=0
        for shape in model.shapes.values():
            count=int(np.prod(shape));best=offset+int(np.argmax(abs(g[offset:offset+count])));old=model.parameters[best];eps=.003
            model.parameters[best]=old+eps;plus=model.compute(rgb,prior,truth,mask)[2]
            model.parameters[best]=old-eps;minus=model.compute(rgb,prior,truth,mask)[2];model.parameters[best]=old
            self.assertAlmostEqual(float(g[best]),(plus-minus)/(2*eps),delta=1e-5)
            offset+=count
        before=model.parameters.copy();model.compute(rgb,prior,truth,mask,update=True)
        self.assertFalse(np.array_equal(before,model.parameters))

    def test_gaussian_reverse_pass_all_six_groups(self):
        model,avatar,v,t,ctrl,view,k,target,mask=appearance_fixture()
        rgba,loss,g=model.compute(v,ctrl,view,k,(15,13),target,mask)
        self.assertGreater(float(rgba[...,3].max()),0);self.assertTrue(np.isfinite(loss).all())
        # Six optimized groups: RGB, opacity, covariance, offset, color basis and
        # global expression matrix. Check each group's strongest derivative.
        local=np.arange(model.n*32).reshape(model.n,32)
        blocks=[local[:,:3].ravel(),local[:,3],local[:,4:7].ravel(),local[:,7],local[:,8:].ravel(),np.arange(model.n*32,len(g))]
        for block in blocks:
            best=block[np.argmax(abs(g[block]))];old=model.parameters[best];eps=.002
            model.parameters[best]=old+eps;plus=model.compute(v,ctrl,view,k,(15,13),target,mask)[1][0]
            model.parameters[best]=old-eps;minus=model.compute(v,ctrl,view,k,(15,13),target,mask)[1][0];model.parameters[best]=old
            self.assertAlmostEqual(float(g[best]),(plus-minus)/(2*eps),delta=2e-6)
        before=model.parameters.copy();model.compute(v,ctrl,view,k,(15,13),target,mask,update=True)
        self.assertTrue(all(np.any(before[block]!=model.parameters[block]) for block in blocks))
        model.export(avatar);avatar.metadata['covariance_policy']='trace-v1';avatar.validate(t)

    def test_bilinear_initializer_zero_padding(self):
        image=np.arange(27,dtype=np.float32).reshape(3,3,3)/30
        points=np.array([[1.5,1.5,1],[-.5,1,1],[30,1,1]],np.float32)
        actual=initialize_rgb(points,image,np.eye(4,dtype=np.float32),np.eye(3,dtype=np.float32))
        np.testing.assert_allclose(actual[0],image[1:3,1:3].mean((0,1)),atol=1e-7)
        np.testing.assert_allclose(actual[1],image[1,0]*.5,atol=1e-7)
        np.testing.assert_array_equal(actual[2],np.full(3,.01,np.float32))

    def test_complete_cue_and_appearance_training_without_frameworks(self):
        script='''
import importlib.abc,sys,json,numpy as np
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorflow','gsplat','diffusers','mediapipe'}:
            raise AssertionError('unexpected training framework: '+fullname)
sys.meta_path.insert(0,Block())
from server.vhuman.reconstruction.learned_cues import train
from server.vhuman.reconstruction.observations import sha256
from server.vhuman.rig import safetensors
from server.vhuman.realtime.src.avatar.train import fit
from server.vhuman.realtime.src.avatar.bundle import GaussianAvatar
root=Path(sys.argv[1]);rng=np.random.default_rng(21);side=8
rgb=rng.random((8,side,side,3),dtype=np.float32)
normal=np.tile([0,0,1],(8,side,side,1)).astype(np.float32);mask=np.ones((8,side,side),bool)
dataset=root/'cues.npz';np.savez(dataset,rgb=rgb,normals=normal,prior=normal,mask=mask,identity=np.repeat(np.arange(4),2))
dataset.with_suffix('.json').write_text(json.dumps(dict(format='vhuman.synthetic_cues.v1',real_media_used=False,dataset_sha256=sha256(dataset))))
report=train(dataset,root/'cue-output',steps=50)
assert report['training_backend']=='repository_cpu_gemm' and not report['real_geometry_gate_passed'] and not report['default_enabled']
weights,_=safetensors.load(root/'cue-output/normal_cue.safetensors')
assert all(np.isfinite(value).all() for value in weights.values())
path=root/'appearance.npz'
vertices=np.array([[[-.1,-.1,1],[.1,-.1,1],[0,.1,1]]],np.float32)
np.savez(path,vertices=vertices,triangles=np.array([[0,1,2]],np.int32),images=np.zeros((1,16,16,3),np.float32),controls=np.zeros((1,1),np.float32),
    view=np.eye(4,dtype=np.float32)[None],intrinsics=np.array([[[50,0,8],[0,50,8],[0,0,1]]],np.float32))
manifest=root/'appearance.json'
manifest.write_text(json.dumps(dict(format='vhuman.appearance_corpus.v1',purpose='diagnostic',data=path.name,control_names=['jawOpen'],
    provenance=[dict(path=path.name,source='original synthetic fixture',revision='v1',license='CC0-1.0',sha256=sha256(path),roles=['appearance-training'])])))
report=fit(manifest,root/'avatar.npz',count=16,steps=3)
assert report['backend']=='repository_cpu_gemm' and np.isfinite(report['loss_l1'])
avatar=GaussianAvatar.load(root/'avatar.npz');assert avatar.metadata['trained'] and avatar.metadata['covariance_policy']=='trace-v1'
assert avatar.metadata['training_device']=='cpu' and avatar.arrays['rgb'].mean()<.02
'''
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            result=subprocess.run([sys.executable,'-c',script,directory],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
