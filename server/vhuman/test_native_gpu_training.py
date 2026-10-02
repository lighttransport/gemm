"""Bounded native CUDA math checks; opt in with python -m ... --gpu -v."""
import ctypes as C
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from .native_training import ROOT, pointer
from .native_gpu_training import GpuTraining, library, check
from .test_native_image_training import cue_fixture, appearance_fixture
from .reconstruction.native_cue_training import CueNet
from .realtime.src.avatar.native_training import AppearanceTrainer

RUN_GPU='--gpu' in sys.argv


@unittest.skipUnless((ROOT/'cuda/vhuman/libvhuman_training_cuda.so').is_file(),'native CUDA training build required')
class CompileTests(unittest.TestCase):
    def test_nvrtc_without_device(self):check(library().vht_compile_probe())


@unittest.skipUnless(RUN_GPU,'explicit --gpu required')
class NativeGpuTrainingTests(unittest.TestCase):
    def test_gemm_all_transposes_with_tail_tiles(self):
        t=GpuTraining(np.zeros(1,np.float32),0,.01,0)
        try:
            rng=np.random.default_rng(8);m,n,k=19,23,37
            for ta in (False,True):
                for tb in (False,True):
                    a=rng.normal(size=(k,m) if ta else (m,k)).astype(np.float32)
                    b=rng.normal(size=(n,k) if tb else (k,n)).astype(np.float32);out=np.empty((m,n),np.float32)
                    check(t.lib.vht_gemm(t.handle,pointer(out),pointer(a),pointer(b),m,n,k,int(ta),int(tb)))
                    np.testing.assert_allclose(out,(a.T if ta else a)@(b.T if tb else b),atol=5e-6,rtol=5e-6)
        finally:t.close()

    def test_cue_forward_gradients_and_resident_adam(self):
        cpu,rgb,prior,truth,mask=cue_fixture();gpu=CueNet(9,device='cuda',resident=True)
        gpu.parameters[:]=cpu.parameters
        try:
            for _ in range(2):
                if _==1:
                    cpu.optimizer.lr=gpu.optimizer.lr=.001
                    cpu.optimizer.decay=gpu.optimizer.decay=.01
                a=cpu.compute(rgb,prior,truth,mask,update=True);b=gpu.compute(rgb,prior,truth,mask,update=True)
                for x,y,tol in zip(a,b,(2e-6,2e-6,2e-6,2e-6)):
                    np.testing.assert_allclose(x,y,atol=tol,rtol=2e-5)
                gpu.sync_parameters();np.testing.assert_allclose(cpu.parameters,gpu.parameters,atol=2e-6,rtol=2e-5)
            # Changing shape and zero supervision mask must retain valid backward.
            mask[:]=0;gpu.compute(rgb[:,:,:7,:8],prior[:,:,:7,:8],truth[:,:,:7,:8],mask[:,:7,:8])
            self.assertLess(gpu._gpu.peak_bytes,2*1048576)
        finally:gpu.close()

    def test_gaussian_branches_all_groups_and_resident_adam(self):
        cpu,avatar,v,tri,ctrl,view,k,target,mask=appearance_fixture()
        gpu=AppearanceTrainer(avatar,tri,device='cuda',resident=True);gpu.parameters[:]=cpu.parameters
        try:
            for _ in range(2):
                a=cpu.compute(v,ctrl,view,k,(15,13),target,mask,update=True)
                b=gpu.compute(v,ctrl,view,k,(15,13),target,mask,update=True)
                for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-7,rtol=2e-5)
                gpu.sync_parameters();np.testing.assert_allclose(cpu.parameters,gpu.parameters,atol=2e-6,rtol=2e-5)
            # Independent CPU path checks saturation and invisible/degenerate mesh.
            for case in ('cap','behind','degenerate','invisible','fov'):
                vertices=v.copy();cpu.local[:,3]=1;cpu.local[:,8:]=.02
                if case=='cap':cpu.local[:,3]=10;cpu.local[:,:3]=10;cpu.local[:,8:]=1
                if case=='behind':vertices[:,2]=-1
                if case=='degenerate':vertices[:]=vertices[0]
                if case=='invisible':cpu.local[:,3]=-20
                if case=='fov':vertices[:,0]+=.2
                gpu.parameters[:]=cpu.parameters;gpu.upload_parameters()
                a=cpu.compute(vertices,ctrl,view,k,(15,13),target,mask)
                b=gpu.compute(vertices,ctrl,view,k,(15,13),target,mask)
                for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-7,rtol=2e-5)
        finally:gpu.close()

    def test_memory_budget_and_invalid_supervision(self):
        cpu,rgb,prior,truth,mask=cue_fixture();gpu=CueNet(9,device='cuda',memory_mb=1)
        try:
            gpu.compute(rgb,prior,truth,mask)
            before=gpu.parameters.copy();mask[0,0,0]=2
            with self.assertRaisesRegex(RuntimeError,'invalid cue mask'):gpu.compute(rgb,prior,truth,mask,update=True)
            np.testing.assert_array_equal(before,gpu.parameters)
            self.assertEqual(gpu.optimizer.iteration,0)
            large=np.zeros((1,3,64,64),np.float32)
            with self.assertRaisesRegex(RuntimeError,'memory budget exceeded'):gpu.compute(large,large)
        finally:gpu.close()

    def test_complete_training_and_export_without_frameworks(self):
        script='''
import importlib.abc,sys,json,numpy as np
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorflow','gsplat','diffusers','mediapipe'}:
            raise AssertionError('unexpected model runtime: '+fullname)
sys.meta_path.insert(0,Block())
from server.vhuman.reconstruction.learned_cues import train
from server.vhuman.reconstruction.observations import sha256
from server.vhuman.realtime.src.avatar.train import fit
from server.vhuman.realtime.src.avatar.bundle import GaussianAvatar
from server.vhuman.test_native_image_training import appearance_fixture
work=Path(sys.argv[1]);rgb=np.random.default_rng(3).random((8,8,8,3),dtype=np.float32)
normal=np.zeros_like(rgb);normal[...,2]=1
dataset=work/'synthetic.npz'
np.savez(dataset,rgb=rgb,normals=normal,prior=normal,mask=np.ones((8,8,8),bool),identity=np.repeat(np.arange(4),2))
dataset.with_suffix('.json').write_text(json.dumps(dict(format='vhuman.synthetic_cues.v1',real_media_used=False,dataset_sha256=sha256(dataset))))
report=train(dataset,work/'cue',50,device='cuda',memory_mb=32)
assert report['training_backend']=='repository_cuda_gemm' and not report['default_enabled']
assert (work/'cue/normal_cue.safetensors').is_file()
model,avatar,v,t,ctrl,view,k,target,mask=appearance_fixture()
np.savez(work/'frames.npz',vertices=v[None],triangles=t,controls=ctrl[None],view=view[None],intrinsics=k[None],images=target[None],masks=mask[None])
receipt=dict(path='frames.npz',source='original synthetic GPU integration fixture',revision='1',license='Apache-2.0',roles=['appearance-training'],sha256=sha256(work/'frames.npz'))
manifest=work/'manifest.json';manifest.write_text(json.dumps(dict(format='vhuman.appearance_corpus.v1',purpose='diagnostic',data='frames.npz',control_names=avatar.metadata['control_names'],provenance=[receipt])))
report=fit(manifest,work/'avatar.npz',count=16,steps=2,device='cuda',memory_mb=32)
assert report['backend']=='repository_cuda_gemm'
checkpoint=GaussianAvatar.load(work/'avatar.npz',t)
assert checkpoint.metadata['training_steps']==2 and checkpoint.metadata['cuda_peak_bytes']>0
'''
        with tempfile.TemporaryDirectory(prefix='native-gpu-training-',dir=ROOT/'tmp') as work:
            result=subprocess.run([sys.executable,'-c',script,work],cwd=ROOT,text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':
    if RUN_GPU:sys.argv.remove('--gpu')
    unittest.main()
