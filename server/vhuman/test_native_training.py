"""CPU algebra checks for native optimizer/PCA; no model framework needed."""
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from .native_training import AdamW, matmul, randomized_basis, set_threads, ROOT


@unittest.skipUnless((ROOT/'cpu/vhuman/libvhuman_training.so').is_file(),'native training build required')
class NativeTrainingMathTests(unittest.TestCase):
    def test_large_transposed_gemm_and_adamw_moments(self):
        rng = np.random.default_rng(9)
        set_threads(4)
        a,b = rng.normal(size=(41,256)).astype(np.float32),rng.normal(size=(41,133)).astype(np.float32)
        np.testing.assert_allclose(matmul(a,b,transpose_a=True),a.T@b,rtol=2e-5,atol=3e-6)
        parameters = rng.normal(size=17).astype(np.float32)
        expected = parameters.astype(np.float64).copy()
        first,second = np.zeros(17),np.zeros(17)
        optimizer = AdamW(parameters,lr=.003,weight_decay=.01,clip=1)
        for step in range(1,6):
            gradient = rng.normal(size=17).astype(np.float32)
            g = gradient*min(1.,1/(np.linalg.norm(gradient.astype(np.float64))+1e-6))
            first=.9*first+.1*g;second=.999*second+.001*g*g
            expected=expected*(1-.003*.01)-.003*first/(1-.9**step)/(np.sqrt(second/(1-.999**step))+1e-8)
            optimizer.step(gradient)
        np.testing.assert_allclose(parameters,expected,atol=3e-7,rtol=2e-6)

    def test_bounded_pca_reconstructs_low_rank_spatial_residual(self):
        rng = np.random.default_rng(7)
        matrix = (rng.normal(size=(70,5))@rng.normal(size=(5,2300))).astype(np.float32)
        basis = randomized_basis(matrix,5)
        np.testing.assert_allclose(basis@basis.T,np.eye(5),atol=2e-6)
        reconstruction = matmul(matmul(matrix,basis,transpose_b=True),basis)
        self.assertLess(np.linalg.norm(matrix-reconstruction)/np.linalg.norm(matrix),2e-6)
        np.testing.assert_array_equal(basis,randomized_basis(matrix,5))

    def test_complete_soft_deformer_training_without_frameworks(self):
        script = '''
import importlib.abc,sys,json,numpy as np
from pathlib import Path
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorflow','mediapipe','gsplat'}:
            raise AssertionError('unexpected training dependency: '+fullname)
sys.meta_path.insert(0,BlockFrameworks())
from server.vhuman.rig import safetensors as st
from server.vhuman.rig.soft_deformer import train
root=Path(sys.argv[1]);rig=root/'rig';rig.mkdir()
identity=np.eye(4).tolist()
definition=dict(controls=[dict(name='jawOpen',min=0,max=1)],correctives=[],joint_matrix=[],blendshapes=[],
                joints=[dict(name='head',parent=None,rest_translation=[0,0,0],rest_rotation=np.eye(3).tolist(),bind=identity)])
(rig/'rig.json').write_text(json.dumps(definition));(rig/'viz.json').write_text('{}')
rest=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)
faces=np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]],np.int32)
st.save(rig/'rig_deformer.safetensors',dict(rest=rest,**{'skin.weights':np.ones((4,1),np.float32),'skin.joints':np.zeros((4,1),np.int32)}))
takes=[]
for take in range(2):
    folder=root/str(take);folder.mkdir();takes.append(folder)
    control=(.5+.4*np.sin(np.arange(24)*.2+take))[:,None].astype(np.float32)
    target=np.tile(rest,(24,1,1));surface=target+control[:,None]*np.full((1,4,3),.0002,np.float32)
    np.savez(folder/'soft_tissue_samples.npz',ids=np.arange(4),faces=faces,target=target,surface=surface,controls=control,fps=30.)
report=train(rig,takes,max_modes=8)
assert report['backend']=='native_cpu' and report['device']=='cpu'
assert np.isfinite(report['heldout_model_rmse_mm'])
arrays,_=st.load(rig/'soft_deformer.safetensors')
assert arrays['basis'].shape==(8,4,3) and arrays['weight'].shape==(8,1)
assert all(np.isfinite(value).all() for value in arrays.values())
'''
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            subprocess.run([sys.executable,'-c',script,directory],cwd=ROOT,check=True,
                           capture_output=True,text=True,timeout=30)


if __name__ == '__main__': unittest.main()
