"""Area-weighted normals and cancellation fallback, including a saved real mesh."""
import argparse
import ctypes as C
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
lib=C.CDLL(str(root/'cpu/pixal3d/libpixal3d_validation.so'))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.px_test_normals.argtypes=[fp,C.c_int,ip,C.c_int,fp];lib.px_test_error.restype=C.c_char_p
v=np.array([[0,0,0],[.02,0,0],[0,.01,0],[.01,.01,.03]],np.float32)
fixtures=[('regular',v,np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)),
          ('cancelled',v[:3],np.array([[0,1,2],[0,2,1]],np.int32))]
if a.mesh:
    mesh=load_file(a.mesh);fixtures.append(('real',mesh['vertices'],mesh['faces']))
torch.set_num_threads(4)
for name,v,f in fixtures:
    actual=np.empty_like(v);assert lib.px_test_normals(v,len(v),f,len(f),actual)==0,lib.px_test_error().decode()
    tv=torch.tensor(v);tf=torch.tensor(f).long()
    cross=torch.cross(tv[tf[:,1]]-tv[tf[:,0]],tv[tf[:,2]]-tv[tf[:,0]],dim=1)
    summed=torch.zeros_like(tv)
    # Match the source-order accumulation for vertices whose contributions cancel.
    for axis in range(3):summed.index_add_(0,tf[:,axis],cross)
    length=summed.norm(dim=1);expected=torch.nn.functional.normalize(summed,dim=1).numpy()
    reliable=length.numpy()>1e-8
    np.testing.assert_allclose(actual[reliable],expected[reliable],atol=2e-5,rtol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1,atol=2e-6,rtol=0)
    if name=='cancelled':np.testing.assert_array_equal(actual,np.tile([0.,0.,1.],(3,1)))
    print(f'{name}: {len(v)} finite unit normals PASS',flush=True)
