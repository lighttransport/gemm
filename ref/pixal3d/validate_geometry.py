"""Native FDG/remeshing vs pinned upstream PyTorch/CuMesh; full PBR GLB smoke."""
import argparse
import ast
import ctypes as C
import io
import json
from pathlib import Path
import struct
import sys
from typing import *
import numpy as np
from PIL import Image
import torch
from upstream_import import ROOT

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=ROOT.parent.parent/'tmp/pixal3d/geometry.glb')
a=p.parse_args()
torch.set_num_threads(8)
assert torch.cuda.is_available() and not torch.version.hip
torch.cuda.set_per_process_memory_fraction(.45)
sys.path.insert(0,str(ROOT/'cumesh-upstream'))
import cumesh
from cumesh import _C
lib=C.CDLL(str(ROOT.parent.parent/'cpu/pixal3d/libpixal3d_validation.so'))
lib.px_test_error.restype=C.c_char_p
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
class Mesh(C.Structure):
    _fields_=[('vertices',C.POINTER(C.c_float)),('triangles',C.POINTER(C.c_int32)),
              ('n_verts',C.c_int),('n_tris',C.c_int),('hash_keys',C.c_void_p),
              ('hash_vals',C.c_void_p),('hash_cap',C.c_int)]
lib.t2_fdg_to_mesh_bxyz.argtypes=[ip,fp,C.c_int,C.c_float,fp]
lib.t2_fdg_to_mesh_bxyz.restype=Mesh
lib.t2_fdg_to_mesh_bzyx.argtypes=lib.t2_fdg_to_mesh_bxyz.argtypes
lib.t2_fdg_to_mesh_bzyx.restype=Mesh
lib.t2_fdg_mesh_free.argtypes=[C.POINTER(Mesh)]
lib.px_test_mesh.argtypes=[fp,C.c_int,ip,C.c_int,C.c_int,C.c_int,C.POINTER(Mesh)]
lib.px_test_postprocess.argtypes=[ip,fp,fp,C.c_int,C.c_char_p]
def unpack(mesh):
    v=np.ctypeslib.as_array(mesh.vertices,shape=(mesh.n_verts,3)).copy()
    f=np.ctypeslib.as_array(mesh.triangles,shape=(mesh.n_tris,3)).copy()
    lib.t2_fdg_mesh_free(C.byref(mesh))
    assert np.isfinite(v).all() and f.min()>=0 and f.max()<len(v)
    return v,f

def native_mesh(v,f,resolution=0,target=0):
    result=Mesh()
    rc=lib.px_test_mesh(np.ascontiguousarray(v),len(v),np.ascontiguousarray(f),len(f),resolution,target,C.byref(result))
    assert rc==0,lib.px_test_error().decode()
    return unpack(result)

def surface_score(name,native,reference,resolution):
    # Distances to triangle surfaces, in both directions, avoid vertex-density bias.
    all_stats=[]
    for source,target in [(native,reference),(reference,native)]:
        v,f=source;tv,tf=target
        points=np.concatenate([v,v[f].mean(1)]).astype(np.float32)
        # cuBVH requires at least nine triangles; repeated faces have the same surface.
        bvh_faces=np.tile(tf,((9+len(tf)-1)//len(tf),1)) if len(tf)<9 else tf
        bvh=cumesh.cuBVH(torch.tensor(tv,device='cuda'),torch.tensor(bvh_faces,device='cuda'))
        distances=bvh.unsigned_distance(torch.tensor(points,device='cuda'))[0].cpu().numpy()
        all_stats.append((float((distances<=2/resolution).mean()),float(distances.max())))
    precision,recall=[s[0] for s in all_stats]
    score=2*precision*recall/max(precision+recall,1e-30)
    print(json.dumps(dict(test=name,native_vertices=len(native[0]),native_faces=len(native[1]),
        reference_vertices=len(reference[0]),reference_faces=len(reference[1]),fscore=score,
        max_surface_distance=max(s[1] for s in all_stats))),flush=True)
    assert score>=.99

# Exercise asymmetric XYZ coordinates and both quad diagonals with original function.
rng=np.random.default_rng(894)
xyz=np.indices((9,11,13)).reshape(3,-1).T.astype(np.int32)+np.array([91,211,371],np.int32)
coords=np.ascontiguousarray(np.column_stack([np.zeros(len(xyz),np.int32),xyz]),dtype=np.int32)
feats=rng.random((len(coords),7),dtype=np.float32)
feats[:,3:6]=rng.choice([-1.,1.],(len(coords),3))
aabb=np.array([-.5,-.5,-.5,.5,.5,.5],np.float32)
native=unpack(lib.t2_fdg_to_mesh_bxyz(coords,feats,len(coords),1/1024,aabb))
legacy=unpack(lib.t2_fdg_to_mesh_bzyx(coords,feats,len(coords),1/1024,aabb))
legacy_expected=(xyz[:,[2,1,0]]+feats[:,:3])/1024-.5
np.testing.assert_allclose(legacy[0],legacy_expected,atol=1e-7,rtol=0)
np.testing.assert_array_equal(legacy[1],native[1])
source=ast.parse((ROOT/'deps/flexible_dual_grid.py').read_text())
nodes=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in
       ['_init_hashmap','flexible_dual_grid_to_mesh']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_flexible_dual_grid','exec'),globals())
ref=flexible_dual_grid_to_mesh(torch.tensor(xyz,device='cuda'),torch.tensor(feats[:,:3],device='cuda'),
    torch.tensor(feats[:,3:6]>0,device='cuda'),torch.tensor(feats[:,6:7],device='cuda'),
    [[-.5]*3,[.5]*3],grid_size=1024)
ref=tuple(t.cpu().numpy() for t in ref)
np.testing.assert_array_equal(native[1],ref[1])
np.testing.assert_allclose(native[0],ref[0],atol=1e-7,rtol=0)
print('FDG vertices, topology, split weights, legacy XYZ mapping: PASS',flush=True)

# Tilted tetrahedron with asymmetric, non-grid-aligned vertices.
v=np.array([[-.13,-.11,-.09],[.19,-.07,-.12],[.03,.17,-.06],[.02,.04,.23]],np.float32)
f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
centers=v[f].mean(1)
new_faces=[[tri[j],tri[(j+1)%3],len(v)+i] for i,tri in enumerate(f) for j in range(3)]
v=np.concatenate([v,centers]).astype(np.float32);f=np.array(new_faces,np.int32)
dirty_v=np.concatenate([v,np.array([[2,2,2],[3,3,3],[4,4,4]],np.float32)])
dirty_f=np.concatenate([f,np.array([[0,0,1],[len(v),len(v)+1,len(v)+2]],np.int32)])
cleaned=native_mesh(dirty_v,dirty_f,resolution=-1)
upstream=cumesh.CuMesh();upstream.init(torch.tensor(dirty_v,device='cuda'),torch.tensor(dirty_f,device='cuda'))
upstream.remove_degenerate_faces();clean_ref=tuple(t.cpu().numpy() for t in upstream.read())
np.testing.assert_array_equal(cleaned[0],clean_ref[0]);np.testing.assert_array_equal(cleaned[1],clean_ref[1])
print('Exact degenerate-face and unreferenced-vertex cleanup: PASS',flush=True)
# One small boundary must be filled; the same boundary at a larger scale must remain open.
for scale in [.01,1.]:
    open_v=np.array([[-.13,-.11,-.09],[.19,-.07,-.12],[.03,.17,-.06],[.02,.04,.23]],np.float32)*scale
    open_f=np.array([[0,2,1],[0,1,3],[1,2,3]],np.int32)
    filled=native_mesh(open_v,open_f,resolution=-2)
    upstream=cumesh.CuMesh();upstream.init(torch.tensor(open_v,device='cuda'),torch.tensor(open_f,device='cuda'))
    upstream.fill_holes(.03)
    filled_ref=tuple(t.cpu().numpy() for t in upstream.read())
    assert tuple(map(len,filled))==tuple(map(len,filled_ref))
    surface_score('fill_holes_'+str(scale),filled,filled_ref,1024)
for resolution in [64,128]:
    native=native_mesh(v,f,resolution)
    rv,rf=cumesh.remeshing.remesh_narrow_band_dc(torch.tensor(v,device='cuda'),torch.tensor(f,device='cuda'),
        torch.zeros(3,device='cuda'),(resolution+3)/resolution,resolution)
    reference=(rv.cpu().numpy(),rf.cpu().numpy())
    surface_score('remesh_'+str(resolution),native,reference,resolution)
    target=len(native[1])//2
    simplified=native_mesh(*native,target=target)
    upstream=cumesh.CuMesh();upstream.init(torch.tensor(native[0],device='cuda'),torch.tensor(native[1],device='cuda'))
    upstream.simplify(target)
    reference=tuple(t.cpu().numpy() for t in upstream.read())
    surface_score('simplify_'+str(resolution),simplified,reference,resolution)

# Compact sphere-like FDG fixture exercises 1024 remesh and 4096 PBR baking.
xyz=np.indices((21,21,21)).reshape(3,-1).T.astype(np.int32)+502
coords=np.ascontiguousarray(np.column_stack([np.zeros(len(xyz),np.int32),xyz]),dtype=np.int32)
shape=np.zeros((len(xyz),7),np.float32)
for axis in range(3):
    lo=xyz.copy();lo[:,[i for i in range(3) if i!=axis]]+=1
    hi=lo.copy();hi[:,axis]+=1
    inside=lambda x:np.linalg.norm(x-[512,512,512],axis=1)<8
    shape[:,axis+3]=np.where(inside(lo)!=inside(hi),10.,-10.)
texture=np.tile(np.array([.7,.3,.1,.25,.65,1],np.float32),(len(xyz),1))
a.output.parent.mkdir(parents=True,exist_ok=True)
rc=lib.px_test_postprocess(coords,shape,texture,len(coords),str(a.output).encode())
assert rc==0,lib.px_test_error().decode()
raw=a.output.read_bytes()
magic,version,length=struct.unpack_from('<III',raw)
assert magic==0x46546c67 and version==2 and length==len(raw)
jlen,jtype=struct.unpack_from('<II',raw,12)
assert jtype==0x4e4f534a
scene=json.loads(raw[20:20+jlen]);blob=raw[28+jlen:]
for item in scene['images']:
    view=scene['bufferViews'][item['bufferView']]
    image=Image.open(io.BytesIO(blob[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']]))
    assert image.size==(4096,4096)
    data=np.asarray(image)
    assert data.max()>0
assert scene['materials'][0]['alphaMode']=='OPAQUE'
print(f'Full 1024 remesh, UV atlas, 4096 PBR and GLB: PASS ({a.output})',flush=True)
