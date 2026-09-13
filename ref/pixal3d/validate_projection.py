"""Compare public native camera projection to the original ProjGrid utilities."""
import ast
import ctypes as C
from typing import *
import numpy as np
from PIL import Image
import torch
from torch import nn
from upstream_import import ROOT
source=ast.parse((ROOT/'upstream/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py').read_text())
nodes=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['project_points_to_image_batch','ProjGrid']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_projection','exec'),globals())
lib=C.CDLL(str(ROOT.parent.parent/'cpu/pixal3d/libpixal3d.so'))
class Camera(C.Structure):
    _fields_=[('fov',C.c_float),('distance',C.c_float),('mesh_scale',C.c_float)]
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.pixal3d_project.argtypes=[ip,C.c_size_t,C.c_int,C.c_int,C.POINTER(Camera),fp]
lib.pixal3d_camera_distance.argtypes=[C.c_float,C.c_float,C.POINTER(C.c_float)]
for grid,image in [(16,512),(32,512),(64,1024)]:
    module=ProjGrid(grid,image)
    xyz=np.indices((grid,)*3).reshape(3,-1).T.astype(np.int32)
    coords=np.ascontiguousarray(np.column_stack([np.zeros(len(xyz),np.int32),xyz]))
    for fov,distance,scale in [(.857556,0,1),(.65,2,.8),(1.1,.9,1.3)]:
        camera=Camera(fov,distance,scale);derived=C.c_float()
        assert lib.pixal3d_camera_distance(fov,scale,C.byref(derived))==0
        d=distance or derived.value
        transform=module.front_view_transform_matrix[None].clone();transform[:,1,3]=-d
        points=module.grid_points[None]/scale/2
        xy,_,_=project_points_to_image_batch(points,transform,torch.tensor([fov]),image)
        expected=((xy[0]+.5)/image*2-1).numpy()
        actual=np.empty((len(coords),2),np.float32)
        assert lib.pixal3d_project(coords,len(coords),grid,image,C.byref(camera),actual)==0
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
        print(f'ProjGrid {grid}, image {image}, FOV {fov}, distance {d}, scale {scale}: PASS',flush=True)
print('Projection PASS')
