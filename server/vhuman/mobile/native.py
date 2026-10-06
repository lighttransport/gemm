"""Framework-free mobile evaluator binding and bind-space package writer."""
import ctypes
import os
from pathlib import Path
import struct
import subprocess
import numpy as np


def write_model(path, model, geometry):
    beta=geometry['gnm_identity'];rest,joints=model.evaluate(beta)
    scale=float(geometry['scale']);rotation=geometry['rotation']
    offset=np.median(geometry['full_neutral']-scale*rest@rotation.T,axis=0)
    bind=(geometry['full_neutral']-offset)@rotation/scale
    arrays=[np.array([scale]),rotation,offset,joints,bind,
            model.data['expression_basis'],model.data['pose_correctives_regressor'],model.data['skinning_weights']]
    with Path(path).open('wb') as f:
        f.write(struct.pack('<8s4I',b'VHGNM001',len(rest),model.expression_dim,4,36))
        f.write(np.asarray(model.parents,dtype='<i4').tobytes())
        for a in arrays:
            if not np.isfinite(a).all():raise ValueError('nonfinite native model')
            f.write(np.asarray(a,dtype='<f4').tobytes())
    return dict(format='vhuman.mobile_gnm.v1',vertices=len(rest),expressions=model.expression_dim,
                residual_transport='native_lbs_bind_space',bytes=Path(path).stat().st_size)


def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    lib=directory/'libvhuman_mobile.so'
    subprocess.run(['c++','-std=c++17','-O3','-Wall','-Wextra','-Wpedantic','-shared','-fPIC',
        str(Path(__file__).with_name('native.cpp')),'-o',str(lib)],check=True,
        env=dict(os.environ,TMPDIR=str(directory.resolve())))
    return lib


class Native:
    def __init__(self,library,model):
        self.lib=ctypes.CDLL(str(library));l=self.lib
        l.vh_mobile_load.argtypes=[ctypes.c_char_p];l.vh_mobile_load.restype=ctypes.c_void_p
        l.vh_mobile_free.argtypes=[ctypes.c_void_p]
        for name in ('vertices','expressions'):
            fn=getattr(l,'vh_mobile_'+name);fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_size_t
        p=ctypes.POINTER(ctypes.c_float)
        l.vh_mobile_eval.argtypes=[ctypes.c_void_p,p,p,p,p];l.vh_mobile_eval.restype=ctypes.c_int
        l.vh_mobile_joint_transform.argtypes=[ctypes.c_void_p,ctypes.c_uint,p]
        l.vh_mobile_joint_transform.restype=ctypes.c_int
        self.handle=l.vh_mobile_load(str(model).encode())
        if not self.handle:raise ValueError('invalid native avatar package')
        self.vertices=l.vh_mobile_vertices(self.handle);self.expressions=l.vh_mobile_expressions(self.handle)

    def evaluate(self,expression,rotations=None,translation=None):
        if not self.handle:raise ValueError('native avatar is closed')
        x=np.ascontiguousarray(expression,dtype=np.float32)
        r=np.ascontiguousarray(np.zeros((4,3)) if rotations is None else rotations,dtype=np.float32)
        t=np.ascontiguousarray(np.zeros(3) if translation is None else translation,dtype=np.float32)
        if x.shape!=(self.expressions,) or r.shape!=(4,3) or t.shape!=(3,):raise ValueError('invalid native pose shapes')
        out=np.empty((self.vertices,3),np.float32)
        p=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if self.lib.vh_mobile_eval(self.handle,p(x),p(r),p(t),p(out)):raise ValueError('invalid native pose')
        return out

    def close(self):
        if self.handle:self.lib.vh_mobile_free(self.handle);self.handle=None

    def joint_transform(self,joint):
        if type(joint) is not int or not 0<=joint<4:raise ValueError('invalid joint')
        out=np.empty(12,np.float32)
        if self.lib.vh_mobile_joint_transform(self.handle,joint,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))):
            raise ValueError('joint transform requires a valid evaluated pose')
        return out[:9].reshape(3,3),out[9:]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
