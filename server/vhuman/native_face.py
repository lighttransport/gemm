"""ctypes boundary for native face computation; NumPy and stdlib only."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import threading
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FP=C.POINTER(C.c_float)
DP=C.POINTER(C.c_double)
IP=C.POINTER(C.c_int32)
OBJECTIVE=C.CFUNCTYPE(C.c_int,C.c_void_p,FP,FP,DP)


@lru_cache(maxsize=1)
def library():
    path=ROOT/'cpu/vhuman/libvhuman_face.so'
    if not path.is_file(): raise RuntimeError('build native face core: make -C cpu/vhuman libvhuman_face.so')
    lib=C.CDLL(str(path))
    declarations={
        'vhf_abi_version':([],C.c_int), 'vhf_error':([],C.c_char_p),
        'vhf_gnm_open':([C.c_int]*4+[IP]+[FP]*7,C.c_void_p),
        'vhf_gnm_close':([C.c_void_p],None),
        'vhf_gnm_eval':([C.c_void_p,C.c_int]+[FP]*13,C.c_int),
        'vhf_gemm':([FP]*3+[C.c_int]*6,C.c_int),
        'vhf_rodrigues':([DP]*3+[C.c_int],C.c_int),
        'vhf_image_sample':([C.c_void_p,C.c_int,C.c_int,DP,DP,C.c_double,C.c_int,C.c_int,FP],C.c_int),
        'vhf_image_resize':([FP]+[C.c_int]*7+[FP],C.c_int),
        'vhf_face_detections':([FP,C.c_float,FP,C.c_int],C.c_int),
        'vhf_adam':([FP]*4+[DP,C.c_int,C.c_int,C.c_double],C.c_int),
        'vhf_adam64':([DP]*5+[C.c_int,C.c_int,C.c_double],C.c_int),
        'vhf_lbfgs':([FP,C.c_int,C.c_int,C.c_int,OBJECTIVE,C.c_void_p],C.c_int),
        'vhf_registration_loss':([DP,C.c_int,IP,C.c_int,DP,DP,IP,C.c_int,DP,DP,DP],C.c_int),
        'vhf_project':([FP,DP,C.c_int,C.c_int,FP,FP],C.c_int),
        'vhf_reprojection':([FP,DP,FP,FP,FP,C.c_int,C.c_int,FP,DP],C.c_int),
        'vhf_quadratic':([FP,FP,C.c_int,C.c_double,FP,DP],C.c_int),
        'vhf_area_loss':([FP,FP,IP,C.c_int,C.c_int,C.c_int,FP,DP],C.c_int),
        'vhf_contact':([FP,FP,FP,DP,C.c_int,C.c_int,IP,C.c_int,IP,FP,C.c_int,FP,FP,FP,FP,FP,DP],C.c_int),
        'vhf_temporal':([FP,FP,DP,C.c_int,C.c_int,FP,FP,DP],C.c_int),
        'vhf_silhouette':([FP,C.c_int,IP,FP,C.c_int,DP,FP,FP,C.c_int,C.c_int,C.c_double,FP,C.c_int,FP,DP],C.c_int),
    }
    for name,(args,result) in declarations.items():
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    if lib.vhf_abi_version()!=1: raise RuntimeError('native face ABI mismatch; rebuild library')
    return lib


def pointer(x): return None if x is None else x.ctypes.data_as(FP)
def check(code):
    if code: raise RuntimeError(library().vhf_error().decode())


def f32(x,shape=None):
    x=np.ascontiguousarray(x,np.float32)
    if (shape is not None and x.shape!=shape) or not np.isfinite(x).all():
        raise ValueError('invalid native face tensor')
    return x


def matmul(a,b, *, transpose_a=False,transpose_b=False,threads=16):
    a,b=f32(a),f32(b)
    if a.ndim!=2 or b.ndim!=2: raise ValueError('GEMM needs matrices')
    m,k=a.shape[::-1] if transpose_a else a.shape
    kk,n=b.shape[::-1] if transpose_b else b.shape
    if k!=kk: raise ValueError('GEMM inner dimensions differ')
    out=np.empty((m,n),np.float32)
    check(library().vhf_gemm(pointer(out),pointer(a),pointer(b),m,n,k,int(transpose_a),int(transpose_b),threads))
    return out


def sample(image,center,size,angle,resolution,zero_border=False):
    image=np.ascontiguousarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3: raise ValueError('expected RGB8 image')
    center=np.ascontiguousarray(center,np.float64);size=np.ascontiguousarray(size,np.float64)
    if center.shape!=(2,) or size.shape!=(2,) or type(resolution) is not int or not 1<=resolution<=2048:raise ValueError('invalid crop')
    out=np.empty((resolution,resolution,3),np.float32)
    check(library().vhf_image_sample(image.ctypes.data,*image.shape[:2],center.ctypes.data_as(DP),size.ctypes.data_as(DP),
                                    angle,resolution,int(zero_border),pointer(out)))
    return out


def resize(image,size, *, nearest=False,quantize=False):
    image=f32(image)
    if image.ndim not in (2,3) or len(size)!=2 or any(type(n) is not int or n<1 or n>4000000 for n in size) or size[0]*size[1]>4000000:
        raise ValueError('invalid resize shape')
    channels=1 if image.ndim==2 else image.shape[2]
    out=np.empty((size[1],size[0],*image.shape[2:]),np.float32)
    check(library().vhf_image_resize(pointer(image),*image.shape[:2],channels,size[1],size[0],int(nearest),int(quantize),pointer(out)))
    return out


def detections(raw,threshold=.5):
    raw=f32(raw,(896*17,));out=np.empty((896,17),np.float32)
    count=library().vhf_face_detections(pointer(raw),threshold,pointer(out),len(out))
    if count<0:check(count)
    return [(row[:16].copy(),float(row[16])) for row in out[:count]]


def rodrigues(rotation, *, derivatives=False):
    rotation=np.ascontiguousarray(rotation,np.float64)
    if rotation.shape[-1:]!=(3,): raise ValueError('invalid rotation shape')
    out=np.empty((*rotation.shape[:-1],3,3),np.float64)
    deriv=np.empty((*rotation.shape[:-1],3,3,3),np.float64) if derivatives else None
    p=lambda a:None if a is None else a.ctypes.data_as(DP)
    check(library().vhf_rodrigues(p(rotation),p(out),p(deriv),rotation.size//3))
    return (out,deriv) if derivatives else out


class Gnm:
    """Owned native sampled/full model, including explicit parameter gradients."""
    def __init__(self,model,vertex_ids=None, *, threads=16):
        self.handle=None;self.lock=threading.RLock();self.lib=library()
        d=model.data
        ids=np.arange(len(d['template_vertex_positions'])) if vertex_ids is None else np.asarray(vertex_ids)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or
                (ids<0).any() or (ids>=len(d['template_vertex_positions'])).any()):
            raise ValueError('invalid native GNM vertex ids')
        self.ids=ids.copy();self.vertices=len(ids);self.identities=model.identity_dim;self.expressions=model.expression_dim
        parents=np.ascontiguousarray(model.parents,np.int32)
        if parents.shape!=(4,): raise ValueError('native GNM requires four joints')
        arrays=[d['template_vertex_positions'][ids],d['template_joint_positions'],d['vertex_identity_basis'][:,ids],
                d['joint_identity_basis'],d['expression_basis'][:,ids],
                d['pose_correctives_regressor'].reshape(36,-1,3)[:,ids],d['skinning_weights'][:,ids]]
        shapes=[(self.vertices,3),(4,3),(self.identities,self.vertices,3),
                (self.identities,4,3),(self.expressions,self.vertices,3),
                (36,self.vertices,3),(4,self.vertices)]
        arrays=[f32(a,shape) for a,shape in zip(arrays,shapes)]
        self.handle=self.lib.vhf_gnm_open(self.vertices,self.identities,self.expressions,threads,
                                         parents.ctypes.data_as(IP),*[pointer(a) for a in arrays])
        if not self.handle: raise RuntimeError(self.lib.vhf_error().decode())

    def evaluate(self,identity,expression,rotations,translation, *, bind_residual=None,upstream=None,joint_upstream=None):
        expression=f32(expression)
        if expression.ndim!=2 or expression.shape[1]!=self.expressions: raise ValueError('invalid expression dimensions')
        count=len(expression)
        if not 1<=count<=1024 or count*self.vertices>67108864//3:
            raise ValueError('native GNM batch exceeds workspace limit')
        identity=f32(identity,(self.identities,));rotations=f32(rotations,(count,4,3));translation=f32(translation,(count,3))
        residual=None if bind_residual is None else f32(bind_residual,(self.vertices,3))
        vertices=np.empty((count,self.vertices,3),np.float32);joints=np.empty((count,4,3),np.float32)
        upstream=None if upstream is None else f32(upstream,vertices.shape)
        joint_upstream=None if joint_upstream is None else f32(joint_upstream,joints.shape)
        backward=upstream is not None or joint_upstream is not None
        gradients=[np.empty_like(x) for x in (identity,expression,rotations,translation)] if backward else [None]*4
        with self.lock:
            if not self.handle: raise RuntimeError('native GNM model closed')
            check(self.lib.vhf_gnm_eval(self.handle,count,*[pointer(x) for x in
                  (identity,expression,rotations,translation,residual,vertices,joints,upstream,joint_upstream,*gradients)]))
        return (vertices,joints,gradients) if backward else (vertices,joints)

    def close(self):
        with self.lock:
            if self.handle: self.lib.vhf_gnm_close(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
    def __del__(self):self.close()
