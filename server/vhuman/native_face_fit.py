"""Array adapters for native objectives/optimizers; no automatic differentiation."""
import ctypes as C
import numpy as np
from .native_face import library,check,pointer,f32,DP,IP,OBJECTIVE


def dp(a):return a.ctypes.data_as(DP)
def ip(a):return a.ctypes.data_as(IP)
def f64(a):return np.ascontiguousarray(a,np.float64)
def i32(a):return np.ascontiguousarray(a,np.int32)


class Adam:
    def __init__(self,x,rates, *, clip=0):
        if x.dtype not in (np.dtype('float32'),np.dtype('float64')) or not x.flags.c_contiguous or not x.flags.writeable:
            raise ValueError('Adam parameters must be writable contiguous floats')
        self.x=x;self.first=np.zeros_like(x);self.second=np.zeros_like(x);self.rates=f64(np.broadcast_to(rates,x.shape))
        self.clip=clip;self.step_count=0
    def step(self,gradient):
        gradient=np.ascontiguousarray(gradient,dtype=self.x.dtype)
        if gradient.shape!=self.x.shape:raise ValueError('Adam gradient shape mismatch')
        p=pointer if self.x.dtype==np.float32 else dp
        fn=library().vhf_adam if self.x.dtype==np.float32 else library().vhf_adam64
        check(fn(*[p(a) for a in (self.x,gradient,self.first,self.second)],dp(self.rates),self.x.size,self.step_count+1,self.clip))
        self.step_count+=1


def lbfgs(parameters,objective,iterations,max_evals):
    if parameters.dtype!=np.float32 or not parameters.flags.c_contiguous or parameters.ndim!=1:
        raise ValueError('L-BFGS needs a contiguous FP32 parameter vector')
    errors=[]
    @OBJECTIVE
    def callback(_,x,g,loss):
        try:
            point=np.ctypeslib.as_array(x,shape=parameters.shape)
            value,gradient=objective(point)
            gradient=f32(gradient,parameters.shape)
            np.ctypeslib.as_array(g,shape=parameters.shape)[:]=gradient;loss[0]=value
            return 0
        except BaseException as exc:
            errors.append(exc);return -1
    code=library().vhf_lbfgs(pointer(parameters),parameters.size,iterations,max_evals,callback,None)
    if errors:raise errors[0]
    check(code)


def camera_array(cameras):
    return f64([np.r_[c.rotation.ravel(),c.origin,c.focal,c.focal_y or c.focal,c.cx,c.cy,c.skew] for c in cameras])


def project(points,cameras):
    points=f32(points);camera=camera_array(cameras)
    if points.ndim!=3 or points.shape[2]!=3 or len(points)!=len(camera):raise ValueError('invalid projection shape')
    pixels=np.empty((*points.shape[:2],2),np.float32);depth=np.empty(points.shape[:2],np.float32)
    check(library().vhf_project(pointer(points),dp(camera),*points.shape[:2],pointer(pixels),pointer(depth)))
    return pixels,depth


def reprojection(points,cameras,target,weight,ipd):
    points=f32(points);camera=camera_array(cameras)
    if points.ndim!=3 or points.shape[2]!=3 or len(points)!=len(camera):raise ValueError('invalid reprojection shape')
    target=f32(target,(*points.shape[:2],2));weight=f32(weight,points.shape[:2]);ipd=f32(ipd,(len(points),))
    gradient=np.empty_like(points);loss=C.c_double()
    check(library().vhf_reprojection(pointer(points),dp(camera),pointer(target),pointer(weight),pointer(ipd),
                                    *points.shape[:2],pointer(gradient),C.byref(loss)))
    return loss.value,gradient


def quadratic(value,weight,reference=None):
    value=f32(value);reference=None if reference is None else f32(reference,value.shape)
    gradient=np.empty_like(value);loss=C.c_double()
    if value.size:check(library().vhf_quadratic(pointer(value),pointer(reference),value.size,weight,pointer(gradient),C.byref(loss)))
    return loss.value,gradient


def area(rest,posed,triangles):
    rest=f32(rest);posed=f32(posed);triangles=i32(triangles)
    if rest.ndim!=2 or rest.shape[1]!=3 or posed.ndim!=3 or posed.shape[1:]!=rest.shape or triangles.ndim!=2 or triangles.shape[1]!=3:raise ValueError('invalid area guard shape')
    gradient=np.empty_like(posed);loss=C.c_double()
    check(library().vhf_area_loss(pointer(rest),pointer(posed),ip(triangles),len(rest),len(triangles),len(posed),pointer(gradient),C.byref(loss)))
    return loss.value,gradient


def contact(points,joints,rotations,alignment,spec,limit=None):
    points=f32(points);joints=f32(joints,(len(points),4,3));rotations=f32(rotations,(len(points),4,3));alignment=f64(alignment)
    if points.ndim!=3 or points.shape[2]!=3 or alignment.shape!=(3,3):raise ValueError('invalid contact shape')
    pairs=i32(spec[0]).reshape(-1,2);eyes=i32([v[:2] for v in spec[1]]).reshape(-1,2);radii=f32([v[2] for v in spec[1]])
    distances=np.empty((len(points),len(pairs)+len(eyes)),np.float32);loss=C.c_double()
    limit=None if limit is None else f32(limit,distances.shape)
    gradients=[np.empty_like(a) for a in (points,joints,rotations)] if limit is not None else [None]*3
    check(library().vhf_contact(pointer(points),pointer(joints),pointer(rotations),dp(alignment),*points.shape[:2],
          ip(pairs),len(pairs),ip(eyes),pointer(radii),len(eyes),pointer(limit),pointer(distances),*[pointer(g) for g in gradients],C.byref(loss)))
    return distances if limit is None else (loss.value,*gradients)


def temporal(expression,rotations,times):
    expression=f32(expression);rotations=f32(rotations,(len(expression),4,3));times=f64(times)
    if expression.ndim!=2 or times.shape!=(len(expression),):raise ValueError('invalid temporal shape')
    ge=np.empty_like(expression);gr=np.empty_like(rotations);loss=C.c_double()
    check(library().vhf_temporal(pointer(expression),pointer(rotations),dp(times),*expression.shape,pointer(ge),pointer(gr),C.byref(loss)))
    return loss.value,ge,gr


def silhouette(vertices,camera,prepared):
    vertices=f32(vertices);tri=i32(prepared['triangles']);bary=f32(prepared['barycentric']);cam=camera_array([camera])
    sdf=f32(prepared['sdf']);allowed=f32(prepared['allowed'],sdf.shape);target=f32(prepared['target'])
    if vertices.ndim!=2 or vertices.shape[1]!=3 or tri.ndim!=2 or tri.shape[1]!=3 or bary.shape!=tri.shape or sdf.ndim!=2 or target.ndim!=2 or target.shape[1]!=2:raise ValueError('invalid silhouette shape')
    gradient=np.empty_like(vertices);loss=C.c_double()
    check(library().vhf_silhouette(pointer(vertices),len(vertices),ip(tri),pointer(bary),len(tri),dp(cam),pointer(sdf),pointer(allowed),*sdf.shape,
                                  prepared['scale'],pointer(target),len(target),pointer(gradient),C.byref(loss)))
    return loss.value,gradient


def registration(x,ids,target,normal,edges,length):
    x=f64(x);ids=i32(ids);target=f64(target);normal=f64(normal);edges=i32(edges);length=f64(length)
    if x.ndim!=2 or x.shape[1]!=3 or ids.ndim!=1 or target.shape!=(len(ids),3) or normal.shape!=target.shape or edges.ndim!=2 or edges.shape[1]!=2 or length.shape!=(len(edges),):raise ValueError('invalid registration shape')
    gradient=np.empty_like(x);energy=np.empty(4,np.float64)
    check(library().vhf_registration_loss(dp(x),len(x),ip(ids),len(ids),dp(target),dp(normal),ip(edges),len(edges),dp(length),dp(gradient),dp(energy)))
    return float(1e6*(energy@[1,.05,.3,.05])),gradient,energy
