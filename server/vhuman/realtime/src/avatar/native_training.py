"""Native CPU/CUDA trace-v1 Gaussian fitting, with no tensor framework."""
import ctypes as C
import numpy as np
from ....native_training import library, pointer, IP, check, AdamW, set_threads
from ....native_gpu_training import device_index, GpuTraining, optional_pointer, check as gpu_check


def f32(value):return np.ascontiguousarray(value,np.float32)


class AppearanceTrainer:
    def __init__(self, avatar, triangles, initial_rgb=None, seed=7, threads=4, device='cpu', resident=False, memory_mb=512):
        avatar.validate(triangles);set_threads(threads);self.threads=threads
        a=avatar.arrays;self.n=len(a['triangle']);self.c=len(avatar.metadata['control_names'])
        self.triangle=a['triangle'].copy();self.control_names=tuple(avatar.metadata['control_names'])
        self.topology=avatar.metadata['topology_sha256']
        self.attachments=np.ascontiguousarray(triangles[a['triangle']],np.int32);self.bary=f32(a['barycentric'])
        self.parameters=np.zeros(self.n*32+self.c*8,np.float32)
        self.local=self.parameters[:self.n*32].reshape(self.n,32);self.expression=self.parameters[self.n*32:].reshape(self.c,8)
        rgb=np.clip(a['rgb'] if initial_rgb is None else f32(initial_rgb),.01,.99)
        if rgb.shape!=(self.n,3):raise ValueError('invalid appearance initializer')
        self.local[:,:3]=np.log(rgb/(1-rgb));self.local[:,3]=1
        self.local[:,4:7]=np.log([.15,.15,.00015])
        self.expression[:]=np.random.default_rng(seed).normal(0,.05,self.expression.shape)
        self.optimizer=AdamW(self.parameters,lr=.01,weight_decay=0)
        self.device=device_index(device);self.resident=resident;self.memory_mb=memory_mb;self._gpu=None

    def close(self):
        if self._gpu:self._gpu.close();self._gpu=None

    def sync_parameters(self):
        if self._gpu:self._gpu.sync()

    def upload_parameters(self):
        if self._gpu:self._gpu.upload()

    def load_avatar(self, avatar):
        """Warm start trace-v1 diagonal assets; optimizer moments start fresh."""
        avatar.validate();a=avatar.arrays
        if (avatar.metadata.get('covariance_policy')!='trace-v1' or len(a['triangle'])!=self.n or
                tuple(avatar.metadata['control_names'])!=self.control_names or avatar.metadata['topology_sha256']!=self.topology or
                not np.array_equal(a['triangle'],self.triangle) or not np.array_equal(a['barycentric'],self.bary)):
            raise ValueError('warm start requires the same trace-v1 binding/control count')
        cov=a['covariance_local'];diagonal=np.diagonal(cov,axis1=1,axis2=2)
        expected=np.zeros_like(cov);expected[:,np.arange(3),np.arange(3)]=diagonal
        if not np.allclose(cov,expected,rtol=0,atol=1e-12):raise ValueError('native fitter requires diagonal local covariance')
        scales=np.sqrt(diagonal)
        if (scales<1e-5).any() or (scales>np.array([1,1,.002])+1e-9).any():raise ValueError('warm start covariance outside training bounds')
        if (a['rgb']>1).any():raise ValueError('warm start radiance outside training bounds')
        logit=lambda x:np.log(np.clip(x,1e-7,1-1e-7)/(1-np.clip(x,1e-7,1-1e-7)))
        self.local[:,:3]=logit(a['rgb']);self.local[:,3]=logit(a['opacity'])
        self.local[:,4:7]=np.log(scales);self.local[:,7]=np.arctanh(np.clip(a['normal_offset']/.005,-1+1e-7,1-1e-7))
        self.local[:,8:]=a['color_basis'].reshape(self.n,24);self.expression[:]=a['expression_matrix']
        self.close();self.optimizer=AdamW(self.parameters,lr=self.optimizer.lr,weight_decay=self.optimizer.decay)

    def compute(self, vertices, controls, view, intrinsics, size, truth=None, mask=None, *, update=False,
                return_gradient=True, return_output=True):
        vertices,controls,view,intrinsics=map(f32,(vertices,controls,view,intrinsics))
        if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<=int(self.attachments.max()):raise ValueError('invalid fitting vertices')
        if controls.shape!=(self.c,) or view.shape!=(4,4) or intrinsics.shape!=(3,3):raise ValueError('invalid fitting camera/controls')
        if (not np.allclose(view[3],[0,0,0,1]) or not np.allclose(intrinsics[2],[0,0,1]) or
                intrinsics[0,1]!=0 or intrinsics[1,0]!=0):raise ValueError('affine view and pinhole camera without skew required')
        if len(size)!=2 or any(type(x) is not int or not 1<=x<=4096 for x in size):raise ValueError('invalid fitting image size')
        width,height=size;camera=f32(np.concatenate((view.ravel(),intrinsics.ravel())))
        gradient=None
        if truth is not None:
            truth,mask=f32(truth),f32(mask)
            if truth.shape!=(height,width,3) or mask.shape!=(height,width):raise ValueError('invalid appearance supervision')
            if return_gradient or self.device is None:gradient=np.empty_like(self.parameters)
        elif update:raise ValueError('appearance supervision required for update')
        rgba=np.empty((height,width,4),np.float32) if return_output or self.device is None else None
        loss=np.zeros(2,np.float64);set_threads(self.threads)
        if self.device is not None:
            if self._gpu is None:self._gpu=GpuTraining(self.parameters,self.device,self.optimizer.lr,self.optimizer.decay,self.memory_mb)
            elif not self.resident:self._gpu.upload()
            if update:self._gpu.configure(self.optimizer.lr,self.optimizer.decay)
            gpu_check(self._gpu.lib.vht_appearance(self._gpu.handle,pointer(vertices),self.attachments.ctypes.data_as(IP),
                pointer(self.bary),pointer(controls),pointer(camera),self.n,len(vertices),self.c,width,height,
                optional_pointer(truth),optional_pointer(mask),optional_pointer(rgba),optional_pointer(gradient),
                loss.ctypes.data_as(C.POINTER(C.c_double)),int(update)))
            if update:
                self.optimizer.iteration+=1
                if not self.resident:self._gpu.sync()
            return rgba,loss,gradient
        check(library().vh_train_appearance(pointer(self.parameters),pointer(vertices),self.attachments.ctypes.data_as(IP),
            pointer(self.bary),pointer(controls),pointer(camera),self.n,len(vertices),self.c,width,height,
            pointer(truth) if truth is not None else None,pointer(mask) if truth is not None else None,
            pointer(rgba),pointer(gradient) if gradient is not None else None,loss.ctypes.data_as(C.POINTER(C.c_double))))
        if update:self.optimizer.step(gradient)
        return rgba if return_output else None,loss,gradient if return_gradient else None

    def export(self, avatar):
        self.sync_parameters()
        p=self.local
        sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-80,80)))
        scales=np.minimum(np.maximum(np.exp(np.clip(p[:,4:7],-80,80)),1e-5),[1,1,.002])
        covariance=np.zeros((self.n,3,3),np.float32);covariance[:,np.arange(3),np.arange(3)]=scales**2
        avatar.arrays.update(rgb=f32(sigmoid(p[:,:3])),opacity=f32(sigmoid(p[:,3])),
            normal_offset=f32(.005*np.tanh(p[:,7])),covariance_local=covariance,
            color_basis=p[:,8:].reshape(self.n,8,3).copy(),expression_matrix=self.expression.copy())
        return avatar


def initialize_rgb(points, image, view, intrinsics):
    """Bilinear sampling with zero padding, equivalent to aligned grid sampling."""
    camera=points@view[:3,:3].T+view[:3,3];screen=camera@intrinsics.T
    uv=screen[:,:2]/np.maximum(screen[:,2:3],1e-6)
    if not np.isfinite(uv).all():raise ValueError('nonfinite appearance initializer projection')
    height,width=image.shape[:2];x,y=uv[:,0],uv[:,1]
    # Clamp far-out coordinates before integer conversion; they sample zero.
    x=np.clip(x,-2,width+1);y=np.clip(y,-2,height+1)
    x0,y0=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);dx,dy=x-x0,y-y0
    color=np.zeros((len(points),3),np.float32)
    for xx,yy,weight in ((x0,y0,(1-dx)*(1-dy)),(x0+1,y0,dx*(1-dy)),(x0,y0+1,(1-dx)*dy),(x0+1,y0+1,dx*dy)):
        valid=(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
        color[valid]+=image[yy[valid],xx[valid]]*weight[valid,None]
    return np.clip(color,.01,.99)
