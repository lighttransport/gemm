"""Native convolution/SiLU/resize/normal and mask-loss training for CueNet v3."""
import ctypes as C
import numpy as np
from ..native_training import library, pointer, check, AdamW, set_threads
from ..native_gpu_training import device_index, GpuTraining, optional_pointer, check as gpu_check


class CueNet:
    def __init__(self, side=64, seed=1234, threads=4, device='cpu', resident=False, memory_mb=512):
        if type(side) is not int or not 1<=side<=128:raise ValueError('native cue side must be 1..128')
        set_threads(threads);self.threads=threads;self.shapes={}
        for prefix,inputs,outputs,kernel in [('encoder.0',6,16,5),('encoder.2',16,24,3),('encoder.4',24,32,3),
                                              ('decoder.0',32,24,3),('decoder.2',24,4,1)]:
            self.shapes[prefix+'.weight']=(outputs,inputs,kernel,kernel);self.shapes[prefix+'.bias']=(outputs,)
        rng=np.random.default_rng(seed);values=[];bound=1
        for name,shape in self.shapes.items():
            if name.endswith('weight'):bound=1/np.sqrt(np.prod(shape[1:]))
            values.append(rng.uniform(-bound,bound,shape).ravel())
        self.parameters=np.concatenate(values).astype(np.float32)
        self.prior=np.zeros((1,3,side,side),np.float32);self.prior[:,2]=1
        self.optimizer=AdamW(self.parameters,lr=.003,weight_decay=1e-4)
        self.device=device_index(device);self.resident=resident;self.memory_mb=memory_mb;self._gpu=None

    def close(self):
        if self._gpu:self._gpu.close();self._gpu=None

    def sync_parameters(self):
        if self._gpu:self._gpu.sync()

    def upload_parameters(self):
        if self._gpu:self._gpu.upload()

    def state_dict(self):
        self.sync_parameters()
        result={'prior':self.prior.copy()};offset=0
        for name,shape in self.shapes.items():
            count=int(np.prod(shape));result[name]=self.parameters[offset:offset+count].reshape(shape).copy();offset+=count
        return result

    def load_state_dict(self, tensors):
        if set(tensors)!=set(self.shapes)|{'prior'}:raise ValueError('cue tensor names differ')
        values=[]
        for name,shape in self.shapes.items():
            value=np.asarray(tensors[name],np.float32)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid cue tensor '+name)
            values.append(value.ravel())
        prior=np.asarray(tensors['prior'],np.float32)
        if prior.shape!=self.prior.shape or not np.isfinite(prior).all():raise ValueError('invalid cue prior')
        self.parameters[:]=np.concatenate(values);self.prior[:]=prior
        self.upload_parameters()

    def compute(self, rgb, prior=None, truth=None, mask=None, *, update=False, return_gradient=True, return_output=True):
        rgb=np.ascontiguousarray(rgb,np.float32)
        if rgb.ndim!=4 or rgb.shape[1]!=3 or not 1<=len(rgb)<=8 or not all(1<=v<=128 for v in rgb.shape[2:]):
            raise ValueError('invalid native cue image batch')
        n,_,h,w=rgb.shape
        prior=self.prior if prior is None else np.ascontiguousarray(prior,np.float32)
        if prior.ndim!=4 or prior.shape[1]!=3 or len(prior) not in (1,n) or not all(1<=v<=128 for v in prior.shape[2:]):
            raise ValueError('invalid cue geometry prior')
        if prior.shape[2:]!=(h,w):
            resized=np.empty((len(prior),3,h,w),np.float32)
            check(library().vh_train_cue_resize(pointer(prior),pointer(resized),len(prior),3,*prior.shape[2:],h,w))
            prior=resized
        inputs=np.ascontiguousarray(np.concatenate((rgb,np.broadcast_to(prior,rgb.shape)),axis=1),np.float32)
        gradient=None
        if truth is not None:
            truth=np.ascontiguousarray(truth,np.float32);mask=np.ascontiguousarray(mask,np.float32)
            if truth.shape!=rgb.shape or mask.shape!=(n,h,w):raise ValueError('invalid cue labels')
            if return_gradient or self.device is None:gradient=np.empty_like(self.parameters)
        elif update:raise ValueError('cue labels required for optimizer update')
        out=np.empty((n,4,h,w),np.float32) if return_output or self.device is None else None
        loss=C.c_double();set_threads(self.threads)
        if self.device is not None:
            if self._gpu is None:self._gpu=GpuTraining(self.parameters,self.device,self.optimizer.lr,self.optimizer.decay,self.memory_mb)
            elif not self.resident:self._gpu.upload()
            if update:self._gpu.configure(self.optimizer.lr,self.optimizer.decay)
            gpu_check(self._gpu.lib.vht_cues(self._gpu.handle,pointer(inputs),optional_pointer(truth),optional_pointer(mask),
                n,h,w,optional_pointer(out),optional_pointer(gradient),C.byref(loss),int(update)))
            if update:
                self.optimizer.iteration+=1
                if not self.resident:self._gpu.sync()
            return (out[:,:3] if out is not None else None,out[:,3:4] if out is not None else None,loss.value,gradient)
        check(library().vh_train_cues(pointer(self.parameters),pointer(inputs),pointer(truth) if truth is not None else None,
            pointer(mask) if truth is not None else None,n,h,w,pointer(out),pointer(gradient) if gradient is not None else None,C.byref(loss)))
        if update:self.optimizer.step(gradient)
        return (out[:,:3] if return_output else None,out[:,3:4] if return_output else None,
                loss.value,gradient if return_gradient else None)

    def __call__(self, rgb, geometry_prior=None):return self.compute(rgb,geometry_prior)[:2]
