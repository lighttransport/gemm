"""Persistent repository CUDA training via driver/NVRTC; no model framework.

Default trainers synchronize their public NumPy parameter storage after steps.
With resident=True, call sync_parameters() before reading/mutating that storage,
then upload_parameters() after external edits. Export helpers synchronize it.
"""
import ctypes as C
from functools import lru_cache
import numpy as np
from .native_training import ROOT, FP, IP, pointer


def device_index(device):
    if device == 'cpu':return None
    if device == 'cuda':return 0
    if isinstance(device,str) and device.startswith('cuda:') and device[5:].isdigit():return int(device[5:])
    raise ValueError('training device must be cpu, cuda, or cuda:N')


@lru_cache(maxsize=1)
def library():
    path=ROOT/'cuda/vhuman/libvhuman_training_cuda.so'
    if not path.is_file():raise RuntimeError('build with make -C cuda/vhuman libvhuman_training_cuda.so')
    lib=C.CDLL(str(path));D=C.POINTER(C.c_double);I=C.c_int;H=C.c_void_p
    declarations={
        'vht_error':([],C.c_char_p),'vht_compile_probe':([],I),
        'vht_open':([I,FP,I,C.c_double,C.c_double,C.c_size_t],H),'vht_close':([H],None),
        'vht_parameters':([H,FP,I],I),'vht_peak_bytes':([H],C.c_size_t),'vht_overlaps':([H],C.c_size_t),
        'vht_optimizer':([H,C.c_double,C.c_double],I),
        'vht_gemm':([H,FP,FP,FP,I,I,I,I,I],I),
        'vht_cues':([H,FP,FP,FP,I,I,I,FP,FP,D,I],I),
        'vht_appearance':([H,FP,IP,FP,FP,FP,I,I,I,I,I,FP,FP,FP,FP,D,I],I),
    }
    for name,(args,result) in declarations.items():
        f=getattr(lib,name);f.argtypes,f.restype=args,result
    return lib


def check(code):
    if code:raise RuntimeError(library().vht_error().decode())


def optional_pointer(value):return None if value is None else pointer(value)


class GpuTraining:
    def __init__(self, parameters, device, lr, decay, memory_mb=512):
        self.handle=None
        if (type(memory_mb) is not int or not 1<=memory_mb<=4096 or parameters.dtype!=np.float32 or
                parameters.ndim!=1 or not parameters.flags.c_contiguous):raise ValueError('invalid CUDA training storage/budget')
        self.parameters=parameters;self.lib=library()
        self.handle=self.lib.vht_open(device,pointer(parameters),len(parameters),lr,decay,memory_mb*1048576)
        if not self.handle:check(-1)

    def close(self):
        if self.handle:self.lib.vht_close(self.handle);self.handle=None

    def __del__(self):self.close()

    def sync(self):check(self.lib.vht_parameters(self.handle,pointer(self.parameters),0))
    def upload(self):check(self.lib.vht_parameters(self.handle,pointer(self.parameters),1))
    def configure(self,lr,decay):check(self.lib.vht_optimizer(self.handle,lr,decay))
    @property
    def peak_bytes(self):return self.lib.vht_peak_bytes(self.handle)
    @property
    def overlaps(self):return self.lib.vht_overlaps(self.handle)
