"""Primary-context native rig with a borrowed CUDA tensor via DLPack.

Views retain the native owner. A view remains valid only until the next submit;
 *copy on the same stream* if it must outlive that submit. close refuses live views.
"""
import ctypes as C
import numpy as np
from ....rig.native import Native, build_gpu_library


class Device(C.Structure):
    _fields_ = [("kind", C.c_int), ("index", C.c_int)]


class Dtype(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class Tensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("device", Device), ("ndim", C.c_int), ("dtype", Dtype),
               ("shape", C.POINTER(C.c_int64)), ("strides", C.POINTER(C.c_int64)), ("offset", C.c_uint64)]


class Managed(C.Structure): pass
Deleter = C.CFUNCTYPE(None, C.POINTER(Managed))
Managed._fields_ = [("tensor", Tensor), ("context", C.c_void_p), ("deleter", Deleter)]
_views = {}


@Deleter
def release(pointer):
    record = _views.pop(C.addressof(pointer.contents), None)
    if record:
        record[2].live_views -= 1


class NativeSharedGPU(Native):
    def __init__(self, package, build_dir, device=0, stream=None):
        import torch
        if torch.version.hip:
            raise RuntimeError("the shared-context Gaussian runtime requires CUDA PyTorch")
        self.torch = torch
        self.device = device
        self.stream = stream or torch.cuda.current_stream(device)
        if self.stream.device.index != device: raise ValueError("stream/device mismatch")
        super().__init__(build_gpu_library(build_dir, backend="cuda"), package)
        self.live_views = 0
        L = self.lib
        L.vh_gpu_create_shared.argtypes = [C.c_void_p, C.c_int, C.c_size_t, C.c_int]
        L.vh_gpu_create_shared.restype = C.c_void_p
        L.vh_gpu_submit.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
        L.vh_gpu_vertices_device.argtypes = [C.c_void_p]
        L.vh_gpu_vertices_device.restype = C.c_size_t
        L.vh_gpu_free.argtypes = [C.c_void_p]
        self.g = L.vh_gpu_create_shared(self.h, device, self.stream.cuda_stream, 0)
        if not self.g:
            super().close(); raise RuntimeError("primary-context native CUDA rig creation failed")

    def submit(self, controls, use_ml=True):
        x = np.ascontiguousarray(controls, np.float32)
        if x.shape != (self.C,) or not np.isfinite(x).all(): raise ValueError("invalid native controls")
        if self.torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("submit and consume within the borrowed CUDA stream context")
        if self.lib.vh_gpu_submit(self.g, self._p(x), int(use_ml)):
            raise RuntimeError("native CUDA submission failed")
        shape = (C.c_int64 * 2)(self.V, 3)
        managed = Managed(Tensor(self.lib.vh_gpu_vertices_device(self.g), Device(2, self.device),
                                 2, Dtype(2, 32, 1), shape, None, 0), None, release)
        address = C.addressof(managed)
        _views[address] = (managed, shape, self)
        self.live_views += 1
        capsule_new = C.pythonapi.PyCapsule_New
        capsule_new.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
        capsule_new.restype = C.py_object
        capsule = capsule_new(address, b"dltensor", None)
        try:
            return self.torch.from_dlpack(capsule)
        except Exception:
            release(C.pointer(managed)); raise

    def close(self):
        if self.live_views:
            raise RuntimeError("release native device views before close")
        if self.g:
            self.lib.vh_gpu_free(self.g); self.g = None
        super().close()
