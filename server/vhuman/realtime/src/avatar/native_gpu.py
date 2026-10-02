"""Framework-free primary-context native rig and optional DLPack interchange.

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
    # Dropping the export releases its DeviceView after the consumer finishes.
    del record


# Capsule destructors also release exports that a consumer never takes ownership of.
_capsule_valid = C.pythonapi.PyCapsule_IsValid
_capsule_valid.argtypes = [C.c_void_p, C.c_char_p]
_capsule_valid.restype = C.c_int
_capsule_pointer = C.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes = [C.c_void_p, C.c_char_p]
_capsule_pointer.restype = C.c_void_p


@C.CFUNCTYPE(None, C.c_void_p)
def capsule_destroy(capsule):
    if _capsule_valid(capsule, b'dltensor'):
        pointer = _capsule_pointer(capsule, b'dltensor')
        release(C.cast(pointer, C.POINTER(Managed)))


class DeviceView:
    """Borrowed FP32 vertices; consumer lifetime retains the native rig."""
    def __init__(self, owner, pointer, shape):
        self.owner, self.pointer, self.shape = owner, pointer, tuple(shape)
        self.runtime = owner.runtime
        owner.live_views += 1

    def __dlpack_device__(self):
        return (2, self.runtime.device)

    def __dlpack__(self, stream=None):
        self.runtime.check_open()
        if stream is not None and stream != self.runtime.cuda_stream and not (stream == 1 and self.runtime.cuda_stream == 0):
            raise ValueError('borrowed vertices must be consumed on their native stream')
        shape = (C.c_int64 * len(self.shape))(*self.shape)
        managed = Managed(Tensor(self.pointer, Device(2, self.runtime.device),
                                 len(self.shape), Dtype(2, 32, 1), shape, None, 0), None, release)
        address = C.addressof(managed)
        _views[address] = (managed, shape, self)
        capsule_new = C.pythonapi.PyCapsule_New
        capsule_new.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
        capsule_new.restype = C.py_object
        try:
            return capsule_new(address, b'dltensor', C.cast(capsule_destroy, C.c_void_p))
        except Exception:
            _views.pop(address, None)
            raise

    def numpy(self):
        return self.runtime.download(self.pointer, self.shape)

    def __del__(self):
        self.owner.live_views -= 1


class NativeSharedGPU(Native):
    def __init__(self, package, build_dir, device=0, stream=None, *, runtime=None):
        from .cuda_runtime import CudaRuntime
        self.g = self.h = None
        self.live_views = 0
        self.runtime = None
        self.owns_runtime = runtime is None
        if runtime is not None and (runtime.device != device or stream is not None):
            raise ValueError('runtime/device/stream mismatch')
        selected = runtime if runtime is not None else CudaRuntime(device, stream)
        selected.check_open()
        self.runtime = selected
        self.device, self.stream = device, selected
        self.runtime.borrowers += 1
        try:
            super().__init__(build_gpu_library(build_dir, backend='cuda'), package)
            L = self.lib
            L.vh_gpu_create_shared.argtypes = [C.c_void_p, C.c_int, C.c_size_t, C.c_int]
            L.vh_gpu_create_shared.restype = C.c_void_p
            L.vh_gpu_submit.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
            L.vh_gpu_vertices_device.argtypes = [C.c_void_p]
            L.vh_gpu_vertices_device.restype = C.c_size_t
            L.vh_gpu_free.argtypes = [C.c_void_p]
            self.g = L.vh_gpu_create_shared(self.h, device, selected.cuda_stream, 0)
            if not self.g:
                raise RuntimeError('primary-context native CUDA rig creation failed')
        except Exception:
            self.close()
            raise

    def submit(self, controls, use_ml=True):
        if not self.g:
            raise RuntimeError('native rig is closed')
        x = np.ascontiguousarray(controls, np.float32)
        if x.shape != (self.C,) or not np.isfinite(x).all():
            raise ValueError('invalid native controls')
        if self.lib.vh_gpu_submit(self.g, self._p(x), int(use_ml)):
            raise RuntimeError('native CUDA submission failed')
        return DeviceView(self, self.lib.vh_gpu_vertices_device(self.g), (self.V, 3))

    def close(self):
        if self.live_views:
            raise RuntimeError('release native device views before close')
        if self.g:
            self.lib.vh_gpu_free(self.g)
            self.g = None
        if self.h:
            super().close()
        if self.runtime:
            self.runtime.borrowers -= 1
            if self.owns_runtime:
                self.runtime.close()
            self.runtime = None
