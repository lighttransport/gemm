"""Native CUDA primary-context streams, completion events and presentation IO."""
import ctypes as C
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]


def library(path=None):
    path = Path(path) if path else ROOT / 'cuda/vhuman/libvhuman_runtime.so'
    if not path.is_file():
        raise RuntimeError('Build native CUDA IO with make -C cuda/vhuman libvhuman_runtime.so')
    lib = C.CDLL(str(path.resolve()))
    signatures = {
        'vh_cuda_open': ([C.c_int, C.c_size_t, C.c_int], C.c_void_p),
        'vh_cuda_close': ([C.c_void_p], None),
        'vh_cuda_stream': ([C.c_void_p], C.c_size_t),
        'vh_cuda_sync': ([C.c_void_p], C.c_int),
        'vh_cuda_record': ([C.c_void_p], C.c_size_t),
        'vh_cuda_wait': ([C.c_void_p, C.c_size_t], C.c_int),
        'vh_cuda_elapsed': ([C.c_void_p, C.c_size_t, C.c_size_t, C.POINTER(C.c_float)], C.c_int),
        'vh_cuda_event_sync': ([C.c_void_p, C.c_size_t], C.c_int),
        'vh_cuda_event_free': ([C.c_void_p, C.c_size_t], None),
        'vh_cuda_download': ([C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t], C.c_int),
        'vh_cuda_pixels': ([C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_float, C.c_int, C.c_void_p], C.c_int),
        'vh_pixels_cpu': ([C.c_void_p, C.c_size_t, C.c_float, C.c_int, C.c_void_p], C.c_int),
        'vh_cuda_compile_probe': ([], C.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


class CudaEvent:
    def __init__(self, runtime):
        self.runtime, self.handle = runtime, 0
        runtime.check_open()
        self.handle = runtime.lib.vh_cuda_record(runtime.handle)
        if not self.handle:
            raise RuntimeError('native CUDA event record failed')
        runtime.events += 1

    def synchronize(self):
        self.runtime.check_open()
        if not self.handle or self.runtime.lib.vh_cuda_event_sync(self.runtime.handle, self.handle):
            raise RuntimeError('native CUDA event synchronize failed')

    def elapsed_time(self, end):
        self.runtime.check_open()
        if end.runtime is not self.runtime or not self.handle or not end.handle:
            raise ValueError('timing events must use the same runtime')
        ms = C.c_float()
        if self.runtime.lib.vh_cuda_elapsed(self.runtime.handle, self.handle, end.handle, C.byref(ms)):
            raise RuntimeError('native CUDA event timing failed')
        return ms.value

    def close(self):
        if self.handle:
            self.runtime.lib.vh_cuda_event_free(self.runtime.handle, self.handle)
            self.runtime.events -= 1
            self.handle = 0

    def __del__(self):
        self.close()


class CudaRuntime:
    def __init__(self, device=0, stream=None, *, lib=None):
        self.handle = None
        self.events = self.borrowers = 0
        if type(device) is not int or device < 0:
            raise ValueError('invalid CUDA device')
        self.device = device
        self.stream_owner = stream
        if stream is not None:
            stream_device = getattr(getattr(stream, 'device', None), 'index', device)
            if stream_device != device:
                raise ValueError('stream/device mismatch')
            pointer = getattr(stream, 'cuda_stream', stream)
            if type(pointer) is not int or pointer < 0:
                raise ValueError('invalid CUDA stream')
        else:
            pointer = 0
        self.lib = lib if lib is not None else library()
        self.handle = self.lib.vh_cuda_open(device, pointer, -1 if stream is None else 1)
        if not self.handle:
            raise RuntimeError('native CUDA context/stream unavailable')
        self.cuda_stream = self.lib.vh_cuda_stream(self.handle)

    def check_open(self):
        if not self.handle:
            raise RuntimeError('native CUDA runtime is closed')

    def synchronize(self):
        self.check_open()
        if self.lib.vh_cuda_sync(self.handle):
            raise RuntimeError('native CUDA stream synchronize failed')

    def record(self):
        return CudaEvent(self)

    def wait(self, event):
        self.check_open()
        if event.runtime.device != self.device or not event.handle:
            raise ValueError('invalid completion event/device')
        if self.lib.vh_cuda_wait(self.handle, event.handle):
            raise RuntimeError('native CUDA stream wait failed')

    def pixels(self, pointer, shape, *, background=.18, straight_alpha=False):
        self.check_open()
        if len(shape) != 3 or shape[-1] != 4 or not all(1 <= int(n) <= 8192 for n in shape[:2]):
            raise ValueError('expected bounded HWC RGBA')
        height, width, _ = shape
        output = np.empty((height, width, 4 if straight_alpha else 3), np.uint8)
        if self.lib.vh_cuda_pixels(self.handle, pointer, width, height, background,
                                   int(straight_alpha), output.ctypes.data):
            raise RuntimeError('native CUDA presentation conversion failed')
        return output

    def download(self, pointer, shape):
        self.check_open()
        if not shape or any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) > 2**28:
            raise ValueError('invalid download shape')
        output = np.empty(shape, np.float32)
        if self.lib.vh_cuda_download(self.handle, pointer, output.ctypes.data, output.nbytes):
            raise RuntimeError('native CUDA download failed')
        return output

    def close(self):
        if self.handle:
            if self.events or self.borrowers:
                raise RuntimeError('release native CUDA events and borrowers before close')
            self.lib.vh_cuda_close(self.handle)
            self.handle = None
            self.stream_owner = None

    def __del__(self):
        if self.handle and not self.events and not self.borrowers:
            self.close()
