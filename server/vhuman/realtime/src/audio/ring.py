"""ctypes bridge; Python is the producer/test consumer, never the device callback."""
import ctypes as C
from pathlib import Path
import subprocess
import numpy as np
from ..pipeline.protocol import integer


def build(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                    "-fPIC", "-shared", str(Path(__file__).with_name("pcm_ring.c")), "-o", str(path)], check=True)
    return path


class PcmRing:
    def __init__(self, library, capacity=48000):
        integer(capacity, "capacity")
        self.lib = C.CDLL(str(Path(library).resolve()))
        pointer = C.POINTER(C.c_float)
        self.lib.vh_pcm_create.argtypes = [C.c_uint64]
        self.lib.vh_pcm_create.restype = C.c_void_p
        for name in ("free", "reset", "depth"):
            getattr(self.lib, "vh_pcm_" + name).argtypes = [C.c_void_p]
        self.lib.vh_pcm_depth.restype = C.c_uint64
        for name in ("write", "read"):
            getattr(self.lib, "vh_pcm_" + name).argtypes = [C.c_void_p, pointer, C.c_uint64]
        self.lib.vh_pcm_read.restype = C.c_uint64
        self.lib.vh_pcm_write.restype = C.c_int
        self.handle = self.lib.vh_pcm_create(capacity)
        if not self.handle:
            raise RuntimeError("PCM queue allocation or lock-free atomics unavailable")
        self.capacity = capacity

    @property
    def depth(self):
        return self.lib.vh_pcm_depth(self.handle)

    def write(self, pcm):
        x = np.ascontiguousarray(pcm, np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("expected finite mono PCM")
        if not self.lib.vh_pcm_write(self.handle, x.ctypes.data_as(C.POINTER(C.c_float)), len(x)):
            raise BufferError("PCM full; retry after consumer progress")

    def read(self, count):
        integer(count, "count")
        if count > self.capacity:
            raise ValueError("callback larger than queue capacity")
        out = np.empty(count, np.float32)
        n = self.lib.vh_pcm_read(self.handle, out.ctypes.data_as(C.POINTER(C.c_float)), count)
        return out, n

    def reset(self):
        """Caller must stop producer and callback first."""
        self.lib.vh_pcm_reset(self.handle)

    def close(self):
        if self.handle:
            self.lib.vh_pcm_free(self.handle)
            self.handle = None
