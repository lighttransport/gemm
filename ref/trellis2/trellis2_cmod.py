"""ctypes bindings for the TRELLIS.2 CPU/CUDA runners.

Loads the runner shared libraries (built via `make libcuda_trellis2.so` /
`make libcpu_trellis2.so`) and drives them in-process, instead of spawning the
test binaries as subprocesses. The headline difference from the subprocess path
is that a runner holds its weights resident across predict() calls.

Mirrors the repo's existing ctypes pattern (rdna4/trellis2/shims/cumesh_xatlas.py).

These classes are NOT thread-safe: each wraps a single GPU/CPU context with
mutated internal scratch. The caller (trellis2_server.py) serializes predict()
with a per-backend lock.
"""

import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

DEFAULT_CUDA_SO = os.path.join(_REPO_ROOT, "cuda", "trellis2", "libcuda_trellis2.so")
DEFAULT_CPU_SO = os.path.join(_REPO_ROOT, "cpu", "trellis2", "libcpu_trellis2.so")

_OCC_N = 64 * 64 * 64


def _enc(s):
    return s.encode("utf-8") if isinstance(s, str) else s


def _occ_from_ptr(lib_free_buffer, ptr):
    """Copy a float[64^3] returned by a runner into an ndarray, then free it
    with the .so's own allocator (avoids cross-allocator free)."""
    if not ptr:
        raise RuntimeError("runner predict() returned NULL")
    occ = np.ctypeslib.as_array(ptr, shape=(_OCC_N,)).copy().reshape(64, 64, 64)
    lib_free_buffer(ptr)
    return occ


class CudaCModRunner:
    """In-process CUDA TRELLIS.2 runner via libcuda_trellis2.so.

    predict(rgb_hw3_uint8, steps, cfg, seed) -> np.float32[64,64,64] occupancy.
    """

    def __init__(self, so_path=None, dinov3=None, stage1=None, decoder=None,
                 device_id=0, verbose=0):
        self._h = None
        self._lib = ctypes.CDLL(so_path or DEFAULT_CUDA_SO)
        lib = self._lib

        lib.cuda_trellis2_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.cuda_trellis2_init.restype = ctypes.c_void_p
        lib.cuda_trellis2_load_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.cuda_trellis2_load_weights.restype = ctypes.c_int
        lib.cuda_trellis2_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_float, ctypes.c_uint32]
        lib.cuda_trellis2_predict.restype = ctypes.POINTER(ctypes.c_float)
        lib.cuda_trellis2_free.argtypes = [ctypes.c_void_p]
        lib.cuda_trellis2_free.restype = None
        lib.cuda_trellis2_free_buffer.argtypes = [ctypes.c_void_p]
        lib.cuda_trellis2_free_buffer.restype = None

        self._h = lib.cuda_trellis2_init(int(device_id), int(verbose))
        if not self._h:
            raise RuntimeError("cuda_trellis2_init failed")
        if dinov3 and stage1 and decoder:
            rc = lib.cuda_trellis2_load_weights(
                self._h, _enc(dinov3), _enc(stage1), _enc(decoder))
            if rc != 0:
                raise RuntimeError(f"cuda_trellis2_load_weights failed (rc={rc})")

    def predict(self, rgb, steps=12, cfg=7.5, seed=42):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB HxWx3 uint8, got shape {rgb.shape}")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        ptr = self._lib.cuda_trellis2_predict(
            self._h, rgb.ctypes.data, w, h,
            int(steps), ctypes.c_float(float(cfg)), ctypes.c_uint32(int(seed)))
        return _occ_from_ptr(self._lib.cuda_trellis2_free_buffer, ptr)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.cuda_trellis2_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CpuCModRunner:
    """In-process CPU TRELLIS.2 runner via libcpu_trellis2.so.

    predict(rgb_hw3_uint8, steps, cfg, seed) -> np.float32[64,64,64].
    The CPU sampler's steps/cfg are fixed by the checkpoint, so those args are
    accepted for a uniform interface but ignored.
    """

    def __init__(self, so_path=None, dinov3=None, stage1=None, decoder=None,
                 n_threads=0, verbose=0):
        self._h = None
        self._lib = ctypes.CDLL(so_path or DEFAULT_CPU_SO)
        lib = self._lib

        lib.cpu_trellis2_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.cpu_trellis2_init.restype = ctypes.c_void_p
        lib.cpu_trellis2_load_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.cpu_trellis2_load_weights.restype = ctypes.c_int
        lib.cpu_trellis2_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint32]
        lib.cpu_trellis2_predict.restype = ctypes.POINTER(ctypes.c_float)
        lib.cpu_trellis2_free.argtypes = [ctypes.c_void_p]
        lib.cpu_trellis2_free.restype = None
        lib.cpu_trellis2_free_buffer.argtypes = [ctypes.c_void_p]
        lib.cpu_trellis2_free_buffer.restype = None

        if n_threads <= 0:
            n_threads = min(8, os.cpu_count() or 4)
        self._h = lib.cpu_trellis2_init(int(n_threads), int(verbose))
        if not self._h:
            raise RuntimeError("cpu_trellis2_init failed")
        if dinov3 and stage1 and decoder:
            rc = lib.cpu_trellis2_load_weights(
                self._h, _enc(dinov3), _enc(stage1), _enc(decoder))
            if rc != 0:
                raise RuntimeError(f"cpu_trellis2_load_weights failed (rc={rc})")

    def predict(self, rgb, steps=12, cfg=7.5, seed=42):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB HxWx3 uint8, got shape {rgb.shape}")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        ptr = self._lib.cpu_trellis2_predict(
            self._h, rgb.ctypes.data, w, h, ctypes.c_uint32(int(seed)))
        return _occ_from_ptr(self._lib.cpu_trellis2_free_buffer, ptr)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.cpu_trellis2_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
