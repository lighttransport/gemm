"""C++/CUDA Gaussian inference with no tensor framework or gsplat dependency."""
import ctypes as C
import time
import numpy as np
from ..avatar.cuda_runtime import CudaRuntime, ROOT
from .frame import FrameHandle


def library():
    path = ROOT / 'cuda/vhuman/libvhuman_splat.so'
    if not path.is_file():
        raise RuntimeError('Build native rendering with make -C cuda/vhuman libvhuman_splat.so')
    lib = C.CDLL(str(path))
    signatures = {
        'vhs_error': ([], C.c_char_p),
        'vhs_open': ([C.c_int, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_int,
                      C.c_void_p, C.c_void_p, C.c_void_p], C.c_void_p),
        'vhs_close': ([C.c_void_p], None),
        'vhs_deform': ([C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p, C.c_void_p], C.c_int),
        'vhs_render': ([C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int], C.c_size_t),
        'vhs_free_frame': ([C.c_void_p, C.c_size_t, C.c_size_t], None),
        'vhs_memory': ([C.c_void_p, C.c_int], C.c_size_t),
        'vhs_reset_peak': ([C.c_void_p], None),
        'vhs_compile_probe': ([], C.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = result
    return lib


class FrameBuffer:
    """Owned output allocation; keeping a frame never aliases a later render."""
    def __init__(self, renderer, pointer, shape):
        self.renderer, self.runtime = renderer, renderer.runtime
        self.pointer, self.shape = pointer, shape
        self.nbytes = int(np.prod(shape))*4
        renderer.live_frames += 1

    def data_ptr(self):
        if not self.pointer:
            raise RuntimeError('frame is closed')
        return self.pointer

    def numpy(self):
        return self.runtime.download(self.data_ptr(), self.shape)

    def close(self):
        if self.pointer:
            self.renderer.lib.vhs_free_frame(self.renderer.handle, self.pointer, self.nbytes)
            self.pointer = 0
            self.renderer.live_frames -= 1

    def __del__(self):
        self.close()


class NativeGaussianRenderer:
    def __init__(self, avatar, triangles, device='cuda:0', *, runtime=None):
        self.handle = self.runtime = None
        self.live_frames = 0
        avatar.validate(triangles)
        if device != 'cuda' and not (isinstance(device, str) and device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError('native renderer requires a CUDA device')
        index = 0 if device == 'cuda' else int(device[5:])
        if runtime is not None and runtime.device != index:
            raise ValueError('renderer/runtime device mismatch')
        self.lib = library()
        self.runtime = runtime if runtime is not None else CudaRuntime(index)
        self.owns_runtime = runtime is None
        self.runtime.check_open()
        self.runtime.borrowers += 1
        try:
            a = avatar.arrays
            self.n, self.controls = len(a['triangle']), len(avatar.metadata['control_names'])
            triangles = np.asarray(triangles)
            if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu' or (triangles < 0).any():
                raise ValueError('invalid triangles')
            self.vertex_count = int(triangles.max()) + 1
            if not 3 <= self.vertex_count <= 2000000:
                raise ValueError('vertex count outside native renderer bounds')
            attachments = np.ascontiguousarray(triangles[a['triangle']], np.int32)
            assets = np.ascontiguousarray(np.column_stack((a['barycentric'], a['normal_offset'],
                a['covariance_local'].reshape(self.n, 9), a['opacity'], a['rgb'],
                a['color_basis'].reshape(self.n, 24))), np.float32)
            expression = np.ascontiguousarray(a['expression_matrix'], np.float32)
            trace = int(avatar.metadata.get('covariance_policy', 'eigen-v1') == 'trace-v1')
            self.handle = self.lib.vhs_open(index, self.runtime.cuda_stream, self.n, self.vertex_count,
                self.controls, trace, attachments.ctypes.data, assets.ctypes.data, expression.ctypes.data)
            if not self.handle:
                raise RuntimeError(self.lib.vhs_error().decode())
        except Exception:
            self.runtime.borrowers -= 1
            if self.owns_runtime:
                self.runtime.close()
            self.runtime = None
            raise

    def __del__(self):
        if self.runtime is not None:
            try:
                self.close()
            except RuntimeError:
                pass  # Explicit close reports outstanding externally owned events.

    def _inputs(self, vertices, controls):
        if not self.handle:
            raise RuntimeError('renderer is closed')
        if hasattr(vertices, 'runtime') and hasattr(vertices, 'pointer'):
            if vertices.runtime is not self.runtime:
                raise ValueError('native rig and renderer must share their runtime')
            if len(vertices.shape) != 2 or vertices.shape[1] != 3 or vertices.shape[0] < self.vertex_count:
                raise ValueError('invalid device vertex shape')
            pointer, host = vertices.pointer, None
        else:
            host = np.ascontiguousarray(vertices, np.float32)
            if host.ndim != 2 or host.shape[1] != 3 or len(host) < self.vertex_count or not np.isfinite(host).all():
                raise ValueError('invalid posed vertices')
            pointer = 0
        controls = np.zeros(self.controls, np.float32) if controls is None else np.ascontiguousarray(controls, np.float32)
        if controls.shape != (self.controls,) or not np.isfinite(controls).all():
            raise ValueError('invalid controls')
        return pointer, host, controls

    def deform(self, vertices, controls=None):
        pointer, host, controls = self._inputs(vertices, controls)
        result = np.empty((self.n, 16), np.float32)
        if self.lib.vhs_deform(self.handle, pointer, None if host is None else host.ctypes.data,
                               controls.ctypes.data, result.ctypes.data):
            raise RuntimeError(self.lib.vhs_error().decode())
        return result[:, :3], result[:, 3:12].reshape(-1, 3, 3), result[:, 12], result[:, 13:]

    def render(self, vertices, view, intrinsics, size=(512, 512), controls=None, sample_position=0):
        pointer, host, controls = self._inputs(vertices, controls)
        view, intrinsics = np.asarray(view, np.float32), np.asarray(intrinsics, np.float32)
        if view.shape != (4, 4) or intrinsics.shape != (3, 3) or not np.isfinite(view).all() or not np.isfinite(intrinsics).all():
            raise ValueError('invalid camera matrices')
        if not np.allclose(view[3], [0, 0, 0, 1]) or not np.allclose(intrinsics[2], [0, 0, 1]) or intrinsics[0, 1] != 0 or intrinsics[1, 0] != 0:
            raise ValueError('expected affine view and pinhole intrinsics without skew')
        if len(size) != 2 or any(type(v) is not int or not 1 <= v <= 4096 for v in size):
            raise ValueError('invalid output size')
        camera = np.concatenate((view.ravel(), intrinsics.ravel()))
        width, height = size
        result = self.lib.vhs_render(self.handle, pointer, None if host is None else host.ctypes.data,
            controls.ctypes.data, camera.ctypes.data, width, height)
        if not result:
            raise RuntimeError(self.lib.vhs_error().decode())
        rgba = FrameBuffer(self, result, (height, width, 4))
        return FrameHandle(rgba, sample_position, self.runtime.record(), time.monotonic_ns(), self.runtime)

    def memory_stats(self):
        return {'native_allocated_mib': self.lib.vhs_memory(self.handle, 0)/2**20,
                'allocated_peak_mib': self.lib.vhs_memory(self.handle, 1)/2**20,
                'reserved_peak_mib': self.lib.vhs_memory(self.handle, 1)/2**20}

    def reset_memory_stats(self):
        self.lib.vhs_reset_peak(self.handle)

    def close(self):
        if self.runtime is not None:
            if self.live_frames or (self.owns_runtime and (self.runtime.events or self.runtime.borrowers != 1)):
                raise RuntimeError('release frames, events and native rig before renderer close')
            self.runtime.synchronize()
            if self.handle:
                self.lib.vhs_close(self.handle)
                self.handle = None
            self.runtime.borrowers -= 1
            if self.owns_runtime:
                self.runtime.close()
            self.runtime = None
