"""Isolated gsplat compatibility backend; callers use native CUDA IO/NumPy."""
import time
from ..avatar.cuda_runtime import CudaRuntime
from .frame import FrameHandle


class GaussianRenderer:
    """Optional offline Torch/gsplat reference backend."""
    def __init__(self, avatar, triangles, device='cuda:0', *, runtime=None):
        import torch
        from gsplat import rasterization
        avatar.validate(triangles)
        selected = torch.device(device)
        if selected.type != 'cuda' or torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('gsplat compatibility backend requires CUDA PyTorch')
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if runtime is not None and runtime.device != index:
            raise ValueError('renderer/runtime device mismatch')
        self.runtime = runtime if runtime is not None else CudaRuntime(index)
        self.owns_runtime = runtime is None
        self.runtime.check_open()
        self.runtime.borrowers += 1
        self.torch, self.rasterization = torch, rasterization
        self.device, self.avatar = device, avatar
        try:
            self.stream = torch.cuda.ExternalStream(self.runtime.cuda_stream, device=device)
            with torch.cuda.stream(self.stream):
                self.triangles = torch.as_tensor(triangles, device=device, dtype=torch.long)
                self.arrays = {k: torch.as_tensor(v, device=device, dtype=torch.long if k == 'triangle' else torch.float32)
                               for k, v in avatar.arrays.items() if k != 'component'}
            self.runtime.synchronize()
        except Exception:
            self.runtime.borrowers -= 1
            if self.owns_runtime:
                self.runtime.close()
            raise

    def _tensor(self, value):
        if hasattr(value, '__dlpack__') and hasattr(value, 'runtime'):
            if value.runtime.device != self.runtime.device:
                raise ValueError('vertex device mismatch')
            if value.runtime.cuda_stream != self.runtime.cuda_stream:
                raise ValueError('native rig and renderer must share their stream')
            return self.torch.from_dlpack(value)
        return self.torch.as_tensor(value, device=self.device)

    def deform(self, vertices, controls=None):
        from ..avatar.geometry import deform_torch
        caller = self.torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        with self.torch.cuda.stream(self.stream):
            vertices = self._tensor(vertices)
            controls = self._tensor(controls) if controls is not None else None
            result = deform_torch(vertices, self.triangles, self.arrays, controls,
                                  self.avatar.metadata.get('covariance_policy', 'eigen-v1'))
        caller.wait_stream(self.stream)
        return result

    def render(self, vertices, view, intrinsics, size=(512, 512), controls=None, sample_position=0):
        t = self.torch
        # Compatibility callers may have produced Torch inputs on another stream.
        self.stream.wait_stream(t.cuda.current_stream(self.device))
        with t.inference_mode(), t.cuda.stream(self.stream):
            centers, cov, opacity, rgb = self.deform(vertices, controls)
            view, intrinsics = self._tensor(view), self._tensor(intrinsics)
            image, alpha, _ = self.rasterization(
                means=centers, quats=None, scales=None, covars=cov, opacities=opacity,
                colors=rgb, viewmats=view[None], Ks=intrinsics[None], width=size[0], height=size[1],
                packed=True, render_mode='RGB')
            rgba = t.cat((image[0], alpha[0]), dim=-1)
            ready = self.runtime.record()
        return FrameHandle(rgba, sample_position, ready, time.monotonic_ns(), self.runtime)

    def memory_stats(self):
        return {'torch_allocated_mib': self.torch.cuda.memory_allocated(self.device) / 2**20,
                'allocated_peak_mib': self.torch.cuda.max_memory_allocated(self.device) / 2**20,
                'reserved_peak_mib': self.torch.cuda.max_memory_reserved(self.device) / 2**20}

    def reset_memory_stats(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def close(self):
        if self.runtime is not None:
            if self.owns_runtime and (self.runtime.events or self.runtime.borrowers != 1):
                raise RuntimeError('release renderer frames and native rig before close')
            self.runtime.synchronize()
            self.arrays = None
            self.triangles = None
            self.runtime.borrowers -= 1
            if self.owns_runtime:
                self.runtime.close()
            self.runtime = None
