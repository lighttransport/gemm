"""gsplat adapter; allocations persist and frame completion is an explicit event."""
from dataclasses import dataclass
import time


@dataclass
class FrameHandle:
    rgba: object
    sample_position: int
    ready: object
    submitted_ns: int


class GaussianRenderer:
    def __init__(self, avatar, triangles, device="cuda:0"):
        import torch
        from gsplat import rasterization
        avatar.validate(triangles)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run the runtime doctor")
        self.torch, self.rasterization = torch, rasterization
        self.device, self.avatar = device, avatar
        self.triangles = torch.as_tensor(triangles, device=device, dtype=torch.long)
        self.arrays = {k: torch.as_tensor(v, device=device, dtype=torch.long if k == "triangle" else torch.float32)
                       for k, v in avatar.arrays.items() if k != "component"}

    def deform(self, vertices, controls=None):
        from ..avatar.geometry import deform_torch
        return deform_torch(vertices, self.triangles, self.arrays, controls,
                            self.avatar.metadata.get("covariance_policy", "eigen-v1"))

    def render(self, vertices, view, intrinsics, size=(512, 512), controls=None, sample_position=0):
        t = self.torch
        with t.inference_mode():
            centers, cov, opacity, rgb = self.deform(vertices, controls)
            image, alpha, _ = self.rasterization(
                means=centers, quats=None, scales=None, covars=cov, opacities=opacity,
                colors=rgb, viewmats=view[None], Ks=intrinsics[None], width=size[0], height=size[1],
                packed=True, render_mode="RGB")
            rgba = t.cat((image[0], alpha[0]), dim=-1)
            ready = t.cuda.Event()
            ready.record(t.cuda.current_stream(self.device))
        return FrameHandle(rgba, sample_position, ready, time.monotonic_ns())
