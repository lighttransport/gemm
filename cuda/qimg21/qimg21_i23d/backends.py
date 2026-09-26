"""Generation backends behind one request type.

Every Image-to-3D operation reduces to GenRequest -> Backend.generate() ->
an RGBA PNG on disk. Backends differ only in how they run the model:

- NativeBackend (native.py): the CUDA runner via cuda/qimg21/native_generate.py,
  single reference image, resident processes when a demo server provides them.
- TorchBackend (torch_backend.py): the PyTorch QwenImage21Pipeline loaded once
  in this process; up to 10 reference images.
- MockBackend (here): deterministic synthetic output for tests.

Semantics every backend implements the same way:
- references are condition images (the model's native image conditioning);
  generated views are never added to them by the library;
- strength < 1 is SDEdit: the init image is encoded at the output size and
  the flow restarts at step K = round((1 - strength) * steps) with the seed's
  noise; strength == 1 is a fresh generation conditioned on the references;
- a mask (255 = may change) is enforced in the denoise loop -- outside it the
  latents are reset to the renoised init image at every step -- and after
  decoding the original pixels are pasted back outside it; the mask can also be
  shown to the model as an extra reference image.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from . import imageops


class BackendError(RuntimeError):
    pass


@dataclass
class GenRequest:
    prompt: str
    out: Path
    width: int = 512
    height: int = 512
    steps: int = 20
    seed: int = 0
    references: tuple = ()
    negative_prompt: str | None = None
    true_cfg_scale: float = 4.0
    init_image: Path | None = None
    strength: float = 1.0
    mask: Path | None = None
    mask_as_reference: bool = True
    tags: dict = field(default_factory=dict)

    def validate(self, max_references: int) -> "GenRequest":
        if not self.prompt or not self.prompt.strip():
            raise BackendError("prompt is empty")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value < 256 or value > 2048 or value % 32:
                raise BackendError(f"{name} must be a multiple of 32 in [256, 2048], got {value}")
        if not 1 <= self.steps <= 100:
            raise BackendError(f"steps must be in [1, 100], got {self.steps}")
        if self.seed < 0:
            raise BackendError(f"seed must be non-negative, got {self.seed}")
        if not 0.0 < self.strength <= 1.0:
            raise BackendError(f"strength must be in (0, 1], got {self.strength}")
        if self.strength < 1.0 and self.init_image is None:
            raise BackendError("strength below 1 needs an init image to start from")
        if self.mask is not None and self.init_image is None:
            raise BackendError("a mask needs the init image it applies to")
        refs = list(self.references) + ([self.mask] if self.mask is not None and self.mask_as_reference else [])
        if len(refs) > max_references:
            raise BackendError(f"{len(refs)} reference images (including the mask) but this backend takes "
                               f"at most {max_references}")
        for path in list(self.references) + [p for p in (self.init_image, self.mask) if p is not None]:
            if not Path(path).is_file():
                raise BackendError(f"missing input image {path}")
        return self

    def restart_step(self) -> int:
        """The first step an SDEdit run executes: (1 - strength) of the
        schedule is taken as done, and at least one step always runs."""
        return min(self.steps - 1, max(0, round((1.0 - self.strength) * self.steps)))

    def fingerprint(self) -> str:
        """Identifies the request for determinism checks and caching."""
        payload = asdict(self)
        payload.pop("out"); payload.pop("tags")
        for key in ("references",):
            payload[key] = [imageops.sha256_file(p) for p in self.references]
        for key in ("init_image", "mask"):
            payload[key] = imageops.sha256_file(payload[key]) if payload[key] else None
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class GenResult:
    path: Path
    seconds: float
    backend: str
    details: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str
    max_references: int

    def generate(self, request: GenRequest) -> GenResult: ...

    def close(self) -> None: ...


class MockBackend:
    """Deterministic stand-in: an antialiased ellipse whose colour and shape
    follow the request fingerprint, on a transparent background. It honours
    SDEdit/mask semantics at the pixel level (outside the mask the init image
    is kept) so the library's bookkeeping can be tested without a GPU."""

    name = "mock"

    def __init__(self, max_references: int = 10):
        self.max_references = max_references
        self.calls: list[GenRequest] = []

    def generate(self, request: GenRequest) -> GenResult:
        request.validate(self.max_references)
        start = time.perf_counter()
        self.calls.append(request)
        digest = hashlib.sha256(request.fingerprint().encode()).digest()
        w, h = request.width, request.height
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rx, ry = w * (0.2 + digest[0] / 255 * 0.15), h * (0.25 + digest[1] / 255 * 0.15)
        dist = np.sqrt(((xx + 0.5 - w / 2) / rx) ** 2 + ((yy + 0.5 - h / 2) / ry) ** 2)
        alpha = np.clip((1.0 - dist) * min(rx, ry), 0.0, 1.0)
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 0], out[..., 1], out[..., 2] = digest[2], digest[3], digest[4]
        out[..., 3] = np.rint(alpha * 255).astype(np.uint8)
        if request.init_image is not None:
            init = imageops.load_rgba(request.init_image)
            if init.shape[:2] != (h, w):
                init = imageops.resize_rgba(init, w, h)
            if request.mask is not None:
                mask = imageops.make_mask((w, h), image=request.mask, resize=True)
                out = imageops.paste_outside(init, out, mask)
            elif request.strength < 1.0:
                out = imageops.paste_outside(init, out, np.full((h, w), round(request.strength * 255), np.uint8))
        imageops.save_png(out, request.out)
        return GenResult(Path(request.out), time.perf_counter() - start, self.name,
                         {"fingerprint": request.fingerprint()})

    def close(self) -> None:
        pass


def select_backend(name: str = "auto", *, references: int = 1, **options) -> Backend:
    """native for up to one reference when the CUDA runner is built, torch
    otherwise; "mock" for tests."""
    if name == "mock":
        return MockBackend()
    if name in ("auto", "native"):
        from .native import NativeBackend
        if NativeBackend.available() and references <= NativeBackend.max_references:
            return NativeBackend(**{k: v for k, v in options.items() if k in NativeBackend.OPTIONS})
        if name == "native":
            raise BackendError("the native backend needs the built CUDA runner and at most "
                               f"{NativeBackend.max_references} reference image; use --backend torch")
    if name in ("auto", "torch"):
        from .torch_backend import TorchBackend
        return TorchBackend(**{k: v for k, v in options.items() if k in TorchBackend.OPTIONS})
    raise BackendError(f"unknown backend {name!r}; use auto, native, torch or mock")
