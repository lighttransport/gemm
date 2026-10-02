"""Default inference renderer; no Torch or gsplat imports.

Training uses its differentiable backend directly. The optional offline oracle
is explicitly available from gsplat_reference.
"""
from .native import NativeGaussianRenderer as GaussianRenderer
from .frame import FrameHandle

__all__ = ['GaussianRenderer', 'FrameHandle']
