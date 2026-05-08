"""Monkey-patch trellis2.modules.sparse.linear.SparseLinear.forward.

Replaces the chunked-F.linear hipBLASLt workaround with a direct HIP F32 GEMM
when the shape fits the kernel envelope; otherwise (BF16/F16, or shapes
outside coverage) falls back to F.linear in chunks safely under the bug
boundary (2**18 rows).

USE_RDNA4_LINEAR_CHUNK env var:
    "1" (default) — install the new HIP F32 kernel + chunked F.linear fallback.
    "0"           — leave SparseLinear.forward untouched (raw hipBLASLt path).
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from . import linear_f32 as lf32


_LOGGED_FALLBACK = False
_FALLBACK_CHUNK = 262144  # < 2**19 fp32 hipBLASLt boundary; halves overhead


def _patched_forward(self, input):
    x = input.feats
    w = self.weight
    b = self.bias
    if lf32.can_handle(x, w):
        out = lf32.linear_f32(x, w, b)
        return input.replace(out)
    # Fallback: chunked F.linear so we still avoid the hipBLASLt M>2^19 bug.
    global _LOGGED_FALLBACK
    if not _LOGGED_FALLBACK:
        _LOGGED_FALLBACK = True
        print(f'[linear_patch] HIP kernel uncovered shape '
              f'(K={x.shape[-1]} N={w.shape[0]} dtype={x.dtype}); '
              f'falling back to chunked F.linear')
    M = x.shape[0]
    if M <= _FALLBACK_CHUNK:
        return input.replace(F.linear(x, w, b))
    out = torch.empty(M, w.shape[0], device=x.device, dtype=x.dtype)
    for s in range(0, M, _FALLBACK_CHUNK):
        out[s:s + _FALLBACK_CHUNK] = F.linear(x[s:s + _FALLBACK_CHUNK].contiguous(), w, b)
    return input.replace(out)


def install():
    """Patch SparseLinear.forward → HIP F32 GEMM (with safe F.linear fallback)."""
    from trellis2.modules.sparse.linear import SparseLinear
    SparseLinear.forward = _patched_forward
    print('[linear_patch] installed: SparseLinear.forward -> rdna4 HIP F32 GEMM '
          '(fallback: chunked F.linear)')
