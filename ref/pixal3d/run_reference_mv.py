"""Run pinned multiview inference without loading RMBG for RGBA-only inputs.

The upstream entry point checks every view's alpha before calling its matting
model, but constructs RMBG eagerly. This launcher replaces only that eager
constructor. An opaque/RGB view still fails instead of silently changing the
reference preprocessing contract.
"""
from pathlib import Path
import functools
import math
import runpy
import sys
import torch

UPSTREAM = Path(__file__).with_name("upstream")
sys.path.insert(0, str(UPSTREAM))

# The official Torch 2.7/cu128 NATTEN wheel predates sm_120 and its CUTLASS
# binary has no Blackwell kernel image. FlexAttention implements the same
# neighborhood operation through PyTorch/Triton and runs on the 5060 Ti.
import natten.functional
import natten

_natten_na2d = natten.functional.na2d


def _chunked_na2d(q, k, v, kernel_size, dilation, scale):
    """Exact inference-only NA for NAF's unequal QK/V head widths."""
    kh, kw = ((kernel_size, kernel_size) if isinstance(kernel_size, int)
              else kernel_size)
    dh, dw = ((dilation, dilation) if isinstance(dilation, int) else dilation)
    height, width = q.shape[1:3]
    row_base = torch.arange(height, device=q.device)
    col_base = torch.arange(width, device=q.device)
    row_start = (row_base - (kh // 2) * dh).clamp(0, height - 1 - (kh - 1) * dh)
    col_start = (col_base - (kw // 2) * dw).clamp(0, width - 1 - (kw - 1) * dw)
    outputs = []
    for first in range(0, height, 8):
        last = min(first + 8, height)
        qc = q[:, first:last]
        logits = []
        for iy in range(kh):
            rows = row_start[first:last] + iy * dh
            for ix in range(kw):
                cols = col_start + ix * dw
                logits.append((qc * k[:, rows[:, None], cols[None, :]]).sum(-1) * scale)
        weights = torch.softmax(torch.stack(logits, dim=-1), dim=-1)
        out = torch.zeros((*qc.shape[:-1], v.shape[-1]), device=v.device, dtype=v.dtype)
        offset = 0
        for iy in range(kh):
            rows = row_start[first:last] + iy * dh
            for ix in range(kw):
                cols = col_start + ix * dw
                out.add_(v[:, rows[:, None], cols[None, :]] * weights[..., offset, None])
                offset += 1
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@functools.wraps(_natten_na2d)
def _na2d_blackwell(*args, **kwargs):
    if kwargs.get("backend") == "cutlass-fna":
        if len(args) >= 3 and args[0].shape[-1] != args[2].shape[-1]:
            q, k, v = args[:3]
            return _chunked_na2d(q, k, v, kwargs["kernel_size"],
                                 kwargs.get("dilation", 1),
                                 kwargs.get("scale", 1.0 / math.sqrt(q.shape[-1])))
        kwargs["backend"] = "flex-fna"
    return _natten_na2d(*args, **kwargs)


natten.functional.na2d = _na2d_blackwell
natten.na2d = _na2d_blackwell

from pixal3d.pipelines import rembg


class AlphaOnlyRmbg:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        raise RuntimeError(
            "reference view lacks useful alpha; install the authorized RMBG-2.0 weights")

    def to(self, device):
        return self

    def cpu(self):
        return self


rembg.BiRefNet = AlphaOnlyRmbg


def main():
    runpy.run_path(str(UPSTREAM / "inference_mv.py"), run_name="__main__")


if __name__ == "__main__":
    main()
