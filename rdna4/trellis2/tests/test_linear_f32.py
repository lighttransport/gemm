"""Correctness + determinism test for kernels/linear_f32 on RDNA4.

Compares against torch.nn.functional.linear with a chunked-along-M reference
(to dodge the hipBLASLt M>2^19 fp32 bug). Asserts:
  * max|d| <= 1e-4 vs chunked F.linear at the bug shape (M=1.5M, K=64, N=6).
  * bit-identical output across 4 trials with the same input.
  * Coverage across (K,N) ∈ {64,128,256} × {6,8,16}.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels import linear_f32 as lf32


def _ref_chunked(x, w, b, chunk=131072):
    M = x.shape[0]
    out = torch.empty(M, w.shape[0], device=x.device, dtype=x.dtype)
    for s in range(0, M, chunk):
        out[s:s + chunk] = torch.nn.functional.linear(x[s:s + chunk].contiguous(), w, b)
    return out


@torch.no_grad()
def case(M, K, N, *, atol=1e-4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = (torch.rand(N, K, device='cuda', dtype=torch.float32) - 0.5) * 0.5
    b = (torch.rand(N, device='cuda', dtype=torch.float32) - 0.5) * 0.1

    y = lf32.linear_f32(x, w, b)
    ref = _ref_chunked(x, w, b)
    diff = (y - ref).abs().max().item()
    assert torch.isfinite(y).all(), f'NaN/Inf at M={M},K={K},N={N}'
    assert diff < atol, f'M={M},K={K},N={N}: max|d|={diff:.2e} > {atol:.0e}'
    print(f'[ok] M={M:>8d} K={K:>3d} N={N:>3d}  max|d|={diff:.2e}')

    # Determinism: 4 trials must be bit-identical
    ys = [lf32.linear_f32(x, w, b) for _ in range(4)]
    for i in range(1, 4):
        assert torch.equal(ys[0], ys[i]), f'non-deterministic at M={M},K={K},N={N}'
    print(f'[ok] deterministic across 4 trials')


def main():
    for M, K, N in [
        (1_500_482, 64, 6),  # the bug shape from rdna4/hipblaslt-issue.md
        (524_289,   64, 6),
        (700_000,  128, 8),
        (524_289,  256, 16),
    ]:
        case(M, K, N)
    print('all linear_f32 cases passed')


if __name__ == '__main__':
    main()
