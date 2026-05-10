"""Sweep the hipBLASLt tall+skinny bug across (M, K, N, dtype).

Goal: determine whether the bug is specific to K=64, N=6, fp32, or whether
any small-N matmul past M=2^19 is affected. Informs how broad the workaround
needs to be (currently only patches SparseLinear).

For each shape, runs one F.linear and compares against a chunked reference.
Reports max|diff| and inf/nan counts.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def probe(M, K, N, dtype):
    dev = 'cuda'
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=dtype)
    W = (torch.rand(N, K, device=dev, dtype=dtype) - 0.5) * 0.68
    b = (torch.rand(N, device=dev, dtype=dtype) - 0.5) * 0.2

    out_single = F.linear(x, W, b)
    out_chunk = torch.empty(M, N, device=dev, dtype=dtype)
    for s in range(0, M, 65536):
        out_chunk[s:s + 65536] = F.linear(x[s:s + 65536], W, b)

    diff = (out_single.float() - out_chunk.float()).abs()
    n_inf = int(torch.isinf(out_single).sum())
    n_nan = int(torch.isnan(out_single).sum())
    max_d = float(diff.max())
    bad = (diff > 1e-2).any(dim=1)
    n_bad = int(bad.sum())
    first_bad = int(torch.where(bad)[0][0]) if n_bad else -1
    return max_d, n_inf, n_nan, n_bad, first_bad


def main():
    print('M       K    N    dtype    max|d|         inf      nan      n_bad    first_bad_row')
    print('-' * 95)
    Ms = [524287, 524288, 524289, 700000, 1_500_482]
    cases = [
        # K, N, dtype
        (64, 6, torch.float32),
        (64, 6, torch.float16),
        (64, 6, torch.bfloat16),
        (64, 8, torch.float32),
        (64, 16, torch.float32),
        (64, 64, torch.float32),
        (32, 6, torch.float32),
        (128, 6, torch.float32),
        (256, 6, torch.float32),
    ]
    for K, N, dt in cases:
        for M in Ms:
            max_d, n_inf, n_nan, n_bad, first = probe(M, K, N, dt)
            tag = ' <<< BAD' if (n_bad > 0 or n_inf > 0 or n_nan > 0) else ''
            print(f'{M:>8d} {K:>4d} {N:>4d} {str(dt).split(".")[-1]:>8s}  '
                  f'{max_d:>12.4g}  {n_inf:>7d}  {n_nan:>7d}  {n_bad:>7d}   {first:>10d}{tag}')


if __name__ == '__main__':
    main()
