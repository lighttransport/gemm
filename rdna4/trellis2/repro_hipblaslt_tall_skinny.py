"""Minimal standalone repro of the ROCm hipBLASLt tall+skinny matmul bug
that breaks TRELLIS-2 tex_slat_decoder on RDNA4 / gfx1201.

Bug: torch.nn.functional.linear (i.e. xW^T + b) on shapes around
     [M=1.5M, K=64] x [N=6, K=64] in fp32 returns ±1e18 garbage from
     clean inputs (|x|<=4, |W|<=0.4).
Fix: chunking the call along M into 65536-row chunks gives the correct
     result. Same dtype, same data, same call — just split.

Repro from:
    /mnt/disk1/work/gemm/trellis2/rdna4/trellis2/.venv/bin/python \
        repro_hipblaslt_tall_skinny.py

Tested: ROCm 7.2.2, PyTorch 2.11.0+rocm7.2.2, gfx1201 (RX 9070 XT).
"""
import torch
import torch.nn.functional as F


def main():
    assert torch.cuda.is_available(), 'need ROCm'
    dev = 'cuda'
    print('torch.version.hip:', getattr(torch.version, 'hip', None))
    print('device:', torch.cuda.get_device_name(0))

    M, K, N = 1_500_482, 64, 6
    torch.manual_seed(0)
    # Match the offending input distribution: x post-LayerNorm (std=1, range~[-3.5,3.5]),
    # W from xavier_uniform on a 6x64 SparseLinear (range~[-0.34, 0.33]).
    x = torch.randn(M, K, device=dev, dtype=torch.float32)
    W = (torch.rand(N, K, device=dev, dtype=torch.float32) - 0.5) * 2 * 0.34
    b = (torch.rand(N, device=dev, dtype=torch.float32) - 0.5) * 0.2

    # Single call (suspect path)
    out_single = F.linear(x, W, b)

    # Chunked reference (known-good path)
    chunk = 65536
    out_chunked = torch.empty(M, N, device=dev, dtype=torch.float32)
    for s in range(0, M, chunk):
        out_chunked[s:s + chunk] = F.linear(x[s:s + chunk], W, b)

    # Sanity: a CPU reference on a small slice
    x_cpu = x[:1024].cpu()
    out_cpu = F.linear(x_cpu, W.cpu(), b.cpu())

    def stats(name, t):
        t = t.detach().float()
        print(f'  {name:>14s}: shape={tuple(t.shape)} '
              f'min={t.min().item():>12.4g} max={t.max().item():>12.4g} '
              f'std={t.std().item():>10.4g} '
              f'inf={int(torch.isinf(t).sum())} nan={int(torch.isnan(t).sum())}')

    print('\nresults:')
    stats('out_single',  out_single)
    stats('out_chunked', out_chunked)
    stats('out_cpu_slc', out_cpu)

    diff = (out_single - out_chunked).abs()
    print(f'\nout_single vs out_chunked: max|d|={diff.max().item():.4g} '
          f'mean|d|={diff.mean().item():.4g}')

    # Where do they disagree? Look at first diverging row + its neighbors
    bad_mask = (diff > 1e-3).any(dim=1)
    n_bad = int(bad_mask.sum())
    print(f'rows where any element diverges by >1e-3: {n_bad} / {M}'
          f' ({100*n_bad/M:.2f}%)')
    if n_bad:
        bad_idx = torch.where(bad_mask)[0][:3].tolist()
        for i in bad_idx:
            print(f'  row {i:>8d}: single={out_single[i].cpu().tolist()}')
            print(f'             chunked={out_chunked[i].cpu().tolist()}')

    # Are zero/near-zero values disproportionately produced by single?
    print(f'\nfraction |x|<1e-3: single={float((out_single.abs() < 1e-3).float().mean()):.4f} '
          f'chunked={float((out_chunked.abs() < 1e-3).float().mean()):.4f}')

    # Sharp signal: rows past index 2^19 = 524288 are wrong in the single call.
    BOUNDARY = 1 << 19
    head = out_single[:BOUNDARY]
    tail = out_single[BOUNDARY:]
    head_zero = float((head.abs() < 1e-3).float().mean())
    tail_zero = float((tail.abs() < 1e-3).float().mean())
    print(f'\nfraction |x|<1e-3 by region:'
          f' rows[0..{BOUNDARY-1}]={head_zero:.4f}'
          f' rows[{BOUNDARY}..]={tail_zero:.4f}')
    if tail_zero > 0.5 and head_zero < 0.01:
        print('>>> hipBLASLt tall+skinny bug REPRODUCED:'
              f' rows past M={BOUNDARY} are computed as denormal/zero <<<')
    elif torch.isinf(out_single).any() or out_single.abs().max() > 1e10:
        print('>>> hipBLASLt tall+skinny bug REPRODUCED (overflow variant) <<<')
    else:
        print('bug not reproduced — single-call result looks fine.')


if __name__ == '__main__':
    main()
