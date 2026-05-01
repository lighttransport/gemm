"""FP8 e4m3 matmul via torch._scaled_mm on RDNA4 (gfx1201).

Tests whether the PyTorch path through hipBLASLt actually works for FP8 on
RX 9070 XT, since the C++ hipBLASLt API enumerates 32 algos and all of them
fail (IMA / Tensile init failure / hang).

Shapes mirror rdna4/fp8/bench_fp8_gemm.c (mm0 = 1024 x 4608 x 4608).
"""
import time
import torch

assert torch.cuda.is_available()
print(f"torch={torch.__version__}  hip={torch.version.hip}  dev={torch.cuda.get_device_name(0)}")

dev = "cuda"
shapes = [
    ("mm0", 1024, 4608, 4608),
    ("mm2", 1024, 5120, 4608),
]

for name, M, N, K in shapes:
    # Y[M,N] = X[M,K] @ W^T[K,N]   with X,W in FP8 e4m3, accum/out FP32
    x = torch.randn(M, K, device=dev, dtype=torch.float32) * 0.1
    w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.1
    xf = x.to(torch.float8_e4m3fn)
    wf = w.to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device=dev, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=dev, dtype=torch.float32)

    # _scaled_mm signature: (a [M,K], b [K,N]) — needs b transposed/contiguous
    b = wf.t().contiguous().t()  # column-major view of [K,N]
    # actually: _scaled_mm wants a row-major and b col-major
    try:
        # Try canonical form: a [M,K] row-major, b [K,N] col-major (=> w.t() since w is [N,K] row-major)
        out = torch._scaled_mm(xf, wf.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
    except Exception as e:
        print(f"  [{name}] _scaled_mm error: {type(e).__name__}: {e}")
        continue

    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        out = torch._scaled_mm(xf, wf.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
    torch.cuda.synchronize()

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        out = torch._scaled_mm(xf, wf.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    print(f"  [{name}] _scaled_mm  M={M} N={N} K={K}  {ms:.4f} ms  {tflops:.1f} TFLOP/s  out.dtype={out.dtype}")

    # Sanity check: cosine similarity vs. FP32 reference.
    ref = (x @ w.t()).float()
    a = out.flatten().float()
    b = ref.flatten().float()
    cos = (a @ b) / (a.norm() * b.norm() + 1e-9)
    print(f"  [{name}] cos_vs_fp32_ref = {cos.item():.4f}")
