"""F32 dense GEMM (Y = X @ W^T + b) for the SparseLinear path on RDNA4.

Replaces F.linear / hipBLASLt for the tall+skinny shape that triggers the
ROCm 7.2.2 hipBLASLt M>2^19 bug (see rdna4/hipblaslt-issue.md). One simple
SIMT FMA kernel — sufficient because the workload is bandwidth-bound at the
shape we care about (M~1.5M, K=64, N=6).

Shape coverage (asserted at call time):
    X: [M, K] float32, M arbitrary, K <= LINEAR_F32_MAX_K (256)
    W: [N, K] float32, N <= LINEAR_F32_MAX_N (16)
    b: [N] float32 or None
    Y: [M, N] float32

Outside this envelope, callers should fall back to F.linear.
"""
from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load_inline


LINEAR_F32_MAX_K = 256
LINEAR_F32_MAX_N = 16

_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

// One thread = one row. One workgroup = BM rows. Workgroup loads (N*K + N)
// floats of W/b into LDS once; each thread then loops K and accumulates N
// outputs in registers. N is bounded at compile time to keep the accumulator
// in registers; we use 16 (covers all SparseLinear shapes in TRELLIS-2's
// post-LayerNorm tex/shape decoders).
constexpr int LF32_MAX_N = 16;

extern "C" __global__ void linear_f32_kernel(
    const float * __restrict__ X,
    const float * __restrict__ W,
    const float * __restrict__ B,
    float       * __restrict__ Y,
    int M, int K, int N) {
    extern __shared__ float lds[];
    float *Wsh = lds;          // N*K floats
    float *bsh = lds + N * K;  // N floats

    int t  = threadIdx.x;
    int bs = blockDim.x;
    int wsize = N * K;
    #pragma unroll 1
    for (int i = t; i < wsize; i += bs) Wsh[i] = W[i];
    if (B != nullptr) {
        for (int i = t; i < N; i += bs) bsh[i] = B[i];
    } else {
        for (int i = t; i < N; i += bs) bsh[i] = 0.f;
    }
    __syncthreads();

    int row = blockIdx.x * bs + t;
    if (row >= M) return;
    const float *xrow = X + (size_t)row * (size_t)K;

    float acc[LF32_MAX_N];
    #pragma unroll
    for (int n = 0; n < LF32_MAX_N; n++) acc[n] = 0.f;
    for (int n = 0; n < N; n++) acc[n] = bsh[n];

    for (int k = 0; k < K; k++) {
        float xv = xrow[k];
        for (int n = 0; n < N; n++) {
            acc[n] += xv * Wsh[n * K + k];
        }
    }

    float *yrow = Y + (size_t)row * (size_t)N;
    for (int n = 0; n < N; n++) yrow[n] = acc[n];
}

torch::Tensor linear_f32(torch::Tensor X, torch::Tensor W, c10::optional<torch::Tensor> B) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "X, W must be on HIP device");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be fp32");
    TORCH_CHECK(W.dtype() == torch::kFloat32, "W must be fp32");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "X and W must be 2D");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = W.size(0);
    TORCH_CHECK(W.size(1) == K, "W.size(1) must equal X.size(1)");
    TORCH_CHECK(N <= LF32_MAX_N, "N exceeds LINEAR_F32_MAX_N");
    const float *bptr = nullptr;
    torch::Tensor Bc;
    if (B.has_value()) {
        Bc = B.value().contiguous();
        TORCH_CHECK(Bc.dtype() == torch::kFloat32, "B must be fp32");
        TORCH_CHECK(Bc.numel() == N, "B size must equal N");
        bptr = Bc.data_ptr<float>();
    }
    auto Y = torch::empty({M, N}, X.options());
    if (M == 0) return Y;

    int BM = 64;
    dim3 grid((unsigned)((M + BM - 1) / BM));
    dim3 block(BM);
    size_t lds_bytes = (size_t)(N * K + N) * sizeof(float);
    auto stream = c10::hip::getCurrentHIPStream();
    linear_f32_kernel<<<grid, block, lds_bytes, stream.stream()>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), bptr, Y.data_ptr<float>(),
        (int)M, (int)K, (int)N);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_f32", &linear_f32, "F32 dense GEMM Y = X @ W^T + b");
}
"""


_ext = None


def _ensure_loaded():
    global _ext
    if _ext is not None:
        return _ext
    build_dir = os.path.expanduser('~/.cache/torch_extensions/linear_f32_rdna4')
    os.makedirs(build_dir, exist_ok=True)
    _ext = load_inline(
        name='linear_f32_rdna4',
        cpp_sources='',
        cuda_sources=_KERNEL_SRC,
        verbose=os.environ.get('LINEAR_F32_VERBOSE') == '1',
        with_cuda=True,
        build_directory=build_dir,
    )
    return _ext


def can_handle(x: torch.Tensor, w: torch.Tensor) -> bool:
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        return False
    if x.dim() != 2 or w.dim() != 2:
        return False
    K = x.shape[1]
    N = w.shape[0]
    return K <= LINEAR_F32_MAX_K and N <= LINEAR_F32_MAX_N


def linear_f32(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    ext = _ensure_loaded()
    x_c = x.contiguous()
    w_c = w.contiguous()
    b_c = b.contiguous() if b is not None else None
    return ext.linear_f32(x_c, w_c, b_c)
