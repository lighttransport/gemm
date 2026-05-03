/*
 * cutile_gemm.cu — CUTLASS-backed GEMMs exposed via C ABI.
 *
 * Built into libcutile_gemm.so by nvcc; dlopen'd by bench_cuda_gemm at runtime.
 *
 * Layout (matches the rest of the bench): row-major
 *   Y[M,N] = X[M,K] * W[N,K]^T
 * We pass A=X (row-major M×K), B=W with ColumnMajor view (K×N, ld=K), C/D row-major.
 *
 * F16/BF16 use cutlass::arch::Sm80 OpClassTensorOp (Ampere HMMA), runs on
 * sm_120 via PTX backwards compatibility.
 *
 * FP8 path is intentionally absent: the canonical CUTLASS 4.x sm_120 FP8
 * blockwise examples (87a, 87b) fail at runtime on RTX 5060 Ti / CUDA 13.x
 * with a TMA descriptor / device-side assertion error. Until that upstream
 * issue is resolved, FP8 cutile is unsupported and the bench skips it.
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

template <typename ElemAB>
struct GemmConfig;

template <>
struct GemmConfig<cutlass::half_t> {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, RowMajor,
        cutlass::half_t, ColMajor,
        float,           RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape< 64,  64, 32>,
        cutlass::gemm::GemmShape< 16,   8, 16>
    >;
};

template <>
struct GemmConfig<cutlass::bfloat16_t> {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::bfloat16_t, RowMajor,
        cutlass::bfloat16_t, ColMajor,
        float,               RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape< 64,  64, 32>,
        cutlass::gemm::GemmShape< 16,   8, 16>
    >;
};

template <typename ElemAB>
static int run_gemm(uintptr_t d_Y, uintptr_t d_W, uintptr_t d_X,
                    int M, int N, int K, void *cu_stream) {
    using Gemm = typename GemmConfig<ElemAB>::Gemm;

    typename Gemm::Arguments args(
        {M, N, K},
        { reinterpret_cast<const ElemAB *>(d_X), K },
        { reinterpret_cast<const ElemAB *>(d_W), K },
        { reinterpret_cast<float *>(d_Y), N },
        { reinterpret_cast<float *>(d_Y), N },
        { 1.0f, 0.0f }
    );

    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutile: can_implement failed (%d)\n", (int)st);
        return -1;
    }
    st = gemm.initialize(args, nullptr, (cudaStream_t)cu_stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutile: initialize failed (%d)\n", (int)st);
        return -1;
    }
    st = gemm((cudaStream_t)cu_stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutile: launch failed (%d)\n", (int)st);
        return -1;
    }
    return 0;
}

extern "C" {

int cutile_gemm_f16_f32(uintptr_t d_Y, uintptr_t d_W, uintptr_t d_X,
                        int M, int N, int K, void *stream) {
    return run_gemm<cutlass::half_t>(d_Y, d_W, d_X, M, N, K, stream);
}

int cutile_gemm_bf16_f32(uintptr_t d_Y, uintptr_t d_W, uintptr_t d_X,
                         int M, int N, int K, void *stream) {
    return run_gemm<cutlass::bfloat16_t>(d_Y, d_W, d_X, M, N, K, stream);
}

int cutile_gemm_fp8_e4m3_f32(uintptr_t /*d_Y*/, uintptr_t /*d_W*/, uintptr_t /*d_X*/,
                              int /*M*/, int /*N*/, int /*K*/, void * /*stream*/) {
    /* Unsupported on sm_120 GeForce — see file header. */
    return -1;
}

} /* extern "C" */
