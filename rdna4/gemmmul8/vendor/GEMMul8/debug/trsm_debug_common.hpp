#pragma once

#include "common.hpp"

namespace trsm_debug {
using namespace debug_common;

template <typename TA, typename TB, gemmul8::Backend backend>
inline void call_gemmul8(
    Context &ctx,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    size_t m,
    size_t n,
    const TB *alpha,
    const TA *A,
    size_t lda,
    TB *B,
    size_t ldb,
    int num_moduli,
    bool fastmode,
    void *work,
    const char *handle_name //
) {
    const bool request_blas = (std::string(handle_name) == "cublas");

    if constexpr (backend == gemmul8::Backend::FP8) {
        gemmul8::trsmLt<TA, backend, TB>(
            ctx.handleLt, side, uplo, trans, diag,
            m, n,
            alpha, A, lda,
            B, ldb,
            num_moduli, fastmode, work,
            ctx.stream);
    } else {
        if (request_blas) {
            gemmul8::trsm<TA, backend, TB>(
                ctx.handle, side, uplo, trans, diag,
                m, n,
                alpha, A, lda,
                B, ldb,
                num_moduli, fastmode, work);
        } else {
            gemmul8::trsmLt<TA, backend, TB>(
                ctx.handleLt, side, uplo, trans, diag,
                m, n,
                alpha, A, lda,
                B, ldb,
                num_moduli, fastmode, work,
                ctx.stream);
        }
    }
}

template <typename T>
__global__ void stabilize_trsm_triangular_kernel(
    const size_t n,
    T *const A,
    const size_t lda,
    const cublasFillMode_t uplo,
    const cublasDiagType_t diag //
) {
    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n * n) return;

    const size_t col = idx / n;
    const size_t row = idx - col * n;

    const bool active =
        (uplo == CUBLAS_FILL_MODE_UPPER) ? (row <= col)
                                         : ((uplo == CUBLAS_FILL_MODE_LOWER) ? (row >= col)
                                                                             : true);

    if (!active) return;

    const size_t pos = row + col * lda;

    if (row == col) {
        if (diag == CUBLAS_DIAG_UNIT) {
            if constexpr (std::is_same_v<T, cuFloatComplex>) {
                A[pos] = cuFloatComplex{1.0f, 0.0f};
            } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                A[pos] = cuDoubleComplex{1.0, 0.0};
            } else {
                A[pos] = T(1);
            }
        } else {
            if constexpr (std::is_same_v<T, cuFloatComplex>) {
                A[pos] = cuFloatComplex{2.0f, 0.0f};
            } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                A[pos] = cuDoubleComplex{2.0, 0.0};
            } else {
                A[pos] = T(2);
            }
        }
        return;
    }

    constexpr double scale = 1.0 / 64.0;

    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        A[pos].x *= static_cast<float>(scale);
        A[pos].y *= static_cast<float>(scale);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        A[pos].x *= scale;
        A[pos].y *= scale;
    } else {
        A[pos] = T(double(A[pos]) * scale);
    }
}

template <typename T>
inline void stabilize_trsm_triangular(
    const size_t n,
    T *const A,
    const size_t lda,
    const cublasFillMode_t uplo,
    const cublasDiagType_t diag,
    const cudaStream_t stream //
) {
    constexpr size_t block_size = 256;
    const size_t grid_size      = (n * n + block_size - 1) / block_size;

    stabilize_trsm_triangular_kernel<T>
        <<<grid_size, block_size, 0, stream>>>(
            n, A, lda, uplo, diag);
}

template <typename TA, typename TB, gemmul8::Backend backend>
inline bool run_case(
    Context &ctx,
    Progress &progress,
    const char *test_name,
    Dim3 dim,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    LdExtra ld_extra,
    AlphaCase<TB> scal,
    unsigned num_moduli //
) {
    static_assert(std::is_same_v<TA, TB>, "TRSM debug assumes TA == TB.");

    using Tref = ref_t<TB>;

    const size_t m    = dim.m;
    const size_t n    = dim.n;
    const size_t ktri = (side == CUBLAS_SIDE_LEFT) ? m : n;

    const size_t lda = ktri + ld_extra.lda;
    const size_t ldb = m + ld_extra.ldb;

    TA *A         = nullptr;
    TB *B         = nullptr;
    TB *B_init    = nullptr;
    Tref *A_ref   = nullptr;
    Tref *B_exact = nullptr;
    void *work    = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A), lda * ktri * sizeof(TA), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B), ldb * n * sizeof(TB), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_init), ldb * n * sizeof(TB), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_ref), lda * ktri * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_exact), ldb * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, ktri, A, phi_default, seedA, ctx.stream);
    stabilize_trsm_triangular<TA>(ktri, A, lda, uplo, diag, ctx.stream);

    makemat::set_ones<TB>(m, n, B, ldb, ctx.stream);
    CHECK_CUDA(cudaMemsetAsync(B_init, 0, ldb * n * sizeof(TB), ctx.stream));

    const TB trmm_alpha = testTraits<TB>::one();
    CHECK_CUBLAS(testTraits<TB>::trmm(
        ctx.handle,
        side,
        uplo,
        ref_op<TB>(trans),
        diag,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        &trmm_alpha,
        A, static_cast<int64_t>(lda),
        B, static_cast<int64_t>(ldb),
        B_init, static_cast<int64_t>(ldb)));

    copy_cast_matrix<TA, Tref>(lda, ktri, A, lda, A_ref, lda, ctx.stream);
    copy_cast_matrix<TB, Tref>(ldb, n, B_init, ldb, B_exact, ldb, ctx.stream);

    const Tref alpha_ref = value_cast<Tref>(scal.alpha);
    CHECK_CUBLAS(testTraits<Tref>::trsm(
        ctx.handle,
        side,
        uplo,
        ref_op<Tref>(trans),
        diag,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        &alpha_ref,
        A_ref,
        static_cast<int64_t>(lda),
        B_exact,
        static_cast<int64_t>(ldb)));

    const size_t lwork = gemmul8::workSizeTrsm<TB, backend>(
        side, m, n, int(num_moduli));
    CHECK_CUDA(cudaMallocAsync(&work, lwork, ctx.stream));

    bool ok = true;

    auto reset_B = [&]() {
        CHECK_CUDA(cudaMemcpy2DAsync(
            B, ldb * sizeof(TB),
            B_init, ldb * sizeof(TB),
            ldb * sizeof(TB), n,
            cudaMemcpyDeviceToDevice,
            ctx.stream));
    };

    auto evaluate = [&](const char *mode, const char *handle) {
        const double rms = calc_rms_error<Tref, TB>(
            m, n, B_exact, ldb, B, ldb, ctx.stream);
        const double pad = calc_padding_rms_error<TB, TB>(
            m, ldb, n, B_init, ldb, B, ldb, ctx.stream);

        const bool failed =
            !std::isfinite(rms) || rms > rms_err_tol ||
            !std::isfinite(pad) || pad > padding_rms_err_tol;

        if (failed) {
            progress.clear_line();
            std::printf(
                "FAILED [%s] op=trsm type=(%c,%c) backend=%s "
                "side=%s uplo=%s trans=%s diag=%s "
                "size=(%zu,%zu) ld=(%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s alpha=%s rms_error=%.6e padding_rms=%.6e\n",
                test_name,
                testTraits<TA>::prefix,
                testTraits<TB>::prefix,
                backend_name(backend),
                side_name(side),
                uplo_name(uplo),
                op_name(trans),
                diag_name(diag),
                m, n,
                lda, ldb,
                num_moduli,
                mode, handle,
                scal.name,
                rms, pad);
            std::fflush(stdout);
            ok = false;
        }
        progress.advance();
    };

    auto run_one = [&](bool fastmode, const char *mode, const char *handle) {
        reset_B();
        call_gemmul8<TA, TB, backend>(
            ctx, side, uplo, trans, diag,
            m, n,
            &scal.alpha,
            A, lda,
            B, ldb,
            int(num_moduli), fastmode, work,
            handle);
        evaluate(mode, handle);
    };

    if constexpr (backend == gemmul8::Backend::INT8) {
        run_one(true, "fast", "cublas");
        run_one(false, "accu", "cublas");
#if defined(__CUDACC__)
        run_one(true, "fast", "cublasLt");
        run_one(false, "accu", "cublasLt");
#endif
    } else {
#if defined(__CUDACC__)
        run_one(true, "fast", "cublasLt");
        run_one(false, "accu", "cublasLt");
#endif
    }

    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    CHECK_CUDA(cudaFreeAsync(work, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B_exact, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B_init, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A, ctx.stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

    return ok;
}

} // namespace trsm_debug
