#pragma once

#include "common.hpp"

namespace trtrmm_debug {
using namespace debug_common;

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline void call_gemmul8(
    Context &ctx,
    cublasFillMode_t uplo_A,
    cublasFillMode_t uplo_B,
    cublasOperation_t trans_A,
    cublasOperation_t trans_B,
    cublasDiagType_t diag_A,
    cublasDiagType_t diag_B,
    size_t n,
    const TC *alpha,
    const TA *A,
    size_t lda,
    const TB *B,
    size_t ldb,
    const TC *beta,
    TC *C,
    size_t ldc,
    int num_moduli,
    bool fastmode,
    void *work,
    const char *handle_name //
) {
    const bool request_blas = (std::string(handle_name) == "cublas");

    if constexpr (backend == gemmul8::Backend::FP8) {
        gemmul8::trtrmmLt<TA, backend, TB, TC>(
            ctx.handleLt,
            uplo_A, uplo_B,
            trans_A, trans_B,
            diag_A, diag_B,
            n,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            num_moduli, fastmode, work,
            nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
    } else {
        if (request_blas) {
            gemmul8::trtrmm<TA, backend, TB, TC>(
                ctx.handle,
                uplo_A, uplo_B,
                trans_A, trans_B,
                diag_A, diag_B,
                n,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                num_moduli, fastmode, work);
        } else {
            gemmul8::trtrmmLt<TA, backend, TB, TC>(
                ctx.handleLt,
                uplo_A, uplo_B,
                trans_A, trans_B,
                diag_A, diag_B,
                n,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                num_moduli, fastmode, work,
                nullptr, nullptr,
                false, false, false, false,
                ctx.stream);
        }
    }
}

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline bool run_case(
    Context &ctx,
    Progress &progress,
    const char *test_name,
    Dim3 dim,
    cublasFillMode_t uplo_A,
    cublasFillMode_t uplo_B,
    cublasOperation_t trans_A,
    cublasOperation_t trans_B,
    cublasDiagType_t diag_A,
    cublasDiagType_t diag_B,
    LdExtra ld_extra,
    ScalarCase<TC> scal,
    unsigned num_moduli //
) {
    using Tref = ref_t<TC>;

    const size_t n = dim.n;

    const size_t lda = n + ld_extra.lda;
    const size_t ldb = n + ld_extra.ldb;
    const size_t ldc = n + ld_extra.ldc;

    TA *A         = nullptr;
    TB *B         = nullptr;
    Tref *A_ref   = nullptr;
    Tref *B_ref   = nullptr;
    Tref *A_full  = nullptr;
    Tref *B_full  = nullptr;
    TC *C         = nullptr;
    TC *C_init    = nullptr;
    Tref *C_exact = nullptr;
    void *work    = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A), lda * n * sizeof(TA), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B), ldb * n * sizeof(TB), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_ref), lda * n * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_ref), ldb * n * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_full), lda * n * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_full), ldb * n * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_init), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_exact), ldc * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, n, A, phi_default, seedA, ctx.stream);
    makemat::randmat<TB>(ldb, n, B, phi_default, seedB, ctx.stream);

    if (diag_A == CUBLAS_DIAG_NON_UNIT) {
        add_diag_shift<TA>(n, A, lda, ctx.stream);
    }
    if (diag_B == CUBLAS_DIAG_NON_UNIT) {
        add_diag_shift<TB>(n, B, ldb, ctx.stream);
    }

    copy_cast_matrix<TA, Tref>(lda, n, A, lda, A_ref, lda, ctx.stream);
    copy_cast_matrix<TB, Tref>(ldb, n, B, ldb, B_ref, ldb, ctx.stream);

    triangular_to_full<Tref, Tref>(
        n, uplo_A, diag_A, A_ref, lda, A_full, lda, ctx.stream);
    triangular_to_full<Tref, Tref>(
        n, uplo_B, diag_B, B_ref, ldb, B_full, ldb, ctx.stream);

    if (scal.beta_is_zero) {
        CHECK_CUDA(cudaMemsetAsync(C_init, 0, ldc * n * sizeof(TC), ctx.stream));
        CHECK_CUDA(cudaMemsetAsync(C_exact, 0, ldc * n * sizeof(Tref), ctx.stream));
    } else {
        makemat::randmat<TC>(ldc, n, C_init, phi_default, seedA + seedB + 17, ctx.stream);
        copy_cast_matrix<TC, Tref>(ldc, n, C_init, ldc, C_exact, ldc, ctx.stream);
    }

    const Tref alpha_ref = value_cast<Tref>(scal.alpha);
    const Tref beta_ref  = value_cast<Tref>(scal.beta);

    CHECK_CUBLAS(testTraits<Tref>::gemm(
        ctx.handle,
        ref_op<Tref>(trans_A),
        ref_op<Tref>(trans_B),
        static_cast<int64_t>(n),
        static_cast<int64_t>(n),
        static_cast<int64_t>(n),
        &alpha_ref,
        A_full,
        static_cast<int64_t>(lda),
        B_full,
        static_cast<int64_t>(ldb),
        &beta_ref,
        C_exact,
        static_cast<int64_t>(ldc)));

    const size_t lwork = gemmul8::workSize<testTraits<TC>::is_complex, backend, gemmul8::Func::trtrmm>(
        n, n, n, int(num_moduli));
    CHECK_CUDA(cudaMallocAsync(&work, lwork, ctx.stream));

    bool ok = true;

    auto reset_C = [&]() {
        CHECK_CUDA(cudaMemcpy2DAsync(
            C, ldc * sizeof(TC),
            C_init, ldc * sizeof(TC),
            ldc * sizeof(TC), n,
            cudaMemcpyDeviceToDevice,
            ctx.stream));
    };

    auto evaluate = [&](const char *mode, const char *handle) {
        const double rms = calc_rms_error<Tref, TC>(
            n, n, C_exact, ldc, C, ldc, ctx.stream);
        const double pad = calc_padding_rms_error<TC, TC>(
            n, ldc, n, C_init, ldc, C, ldc, ctx.stream);

        const bool failed =
            !std::isfinite(rms) || rms > rms_err_tol ||
            !std::isfinite(pad) || pad > padding_rms_err_tol;

        if (failed) {
            progress.clear_line();
            std::printf(
                "FAILED [%s] op=trtrmm type=(%c,%c,%c) backend=%s "
                "uplo=(%s,%s) trans=(%s,%s) diag=(%s,%s) "
                "n=%zu ld=(%zu,%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s scalar=%s rms_error=%.6e padding_rms=%.6e\n",
                test_name,
                testTraits<TA>::prefix,
                testTraits<TB>::prefix,
                testTraits<TC>::prefix,
                backend_name(backend),
                uplo_name(uplo_A),
                uplo_name(uplo_B),
                op_name(trans_A),
                op_name(trans_B),
                diag_name(diag_A),
                diag_name(diag_B),
                n,
                lda, ldb, ldc,
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
        reset_C();
        call_gemmul8<TA, TB, TC, backend>(
            ctx,
            uplo_A, uplo_B,
            trans_A, trans_B,
            diag_A, diag_B,
            n,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
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
    CHECK_CUDA(cudaFreeAsync(C_exact, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(C_init, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(C, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B_full, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A_full, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A, ctx.stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

    return ok;
}

} // namespace trtrmm_debug
