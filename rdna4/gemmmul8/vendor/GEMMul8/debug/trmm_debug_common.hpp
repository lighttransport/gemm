#pragma once

#include "common.hpp"

namespace trmm_debug {
using namespace debug_common;

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline void call_gemmul8(
    Context &ctx,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    size_t m,
    size_t n,
    const TC *alpha,
    const TA *A,
    size_t lda,
    const TB *B,
    size_t ldb,
    TC *C,
    size_t ldc,
    int num_moduli,
    bool fastmode,
    void *work,
    const char *handle_name //
) {
    const bool request_blas = (std::string(handle_name) == "cublas");

    if constexpr (backend == gemmul8::Backend::FP8) {
        gemmul8::trmmLt<TA, backend, TB, TC>(
            ctx.handleLt, side, uplo, trans, diag,
            m, n,
            alpha, A, lda,
            B, ldb,
            C, ldc,
            num_moduli, fastmode, work,
            nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
    } else {
        if (request_blas) {
            gemmul8::trmm<TA, backend, TB, TC>(
                ctx.handle, side, uplo, trans, diag,
                m, n,
                alpha, A, lda,
                B, ldb,
                C, ldc,
                num_moduli, fastmode, work);
        } else {
            gemmul8::trmmLt<TA, backend, TB, TC>(
                ctx.handleLt, side, uplo, trans, diag,
                m, n,
                alpha, A, lda,
                B, ldb,
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
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    LdExtra ld_extra,
    AlphaCase<TC> scal,
    unsigned num_moduli //
) {
    using Tref = ref_t<TC>;

    const size_t m    = dim.m;
    const size_t n    = dim.n;
    const size_t ktri = (side == CUBLAS_SIDE_LEFT) ? m : n;

    const size_t lda = ktri + ld_extra.lda;
    const size_t ldb = m + ld_extra.ldb;
    const size_t ldc = m + ld_extra.ldc;

    TA *A         = nullptr;
    TB *B         = nullptr;
    Tref *A_ref   = nullptr;
    Tref *B_ref   = nullptr;
    TC *C         = nullptr;
    TC *C_init    = nullptr;
    Tref *C_exact = nullptr;
    void *work    = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A), lda * ktri * sizeof(TA), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B), ldb * n * sizeof(TB), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_ref), lda * ktri * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_ref), ldb * n * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_init), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_exact), ldc * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, ktri, A, phi_default, seedA, ctx.stream);
    makemat::randmat<TB>(ldb, n, B, phi_default, seedB, ctx.stream);

    if (diag == CUBLAS_DIAG_NON_UNIT) {
        add_diag_shift<TA>(ktri, A, lda, ctx.stream);
    }

    copy_cast_matrix<TA, Tref>(lda, ktri, A, lda, A_ref, lda, ctx.stream);
    copy_cast_matrix<TB, Tref>(ldb, n, B, ldb, B_ref, ldb, ctx.stream);

    makemat::randmat<TC>(ldc, n, C_init, phi_default, seedA + seedB + 17, ctx.stream);
    CHECK_CUDA(cudaMemsetAsync(C_exact, 0, ldc * n * sizeof(Tref), ctx.stream));

    const Tref alpha_ref = value_cast<Tref>(scal.alpha);
    CHECK_CUBLAS(testTraits<Tref>::trmm(
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
        B_ref,
        static_cast<int64_t>(ldb),
        C_exact,
        static_cast<int64_t>(ldc)));

    const size_t lwork = gemmul8::workSize<testTraits<TC>::is_complex, backend, gemmul8::Func::trmm>(
        m, n, ktri, int(num_moduli));
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
            m, n, C_exact, ldc, C, ldc, ctx.stream);
        const double pad = calc_padding_rms_error<TC, TC>(
            m, ldc, n, C_init, ldc, C, ldc, ctx.stream);

        const bool failed =
            !std::isfinite(rms) || rms > rms_err_tol ||
            !std::isfinite(pad) || pad > padding_rms_err_tol;

        if (failed) {
            progress.clear_line();
            std::printf(
                "FAILED [%s] op=trmm type=(%c,%c,%c) backend=%s "
                "side=%s uplo=%s trans=%s diag=%s "
                "size=(%zu,%zu) ld=(%zu,%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s alpha=%s rms_error=%.6e padding_rms=%.6e\n",
                test_name,
                testTraits<TA>::prefix,
                testTraits<TB>::prefix,
                testTraits<TC>::prefix,
                backend_name(backend),
                side_name(side),
                uplo_name(uplo),
                op_name(trans),
                diag_name(diag),
                m, n,
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
            ctx, side, uplo, trans, diag,
            m, n,
            &scal.alpha,
            A, lda,
            B, ldb,
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
    CHECK_CUDA(cudaFreeAsync(B_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(B, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A, ctx.stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

    return ok;
}

} // namespace trmm_debug
