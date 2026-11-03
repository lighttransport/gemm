#pragma once

#include "common.hpp"

namespace rankk_debug {
using namespace debug_common;

template <gemmul8::Func FUNC>
inline const char *func_name();

template <>
inline const char *func_name<gemmul8::Func::syrk>() { return "syrk"; }

template <>
inline const char *func_name<gemmul8::Func::syr2k>() { return "syr2k"; }

template <>
inline const char *func_name<gemmul8::Func::syrkx>() { return "syrkx"; }

template <>
inline const char *func_name<gemmul8::Func::herk>() { return "herk"; }

template <>
inline const char *func_name<gemmul8::Func::her2k>() { return "her2k"; }

template <>
inline const char *func_name<gemmul8::Func::herkx>() { return "herkx"; }

template <gemmul8::Func FUNC>
inline constexpr bool is_hermitian_func() {
    return FUNC == gemmul8::Func::herk ||
           FUNC == gemmul8::Func::her2k ||
           FUNC == gemmul8::Func::herkx;
}

template <gemmul8::Func FUNC>
inline constexpr bool is_single_input_func() {
    return FUNC == gemmul8::Func::syrk || FUNC == gemmul8::Func::herk;
}

template <gemmul8::Func FUNC>
inline constexpr bool force_B_equal_A_func() {
    return FUNC == gemmul8::Func::syrkx || FUNC == gemmul8::Func::herkx;
}

inline size_t rk_rows(const size_t n, const size_t k, const cublasOperation_t trans) {
    return (trans == CUBLAS_OP_N) ? n : k;
}

inline size_t rk_cols(const size_t n, const size_t k, const cublasOperation_t trans) {
    return (trans == CUBLAS_OP_N) ? k : n;
}

template <gemmul8::Func FUNC, typename Tref>
inline void cublas_ref_single(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    size_t n,
    size_t k,
    const void *alpha,
    const Tref *A,
    size_t lda,
    const void *beta,
    Tref *C,
    size_t ldc //
) {
    if constexpr (FUNC == gemmul8::Func::syrk) {
        CHECK_CUBLAS(testTraits<Tref>::syrk(
            handle,
            uplo,
            ref_op<Tref>(trans),
            static_cast<int64_t>(n),
            static_cast<int64_t>(k),
            static_cast<const Tref *>(alpha),
            A,
            static_cast<int64_t>(lda),
            static_cast<const Tref *>(beta),
            C,
            static_cast<int64_t>(ldc)));
    } else {
        CHECK_CUBLAS(testTraits<Tref>::herk(
            handle,
            uplo,
            trans,
            static_cast<int64_t>(n),
            static_cast<int64_t>(k),
            static_cast<const double *>(alpha),
            A,
            static_cast<int64_t>(lda),
            static_cast<const double *>(beta),
            C,
            static_cast<int64_t>(ldc)));
    }
}

template <gemmul8::Func FUNC, typename Tref>
inline void cublas_ref_double_input(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    size_t n,
    size_t k,
    const void *alpha,
    const Tref *A,
    size_t lda,
    const Tref *B,
    size_t ldb,
    const void *beta,
    Tref *C,
    size_t ldc //
) {
    if constexpr (FUNC == gemmul8::Func::syr2k) {
        CHECK_CUBLAS(testTraits<Tref>::syr2k(
            handle, uplo, ref_op<Tref>(trans),
            static_cast<int64_t>(n), static_cast<int64_t>(k),
            static_cast<const Tref *>(alpha),
            A, static_cast<int64_t>(lda),
            B, static_cast<int64_t>(ldb),
            static_cast<const Tref *>(beta),
            C, static_cast<int64_t>(ldc)));
    } else if constexpr (FUNC == gemmul8::Func::syrkx) {
        CHECK_CUBLAS(testTraits<Tref>::syrkx(
            handle, uplo, ref_op<Tref>(trans),
            static_cast<int64_t>(n), static_cast<int64_t>(k),
            static_cast<const Tref *>(alpha),
            A, static_cast<int64_t>(lda),
            B, static_cast<int64_t>(ldb),
            static_cast<const Tref *>(beta),
            C, static_cast<int64_t>(ldc)));
    } else if constexpr (FUNC == gemmul8::Func::her2k) {
        CHECK_CUBLAS(testTraits<Tref>::her2k(
            handle, uplo, trans,
            static_cast<int64_t>(n), static_cast<int64_t>(k),
            static_cast<const Tref *>(alpha),
            A, static_cast<int64_t>(lda),
            B, static_cast<int64_t>(ldb),
            static_cast<const double *>(beta),
            C, static_cast<int64_t>(ldc)));
    } else {
        CHECK_CUBLAS(testTraits<Tref>::herkx(
            handle, uplo, trans,
            static_cast<int64_t>(n), static_cast<int64_t>(k),
            static_cast<const Tref *>(alpha),
            A, static_cast<int64_t>(lda),
            B, static_cast<int64_t>(ldb),
            static_cast<const double *>(beta),
            C, static_cast<int64_t>(ldc)));
    }
}

template <gemmul8::Func FUNC, typename TA, typename TC, gemmul8::Backend backend>
inline void call_gemmul8_single(
    Context &ctx,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    size_t n,
    size_t k,
    const void *alpha,
    const TA *A,
    size_t lda,
    const void *beta,
    TC *C,
    size_t ldc,
    int num_moduli,
    bool fastmode,
    void *work,
    const char *handle_name //
) {
    const bool request_blas = (std::string(handle_name) == "cublas");

    if constexpr (FUNC == gemmul8::Func::syrk) {
        const TC *a = static_cast<const TC *>(alpha);
        const TC *b = static_cast<const TC *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::syrkLt<TA, backend, TC>(
                ctx.handleLt, uplo, trans, n, k,
                a, A, lda,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr,
                false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::syrk<TA, backend, TC>(
                    ctx.handle, uplo, trans, n, k,
                    a, A, lda,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::syrkLt<TA, backend, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    a, A, lda,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr,
                    false, false,
                    ctx.stream);
            }
        }
    } else {
        using R    = real_scalar_t<TC>;
        const R *a = static_cast<const R *>(alpha);
        const R *b = static_cast<const R *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::herkLt<TA, backend, TC>(
                ctx.handleLt, uplo, trans, n, k,
                a, A, lda,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr,
                false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::herk<TA, backend, TC>(
                    ctx.handle, uplo, trans, n, k,
                    a, A, lda,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::herkLt<TA, backend, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    a, A, lda,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr,
                    false, false,
                    ctx.stream);
            }
        }
    }
}

template <gemmul8::Func FUNC, typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline void call_gemmul8_double_input(
    Context &ctx,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    size_t n,
    size_t k,
    const TC *alpha,
    const TA *A,
    size_t lda,
    const TB *B,
    size_t ldb,
    const void *beta,
    TC *C,
    size_t ldc,
    int num_moduli,
    bool fastmode,
    void *work,
    const char *handle_name //
) {
    const bool request_blas = (std::string(handle_name) == "cublas");

    if constexpr (FUNC == gemmul8::Func::syr2k) {
        const TC *b = static_cast<const TC *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::syr2kLt<TA, backend, TB, TC>(
                ctx.handleLt, uplo, trans, n, k,
                alpha, A, lda,
                B, ldb,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr, nullptr,
                false, false, false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::syr2k<TA, backend, TB, TC>(
                    ctx.handle, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::syr2kLt<TA, backend, TB, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr, nullptr,
                    false, false, false, false,
                    ctx.stream);
            }
        }
    } else if constexpr (FUNC == gemmul8::Func::syrkx) {
        const TC *b = static_cast<const TC *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::syrkxLt<TA, backend, TB, TC>(
                ctx.handleLt, uplo, trans, n, k,
                alpha, A, lda,
                B, ldb,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr, nullptr,
                false, false, false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::syrkx<TA, backend, TB, TC>(
                    ctx.handle, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::syrkxLt<TA, backend, TB, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr, nullptr,
                    false, false, false, false,
                    ctx.stream);
            }
        }
    } else if constexpr (FUNC == gemmul8::Func::her2k) {
        using R    = real_scalar_t<TC>;
        const R *b = static_cast<const R *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::her2kLt<TA, backend, TB, TC>(
                ctx.handleLt, uplo, trans, n, k,
                alpha, A, lda,
                B, ldb,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr, nullptr,
                false, false, false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::her2k<TA, backend, TB, TC>(
                    ctx.handle, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::her2kLt<TA, backend, TB, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr, nullptr,
                    false, false, false, false,
                    ctx.stream);
            }
        }
    } else {
        using R    = real_scalar_t<TC>;
        const R *b = static_cast<const R *>(beta);

        if constexpr (backend == gemmul8::Backend::FP8) {
            gemmul8::herkxLt<TA, backend, TB, TC>(
                ctx.handleLt, uplo, trans, n, k,
                alpha, A, lda,
                B, ldb,
                b, C, ldc,
                num_moduli, fastmode, work,
                nullptr, nullptr,
                false, false, false, false,
                ctx.stream);
        } else {
            if (request_blas) {
                gemmul8::herkx<TA, backend, TB, TC>(
                    ctx.handle, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work);
            } else {
                gemmul8::herkxLt<TA, backend, TB, TC>(
                    ctx.handleLt, uplo, trans, n, k,
                    alpha, A, lda,
                    B, ldb,
                    b, C, ldc,
                    num_moduli, fastmode, work,
                    nullptr, nullptr,
                    false, false, false, false,
                    ctx.stream);
            }
        }
    }
}

template <gemmul8::Func FUNC, typename TA, typename TC, gemmul8::Backend backend>
inline bool run_case_single(
    Context &ctx,
    Progress &progress,
    const char *test_name,
    Dim3 dim,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    LdExtra ld_extra,
    unsigned num_moduli,
    bool use_scalar_sweep = false,
    size_t scalar_idx     = 0 //
) {
    using Tref = ref_t<TC>;

    const size_t n = dim.n;
    const size_t k = dim.k;

    const size_t a_rows = rk_rows(n, k, trans);
    const size_t a_cols = rk_cols(n, k, trans);
    const size_t lda    = a_rows + ld_extra.lda;
    const size_t ldc    = n + ld_extra.ldc;

    TA *A         = nullptr;
    Tref *A_ref   = nullptr;
    TC *C         = nullptr;
    TC *C_init    = nullptr;
    Tref *C_exact = nullptr;
    void *work    = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A), lda * a_cols * sizeof(TA), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_ref), lda * a_cols * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_init), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_exact), ldc * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, a_cols, A, phi_default, seedA, ctx.stream);
    copy_cast_matrix<TA, Tref>(lda, a_cols, A, lda, A_ref, lda, ctx.stream);

    ScalarCase<TC> complex_scal  = default_scalar_case<TC>();
    RealScalarCase<TC> real_scal = default_real_scalar_case<TC>();

    if (use_scalar_sweep) {
        if constexpr (is_hermitian_func<FUNC>()) {
            real_scal = real_scalar_cases<TC>().at(scalar_idx);
        } else {
            complex_scal = scalar_cases<TC>().at(scalar_idx);
        }
    }

    const bool beta_zero = is_hermitian_func<FUNC>()
                               ? real_scal.beta_is_zero
                               : complex_scal.beta_is_zero;

    if (beta_zero) {
        CHECK_CUDA(cudaMemsetAsync(C_init, 0, ldc * n * sizeof(TC), ctx.stream));
        CHECK_CUDA(cudaMemsetAsync(C_exact, 0, ldc * n * sizeof(Tref), ctx.stream));
    } else {
        makemat::randmat<TC>(ldc, n, C_init, phi_default, seedA + seedB + 17, ctx.stream);
        if constexpr (is_hermitian_func<FUNC>()) {
            zero_diag_imag<TC>(n, C_init, ldc, ctx.stream);
        }
        copy_cast_matrix<TC, Tref>(ldc, n, C_init, ldc, C_exact, ldc, ctx.stream);
    }

    const Tref alpha_ref_complex = value_cast<Tref>(complex_scal.alpha);
    const Tref beta_ref_complex  = value_cast<Tref>(complex_scal.beta);
    const double alpha_ref_real  = static_cast<double>(real_scal.alpha);
    const double beta_ref_real   = static_cast<double>(real_scal.beta);

    if constexpr (is_hermitian_func<FUNC>()) {
        cublas_ref_single<FUNC, Tref>(
            ctx.handle, uplo, trans, n, k,
            &alpha_ref_real, A_ref, lda,
            &beta_ref_real, C_exact, ldc);
    } else {
        cublas_ref_single<FUNC, Tref>(
            ctx.handle, uplo, trans, n, k,
            &alpha_ref_complex, A_ref, lda,
            &beta_ref_complex, C_exact, ldc);
    }

    const size_t lwork = gemmul8::workSize<testTraits<TC>::is_complex, backend, FUNC>(
        n, n, k, int(num_moduli));
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
        const double active = calc_uplo_rms_error<Tref, TC>(
            n, uplo, true, C_exact, ldc, C, ldc, ctx.stream);
        const double inactive = calc_uplo_rms_error<TC, TC>(
            n, uplo, false, C_init, ldc, C, ldc, ctx.stream);
        const double pad = calc_padding_rms_error<TC, TC>(
            n, ldc, n, C_init, ldc, C, ldc, ctx.stream);

        const bool failed =
            !std::isfinite(active) || active > rms_err_tol ||
            !std::isfinite(inactive) || inactive > inactive_uplo_rms_err_tol ||
            !std::isfinite(pad) || pad > padding_rms_err_tol;

        if (failed) {
            progress.clear_line();
            std::printf(
                "FAILED [%s] op=%s type=(%c,%c) backend=%s uplo=%s trans=%s "
                "size=(%zu,%zu) ld=(%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s active_rms=%.6e inactive_rms=%.6e padding_rms=%.6e\n",
                test_name, func_name<FUNC>(),
                testTraits<TA>::prefix, testTraits<TC>::prefix,
                backend_name(backend), uplo_name(uplo), op_name(trans),
                n, k, lda, ldc, num_moduli,
                mode, handle, active, inactive, pad);
            std::fflush(stdout);
            ok = false;
        }
        progress.advance();
    };

    auto run_one = [&](bool fastmode, const char *mode, const char *handle) {
        reset_C();
        if constexpr (is_hermitian_func<FUNC>()) {
            call_gemmul8_single<FUNC, TA, TC, backend>(
                ctx, uplo, trans, n, k,
                &real_scal.alpha, A, lda,
                &real_scal.beta, C, ldc,
                int(num_moduli), fastmode, work, handle);
        } else {
            call_gemmul8_single<FUNC, TA, TC, backend>(
                ctx, uplo, trans, n, k,
                &complex_scal.alpha, A, lda,
                &complex_scal.beta, C, ldc,
                int(num_moduli), fastmode, work, handle);
        }
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
    CHECK_CUDA(cudaFreeAsync(A_ref, ctx.stream));
    CHECK_CUDA(cudaFreeAsync(A, ctx.stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

    return ok;
}

template <gemmul8::Func FUNC, typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline bool run_case_double_input(
    Context &ctx,
    Progress &progress,
    const char *test_name,
    Dim3 dim,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    LdExtra ld_extra,
    ScalarCase<TC> scal,
    unsigned num_moduli //
) {
    using Tref = ref_t<TC>;

    const size_t n = dim.n;
    const size_t k = dim.k;

    const size_t a_rows = rk_rows(n, k, trans);
    const size_t a_cols = rk_cols(n, k, trans);
    const size_t b_rows = a_rows;
    const size_t b_cols = a_cols;
    const size_t lda    = a_rows + ld_extra.lda;
    const size_t ldb    = b_rows + ld_extra.ldb;
    const size_t ldc    = n + ld_extra.ldc;

    TA *A         = nullptr;
    TB *B         = nullptr;
    Tref *A_ref   = nullptr;
    Tref *B_ref   = nullptr;
    TC *C         = nullptr;
    TC *C_init    = nullptr;
    Tref *C_exact = nullptr;
    void *work    = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A), lda * a_cols * sizeof(TA), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B), ldb * b_cols * sizeof(TB), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&A_ref), lda * a_cols * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&B_ref), ldb * b_cols * sizeof(Tref), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_init), ldc * n * sizeof(TC), ctx.stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_exact), ldc * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, a_cols, A, phi_default, seedA, ctx.stream);

    if constexpr (force_B_equal_A_func<FUNC>()) {
        CHECK_CUDA(cudaMemsetAsync(B, 0, ldb * b_cols * sizeof(TB), ctx.stream));
        copy_cast_matrix<TA, TB>(a_rows, a_cols, A, lda, B, ldb, ctx.stream);
    } else {
        makemat::randmat<TB>(ldb, b_cols, B, phi_default, seedB, ctx.stream);
    }

    copy_cast_matrix<TA, Tref>(lda, a_cols, A, lda, A_ref, lda, ctx.stream);
    copy_cast_matrix<TB, Tref>(ldb, b_cols, B, ldb, B_ref, ldb, ctx.stream);

    if (scal.beta_is_zero) {
        CHECK_CUDA(cudaMemsetAsync(C_init, 0, ldc * n * sizeof(TC), ctx.stream));
        CHECK_CUDA(cudaMemsetAsync(C_exact, 0, ldc * n * sizeof(Tref), ctx.stream));
    } else {
        makemat::randmat<TC>(ldc, n, C_init, phi_default, seedA + seedB + 17, ctx.stream);
        if constexpr (is_hermitian_func<FUNC>()) {
            zero_diag_imag<TC>(n, C_init, ldc, ctx.stream);
        }
        copy_cast_matrix<TC, Tref>(ldc, n, C_init, ldc, C_exact, ldc, ctx.stream);
    }

    const Tref alpha_ref        = value_cast<Tref>(scal.alpha);
    const Tref beta_ref_complex = value_cast<Tref>(scal.beta);
    const double beta_ref_real  = host_real_part<TC>(scal.beta);

    if constexpr (is_hermitian_func<FUNC>()) {
        cublas_ref_double_input<FUNC, Tref>(
            ctx.handle, uplo, trans, n, k,
            &alpha_ref, A_ref, lda, B_ref, ldb,
            &beta_ref_real, C_exact, ldc);
    } else {
        cublas_ref_double_input<FUNC, Tref>(
            ctx.handle, uplo, trans, n, k,
            &alpha_ref, A_ref, lda, B_ref, ldb,
            &beta_ref_complex, C_exact, ldc);
    }

    const size_t lwork = gemmul8::workSize<testTraits<TC>::is_complex, backend, FUNC>(
        n, n, k, int(num_moduli));
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
        const double active = calc_uplo_rms_error<Tref, TC>(
            n, uplo, true, C_exact, ldc, C, ldc, ctx.stream);
        const double inactive = calc_uplo_rms_error<TC, TC>(
            n, uplo, false, C_init, ldc, C, ldc, ctx.stream);
        const double pad = calc_padding_rms_error<TC, TC>(
            n, ldc, n, C_init, ldc, C, ldc, ctx.stream);

        const bool failed =
            !std::isfinite(active) || active > rms_err_tol ||
            !std::isfinite(inactive) || inactive > inactive_uplo_rms_err_tol ||
            !std::isfinite(pad) || pad > padding_rms_err_tol;

        if (failed) {
            progress.clear_line();
            std::printf(
                "FAILED [%s] op=%s type=(%c,%c,%c) backend=%s uplo=%s trans=%s "
                "size=(%zu,%zu) ld=(%zu,%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s active_rms=%.6e inactive_rms=%.6e padding_rms=%.6e\n",
                test_name, func_name<FUNC>(),
                testTraits<TA>::prefix, testTraits<TB>::prefix, testTraits<TC>::prefix,
                backend_name(backend), uplo_name(uplo), op_name(trans),
                n, k, lda, ldb, ldc, num_moduli,
                mode, handle, active, inactive, pad);
            std::fflush(stdout);
            ok = false;
        }
        progress.advance();
    };

    auto run_one = [&](bool fastmode, const char *mode, const char *handle) {
        reset_C();
        if constexpr (is_hermitian_func<FUNC>()) {
            real_scalar_t<TC> beta_real = real_scalar_t<TC>(host_real_part<TC>(scal.beta));
            call_gemmul8_double_input<FUNC, TA, TB, TC, backend>(
                ctx, uplo, trans, n, k,
                &scal.alpha, A, lda, B, ldb,
                &beta_real, C, ldc,
                int(num_moduli), fastmode, work, handle);
        } else {
            call_gemmul8_double_input<FUNC, TA, TB, TC, backend>(
                ctx, uplo, trans, n, k,
                &scal.alpha, A, lda, B, ldb,
                &scal.beta, C, ldc,
                int(num_moduli), fastmode, work, handle);
        }
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

} // namespace rankk_debug
