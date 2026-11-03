#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../include/gemmul8.hpp"
#include "../test/common/common.hpp"

namespace gemm_debug {

#define GEMM_DEBUG_GEMM_TEMPLATE_ARGS     TA, backend, TB, TC
#define GEMM_DEBUG_WORKSIZE_TEMPLATE_ARGS testTraits<TC>::is_complex, backend, gemmul8::Func::gemm

inline constexpr double phi_default         = 0.0;
inline constexpr double rms_err_tol         = 1.e-1;
inline constexpr double padding_rms_err_tol = 0.0;
inline constexpr bool print_each_result     = true;
inline constexpr bool print_case_begin      = false;

template <typename> inline constexpr bool dependent_false_type_v = false;

template <typename TC>
using ref_t = std::conditional_t<testTraits<TC>::is_complex, cuDoubleComplex, double>;

struct Dim3 {
    size_t m;
    size_t n;
    size_t k;
};

struct LdExtra {
    size_t lda;
    size_t ldb;
    size_t ldc;
};

class Progress {
  private:
    size_t test_idx_{0};
    size_t total_tests_{0};
    int bar_width_{30};

  public:
    explicit Progress(const size_t total_tests, const int bar_width = 30)
        : total_tests_(total_tests), bar_width_(bar_width) {}

    void advance() {
        ++test_idx_;
        print();
    }

    void print() const {
        const double progress =
            (total_tests_ == 0) ? 1.0 : double(test_idx_) / double(total_tests_);

        int pos = int(bar_width_ * progress);
        if (pos > bar_width_) pos = bar_width_;

        std::cout << "\r[";
        for (int i = 0; i < bar_width_; ++i) {
            if (i < pos) {
                std::cout << "=";
            } else if (i == pos && test_idx_ < total_tests_) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] (" << test_idx_ << "/" << total_tests_ << ")" << std::flush;
    }

    void clear_line() const {
        std::cout << "\r";
        for (int i = 0; i < bar_width_ + 64; ++i) {
            std::cout << " ";
        }
        std::cout << "\r" << std::flush;
    }

    void finish() {
        test_idx_ = total_tests_;
        print();
        std::cout << std::endl;
    }
};

template <gemmul8::Backend backend>
inline constexpr size_t evaluations_per_case() {
#if defined(__CUDACC__)
    if constexpr (backend == gemmul8::Backend::INT8) {
        return 4; // gemm fast/accu + gemmLt fast/accu
    } else {
        return 2; // gemmLt fast/accu
    }
#else
    if constexpr (backend == gemmul8::Backend::INT8) {
        return 2; // gemm fast/accu only
    } else {
        return 0;
    }
#endif
}

inline const char *op_name(cublasOperation_t op) {
    switch (op) {
    case CUBLAS_OP_N:
        return "N";
    case CUBLAS_OP_T:
        return "T";
    case CUBLAS_OP_C:
        return "C";
    default:
        return "?";
    }
}

inline const char *backend_name(gemmul8::Backend backend) {
    return (backend == gemmul8::Backend::INT8) ? "INT8" : "FP8";
}

template <typename T>
inline T scalar(double r, double i = 0.0) {
    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        return cuFloatComplex{static_cast<float>(r), static_cast<float>(i)};
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return cuDoubleComplex{r, i};
    } else {
        (void)i;
        return static_cast<T>(r);
    }
}

template <typename T>
struct ScalarCase {
    T alpha;
    T beta;
    const char *name;
    bool beta_is_zero;
};

template <typename T>
inline ScalarCase<T> default_scalar_case() {
    return {testTraits<T>::one(), testTraits<T>::zero(), "alpha=1,beta=0", true};
}

template <typename T>
inline std::vector<ScalarCase<T>> scalar_cases() {
    return {
        {scalar<T>(1.0), scalar<T>(0.0), "alpha=1,beta=0", true},
        {scalar<T>(-1.0), scalar<T>(0.0), "alpha=-1,beta=0", true},
        {scalar<T>(1.0), scalar<T>(1.0), "alpha=1,beta=1", false},
        {scalar<T>(0.5, 0.25), scalar<T>(-0.5, 0.125), "alpha=mixed,beta=mixed", false},
    };
}

template <typename TC>
inline unsigned default_num_moduli() {
    return testTraits<TC>::is_double ? 15u : 9u;
}

template <typename TC>
inline unsigned num_moduli_min_for_range() {
    return testTraits<TC>::is_double ? 12u : 8u;
}

template <typename TC>
inline unsigned num_moduli_max_for_range() {
    return testTraits<TC>::is_double ? 20u : 13u;
}

template <typename T>
inline std::vector<cublasOperation_t> op_list(bool include_real_conj) {
    if constexpr (testTraits<T>::is_complex) {
        return {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    } else {
        if (include_real_conj) {
            return {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
        } else {
            return {CUBLAS_OP_N, CUBLAS_OP_T};
        }
    }
}

template <typename T>
inline cublasOperation_t ref_op(cublasOperation_t op) {
    if constexpr (testTraits<T>::is_complex) {
        return op;
    } else {
        return (op == CUBLAS_OP_C) ? CUBLAS_OP_T : op;
    }
}

inline size_t rows_A(size_t m, size_t k, cublasOperation_t transa) {
    return (transa == CUBLAS_OP_N) ? m : k;
}

inline size_t cols_A(size_t m, size_t k, cublasOperation_t transa) {
    return (transa == CUBLAS_OP_N) ? k : m;
}

inline size_t rows_B(size_t n, size_t k, cublasOperation_t transb) {
    return (transb == CUBLAS_OP_N) ? k : n;
}

inline size_t cols_B(size_t n, size_t k, cublasOperation_t transb) {
    return (transb == CUBLAS_OP_N) ? n : k;
}

template <typename Tout, typename Tin>
__host__ __device__ __forceinline__ Tout value_cast(const Tin x) {
    if constexpr (std::is_same_v<Tout, Tin>) {
        return x;
    } else if constexpr (!testTraits<Tout>::is_complex && !testTraits<Tin>::is_complex) {
        return static_cast<Tout>(x);
    } else if constexpr (std::is_same_v<Tout, cuDoubleComplex> && std::is_same_v<Tin, cuFloatComplex>) {
        return cuDoubleComplex{static_cast<double>(x.x), static_cast<double>(x.y)};
    } else if constexpr (std::is_same_v<Tout, cuFloatComplex> && std::is_same_v<Tin, cuDoubleComplex>) {
        return cuFloatComplex{static_cast<float>(x.x), static_cast<float>(x.y)};
    } else if constexpr (std::is_same_v<Tout, cuDoubleComplex> && std::is_same_v<Tin, cuDoubleComplex>) {
        return x;
    } else if constexpr (std::is_same_v<Tout, cuFloatComplex> && std::is_same_v<Tin, cuFloatComplex>) {
        return x;
    } else {
        static_assert(dependent_false_type_v<Tout>, "Unsupported value_cast.");
    }
}

template <typename Tin, typename Tout>
__global__ void copy_cast_matrix_kernel(
    const size_t rows,
    const size_t cols,
    const Tin *const __restrict__ in,
    const size_t ldin,
    Tout *const __restrict__ out,
    const size_t ldout //
) {
    const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (idx >= rows * cols) return;

    const size_t row       = idx % rows;
    const size_t col       = idx / rows;
    out[col * ldout + row] = value_cast<Tout>(in[col * ldin + row]);
}

template <typename Tin, typename Tout>
inline void copy_cast_matrix(
    const size_t rows,
    const size_t cols,
    const Tin *const in,
    const size_t ldin,
    Tout *const out,
    const size_t ldout,
    const cudaStream_t stream //
) {
    if (rows == 0 || cols == 0) return;
    copy_cast_matrix_kernel<Tin, Tout>
        <<<(rows * cols + 255) / 256, 256, 0, stream>>>(rows, cols, in, ldin, out, ldout);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ double real_part(const T x) {
    return static_cast<double>(x);
}

template <>
__device__ __forceinline__ double real_part<cuFloatComplex>(const cuFloatComplex x) {
    return static_cast<double>(x.x);
}

template <>
__device__ __forceinline__ double real_part<cuDoubleComplex>(const cuDoubleComplex x) {
    return x.x;
}

template <typename T>
__device__ __forceinline__ double imag_part(const T) {
    return 0.0;
}

template <>
__device__ __forceinline__ double imag_part<cuFloatComplex>(const cuFloatComplex x) {
    return static_cast<double>(x.y);
}

template <>
__device__ __forceinline__ double imag_part<cuDoubleComplex>(const cuDoubleComplex x) {
    return x.y;
}

template <typename Texact, typename Tcalc>
__device__ __forceinline__ double l2_diff_sq(const Texact exact, const Tcalc calc) {
    const double err_re = real_part<Texact>(exact) - real_part<Tcalc>(calc);
    const double err_im = imag_part<Texact>(exact) - imag_part<Tcalc>(calc);
    return fma(err_re, err_re, err_im * err_im);
}

template <typename Texact, typename Tcalc>
__global__ void rms_error_kernel(
    const size_t rows,
    const size_t cols,
    const Texact *const __restrict__ exact,
    const size_t ld_exact,
    const Tcalc *const __restrict__ calc,
    const size_t ld_calc,
    double *const __restrict__ partial //
) {
    extern __shared__ double s[];

    const size_t tid    = threadIdx.x;
    const size_t gid    = size_t(blockIdx.x) * blockDim.x + tid;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    const size_t size   = rows * cols;

    double sum = 0.0;
    for (size_t idx = gid; idx < size; idx += stride) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;

        sum += l2_diff_sq<Texact, Tcalc>(
            exact[col * ld_exact + row],
            calc[col * ld_calc + row]);
    }

    s[tid] = sum;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s[tid] += s[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = s[0];
    }
}

template <typename Tinit, typename Tcalc>
__global__ void padding_error_kernel(
    const size_t valid_rows,
    const size_t ld,
    const size_t cols,
    const Tinit *const __restrict__ init,
    const size_t ld_init,
    const Tcalc *const __restrict__ calc,
    const size_t ld_calc,
    double *const __restrict__ partial //
) {
    extern __shared__ double s[];

    const size_t pad_rows = ld - valid_rows;
    const size_t size     = pad_rows * cols;

    const size_t tid    = threadIdx.x;
    const size_t gid    = size_t(blockIdx.x) * blockDim.x + tid;
    const size_t stride = size_t(blockDim.x) * gridDim.x;

    double sum = 0.0;
    for (size_t idx = gid; idx < size; idx += stride) {
        const size_t pad_row = idx % pad_rows;
        const size_t col     = idx / pad_rows;
        const size_t row     = valid_rows + pad_row;

        sum += l2_diff_sq<Tinit, Tcalc>(
            init[col * ld_init + row],
            calc[col * ld_calc + row]);
    }

    s[tid] = sum;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s[tid] += s[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = s[0];
    }
}

__global__ void reduce_double_kernel(
    const size_t size,
    const double *const __restrict__ in,
    double *const __restrict__ out //
) {
    extern __shared__ double s[];

    const size_t tid = threadIdx.x;
    const size_t idx = size_t(blockIdx.x) * blockDim.x + tid;

    double sum = 0.0;
    if (idx < size) {
        sum = in[idx];
    }

    s[tid] = sum;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s[tid] += s[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = s[0];
    }
}

inline size_t div_ceil(const size_t x, const size_t y) {
    return (x + y - 1) / y;
}

inline size_t reduction_blocks(const size_t size) {
    return std::min<size_t>(div_ceil(size, 256), 65535);
}

inline double finish_rms_reduction(
    const size_t count,
    double *const buf0,
    double *const buf1,
    const cudaStream_t stream //
) {
    size_t cur_size = reduction_blocks(count);
    double *in      = buf0;
    double *out     = buf1;

    while (cur_size > 1) {
        const size_t blocks = reduction_blocks(cur_size);
        reduce_double_kernel<<<blocks, 256, 256 * sizeof(double), stream>>>(cur_size, in, out);
        CHECK_CUDA(cudaGetLastError());

        cur_size = blocks;
        std::swap(in, out);
    }

    double sumsq = 0.0;
    CHECK_CUDA(cudaMemcpyAsync(&sumsq, in, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    return std::sqrt(sumsq / static_cast<double>(count));
}

template <typename Texact, typename Tcalc>
inline double calc_rms_error(
    const size_t rows,
    const size_t cols,
    const Texact *const exact,
    const size_t ld_exact,
    const Tcalc *const calc,
    const size_t ld_calc,
    const cudaStream_t stream //
) {
    const size_t count = rows * cols;
    if (count == 0) return 0.0;

    const size_t blocks0 = reduction_blocks(count);

    double *buf0 = nullptr;
    double *buf1 = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf0), blocks0 * sizeof(double), stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf1), blocks0 * sizeof(double), stream));

    rms_error_kernel<Texact, Tcalc>
        <<<blocks0, 256, 256 * sizeof(double), stream>>>(
            rows, cols,
            exact, ld_exact,
            calc, ld_calc,
            buf0);
    CHECK_CUDA(cudaGetLastError());

    const double rms = finish_rms_reduction(count, buf0, buf1, stream);

    CHECK_CUDA(cudaFreeAsync(buf1, stream));
    CHECK_CUDA(cudaFreeAsync(buf0, stream));

    return rms;
}

template <typename Tinit, typename Tcalc>
inline double calc_padding_rms_error(
    const size_t valid_rows,
    const size_t ld,
    const size_t cols,
    const Tinit *const init,
    const size_t ld_init,
    const Tcalc *const calc,
    const size_t ld_calc,
    const cudaStream_t stream //
) {
    if (ld <= valid_rows || cols == 0) return 0.0;

    const size_t count   = (ld - valid_rows) * cols;
    const size_t blocks0 = reduction_blocks(count);

    double *buf0 = nullptr;
    double *buf1 = nullptr;

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf0), blocks0 * sizeof(double), stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf1), blocks0 * sizeof(double), stream));

    padding_error_kernel<Tinit, Tcalc>
        <<<blocks0, 256, 256 * sizeof(double), stream>>>(
            valid_rows, ld, cols,
            init, ld_init,
            calc, ld_calc,
            buf0);
    CHECK_CUDA(cudaGetLastError());

    const double rms = finish_rms_reduction(count, buf0, buf1, stream);

    CHECK_CUDA(cudaFreeAsync(buf1, stream));
    CHECK_CUDA(cudaFreeAsync(buf0, stream));

    return rms;
}

class Context {
  public:
    cublasHandle_t handle{};
    cublasLtHandle_t handleLt{};
    cudaStream_t stream{};

    Context() {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasLtCreate(&handleLt));
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUBLAS(cublasSetStream(handle, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    ~Context() {
        if (stream) cudaStreamSynchronize(stream);
        if (handleLt) cublasLtDestroy(handleLt);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

template <typename TA, typename TB, typename TC, gemmul8::Backend backend>
inline bool run_case(
    Context &ctx,
    Progress &progress,
    const char *test_name,
    Dim3 dim,
    cublasOperation_t transa,
    cublasOperation_t transb,
    LdExtra ld_extra,
    ScalarCase<TC> scal,
    unsigned num_moduli //
) {

    using Tref = ref_t<TC>;

    const size_t m = dim.m;
    const size_t n = dim.n;
    const size_t k = dim.k;

    const size_t a_rows = rows_A(m, k, transa);
    const size_t a_cols = cols_A(m, k, transa);
    const size_t b_rows = rows_B(n, k, transb);
    const size_t b_cols = cols_B(n, k, transb);

    const size_t lda       = a_rows + ld_extra.lda;
    const size_t ldb       = b_rows + ld_extra.ldb;
    const size_t ldc       = m + ld_extra.ldc;
    const size_t ldc_exact = ldc;

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
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&C_exact), ldc_exact * n * sizeof(Tref), ctx.stream));

    makemat::randmat<TA>(lda, a_cols, A, phi_default, seedA, ctx.stream);
    makemat::randmat<TB>(ldb, b_cols, B, phi_default, seedB, ctx.stream);

    copy_cast_matrix<TA, Tref>(lda, a_cols, A, lda, A_ref, lda, ctx.stream);
    copy_cast_matrix<TB, Tref>(ldb, b_cols, B, ldb, B_ref, ldb, ctx.stream);

    if (scal.beta_is_zero) {
        CHECK_CUDA(cudaMemsetAsync(C_init, 0, ldc * n * sizeof(TC), ctx.stream));
        CHECK_CUDA(cudaMemsetAsync(C_exact, 0, ldc_exact * n * sizeof(Tref), ctx.stream));
    } else {
        makemat::randmat<TC>(ldc, n, C_init, phi_default, seedA + seedB + 17, ctx.stream);
        copy_cast_matrix<TC, Tref>(ldc, n, C_init, ldc, C_exact, ldc_exact, ctx.stream);
    }

    const Tref alpha_ref = value_cast<Tref>(scal.alpha);
    const Tref beta_ref  = value_cast<Tref>(scal.beta);

    CHECK_CUBLAS(testTraits<Tref>::gemm(
        ctx.handle,
        ref_op<Tref>(transa),
        ref_op<Tref>(transb),
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        static_cast<int64_t>(k),
        &alpha_ref,
        A_ref, static_cast<int64_t>(lda),
        B_ref, static_cast<int64_t>(ldb),
        &beta_ref,
        C_exact, static_cast<int64_t>(ldc_exact)));

    const size_t lwork = gemmul8::workSize<GEMM_DEBUG_WORKSIZE_TEMPLATE_ARGS>(m, n, k, int(num_moduli));
    CHECK_CUDA(cudaMallocAsync(&work, lwork, ctx.stream));

    bool ok = true;

    auto print_line = [&](
                          const char *status,
                          const char *mode_name,
                          const char *handle_name,
                          const double rms,
                          const double padding_rms) {
        std::printf(
            "%s [%s] type=(%c,%c,%c) backend=%s op=(%s,%s) "
            "size=(%zu,%zu,%zu) ld=(%zu,%zu,%zu) num_moduli=%u "
            "mode=%s handle=%s scalar=%s rms_error=%.6e padding_rms_error=%.6e\n",
            status,
            test_name,
            testTraits<TA>::prefix,
            testTraits<TB>::prefix,
            testTraits<TC>::prefix,
            backend_name(backend),
            op_name(transa),
            op_name(transb),
            m, n, k,
            lda, ldb, ldc,
            num_moduli,
            mode_name,
            handle_name,
            scal.name,
            rms,
            padding_rms);
        std::fflush(stdout);
    };

    if constexpr (print_case_begin) {
        print_line("BEGIN", "ref", "cublas-fp64", -1.0, -1.0);
    }

    auto evaluate = [&](const char *mode_name, const char *handle_name) {
        const double rms = calc_rms_error<Tref, TC>(
            m, n,
            C_exact, ldc_exact,
            C, ldc,
            ctx.stream);
        const double padding_rms = calc_padding_rms_error<TC, TC>(
            m, ldc, n,
            C_init, ldc,
            C, ldc,
            ctx.stream);

        const bool rms_failed     = (!std::isfinite(rms) || rms > rms_err_tol);
        const bool padding_failed = (!std::isfinite(padding_rms) || padding_rms > padding_rms_err_tol);
        const bool failed         = rms_failed || padding_failed;

        if (failed) {
            progress.clear_line();

            std::printf(
                "FAILED [%s] type=(%c,%c,%c) backend=%s op=(%s,%s) "
                "size=(%zu,%zu,%zu) ld=(%zu,%zu,%zu) num_moduli=%u "
                "mode=%s handle=%s scalar=%s rms_error=%.6e padding_rms_error=%.6e\n",
                test_name,
                testTraits<TA>::prefix,
                testTraits<TB>::prefix,
                testTraits<TC>::prefix,
                backend_name(backend),
                op_name(transa),
                op_name(transb),
                m, n, k,
                lda, ldb, ldc,
                num_moduli,
                mode_name,
                handle_name,
                scal.name,
                rms,
                padding_rms);
            std::fflush(stdout);

            ok = false;
        }

        progress.advance();
    };

    auto reset_C = [&]() {
        CHECK_CUDA(cudaMemcpy2DAsync(
            C, ldc * sizeof(TC),
            C_init, ldc * sizeof(TC),
            ldc * sizeof(TC), n,
            cudaMemcpyDeviceToDevice,
            ctx.stream));
    };

    if constexpr (backend == gemmul8::Backend::INT8) {
        reset_C();
        gemmul8::gemm<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handle,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), true,
            work);
        evaluate("fast", "cublas");

        reset_C();
        gemmul8::gemm<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handle,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), false,
            work);
        evaluate("accu", "cublas");

#if defined(__CUDACC__)
        reset_C();
        gemmul8::gemmLt<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handleLt,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), true,
            work, nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
        evaluate("fast", "cublasLt");

        reset_C();
        gemmul8::gemmLt<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handleLt,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), false,
            work, nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
        evaluate("accu", "cublasLt");
#endif

    } else {
#if defined(__CUDACC__)
        reset_C();
        gemmul8::gemmLt<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handleLt,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), true,
            work, nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
        evaluate("fast", "cublasLt");

        reset_C();
        gemmul8::gemmLt<GEMM_DEBUG_GEMM_TEMPLATE_ARGS>(
            ctx.handleLt,
            transa, transb,
            m, n, k,
            &scal.alpha,
            A, lda,
            B, ldb,
            &scal.beta,
            C, ldc,
            int(num_moduli), false,
            work, nullptr, nullptr,
            false, false, false, false,
            ctx.stream);
        evaluate("accu", "cublasLt");
#else
        (void)evaluate;
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

template <typename Runner>
inline bool run_all_type_backend_cases(Runner &&runner) {
    bool ok = true;

    ok &= runner.template operator()<float, float, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<float, float, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<float, double, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, float, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<float, double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, float, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, double, gemmul8::Backend::INT8>();

    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();

    ok &= runner.template operator()<float, float, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<float, float, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<float, double, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, float, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<float, double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, float, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, double, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, double, double, gemmul8::Backend::FP8>();

    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();

    return ok;
}

template <typename Runner>
inline bool run_representative_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<double, double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

} // namespace gemm_debug
