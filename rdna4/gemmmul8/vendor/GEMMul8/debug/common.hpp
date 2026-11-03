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

namespace debug_common {

inline constexpr double phi_default               = 0.0;
inline constexpr double rms_err_tol               = 1.e-1;
inline constexpr double padding_rms_err_tol       = 0.0;
inline constexpr double inactive_uplo_rms_err_tol = 0.0;
inline constexpr double triangular_diagonal_shift = 16.0;

template <typename> inline constexpr bool dependent_false_type_v = false;

template <typename TC>
using ref_t = std::conditional_t<testTraits<TC>::is_complex, cuDoubleComplex, double>;

template <typename TC>
using real_scalar_t = std::conditional_t<std::is_same_v<TC, cuDoubleComplex> || std::is_same_v<TC, double>, double, float>;

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
        for (int i = 0; i < bar_width_ + 96; ++i) std::cout << " ";
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
    if constexpr (backend == gemmul8::Backend::INT8) return 4; // blas fast/accu + Lt fast/accu
    else return 2;                                             // Lt fast/accu
#else
    if constexpr (backend == gemmul8::Backend::INT8) return 2;
    else return 0;
#endif
}

inline const char *op_name(cublasOperation_t op) {
    switch (op) {
    case CUBLAS_OP_N: return "N";
    case CUBLAS_OP_T: return "T";
    case CUBLAS_OP_C: return "C";
    default: return "?";
    }
}

inline const char *side_name(cublasSideMode_t side) {
    return (side == CUBLAS_SIDE_LEFT) ? "L" : "R";
}

inline const char *uplo_name(cublasFillMode_t uplo) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? "U" : "L";
}

inline const char *diag_name(cublasDiagType_t diag) {
    return (diag == CUBLAS_DIAG_UNIT) ? "U" : "N";
}

inline const char *backend_name(gemmul8::Backend backend) {
    return (backend == gemmul8::Backend::INT8) ? "INT8" : "FP8";
}

template <typename T>
__host__ __device__ inline T scalar(double r, double i = 0.0) {
    if constexpr (std::is_same_v<T, cuFloatComplex>) return cuFloatComplex{static_cast<float>(r), static_cast<float>(i)};
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) return cuDoubleComplex{r, i};
    else {
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

template <typename T>
inline std::vector<ScalarCase<T>> herkx_scalar_cases() {
    return {
        { scalar<T>(1.0),  scalar<T>(0.0),      "alpha=1,beta=0",  true},
        {scalar<T>(-1.0),  scalar<T>(0.0),     "alpha=-1,beta=0",  true},
        { scalar<T>(1.0),  scalar<T>(1.0),      "alpha=1,beta=1", false},
        { scalar<T>(0.5), scalar<T>(-0.5), "alpha=0.5,beta=-0.5", false},
    };
}

template <typename T>
struct AlphaCase {
    T alpha;
    const char *name;
};

template <typename T>
inline AlphaCase<T> default_alpha_case() {
    return {testTraits<T>::one(), "alpha=1"};
}

template <typename T>
inline std::vector<AlphaCase<T>> alpha_cases() {
    return {
        {scalar<T>(1.0), "alpha=1"},
        {scalar<T>(-1.0), "alpha=-1"},
        {scalar<T>(0.5, 0.25), "alpha=mixed"},
    };
}

template <typename T>
struct RealScalarCase {
    real_scalar_t<T> alpha;
    real_scalar_t<T> beta;
    const char *name;
    bool beta_is_zero;
};

template <typename T>
inline RealScalarCase<T> default_real_scalar_case() {
    return {real_scalar_t<T>(1), real_scalar_t<T>(0), "alpha=1,beta=0", true};
}

template <typename T>
inline std::vector<RealScalarCase<T>> real_scalar_cases() {
    return {
        { real_scalar_t<T>(1.0),  real_scalar_t<T>(0.0),      "alpha=1,beta=0",  true},
        {real_scalar_t<T>(-1.0),  real_scalar_t<T>(0.0),     "alpha=-1,beta=0",  true},
        { real_scalar_t<T>(1.0),  real_scalar_t<T>(1.0),      "alpha=1,beta=1", false},
        { real_scalar_t<T>(0.5), real_scalar_t<T>(-0.5), "alpha=0.5,beta=-0.5", false},
    };
}

template <typename TC>
inline unsigned default_num_moduli() { return testTraits<TC>::is_double ? 15u : 9u; }

template <typename TC>
inline unsigned num_moduli_min_for_range() { return testTraits<TC>::is_double ? 12u : 8u; }

template <typename TC>
inline unsigned num_moduli_max_for_range() { return testTraits<TC>::is_double ? 20u : 13u; }

template <typename T>
inline std::vector<cublasOperation_t> op_list(bool include_real_conj) {
    if constexpr (testTraits<T>::is_complex) return {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    else return include_real_conj ? std::vector<cublasOperation_t>{CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}
                                  : std::vector<cublasOperation_t>{CUBLAS_OP_N, CUBLAS_OP_T};
}

template <typename T>
inline std::vector<cublasOperation_t> rk_op_list(bool include_real_conj) {
    if constexpr (testTraits<T>::is_complex) {
        return {CUBLAS_OP_N, CUBLAS_OP_T};
    } else {
        return include_real_conj
                   ? std::vector<cublasOperation_t>{CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}
                   : std::vector<cublasOperation_t>{CUBLAS_OP_N, CUBLAS_OP_T};
    }
}

template <typename T>
inline std::vector<cublasOperation_t> her_op_list() {
    (void)sizeof(T);
    return {CUBLAS_OP_N, CUBLAS_OP_C};
}

template <typename T>
inline cublasOperation_t ref_op(cublasOperation_t op) {
    if constexpr (testTraits<T>::is_complex) return op;
    else return (op == CUBLAS_OP_C) ? CUBLAS_OP_T : op;
}

template <typename Tout, typename Tin>
__host__ __device__ __forceinline__ Tout value_cast(const Tin x) {
    if constexpr (std::is_same_v<Tout, Tin>) return x;
    else if constexpr (!testTraits<Tout>::is_complex && !testTraits<Tin>::is_complex) return static_cast<Tout>(x);
    else if constexpr (std::is_same_v<Tout, cuDoubleComplex> &&
                       std::is_same_v<Tin, cuFloatComplex>) {
        return cuDoubleComplex{static_cast<double>(x.x), static_cast<double>(x.y)};
    } else if constexpr (std::is_same_v<Tout, cuFloatComplex> &&
                         std::is_same_v<Tin, cuDoubleComplex>) {
        return cuFloatComplex{static_cast<float>(x.x), static_cast<float>(x.y)};
    } else static_assert(dependent_false_type_v<Tout>, "Unsupported value_cast.");
}

template <typename T>
__host__ __device__ __forceinline__ T zero_value() { return scalar<T>(0.0); }

template <typename T>
__host__ __device__ __forceinline__ T one_value() { return scalar<T>(1.0); }

template <typename Tin, typename Tout>
__global__ void copy_cast_matrix_kernel(size_t rows, size_t cols, const Tin *__restrict__ in, size_t ldin, Tout *__restrict__ out, size_t ldout) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const size_t row       = idx % rows;
    const size_t col       = idx / rows;
    out[col * ldout + row] = value_cast<Tout>(in[col * ldin + row]);
}

template <typename Tin, typename Tout>
inline void copy_cast_matrix(size_t rows, size_t cols, const Tin *in, size_t ldin, Tout *out, size_t ldout, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;
    copy_cast_matrix_kernel<Tin, Tout><<<(rows * cols + 255) / 256, 256, 0, stream>>>(rows, cols, in, ldin, out, ldout);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T> __device__ __forceinline__ double real_part(const T x) { return static_cast<double>(x); }
template <> __device__ __forceinline__ double real_part<cuFloatComplex>(const cuFloatComplex x) { return static_cast<double>(x.x); }
template <> __device__ __forceinline__ double real_part<cuDoubleComplex>(const cuDoubleComplex x) { return x.x; }

template <typename T> __device__ __forceinline__ double imag_part(const T) { return 0.0; }
template <> __device__ __forceinline__ double imag_part<cuFloatComplex>(const cuFloatComplex x) { return static_cast<double>(x.y); }
template <> __device__ __forceinline__ double imag_part<cuDoubleComplex>(const cuDoubleComplex x) { return x.y; }

template <typename Texact, typename Tcalc>
__device__ __forceinline__ double l2_diff_sq(const Texact exact, const Tcalc calc) {
    const double err_re = real_part<Texact>(exact) - real_part<Tcalc>(calc);
    const double err_im = imag_part<Texact>(exact) - imag_part<Tcalc>(calc);
    return fma(err_re, err_re, err_im * err_im);
}

template <typename Texact, typename Tcalc>
__global__ void rms_error_kernel(
    size_t rows,
    size_t cols,
    const Texact *__restrict__ exact,
    size_t ld_exact,
    const Tcalc *__restrict__ calc,
    size_t ld_calc,
    double *__restrict__ partial //
) {
    extern __shared__ double s[];
    const size_t tid    = threadIdx.x;
    const size_t gid    = size_t(blockIdx.x) * blockDim.x + tid;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    const size_t size   = rows * cols;
    double sum          = 0.0;
    for (size_t idx = gid; idx < size; idx += stride) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        sum += l2_diff_sq<Texact, Tcalc>(exact[col * ld_exact + row], calc[col * ld_calc + row]);
    }
    s[tid] = sum;
    __syncthreads();
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

template <typename Tinit, typename Tcalc>
__global__ void padding_error_kernel(
    size_t valid_rows,
    size_t ld,
    size_t cols,
    const Tinit *__restrict__ init,
    size_t ld_init,
    const Tcalc *__restrict__ calc,
    size_t ld_calc,
    double *__restrict__ partial //
) {
    extern __shared__ double s[];
    const size_t pad_rows = ld - valid_rows;
    const size_t size     = pad_rows * cols;
    const size_t tid      = threadIdx.x;
    const size_t gid      = size_t(blockIdx.x) * blockDim.x + tid;
    const size_t stride   = size_t(blockDim.x) * gridDim.x;
    double sum            = 0.0;
    for (size_t idx = gid; idx < size; idx += stride) {
        const size_t pad_row = idx % pad_rows;
        const size_t col     = idx / pad_rows;
        const size_t row     = valid_rows + pad_row;
        sum += l2_diff_sq<Tinit, Tcalc>(init[col * ld_init + row], calc[col * ld_calc + row]);
    }
    s[tid] = sum;
    __syncthreads();
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

__device__ __forceinline__ bool in_uplo(cublasFillMode_t uplo, size_t row, size_t col) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? (row <= col) : (row >= col);
}

template <typename Texact, typename Tcalc>
__global__ void uplo_error_kernel(
    size_t n,
    cublasFillMode_t uplo,
    bool active,
    const Texact *__restrict__ exact,
    size_t ld_exact,
    const Tcalc *__restrict__ calc,
    size_t ld_calc,
    double *__restrict__ partial //
) {
    extern __shared__ double s[];
    const size_t tid    = threadIdx.x;
    const size_t gid    = size_t(blockIdx.x) * blockDim.x + tid;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    const size_t size   = n * n;
    double sum          = 0.0;
    for (size_t idx = gid; idx < size; idx += stride) {
        const size_t row = idx % n;
        const size_t col = idx / n;
        if (in_uplo(uplo, row, col) == active) {
            sum += l2_diff_sq<Texact, Tcalc>(exact[col * ld_exact + row], calc[col * ld_calc + row]);
        }
    }
    s[tid] = sum;
    __syncthreads();
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

__global__ void reduce_double_kernel(size_t size, const double *__restrict__ in, double *__restrict__ out) {
    extern __shared__ double s[];
    const size_t tid = threadIdx.x;
    const size_t idx = size_t(blockIdx.x) * blockDim.x + tid;
    s[tid]           = (idx < size) ? in[idx] : 0.0;
    __syncthreads();
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

inline size_t div_ceil(size_t x, size_t y) { return (x + y - 1) / y; }
inline size_t reduction_blocks(size_t size) { return std::min<size_t>(div_ceil(size, 256), 65535); }

inline double finish_rms_reduction(size_t num_partials, size_t rms_count, double *buf0, double *buf1, cudaStream_t stream) {
    size_t cur_size = num_partials;
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
    return std::sqrt(sumsq / static_cast<double>(rms_count));
}

template <typename KernelLaunch>
inline double calc_error_common(size_t launch_count, size_t rms_count, KernelLaunch &&launch, cudaStream_t stream) {
    if (rms_count == 0) return 0.0;
    const size_t blocks0 = reduction_blocks(std::max<size_t>(launch_count, 1));
    double *buf0         = nullptr;
    double *buf1         = nullptr;
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf0), blocks0 * sizeof(double), stream));
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void **>(&buf1), blocks0 * sizeof(double), stream));
    launch(blocks0, buf0);
    const double rms = finish_rms_reduction(blocks0, rms_count, buf0, buf1, stream);
    CHECK_CUDA(cudaFreeAsync(buf1, stream));
    CHECK_CUDA(cudaFreeAsync(buf0, stream));
    return rms;
}

template <typename Texact, typename Tcalc>
inline double calc_rms_error(size_t rows, size_t cols, const Texact *exact, size_t ld_exact, const Tcalc *calc, size_t ld_calc, cudaStream_t stream) {
    const size_t count = rows * cols;
    return calc_error_common(count, count, [&](size_t blocks, double *buf) {
        rms_error_kernel<Texact,Tcalc><<<blocks, 256, 256 * sizeof(double), stream>>>(rows, cols, exact, ld_exact, calc, ld_calc, buf);
        CHECK_CUDA(cudaGetLastError()); }, stream);
}

template <typename Tinit, typename Tcalc>
inline double calc_padding_rms_error(
    size_t valid_rows,
    size_t ld,
    size_t cols,
    const Tinit *init,
    size_t ld_init,
    const Tcalc *calc,
    size_t ld_calc,
    cudaStream_t stream) {
    if (ld <= valid_rows || cols == 0) return 0.0;
    const size_t count = (ld - valid_rows) * cols;
    return calc_error_common(count, count, [&](size_t blocks, double *buf) {
        padding_error_kernel<Tinit,Tcalc><<<blocks, 256, 256 * sizeof(double), stream>>>(valid_rows, ld, cols, init, ld_init, calc, ld_calc, buf);
        CHECK_CUDA(cudaGetLastError()); }, stream);
}

template <typename Texact, typename Tcalc>
inline double calc_uplo_rms_error(
    size_t n,
    cublasFillMode_t uplo,
    bool active,
    const Texact *exact,
    size_t ld_exact,
    const Tcalc *calc,
    size_t ld_calc,
    cudaStream_t stream //
) {
    const size_t count = active ? (n * (n + 1) / 2) : (n * (n - 1) / 2);
    return calc_error_common(n * n, count, [&](size_t blocks, double *buf) {
        uplo_error_kernel<Texact,Tcalc><<<blocks, 256, 256 * sizeof(double), stream>>>(n, uplo, active, exact, ld_exact, calc, ld_calc, buf);
        CHECK_CUDA(cudaGetLastError()); }, stream);
}

template <typename T>
__global__ void zero_diag_imag_kernel(size_t n, T *A, size_t lda) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if constexpr (std::is_same_v<T, cuFloatComplex> || std::is_same_v<T, cuDoubleComplex>) A[i * lda + i].y = 0;
}

template <typename T>
inline void zero_diag_imag(size_t n, T *A, size_t lda, cudaStream_t stream) {
    zero_diag_imag_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(n, A, lda);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void add_diag_shift_kernel(size_t n, T *A, size_t lda, double shift) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if constexpr (std::is_same_v<T, cuFloatComplex> || std::is_same_v<T, cuDoubleComplex>) A[i * lda + i].x += decltype(A[i * lda + i].x)(shift);
    else A[i * lda + i] += static_cast<T>(shift);
}

template <typename T>
inline void add_diag_shift(size_t n, T *A, size_t lda, cudaStream_t stream, double shift = triangular_diagonal_shift) {
    add_diag_shift_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(n, A, lda, shift);
    CHECK_CUDA(cudaGetLastError());
}

template <typename Tin, typename Tout>
__global__ void triangular_to_full_kernel(
    size_t n,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    const Tin *__restrict__ in,
    size_t ldin,
    Tout *__restrict__ out,
    size_t ldout //
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const size_t row = idx % n;
    const size_t col = idx / n;
    Tout v           = zero_value<Tout>();
    if (row == col && diag == CUBLAS_DIAG_UNIT) v = one_value<Tout>();
    else if (in_uplo(uplo, row, col)) v = value_cast<Tout>(in[col * ldin + row]);
    out[col * ldout + row] = v;
}

template <typename Tin, typename Tout>
inline void triangular_to_full(
    size_t n,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    const Tin *in,
    size_t ldin,
    Tout *out,
    size_t ldout,
    cudaStream_t stream //
) {
    triangular_to_full_kernel<Tin, Tout><<<(n * n + 255) / 256, 256, 0, stream>>>(n, uplo, diag, in, ldin, out, ldout);
    CHECK_CUDA(cudaGetLastError());
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

// Type runners
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

template <typename Runner>
inline bool run_all_single_input_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<float, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<float, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<float, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<float, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

template <typename Runner>
inline bool run_representative_single_input_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

template <typename Runner>
inline bool run_all_complex_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
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
inline bool run_all_complex_single_input_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

template <typename Runner>
inline bool run_representative_complex_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

template <typename Runner>
inline bool run_representative_complex_single_input_type_backend_cases(Runner &&runner) {
    bool ok = true;
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();
    return ok;
}

template <typename Runner>
inline bool run_all_trsm_type_backend_cases(Runner &&runner) {
    bool ok = true;

    ok &= runner.template operator()<float, float, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();

    ok &= runner.template operator()<float, float, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuFloatComplex, cuFloatComplex, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();

    return ok;
}

template <typename Runner>
inline bool run_representative_trsm_type_backend_cases(Runner &&runner) {
    bool ok = true;

    // gemmul8::trsm and gemmul8::trsmLt are instantiated only for TA == TB.
    ok &= runner.template operator()<double, double, gemmul8::Backend::INT8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::INT8>();

    ok &= runner.template operator()<double, double, gemmul8::Backend::FP8>();
    ok &= runner.template operator()<cuDoubleComplex, cuDoubleComplex, gemmul8::Backend::FP8>();

    return ok;
}

template <typename T>
inline double host_real_part(const T x) { return static_cast<double>(x); }

template <>
inline double host_real_part<cuFloatComplex>(const cuFloatComplex x) { return static_cast<double>(x.x); }

template <>
inline double host_real_part<cuDoubleComplex>(const cuDoubleComplex x) { return x.x; }

} // namespace debug_common
