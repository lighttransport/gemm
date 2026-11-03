#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <type_traits>

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cuComplex.h>
#endif

#include "self_hipify.hpp"

struct DD_real {
    double hi;
    double lo;

    __host__ __device__ constexpr DD_real() : hi(0.0), lo(0.0) {}
    __host__ __device__ constexpr DD_real(double hi_) : hi(hi_), lo(0.0) {}
    __host__ __device__ constexpr DD_real(double hi_, double lo_) : hi(hi_), lo(lo_) {}
};

struct DD_complex {
    DD_real re;
    DD_real im;

    __host__ __device__ constexpr DD_complex() : re(0.0), im(0.0) {}
    __host__ __device__ constexpr DD_complex(double re_) : re(re_), im(0.0) {}
    __host__ __device__ constexpr DD_complex(double re_, double im_) : re(re_), im(im_) {}
    __host__ __device__ constexpr DD_complex(DD_real re_) : re(re_), im(0.0) {}
    __host__ __device__ constexpr DD_complex(DD_real re_, DD_real im_) : re(re_), im(im_) {}
    __host__ __device__ constexpr DD_complex(cuFloatComplex reim) : re(double(reim.x)), im(double(reim.y)) {}
    __host__ __device__ constexpr DD_complex(cuDoubleComplex reim) : re(reim.x), im(reim.y) {}
};

struct DD_real_storage {
    double hi;
    double lo;
};

struct DD_complex_storage {
    DD_real_storage re;
    DD_real_storage im;
};

template <typename T>
struct eval_shared_storage_traits {
    using storage_t = T;

    __device__ __forceinline__ static storage_t pack(const T x) {
        return x;
    }

    __device__ __forceinline__ static T unpack(const storage_t x) {
        return x;
    }
};

template <>
struct eval_shared_storage_traits<DD_real> {
    using storage_t = DD_real_storage;

    __device__ __forceinline__ static storage_t pack(const DD_real x) {
        return DD_real_storage{x.hi, x.lo};
    }

    __device__ __forceinline__ static DD_real unpack(const storage_t x) {
        return DD_real{x.hi, x.lo};
    }
};

template <>
struct eval_shared_storage_traits<DD_complex> {
    using storage_t = DD_complex_storage;

    __device__ __forceinline__ static storage_t pack(const DD_complex x) {
        return DD_complex_storage{
            DD_real_storage{x.re.hi, x.re.lo},
            DD_real_storage{x.im.hi, x.im.lo}
        };
    }

    __device__ __forceinline__ static DD_complex unpack(const storage_t x) {
        return DD_complex{
            DD_real{x.re.hi, x.re.lo},
            DD_real{x.im.hi, x.im.lo}
        };
    }
};

namespace {

template <typename T> __host__ __device__ __forceinline__ T Tzero() { return T(0); };
template <> __host__ __device__ __forceinline__ cuFloatComplex Tzero<cuFloatComplex>() { return {0.0f, 0.0f}; };
template <> __host__ __device__ __forceinline__ cuDoubleComplex Tzero<cuDoubleComplex>() { return {0.0, 0.0}; };

template <typename T> __host__ __device__ __forceinline__ T Tone() { return T(1); };
template <> __host__ __device__ __forceinline__ cuFloatComplex Tone<cuFloatComplex>() { return {1.0f, 0.0f}; };
template <> __host__ __device__ __forceinline__ cuDoubleComplex Tone<cuDoubleComplex>() { return {1.0, 0.0}; };

template <typename T> __host__ __device__ __forceinline__ T Tmone() { return T(-1); };
template <> __host__ __device__ __forceinline__ cuFloatComplex Tmone<cuFloatComplex>() { return {-1.0f, 0.0f}; };
template <> __host__ __device__ __forceinline__ cuDoubleComplex Tmone<cuDoubleComplex>() { return {-1.0, 0.0}; };

template <typename T> __device__ __forceinline__ T Tabs(T in);
template <> __device__ __forceinline__ double Tabs<double>(double in) { return fabs(in); }
template <> __device__ __forceinline__ float Tabs<float>(float in) { return fabsf(in); }
template <> __device__ __forceinline__ cuDoubleComplex Tabs<cuDoubleComplex>(cuDoubleComplex in) { return cuDoubleComplex{fabs(in.x), fabs(in.y)}; }
template <> __device__ __forceinline__ cuFloatComplex Tabs<cuFloatComplex>(cuFloatComplex in) { return cuFloatComplex{fabsf(in.x), fabsf(in.y)}; }

template <typename T> inline constexpr bool isComplex        = false;
template <> inline constexpr bool isComplex<cuFloatComplex>  = true;
template <> inline constexpr bool isComplex<cuDoubleComplex> = true;
template <> inline constexpr bool isComplex<DD_complex>      = true;

template <typename T0, typename... Ts>
inline constexpr bool same_complex_domain_v = ((isComplex<Ts> == isComplex<T0>) && ...);

template <typename T> struct underlying_type {
    using type = T;
};
template <> struct underlying_type<cuFloatComplex> {
    using type = float;
};
template <> struct underlying_type<cuDoubleComplex> {
    using type = double;
};
template <> struct underlying_type<DD_complex> {
    using type = DD_real;
};
template <typename T> using underlying_t = typename underlying_type<T>::type;

template <typename T> struct ACCU_type {
    using type = DD_real;
};
template <> struct ACCU_type<cuFloatComplex> {
    using type = DD_complex;
};
template <> struct ACCU_type<cuDoubleComplex> {
    using type = DD_complex;
};
template <> struct ACCU_type<DD_complex> {
    using type = DD_complex;
};
template <typename T> using ACCU_t = typename ACCU_type<T>::type;

template <typename T> __host__ __device__ __forceinline__ T cast_from_dd(const DD_real a) {
    return a;
}
template <> __host__ __device__ __forceinline__ double cast_from_dd<double>(const DD_real a) {
    return a.hi;
}
template <> __host__ __device__ __forceinline__ float cast_from_dd<float>(const DD_real a) {
    return float(a.hi);
}

template <typename T> __host__ __device__ __forceinline__ T cast_from_dd(const DD_complex a) {
    return a;
}
template <> __host__ __device__ __forceinline__ cuDoubleComplex cast_from_dd<cuDoubleComplex>(const DD_complex a) {
    return cuDoubleComplex{a.re.hi, a.im.hi};
}
template <> __host__ __device__ __forceinline__ cuFloatComplex cast_from_dd<cuFloatComplex>(const DD_complex a) {
    return cuFloatComplex{float(a.re.hi), float(a.im.hi)};
}

__host__ __device__ __forceinline__ double FMA(const double a, const double b, const double c) {
#ifdef __CUDA_ARCH__
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

__device__ __host__ __forceinline__ DD_real fast_two_sum(const double a, const double b) {
    const double s = a + b;
    const double e = (a - s) + b;
    return DD_real{s, e};
}

__device__ __host__ __forceinline__ DD_real two_sum(const double a, const double b) {
    const double s  = a + b;
    const double bb = s - a;
    const double t  = b - bb;
    const double u  = s - bb;
    const double e  = (a - u) + t;
    return DD_real{s, e};
}

__device__ __host__ __forceinline__ DD_real two_prod(const double a, const double b) {
    const double s = a * b;
    const double e = FMA(a, b, -s);
    return DD_real{s, e};
}

// dd arithmetic (real)
__device__ __host__ __forceinline__ DD_real operator+(const DD_real a, const DD_real b) {
    const DD_real s = two_sum(a.hi, b.hi);
    const double e  = s.lo + a.lo + b.lo;
    return fast_two_sum(s.hi, e);
}

__device__ __host__ __forceinline__ DD_real operator-(const DD_real a) {
    return DD_real{-a.hi, -a.lo};
}

__device__ __host__ __forceinline__ DD_real operator-(const DD_real a, const DD_real b) {
    return a + (-b);
}

__device__ __host__ __forceinline__ DD_real operator*(const DD_real a, const DD_real b) {
    const DD_real p = two_prod(a.hi, b.hi);
    const double e  = FMA(a.hi, b.lo, FMA(a.lo, b.hi, p.lo));
    return fast_two_sum(p.hi, e);
}

__device__ __host__ __forceinline__ DD_real operator/(const DD_real a, const DD_real b) {
    const double q1 = a.hi / b.hi;
    const DD_real r = two_prod(q1, b.hi);
    const double s  = a.hi - r.hi;
    const double t  = s - r.lo;
    const double u  = t + a.lo;
    const double v  = FMA(-q1, b.lo, u);
    const double q2 = v / b.hi;
    return fast_two_sum(q1, q2);
}

__device__ __host__ __forceinline__ DD_real &operator+=(DD_real &a, const DD_real b) {
    a = a + b;
    return a;
}

__device__ __host__ __forceinline__ DD_real &operator-=(DD_real &a, const DD_real b) {
    a = a - b;
    return a;
}

__device__ __host__ __forceinline__ DD_real &operator*=(DD_real &a, const DD_real b) {
    a = a * b;
    return a;
}

__device__ __host__ __forceinline__ DD_real &operator/=(DD_real &a, const DD_real b) {
    a = a / b;
    return a;
}

// dd arithmetic (complex)
__device__ __host__ __forceinline__ DD_complex operator+(const DD_complex a, const DD_complex b) {
    return DD_complex{a.re + b.re, a.im + b.im};
}

__device__ __host__ __forceinline__ DD_complex operator-(const DD_complex a) {
    return DD_complex{-a.re, -a.im};
}

__device__ __host__ __forceinline__ DD_complex operator-(const DD_complex a, const DD_complex b) {
    return DD_complex{a.re - b.re, a.im - b.im};
}

__device__ __host__ __forceinline__ DD_complex operator*(const DD_complex a, const DD_complex b) {
    return DD_complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __host__ __forceinline__ DD_complex operator/(const DD_complex a, const DD_complex b) {
    const DD_real den = b.re * b.re + b.im * b.im;
    return DD_complex{(a.re * b.re + a.im * b.im) / den, (a.im * b.re - a.re * b.im) / den};
}

__device__ __host__ __forceinline__ DD_complex &operator+=(DD_complex &a, const DD_complex b) {
    a = a + b;
    return a;
}

__device__ __host__ __forceinline__ DD_complex &operator-=(DD_complex &a, const DD_complex b) {
    a = a - b;
    return a;
}

__device__ __host__ __forceinline__ DD_complex &operator*=(DD_complex &a, const DD_complex b) {
    a = a * b;
    return a;
}

__device__ __host__ __forceinline__ DD_complex &operator/=(DD_complex &a, const DD_complex b) {
    a = a / b;
    return a;
}

template <typename T> __host__ __device__ __forceinline__ T conj(const T x) {
    return x;
}
template <> __host__ __device__ __forceinline__ cuFloatComplex conj<cuFloatComplex>(const cuFloatComplex x) {
    return cuFloatComplex{x.x, -x.y};
}
template <> __host__ __device__ __forceinline__ cuDoubleComplex conj<cuDoubleComplex>(const cuDoubleComplex x) {
    return cuDoubleComplex{x.x, -x.y};
}
template <> __host__ __device__ __forceinline__ DD_complex conj<DD_complex>(const DD_complex x) {
    return DD_complex{x.re, -x.im};
}

template <typename T>
__global__ void DDaxpby_kernel(
    const size_t m, const size_t n,
    const T alpha,
    const T *const __restrict__ X, size_t ldx,
    const T beta,
    T *const __restrict__ Y, size_t ldy,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL //
) {
    const size_t idx = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
    if (idx >= m * n) return;

    const size_t col = idx / m;
    const size_t row = idx - col * m;

    if (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (row > col) return;
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (row < col) return;
    }

    const size_t idx_x = col * ldx + row;
    const size_t idx_y = col * ldy + row;

    using accu_t = ACCU_t<T>;

    Y[idx_y] = cast_from_dd<T>(accu_t(alpha) * accu_t(X[idx_x]) + accu_t(beta) * accu_t(Y[idx_y]));
}

template <typename T>
void DDaxpby(
    const size_t m, const size_t n,
    const T alpha,
    const T *const X, size_t ldx,
    const T beta,
    T *const Y, size_t ldy,
    const cudaStream_t stream   = 0,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL //
) {
    DDaxpby_kernel<T><<<(m * n + 255) / 256, 256, 0, stream>>>(
        m, n, alpha, X, ldx, beta, Y, ldy, UPLO);
}

template <bool CONJ, cublasFillMode_t UPLO, typename T>
__global__ void tri_2_sym_kernel(
    const size_t n,
    const T *__restrict__ A, const size_t lda,
    T *__restrict__ B, const size_t ldb //
) {
    constexpr int TILE = 32;

    using ShmT = typename eval_shared_storage_traits<T>::storage_t;
    __shared__ ShmT smem[TILE][TILE + 1];

    size_t load_row = blockDim.x * blockIdx.x + threadIdx.x;
    size_t load_col = blockDim.y * blockIdx.y + threadIdx.y;

    if (load_row < n && load_col < n) {
        bool load = true;

        if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
            load = (load_row < load_col);
        } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
            load = (load_row > load_col);
        }

        if (load) {
            smem[threadIdx.x][threadIdx.y] = eval_shared_storage_traits<T>::pack(A[load_col * lda + load_row]);
        }
    }

    __syncthreads();

    const size_t row = blockDim.y * blockIdx.y + threadIdx.x;
    const size_t col = blockDim.x * blockIdx.x + threadIdx.y;

    if (row >= n || col >= n) return;

    if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (row <= col) return; // fill lower triangle only
    } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (row >= col) return; // fill upper triangle only
    }

    const T tmp = eval_shared_storage_traits<T>::unpack(smem[threadIdx.y][threadIdx.x]);
    if constexpr (CONJ) {
        B[col * ldb + row] = conj(tmp);
    } else {
        B[col * ldb + row] = tmp;
    }
}

template <bool CONJ, cublasFillMode_t UPLO, typename T>
__forceinline__ void tri_2_sym_launch(
    const size_t n,
    const T *A, const size_t lda,
    T *B, const size_t ldb,
    const cudaStream_t stream = 0 //
) {
    cudaMemcpy2DAsync(B, ldb * sizeof(T),
                      A, lda * sizeof(T),
                      n * sizeof(T), n, cudaMemcpyDeviceToDevice, stream);

    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) return;

    dim3 threads(32, 32);
    dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (n + threads.y - 1) / threads.y);

    tri_2_sym_kernel<CONJ, UPLO, T>
        <<<blocks, threads, 0, stream>>>(
            n, A, lda, B, ldb);
}

template <bool CONJ, typename T>
void tri_2_sym(
    const size_t n,
    const T *const A, const size_t lda,
    T *const B, const size_t ldb,
    const cudaStream_t stream   = 0,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL //
) {
    if (UPLO == CUBLAS_FILL_MODE_UPPER) {
        tri_2_sym_launch<CONJ, CUBLAS_FILL_MODE_UPPER, T>(
            n, A, lda, B, ldb, stream);
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER) {
        tri_2_sym_launch<CONJ, CUBLAS_FILL_MODE_LOWER, T>(
            n, A, lda, B, ldb, stream);
    } else {
        tri_2_sym_launch<CONJ, CUBLAS_FILL_MODE_FULL, T>(
            n, A, lda, B, ldb, stream);
    }
}

template <cublasFillMode_t UPLO, cublasDiagType_t DIAG, typename T>
__global__ void tri_2_full_kernel(
    const size_t n,
    T *__restrict__ A, const size_t lda //
) {
    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) return;

    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n * n) return;

    const auto col    = idx / n;
    const auto row    = idx - col * n;
    const auto idx_in = col * lda + row;

    if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (row > col) {
            A[idx_in] = Tzero<T>();
        } else if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
            if (row == col) A[idx_in] = Tone<T>();
        }
    } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (row < col) {
            A[idx_in] = Tzero<T>();
        } else if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
            if (row == col) A[idx_in] = Tone<T>();
        }
    }
}

template <typename T>
__global__ void eye_kernel(const size_t n, T *__restrict__ A) {
    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n * n) return;
    const auto col = idx / n;
    const auto row = idx - col * n;
    A[idx]         = (row == col) ? Tone<T>() : Tzero<T>();
}

template <typename T>
inline void eye(const size_t n, T *A, const cudaStream_t stream = 0) {
    eye_kernel<T><<<(n * n + 255) / 256, 256, 0, stream>>>(n, A);
}

template <typename Tin, typename Tout>
__global__ void addvec_kernel(
    const size_t n,
    const Tin *__restrict__ A,
    Tout *__restrict__ B //
) {
    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    using accu_t = ACCU_t<Tout>;
    B[idx]       = cast_from_dd<Tout>(accu_t(A[idx]) + accu_t(B[idx]));
}

template <typename Tin, typename Tout> __device__ __forceinline__ Tout casting(Tin in) { return Tout(in); };

template <> __device__ __forceinline__ float casting<double, float>(double in) { return __double2float_rn(in); }
template <> __device__ __forceinline__ double casting<double, double>(double in) { return in; }
template <> __device__ __forceinline__ cuFloatComplex casting<double, cuFloatComplex>(double in) { return cuFloatComplex{__double2float_rn(in), 0.0f}; }
template <> __device__ __forceinline__ cuDoubleComplex casting<double, cuDoubleComplex>(double in) { return cuDoubleComplex{in, 0.0}; }

// --- Tin = float ---
template <> __device__ __forceinline__ float casting<float, float>(float in) { return in; }
template <> __device__ __forceinline__ double casting<float, double>(float in) { return static_cast<double>(in); }
template <> __device__ __forceinline__ cuFloatComplex casting<float, cuFloatComplex>(float in) { return cuFloatComplex{in, 0.0f}; }
template <> __device__ __forceinline__ cuDoubleComplex casting<float, cuDoubleComplex>(float in) { return cuDoubleComplex{double(in), 0.0}; }

// --- Tin = cuFloatComplex ---
template <> __device__ __forceinline__ float casting<cuFloatComplex, float>(cuFloatComplex in) { return in.x; }
template <> __device__ __forceinline__ double casting<cuFloatComplex, double>(cuFloatComplex in) { return static_cast<double>(in.x); }
template <> __device__ __forceinline__ cuFloatComplex casting<cuFloatComplex, cuFloatComplex>(cuFloatComplex in) { return in; }
template <> __device__ __forceinline__ cuDoubleComplex casting<cuFloatComplex, cuDoubleComplex>(cuFloatComplex in) { return cuDoubleComplex{static_cast<double>(in.x), static_cast<double>(in.y)}; }

// --- Tin = cuDoubleComplex ---
template <> __device__ __forceinline__ float casting<cuDoubleComplex, float>(cuDoubleComplex in) { return __double2float_rn(in.x); }
template <> __device__ __forceinline__ double casting<cuDoubleComplex, double>(cuDoubleComplex in) { return in.x; }
template <> __device__ __forceinline__ cuFloatComplex casting<cuDoubleComplex, cuFloatComplex>(cuDoubleComplex in) { return cuFloatComplex{__double2float_rn(in.x), __double2float_rn(in.y)}; }
template <> __device__ __forceinline__ cuDoubleComplex casting<cuDoubleComplex, cuDoubleComplex>(cuDoubleComplex in) { return in; }

template <typename Tin, typename Tout = Tin>
__global__ void cast_kernel(
    const size_t n,
    const Tin *A, const size_t lda,
    Tout *B, const size_t ldb //
) {
    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= n * n) return;
    const size_t col     = idx / n;
    const size_t row     = idx - col * n;
    const size_t idx_in  = col * lda + row;
    const size_t idx_out = col * ldb + row;
    B[idx_out]           = casting<Tin, Tout>(A[idx_in]);
}

template <cublasFillMode_t UPLO, cublasDiagType_t DIAG, typename Tin, typename Tout = Tin>
__forceinline__ void tri_2_full_launch(
    const size_t n,
    const Tin *A, const size_t lda,
    Tout *B, const size_t ldb,
    const cudaStream_t stream = 0 //
) {
    if constexpr (std::is_same_v<Tin, Tout>) {
        cudaMemcpy2DAsync(B, ldb * sizeof(Tin),
                          A, lda * sizeof(Tin),
                          n * sizeof(Tin), n, cudaMemcpyDeviceToDevice, stream);
    } else {
        cast_kernel<Tin, Tout><<<(n * n + 255) / 256, 256, 0, stream>>>(
            n, A, lda, B, ldb);
    }

    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) return;

    tri_2_full_kernel<UPLO, DIAG, Tout>
        <<<(n * n + 255) / 256, 256, 0, stream>>>(
            n, B, ldb);
}

template <cublasFillMode_t UPLO, cublasDiagType_t DIAG, typename Tin>
__forceinline__ void tri_2_full_launch(
    const size_t n,
    Tin *A, const size_t lda,
    const cudaStream_t stream = 0 //
) {
    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) return;

    tri_2_full_kernel<UPLO, DIAG, Tin>
        <<<(n * n + 255) / 256, 256, 0, stream>>>(
            n, A, lda);
}

// C := D + D^H + beta*C
template <bool CONJ, cublasFillMode_t UPLO, typename TD, typename TC>
__global__ void her2k_final_kernel(
    const size_t n,
    const TD *__restrict__ D, const size_t ldd,
    const TC beta,
    TC *__restrict__ C, const size_t ldc //
) {
    constexpr int TILE = 32;

    using ShmTD = typename eval_shared_storage_traits<TD>::storage_t;

    __shared__ ShmTD smem[TILE][TILE + 1];

    size_t load_row = blockDim.x * blockIdx.x + threadIdx.x;
    size_t load_col = blockDim.y * blockIdx.y + threadIdx.y;

    if (load_row < n && load_col < n) {
        bool load = true;

        if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
            load = (load_row >= load_col);
        } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
            load = (load_row <= load_col);
        }

        if (load) {
            smem[threadIdx.x][threadIdx.y] = eval_shared_storage_traits<TD>::pack(D[load_col * ldd + load_row]);
        }
    }

    __syncthreads();

    const size_t row = blockDim.y * blockIdx.y + threadIdx.x;
    const size_t col = blockDim.x * blockIdx.x + threadIdx.y;

    if (row >= n || col >= n) return;

    if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (row > col) return;
    } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (row < col) return;
    }

    const TD D_ij     = D[col * ldd + row];
    const TD D_ji_raw = eval_shared_storage_traits<TD>::unpack(smem[threadIdx.y][threadIdx.x]);
    const TD D_ji     = (CONJ) ? conj(D_ji_raw) : D_ji_raw;

    using accu_t      = ACCU_t<TC>;
    const size_t idxC = col * ldc + row;
    const accu_t out  = accu_t(D_ij) + accu_t(D_ji) + accu_t(beta) * accu_t(C[idxC]);
    C[idxC]           = cast_from_dd<TC>(out);
}

template <bool CONJ, cublasFillMode_t UPLO, typename TD, typename TC>
__forceinline__ void her2k_final_launch(
    const size_t n,
    const TD *const D, const size_t ldd,
    const TC beta,
    TC *const C, const size_t ldc,
    const cudaStream_t stream = 0 //
) {
    dim3 threads(32, 32);
    dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (n + threads.y - 1) / threads.y);

    her2k_final_kernel<CONJ, UPLO, TD, TC>
        <<<blocks, threads, 0, stream>>>(
            n, D, ldd, beta, C, ldc);
}

template <bool CONJ, typename TD, typename TC>
void her2k_final(
    const size_t n,
    const TD *const D, const size_t ldd,
    const TC beta,
    TC *const C, const size_t ldc,
    const cudaStream_t stream   = 0,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL //
) {
    if (UPLO == CUBLAS_FILL_MODE_UPPER) {
        her2k_final_launch<CONJ, CUBLAS_FILL_MODE_UPPER, TD, TC>(
            n, D, ldd, beta, C, ldc, stream);
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER) {
        her2k_final_launch<CONJ, CUBLAS_FILL_MODE_LOWER, TD, TC>(
            n, D, ldd, beta, C, ldc, stream);
    } else {
        her2k_final_launch<CONJ, CUBLAS_FILL_MODE_FULL, TD, TC>(
            n, D, ldd, beta, C, ldc, stream);
    }
}

template <typename T, cublasOperation_t OP_A, bool CONJ = false>
__device__ __forceinline__ T load_A_element(
    const T *const __restrict__ A, const size_t lda,
    const size_t m, const size_t k,
    const size_t row, const size_t a_col //
) {
    T v;

    if constexpr (OP_A == CUBLAS_OP_N) {
        // op(A)(row, a_col) = A(row, a_col)
        v = (row < m && a_col < k) ? A[row + a_col * lda] : T{};
    } else {
        // op(A)(row, a_col) = A(a_col, row)
        v = (row < m && a_col < k) ? A[a_col + row * lda] : T{};
    }

    if constexpr (OP_A == CUBLAS_OP_C || CONJ) {
        return conj(v);
    } else {
        return v;
    }
}

template <typename T, cublasOperation_t OP_B, bool CONJ = false>
__device__ __forceinline__ T load_B_element(
    const T *const __restrict__ B, const size_t ldb,
    const size_t k, const size_t n,
    const size_t b_row, const size_t col //
) {
    T v;

    if constexpr (OP_B == CUBLAS_OP_N) {
        // op(B)(b_row, col) = B(b_row, col)
        v = (b_row < k && col < n) ? B[b_row + col * ldb] : T{};
    } else {
        // op(B)(b_row, col) = B(col, b_row)
        v = (b_row < k && col < n) ? B[col + b_row * ldb] : T{};
    }

    if constexpr (OP_B == CUBLAS_OP_C || CONJ) {
        return conj(v);
    } else {
        return v;
    }
}

template <typename TA, typename TB>
inline constexpr int DDGEMM_TILE = ((sizeof(TA) + sizeof(TB)) >= 48) ? 16 : 32;

template <cublasOperation_t opA, cublasOperation_t opB, bool CONJ_B,
          typename TA, typename TB, typename TC, typename accu_t>
__global__ void DDgemm_kernel(
    size_t m, size_t n, size_t k,
    const TC alpha,
    const TA *const __restrict__ A, size_t lda,
    const TB *const __restrict__ B, size_t ldb,
    const TC beta,
    TC *const __restrict__ C, size_t ldc //
) {
    constexpr int TILE = DDGEMM_TILE<TA, TB>;
    using ShmTA        = typename eval_shared_storage_traits<TA>::storage_t;
    using ShmTB        = typename eval_shared_storage_traits<TB>::storage_t;
    __shared__ ShmTA Asub[TILE][TILE + 1];
    __shared__ ShmTB Bsub[TILE][TILE + 1];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    accu_t sum{};

    int numTiles = (int)((k + TILE - 1) / TILE);

    for (int t = 0; t < numTiles; ++t) {
        size_t a_col                   = t * TILE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = eval_shared_storage_traits<TA>::pack(load_A_element<TA, opA>(A, lda, m, k, row, a_col));

        size_t b_row                   = t * TILE + threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] = eval_shared_storage_traits<TB>::pack(load_B_element<TB, opB, CONJ_B>(B, ldb, k, n, b_row, col));

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i) {

            const TA a_val = eval_shared_storage_traits<TA>::unpack(Asub[threadIdx.y][i]);
            const TB b_val = eval_shared_storage_traits<TB>::unpack(Bsub[i][threadIdx.x]);
            auto a         = accu_t(a_val);
            auto b         = accu_t(b_val);

            sum = a * b + sum;
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[col * ldc + row] = cast_from_dd<TC>(accu_t(alpha) * sum + accu_t(beta) * accu_t(C[col * ldc + row]));
    }
}

template <typename TC>
__host__ __device__ __forceinline__ TC make_scalar(underlying_t<TC> x) {
    if constexpr (isComplex<TC>) {
        return TC{x, underlying_t<TC>(0)};
    } else {
        return TC(x);
    }
}

} // namespace

namespace eval {

template <typename Tin, typename Tout>
inline void addvec(
    const size_t n,
    const Tin *A,
    Tout *B,
    const cudaStream_t stream = 0 //
) {
    addvec_kernel<Tin, Tout><<<(n + 255) / 256, 256, 0, stream>>>(n, A, B);
}

template <typename Tin, typename Tout = Tin>
void tri_2_full(
    const size_t n,
    const Tin *const A, const size_t lda,
    Tout *const B, const size_t ldb,
    const cudaStream_t stream   = 0,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL,
    const cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT //
) {
    if (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (DIAG == CUBLAS_DIAG_NON_UNIT) {
            tri_2_full_launch<CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, Tin, Tout>(
                n, A, lda, B, ldb, stream);
        } else {
            tri_2_full_launch<CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_UNIT, Tin, Tout>(
                n, A, lda, B, ldb, stream);
        }
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (DIAG == CUBLAS_DIAG_NON_UNIT) {
            tri_2_full_launch<CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT, Tin, Tout>(
                n, A, lda, B, ldb, stream);
        } else {
            tri_2_full_launch<CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, Tin, Tout>(
                n, A, lda, B, ldb, stream);
        }
    } else {
        tri_2_full_launch<CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, Tin, Tout>(
            n, A, lda, B, ldb, stream);
    }
}

template <typename Tin>
void tri_2_full(
    const size_t n,
    Tin *const A, const size_t lda,
    const cudaStream_t stream   = 0,
    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL,
    const cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT //
) {
    if (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (DIAG == CUBLAS_DIAG_NON_UNIT) {
            tri_2_full_launch<CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, Tin>(
                n, A, lda, stream);
        } else {
            tri_2_full_launch<CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_UNIT, Tin>(
                n, A, lda, stream);
        }
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (DIAG == CUBLAS_DIAG_NON_UNIT) {
            tri_2_full_launch<CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT, Tin>(
                n, A, lda, stream);
        } else {
            tri_2_full_launch<CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, Tin>(
                n, A, lda, stream);
        }
    } else {
        tri_2_full_launch<CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, Tin>(
            n, A, lda, stream);
    }
}

// C := op(A)*op(B)
template <typename TA, typename TB, typename TC>
void DDgemm(
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    size_t m, size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0,
    const bool conj_b         = false //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDgemm requires TA, TB, and TC to be all real or all complex.");

    using accu_t       = ACCU_t<TC>;
    constexpr int TILE = DDGEMM_TILE<TA, TB>;
    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (op_A == CUBLAS_OP_N) {
        if (op_B == CUBLAS_OP_N) {
            if (conj_b) {
                DDgemm_kernel<CUBLAS_OP_N, CUBLAS_OP_N, true, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_N, CUBLAS_OP_N, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }

        } else {
            if (op_B == CUBLAS_OP_C || conj_b) {
                DDgemm_kernel<CUBLAS_OP_N, CUBLAS_OP_C, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_N, CUBLAS_OP_T, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }
    } else if (op_A == CUBLAS_OP_T) {
        if (op_B == CUBLAS_OP_N) {
            if (conj_b) {
                DDgemm_kernel<CUBLAS_OP_T, CUBLAS_OP_N, true, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_T, CUBLAS_OP_N, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        } else {
            if (op_B == CUBLAS_OP_C || conj_b) {
                DDgemm_kernel<CUBLAS_OP_T, CUBLAS_OP_C, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_T, CUBLAS_OP_T, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }
    } else {
        if (op_B == CUBLAS_OP_N) {
            if (conj_b) {
                DDgemm_kernel<CUBLAS_OP_C, CUBLAS_OP_N, true, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_C, CUBLAS_OP_N, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        } else {
            if (op_B == CUBLAS_OP_C || conj_b) {
                DDgemm_kernel<CUBLAS_OP_C, CUBLAS_OP_C, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                DDgemm_kernel<CUBLAS_OP_C, CUBLAS_OP_T, false, TA, TB, TC, accu_t>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }
    }
}

template <typename T>
__global__ void make_hermitian_diag_real_kernel(
    const size_t n,
    T *const A,
    const size_t lda //
) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        A[i + i * lda].y = 0.0f;
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        A[i + i * lda].y = 0.0;
    } else if constexpr (std::is_same_v<T, DD_complex>) {
        A[i + i * lda].im = DD_real(0.0);
    }
}

template <typename T>
inline void make_hermitian_diag_real(
    const size_t n,
    T *const A,
    const size_t lda,
    const cudaStream_t stream //
) {
    make_hermitian_diag_real_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(n, A, lda);
}

// C := alpha*op(A)*op(B)^H + beta*C
template <typename TA, typename TC>
void DDherk(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const underlying_t<TC> alpha,
    const TA *const A, size_t lda,
    const underlying_t<TC> beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TC>,
                  "DDherk requires TA and TC to be both real or both complex.");
    static_assert(isComplex<TA>, "DDherk requires complex input type.");

    const cublasOperation_t transA = trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_C : CUBLAS_OP_N;
    const bool conj_B              = (trans == CUBLAS_OP_T) ? true : false;

    const TC alpha_ = make_scalar<TC>(alpha);
    const TC beta_  = make_scalar<TC>(beta);

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // workC := alpha*op(A)*op(A)^H
    DDgemm<TA, TA, TC>(transA, transB, n, n, k, alpha_, A, lda, A, lda, Tzero<TC>(), workC, n, stream, conj_B);

    // C := workC + beta*C
    DDaxpby(n, n, Tone<TC>(), workC, n, beta_, C, ldc, stream, uplo);

    make_hermitian_diag_real(n, C, ldc, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

// C := alpha*op(A)*op(B)^H + beta*C
template <typename TA, typename TB, typename TC>
void DDherkx(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const underlying_t<TC> beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDherkx requires TA, TB, and TC to be all real or all complex.");
    static_assert(isComplex<TA>, "DDherkx requires complex input type.");

    const cublasOperation_t transA = trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_C : CUBLAS_OP_N;
    const bool conj_B              = (trans == CUBLAS_OP_T) ? true : false;

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const TC beta_ = make_scalar<TC>(beta);

    // workC := alpha*op(A)*op(B)^H
    DDgemm<TA, TB, TC>(transA, transB, n, n, k, alpha, A, lda, B, ldb, Tzero<TC>(), workC, n, stream, conj_B);

    // C := workC + beta*C
    DDaxpby(n, n, Tone<TC>(), workC, n, beta_, C, ldc, stream, uplo);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

// C := alpha*op(A)*op(B)^H + conj(alpha)*op(B)*op(A)^H + beta*C
template <typename TA, typename TB, typename TC>
void DDher2k(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const underlying_t<TC> beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDher2k requires TA, TB, and TC to be all real or all complex.");
    static_assert(isComplex<TA>, "DDher2k requires complex input type.");

    const cublasOperation_t transA = trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_C : CUBLAS_OP_N;
    const bool conj_B              = (trans == CUBLAS_OP_T) ? true : false;

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const TC beta_ = make_scalar<TC>(beta);

    // workC := alpha*op(A)*op(B)^H
    DDgemm<TA, TB, TC>(transA, transB, n, n, k, alpha, A, lda, B, ldb, Tzero<TC>(), workC, n, stream, conj_B);

    // C := workC + workC^H + beta*C
    her2k_final<true, TC, TC>(n, workC, n, beta_, C, ldc, stream, uplo);

    make_hermitian_diag_real(n, C, ldc, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

template <typename TA, typename TC>
void DDsyrk(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TC>,
                  "DDsyrk requires TA and TC to be both real or both complex.");

    const cublasOperation_t transA = (trans == CUBLAS_OP_C) ? CUBLAS_OP_T : trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // workC := alpha*op(A)*op(A)^T
    DDgemm<TA, TA, TC>(transA, transB, n, n, k, alpha, A, lda, A, lda, Tzero<TC>(), workC, n, stream);

    // C := workC + beta*C
    DDaxpby(n, n, Tone<TC>(), workC, n, beta, C, ldc, stream, uplo);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

template <typename TA, typename TB, typename TC>
void DDsyrkx(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDsyrkx requires TA, TB, and TC to be all real or all complex.");

    const cublasOperation_t transA = (trans == CUBLAS_OP_C) ? CUBLAS_OP_T : trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // workC := alpha*op(A)*op(B)^T
    DDgemm<TA, TB, TC>(transA, transB, n, n, k, alpha, A, lda, B, ldb, Tzero<TC>(), workC, n, stream);

    // C := workC + beta*C
    DDaxpby(n, n, Tone<TC>(), workC, n, beta, C, ldc, stream, uplo);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

template <typename TA, typename TB, typename TC>
void DDsyr2k(
    const cublasFillMode_t uplo,
    const cublasOperation_t trans,
    size_t n, size_t k,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDsyr2k requires TA, TB, and TC to be all real or all complex.");

    const cublasOperation_t transA = (trans == CUBLAS_OP_C) ? CUBLAS_OP_T : trans;
    const cublasOperation_t transB = (trans == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;

    TC *workC;
    cudaMallocAsync(reinterpret_cast<void **>(&workC), n * n * sizeof(TC), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // workC := alpha*op(A)*op(B)^T
    DDgemm<TA, TB, TC>(transA, transB, n, n, k, alpha, A, lda, B, ldb, Tzero<TC>(), workC, n, stream);

    // C := workC + workC^T + beta*C
    her2k_final<false, TC, TC>(n, workC, n, beta, C, ldc, stream, uplo);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workC, stream);
}

template <typename TA, typename TB, typename TC>
void DDsymm(
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    size_t m, size_t n,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDsymm requires TA, TB, and TC to be all real or all complex.");

    TA *workA;

    if (side == CUBLAS_SIDE_LEFT) {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), m * m * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_sym<false, TA>(m, A, lda, workA, m, stream, uplo);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, alpha, workA, m, B, ldb, beta, C, ldc, stream);

    } else {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), n * n * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_sym<false, TA>(n, A, lda, workA, n, stream, uplo);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, B, ldb, workA, n, beta, C, ldc, stream);
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workA, stream);
}

template <typename TA, typename TB, typename TC>
void DDhemm(
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    size_t m, size_t n,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDhemm requires TA, TB, and TC to be all real or all complex.");

    TA *workA;

    if (side == CUBLAS_SIDE_LEFT) {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), m * m * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_sym<true, TA>(m, A, lda, workA, m, stream, uplo);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, alpha, workA, m, B, ldb, beta, C, ldc, stream);

    } else {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), n * n * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_sym<true, TA>(n, A, lda, workA, n, stream, uplo);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, B, ldb, workA, n, beta, C, ldc, stream);
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workA, stream);
}

template <typename TA, typename TB, typename TC>
void DDtrmm(
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    size_t m, size_t n,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDtrmm requires TA, TB, and TC to be all real or all complex.");

    TA *workA;

    if (side == CUBLAS_SIDE_LEFT) {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), m * m * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_full<TA>(m, A, lda, workA, m, stream, uplo, diag);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(trans, CUBLAS_OP_N, m, n, m, alpha, workA, m, B, ldb, Tzero<TC>(), C, ldc, stream);

    } else {

        cudaMallocAsync(reinterpret_cast<void **>(&workA), n * n * sizeof(TA), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        tri_2_full<TA>(n, A, lda, workA, n, stream, uplo, diag);

        // C := alpha*AB + beta*C
        DDgemm<TA, TB, TC>(CUBLAS_OP_N, trans, m, n, n, alpha, B, ldb, workA, n, Tzero<TC>(), C, ldc, stream);
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workA, stream);
}

template <typename TA, typename TB, typename TC>
void DDtrtrmm(
    cublasFillMode_t uplo_A, cublasFillMode_t uplo_B,
    cublasOperation_t trans_A, cublasOperation_t trans_B,
    cublasDiagType_t diag_A, cublasDiagType_t diag_B,
    size_t n,
    const TC alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC beta,
    TC *const C, size_t ldc,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB, TC>,
                  "DDtrtrmm requires TA, TB, and TC to be all real or all complex.");

    TA *workA;
    TB *workB;

    cudaMallocAsync(reinterpret_cast<void **>(&workA), n * n * sizeof(TA), stream);
    cudaMallocAsync(reinterpret_cast<void **>(&workB), n * n * sizeof(TB), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    tri_2_full<TA>(n, A, lda, workA, n, stream, uplo_A, diag_A);
    tri_2_full<TB>(n, B, ldb, workB, n, stream, uplo_B, diag_B);

    // C := alpha*AB + beta*C
    DDgemm<TA, TB, TC>(trans_A, trans_B, n, n, n, alpha, workA, n, workB, n, beta, C, ldc, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workA, stream);
    cudaFreeAsync(workB, stream);
}

template <typename TA, typename TB>
void DDtrsm(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    size_t m, size_t n,
    const TB alpha,
    const TA *const A, size_t lda,
    TB *const B, size_t ldb,
    const cudaStream_t stream = 0 //
) {
    static_assert(same_complex_domain_v<TA, TB>,
                  "DDtrsm requires TA and TB to be both real or both complex.");

    using T      = std::conditional_t<isComplex<TB>, cuDoubleComplex, double>;
    using accu_t = ACCU_t<T>;

    const size_t size_A = (side == CUBLAS_SIDE_LEFT) ? m : n;

    T *workA         = nullptr;
    T *workX         = nullptr;
    T *workR         = nullptr;
    TB *workB        = nullptr;
    accu_t *workX_hi = nullptr;

    cudaMallocAsync(reinterpret_cast<void **>(&workA), size_A * size_A * sizeof(T), stream);
    cudaMallocAsync(reinterpret_cast<void **>(&workX), size_A * size_A * sizeof(T), stream);
    cudaMallocAsync(reinterpret_cast<void **>(&workR), size_A * size_A * sizeof(T), stream);
    cudaMallocAsync(reinterpret_cast<void **>(&workX_hi), size_A * size_A * sizeof(accu_t), stream);
    cudaMallocAsync(reinterpret_cast<void **>(&workB), ldb * n * sizeof(TB), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    tri_2_full<TA, T>(size_A, A, lda, workA, size_A, stream, uplo, diag);
    cudaMemsetAsync(workX_hi, 0, size_A * size_A * sizeof(accu_t), stream);

    T one   = Tone<T>();
    T mone  = Tmone<T>();
    TB zero = Tzero<TB>();

    cudaStream_t old_stream{};
    cublasPointerMode_t old_ptr_mode{};
    cublasGetStream(handle, &old_stream);
    cublasGetPointerMode(handle, &old_ptr_mode);
    cublasSetStream(handle, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // workX := inv(op(A)) in FP64
    eye<T>(size_A, workX, stream);

    if constexpr (isComplex<TB>) {
        CHECK_CUBLAS(cublasZtrsm(
            handle, side, uplo, trans, CUBLAS_DIAG_NON_UNIT,
            int(size_A), int(size_A),
            &one,
            workA, int(size_A),
            workX, int(size_A)));
    } else {
        CHECK_CUBLAS(cublasDtrsm(
            handle, side, uplo, trans, CUBLAS_DIAG_NON_UNIT,
            int(size_A), int(size_A),
            &one,
            workA, int(size_A),
            workX, int(size_A)));
    }

    // workX_hi := workX
    addvec<T, accu_t>(size_A * size_A, workX, workX_hi, stream);

    // mixed-precision iterative refinement
    // workX_hi := inv(op(A))
    for (int iter = 0; iter < 3; ++iter) {

        eye<T>(size_A, workR, stream);

        if (side == CUBLAS_SIDE_LEFT) {
            // workR := I - op(A) * workX_hi
            DDgemm<T, accu_t, T>(
                trans, CUBLAS_OP_N,
                size_A, size_A, size_A,
                mone,
                workA, size_A,
                workX_hi, size_A,
                one,
                workR, size_A,
                stream);
        } else {
            // workR := I - workX_hi * op(A)
            DDgemm<accu_t, T, T>(
                CUBLAS_OP_N, trans,
                size_A, size_A, size_A,
                mone,
                workX_hi, size_A,
                workA, size_A,
                one,
                workR, size_A,
                stream);
        }

        // Solve correction equation in FP64
        if constexpr (isComplex<TB>) {
            cublasZtrsm(
                handle, side, uplo, trans, CUBLAS_DIAG_NON_UNIT,
                int(size_A), int(size_A),
                &one,
                workA, int(size_A),
                workR, int(size_A));
        } else {
            cublasDtrsm(
                handle, side, uplo, trans, CUBLAS_DIAG_NON_UNIT,
                int(size_A), int(size_A),
                &one,
                workA, int(size_A),
                workR, int(size_A));
        }

        // workX_hi += R
        addvec<T, accu_t>(size_A * size_A, workR, workX_hi, stream);
    }

    // Preserve input B because output B aliases input B.
    cudaMemcpyAsync(workB, B, ldb * n * sizeof(TB), cudaMemcpyDeviceToDevice, stream);

    if (side == CUBLAS_SIDE_LEFT) {
        DDgemm<accu_t, TB, TB>(
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, m,
            alpha,
            workX_hi, size_A,
            workB, ldb,
            zero,
            B, ldb,
            stream);
    } else {
        DDgemm<TB, accu_t, TB>(
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, n,
            alpha,
            workB, ldb,
            workX_hi, size_A,
            zero,
            B, ldb,
            stream);
    }

    cublasSetPointerMode(handle, old_ptr_mode);
    cublasSetStream(handle, old_stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFreeAsync(workB, stream);
    cudaFreeAsync(workX_hi, stream);
    cudaFreeAsync(workR, stream);
    cudaFreeAsync(workX, stream);
    cudaFreeAsync(workA, stream);
}

//------------------------------
// evaluate error
//------------------------------

__host__ __device__ __forceinline__ bool is_zero(const DD_real a) {
    return a.hi == 0.0 && a.lo == 0.0;
}

// element-wise relative error
template <typename T, cublasFillMode_t UPLO>
__global__ void calc_err_kernel(
    const size_t m, const size_t n,
    T *const __restrict__ C,                     // calculated value
    const size_t ldc,                            //
    const ACCU_t<T> *const __restrict__ C_exact, // true value
    const size_t ldc_exact                       //
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const size_t col = idx / m;
    const size_t row = idx - col * m;

    const size_t idx_C_calc = col * ldc + row;
    const size_t idx_C_accu = col * ldc_exact + row;

    if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        if (row > col) {
            C[idx_C_calc] = T{};
            return;
        }
    } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
        if (row < col) {
            C[idx_C_calc] = T{};
            return;
        }
    }

    if constexpr (isComplex<T>) {

        using U = std::conditional_t<std::is_same_v<T, cuDoubleComplex>, double, float>;

        const DD_complex C_calc = DD_complex(C[idx_C_calc]);
        const DD_complex C_accu = C_exact[idx_C_accu];

        // real part
        const DD_real C_calc_re = C_calc.re;
        const DD_real C_accu_re = C_accu.re;
        const DD_real gap_re    = C_calc_re - C_accu_re;
        U out_re;
        if (is_zero(C_accu_re)) {
            out_re = U(fabs(gap_re.hi));
        } else {
            const DD_real err_re = gap_re / C_accu_re;
            out_re               = U(fabs(err_re.hi));
        }

        // complex part
        const DD_real C_calc_im = C_calc.im;
        const DD_real C_accu_im = C_accu.im;
        const DD_real gap_im    = C_calc_im - C_accu_im;
        U out_im;
        if (is_zero(C_accu_im)) {
            out_im = U(fabs(gap_im.hi));
        } else {
            const DD_real err_im = gap_im / C_accu_im;
            out_im               = U(fabs(err_im.hi));
        }

        C[idx_C_calc] = T{out_re, out_im};

    } else {

        const DD_real C_calc = DD_real(C[idx_C_calc]);
        const DD_real C_accu = C_exact[idx_C_accu];
        const DD_real gap    = C_calc - C_accu;
        if (is_zero(C_accu)) {
            C[idx_C_calc] = T(fabs(gap.hi));
        } else {
            const DD_real err = gap / C_accu;
            C[idx_C_calc]     = T(fabs(err.hi));
        }
    }
}

template <typename T>
inline double median(std::vector<T> &vec, const size_t n) {
    if (n & 1) {
        return double(vec[n / 2]);
    } else {
        return ((double(vec[n / 2]) + double(vec[n / 2 - 1])) * 0.5);
    }
}

template <typename T, cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL>
inline double2 calc_err(
    const size_t m, const size_t n,
    T *const C, const size_t ldc,                           // calculated value
    const ACCU_t<T> *const C_exact, const size_t ldc_exact, // true value
    const cudaStream_t stream = 0                           //
) {
    // UPLO != CUBLAS_FILL_MODE_FULL => m = n

    CHECK_CUDA(cudaStreamSynchronize(stream));
    calc_err_kernel<T, UPLO><<<(m * n + 255) / 256, 256, 0, stream>>>(m, n, C, ldc, C_exact, ldc_exact);

    if constexpr (isComplex<T>) {

        size_t sizeC = m * n * 2;
        using U      = std::conditional_t<(std::is_same_v<T, cuDoubleComplex>), double, float>;
        std::vector<U> hC(sizeC);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpyAsync(hC.data(), C, sizeC * sizeof(U), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::sort(hC.begin(), hC.end());
        double err1             = double(hC[sizeC - 1]);
        const size_t valid_size = (UPLO == CUBLAS_FILL_MODE_FULL) ? sizeC : (m * (m + 1));
        double err2             = median(hC, valid_size);
        return {err1, err2};

    } else {

        size_t sizeC = m * n;
        std::vector<T> hC(sizeC);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpyAsync(hC.data(), C, sizeC * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::sort(hC.begin(), hC.end());
        double err1             = double(hC[sizeC - 1]);
        const size_t valid_size = (UPLO == CUBLAS_FILL_MODE_FULL) ? sizeC : (m * (m + 1) / 2);
        double err2             = median(hC, valid_size);
        return {err1, err2};
    }
}

} // namespace eval
