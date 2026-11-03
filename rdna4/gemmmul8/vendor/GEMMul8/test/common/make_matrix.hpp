#pragma once
#include <cmath>
#include <type_traits>

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include <cuComplex.h>
    #include <curand_kernel.h>
#endif

#include "self_hipify.hpp"

namespace makemat {

template <typename T>
__global__ void randmat_kernel(size_t m,                                            // rows of A
                               size_t n,                                            // columns of A
                               T *const A,                                          // output
                               double phi,                                          // difficulty for matrix multiplication
                               const unsigned long long seed,                       // seed for random numbers
                               const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL, //
                               const cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT   //
) {
    const size_t idx = size_t(threadIdx.x) + size_t(blockIdx.x) * size_t(blockDim.x);
    if (idx >= m * n) return;

    const size_t col = idx / m;
    const size_t row = idx - col * m;
    if (UPLO == CUBLAS_FILL_MODE_UPPER && row > col) {
        return;
    } else if (UPLO == CUBLAS_FILL_MODE_LOWER && row < col) {
        return;
    }
    if (DIAG == CUBLAS_DIAG_UNIT && row == col) {
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            A[idx] = T{1.0, 0.0};
        } else if constexpr (std::is_same_v<T, cuFloatComplex>) {
            A[idx] = T{1.0f, 0.0f};
        } else {
            A[idx] = T(1);
        }
        return;
    }

    curandState state;
    curand_init(seed, idx, 0, &state);

    T out;
    if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        const double rand_r  = curand_uniform_double(&state);
        const double rand_i  = curand_uniform_double(&state);
        const double randn_r = curand_normal_double(&state);
        const double randn_i = curand_normal_double(&state);
        if (phi < 0) {
            out.x = randn_r;
            out.y = randn_i;
        } else {
            out.x = (rand_r - 0.5) * exp(randn_r * phi);
            out.y = (rand_i - 0.5) * exp(randn_i * phi);
        }

    } else if constexpr (std::is_same_v<T, cuFloatComplex>) {
        const double rand_r  = curand_uniform_double(&state);
        const double rand_i  = curand_uniform_double(&state);
        const double randn_r = curand_normal_double(&state);
        const double randn_i = curand_normal_double(&state);
        if (phi < 0) {
            out.x = static_cast<float>(randn_r);
            out.y = static_cast<float>(randn_i);
        } else {
            out.x = static_cast<float>((rand_r - 0.5) * exp(randn_r * phi));
            out.y = static_cast<float>((rand_i - 0.5) * exp(randn_i * phi));
        }

    } else {
        const double rand  = curand_uniform_double(&state);
        const double randn = curand_normal_double(&state);
        if (phi < 0) {
            out = static_cast<T>(randn);
        } else {
            out = static_cast<T>((rand - 0.5) * exp(randn * phi));
        }
    }
    A[idx] = out;
}

template <typename T>
inline void randmat(size_t m,                                            // rows of A
                    size_t n,                                            // columns of A
                    T *const A,                                          // output
                    double phi,                                          // difficulty for matrix multiplication
                    const unsigned long long seed,                       // seed for random numbers
                    const cudaStream_t stream   = 0,                     //
                    const cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL, //
                    const cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT   //
) {
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<T><<<grid_size, block_size, 0, stream>>>(m, n, A, phi, seed, UPLO, DIAG);
    CHECK_CUDA(cudaStreamSynchronize(stream));
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
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename T> __host__ __device__ __forceinline__ T one_value() { return T(1); }
template <> __host__ __device__ __forceinline__ cuFloatComplex one_value<cuFloatComplex>() { return cuFloatComplex{1.0f, 0.0f}; }
template <> __host__ __device__ __forceinline__ cuDoubleComplex one_value<cuDoubleComplex>() { return cuDoubleComplex{1.0, 0.0}; }

template <typename T>
__global__ void set_ones_kernel(
    const size_t rows,
    const size_t cols,
    T *const A,
    const size_t lda //
) {
    const size_t idx = size_t(threadIdx.x) + size_t(blockIdx.x) * size_t(blockDim.x);
    if (idx >= rows * cols) return;

    const size_t col = idx / rows;
    const size_t row = idx - col * rows;

    A[col * lda + row] = one_value<T>();
}

template <typename T>
inline void set_ones(
    const size_t rows,
    const size_t cols,
    T *const A,
    const size_t lda,
    const cudaStream_t stream = 0 //
) {
    constexpr size_t block_size = 256;
    const size_t grid_size      = (rows * cols + block_size - 1) / block_size;

    set_ones_kernel<T><<<grid_size, block_size, 0, stream>>>(rows, cols, A, lda);

    CHECK_CUDA(cudaGetLastError());
}

} // namespace makemat
