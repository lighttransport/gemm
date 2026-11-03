#pragma once
#include "store.hpp"

#include "../general/helper_triangular.hpp"
#include "../general/roundup.hpp"

namespace gemmul8::scaling::accu {

template <typename T, Backend BACKEND>
__device__ __forceinline__ void extract_colwise_full_kernel(
    const unsigned rows_A,
    const T *const __restrict__ in,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo4,
    const int32_t sft //
) {
    using ValT = upperBound_t<T, BACKEND>;
    using Low4 = common::lowx4_t<BACKEND>;

    const unsigned col_idx = blockIdx.x;
    const unsigned rows4   = rows_A >> 2;
    const unsigned tail    = rows_A & 3u;

    if constexpr (common::isComplex<T>) {
        Low4 *__restrict__ out_1 = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;
        Low4 *__restrict__ out_2 = reinterpret_cast<Low4 *>(A_lo.ptr1) + col_idx * lda_lo4;
        Low4 *__restrict__ out_3 = reinterpret_cast<Low4 *>(A_lo.ptr2) + col_idx * lda_lo4;

        unsigned i = threadIdx.x;
        for (; i < rows4; i += blockDim.x) {
            const unsigned idx = i << 2;

            extract_colwise_store4_complex<T, BACKEND>(
                out_1, out_2, out_3, i,
                in[idx], in[idx + 1U], in[idx + 2U], in[idx + 3U], sft);
        }

        for (; i < lda_lo4; i += blockDim.x) {
            if (tail != 0u && i == rows4) {
                const unsigned idx = i << 2;
                const T z          = common::Tconst<T>::zero();

                extract_colwise_store4_complex<T, BACKEND>(
                    out_1, out_2, out_3, i,
                    in[idx], (tail >= 2U) ? in[idx + 1U] : z, (tail >= 3U) ? in[idx + 2U] : z, z, sft);
            } else {
                const Low4 z = common::Tconst<Low4>::zero();
                out_1[i]     = z;
                out_2[i]     = z;
                out_3[i]     = z;
            }
        }
    } else {
        Low4 *__restrict__ out = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;

        unsigned i = threadIdx.x;
        for (; i < rows4; i += blockDim.x) {
            const unsigned idx = i << 2;

            extract_colwise_store4_real<T, BACKEND>(
                out, i,
                in[idx], in[idx + 1U], in[idx + 2U], in[idx + 3U], sft);
        }

        for (; i < lda_lo4; i += blockDim.x) {
            if (tail != 0u && i == rows4) {
                const unsigned idx = i << 2;
                const T z          = common::Tconst<T>::zero();

                extract_colwise_store4_real<T, BACKEND>(
                    out, i,
                    in[idx], (tail >= 2U) ? in[idx + 1U] : z, (tail >= 3U) ? in[idx + 2U] : z, z, sft);
            } else {
                out[i] = common::Tconst<Low4>::zero();
            }
        }
    }
}

template <typename T, Backend BACKEND, cublasDiagType_t DIAG>
__device__ __forceinline__ void extract_colwise_upper_kernel(
    const unsigned rows_A,
    const T *const __restrict__ in,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo4,
    const int32_t sft //
) {
    using Low4             = common::lowx4_t<BACKEND>;
    const unsigned col_idx = blockIdx.x;

    unsigned boundary_i;
    bool has_boundary;
    if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
        const bool diag_flag = (col_idx < rows_A);
        boundary_i           = (diag_flag ? col_idx : rows_A) >> 2;
        has_boundary         = diag_flag ? true : ((rows_A & 3U) != 0U);
    } else {
        const unsigned active_end = min(rows_A, col_idx + 1U);
        boundary_i                = active_end >> 2;
        has_boundary              = ((active_end & 3U) != 0U);
    }

    if constexpr (common::isComplex<T>) {
        Low4 *__restrict__ out_1 = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;
        Low4 *__restrict__ out_2 = reinterpret_cast<Low4 *>(A_lo.ptr1) + col_idx * lda_lo4;
        Low4 *__restrict__ out_3 = reinterpret_cast<Low4 *>(A_lo.ptr2) + col_idx * lda_lo4;

        for (unsigned i = threadIdx.x; i < boundary_i; i += blockDim.x) {
            const unsigned row0 = i << 2;

            extract_colwise_store4_complex<T, BACKEND>(
                out_1, out_2, out_3, i,
                in[row0], in[row0 + 1U], in[row0 + 2U], in[row0 + 3U], sft);
        }

        if (threadIdx.x == 0 && has_boundary) {
            const unsigned row0 = boundary_i << 2;

            const T a0 = general::tri_col_value<true, T, DIAG>(in, row0, col_idx, rows_A);
            const T a1 = general::tri_col_value<true, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
            const T a2 = general::tri_col_value<true, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
            const T a3 = general::tri_col_value<true, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

            extract_colwise_store4_complex<T, BACKEND>(
                out_1, out_2, out_3, boundary_i, a0, a1, a2, a3, sft);
        }

    } else {
        Low4 *__restrict__ out = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;

        for (unsigned i = threadIdx.x; i < boundary_i; i += blockDim.x) {
            const unsigned row0 = i << 2;

            extract_colwise_store4_real<T, BACKEND>(
                out, i,
                in[row0], in[row0 + 1U], in[row0 + 2U], in[row0 + 3U], sft);
        }

        if (threadIdx.x == 0 && has_boundary) {
            const unsigned row0 = boundary_i << 2;

            const T a0 = general::tri_col_value<true, T, DIAG>(in, row0, col_idx, rows_A);
            const T a1 = general::tri_col_value<true, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
            const T a2 = general::tri_col_value<true, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
            const T a3 = general::tri_col_value<true, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

            extract_colwise_store4_real<T, BACKEND>(
                out, boundary_i, a0, a1, a2, a3, sft);
        }
    }
}

template <typename T, Backend BACKEND, cublasDiagType_t DIAG>
__device__ __forceinline__ void extract_colwise_lower_kernel(
    const unsigned rows_A,
    const T *const __restrict__ in,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo4,
    const int32_t sft //
) {
    using Low4             = common::lowx4_t<BACKEND>;
    const unsigned col_idx = blockIdx.x;
    if (col_idx >= rows_A) return;

    const unsigned first4 = col_idx >> 2;
    const unsigned tail4  = rows_A >> 2;
    bool has_head_boundary;
    if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
        has_head_boundary = true;
    } else {
        has_head_boundary = ((col_idx & 3U) != 0U);
    }
    const bool has_tail_boundary = ((rows_A & 3U) != 0U);
    const unsigned dense_begin4  = first4 + (has_head_boundary ? 1U : 0U);

    if constexpr (common::isComplex<T>) {
        Low4 *__restrict__ out_1 = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;
        Low4 *__restrict__ out_2 = reinterpret_cast<Low4 *>(A_lo.ptr1) + col_idx * lda_lo4;
        Low4 *__restrict__ out_3 = reinterpret_cast<Low4 *>(A_lo.ptr2) + col_idx * lda_lo4;

        for (unsigned i = dense_begin4 + threadIdx.x; i < tail4; i += blockDim.x) {
            const unsigned row0 = i << 2;

            extract_colwise_store4_complex<T, BACKEND>(
                out_1, out_2, out_3, i,
                in[row0], in[row0 + 1U], in[row0 + 2U], in[row0 + 3U], sft);
        }

        if (threadIdx.x == 0) {
            if (has_head_boundary) {
                const unsigned row0 = first4 << 2;

                const T a0 = general::tri_col_value<false, T, DIAG>(in, row0, col_idx, rows_A);
                const T a1 = general::tri_col_value<false, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
                const T a2 = general::tri_col_value<false, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
                const T a3 = general::tri_col_value<false, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

                extract_colwise_store4_complex<T, BACKEND>(
                    out_1, out_2, out_3, first4, a0, a1, a2, a3, sft);
            }

            if (has_tail_boundary && (!has_head_boundary || tail4 != first4)) {
                const unsigned row0 = tail4 << 2;

                const T a0 = general::tri_col_value<false, T, DIAG>(in, row0, col_idx, rows_A);
                const T a1 = general::tri_col_value<false, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
                const T a2 = general::tri_col_value<false, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
                const T a3 = general::tri_col_value<false, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

                extract_colwise_store4_complex<T, BACKEND>(
                    out_1, out_2, out_3, tail4, a0, a1, a2, a3, sft);
            }
        }

    } else {
        Low4 *__restrict__ out = reinterpret_cast<Low4 *>(A_lo.ptr0) + col_idx * lda_lo4;

        for (unsigned i = dense_begin4 + threadIdx.x; i < tail4; i += blockDim.x) {
            const unsigned row0 = i << 2;

            extract_colwise_store4_real<T, BACKEND>(
                out, i,
                in[row0], in[row0 + 1U], in[row0 + 2U], in[row0 + 3U], sft);
        }

        if (threadIdx.x == 0) {
            if (has_head_boundary) {
                const unsigned row0 = first4 << 2;

                const T a0 = general::tri_col_value<false, T, DIAG>(in, row0, col_idx, rows_A);
                const T a1 = general::tri_col_value<false, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
                const T a2 = general::tri_col_value<false, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
                const T a3 = general::tri_col_value<false, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

                extract_colwise_store4_real<T, BACKEND>(
                    out, first4, a0, a1, a2, a3, sft);
            }

            if (has_tail_boundary && (!has_head_boundary || tail4 != first4)) {
                const unsigned row0 = tail4 << 2;

                const T a0 = general::tri_col_value<false, T, DIAG>(in, row0, col_idx, rows_A);
                const T a1 = general::tri_col_value<false, T, DIAG>(in, row0 + 1U, col_idx, rows_A);
                const T a2 = general::tri_col_value<false, T, DIAG>(in, row0 + 2U, col_idx, rows_A);
                const T a3 = general::tri_col_value<false, T, DIAG>(in, row0 + 3U, col_idx, rows_A);

                extract_colwise_store4_real<T, BACKEND>(
                    out, tail4, a0, a1, a2, a3, sft);
            }
        }
    }
}

template <typename T, Backend BACKEND, cublasFillMode_t UPLO, cublasDiagType_t DIAG>
__global__ void extract_colwise(
    const unsigned rows_A,
    const T *const __restrict__ A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> A_lo,
    const size_t lda_lo4,
    int16_t *const __restrict__ sftA //
) {
    using U = common::underlying_t<T>;
    __shared__ U samax[32];

    const unsigned col_idx         = blockIdx.x;
    const T *const __restrict__ in = A + col_idx * lda;

    const int32_t sft = calc_sft_before_colwise<T, BACKEND, UPLO, DIAG>(
        rows_A, in, sftA, samax);

    if constexpr (UPLO == CUBLAS_FILL_MODE_FULL) {
        extract_colwise_full_kernel<T, BACKEND>(
            rows_A, in, A_lo, lda_lo4, sft);

    } else if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        extract_colwise_upper_kernel<T, BACKEND, DIAG>(
            rows_A, in, A_lo, lda_lo4, sft);

    } else {
        extract_colwise_lower_kernel<T, BACKEND, DIAG>(
            rows_A, in, A_lo, lda_lo4, sft);
    }
}

} // namespace gemmul8::scaling::accu
