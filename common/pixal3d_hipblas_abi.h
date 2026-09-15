/* Minimal hipBLAS 3 ABI for runtime-only ROCm installations.
 * Signatures and enum values follow ROCm/hipBLAS and hipBLAS-common
 * release rocm-7.2.2. Prefer the SDK header whenever it is installed.
 * SPDX-License-Identifier: MIT */
#pragma once
#include <hip/library_types.h>
typedef void *hipblasHandle_t;
typedef enum { HIPBLAS_STATUS_SUCCESS = 0 } hipblasStatus_t;
typedef enum { HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112 } hipblasOperation_t;
typedef enum { HIPBLAS_COMPUTE_32F = 2 } hipblasComputeType_t;
typedef enum { HIPBLAS_GEMM_DEFAULT = 160 } hipblasGemmAlgo_t;
extern "C" {
hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream);
hipblasStatus_t hipblasCreate(hipblasHandle_t *);
hipblasStatus_t hipblasDestroy(hipblasHandle_t);
hipblasStatus_t hipblasSgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                             const float *, const float *, int, const float *, int, const float *, float *,
                             int);
hipblasStatus_t hipblasGemmEx(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                              const void *, const void *, hipDataType, int, const void *, hipDataType, int,
                              const void *, void *, hipDataType, int, hipblasComputeType_t,
                              hipblasGemmAlgo_t);
}
