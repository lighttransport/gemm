/**
 * Common public types
 * -------------------
 * This header defines the common public types used by the GEMMul8 API.
 *
 * The main backend selector is gemmul8::Backend:
 *
 *   - INT8-based emulation: gemmul8::Backend::INT8
 *   - FP8-based emulation : gemmul8::Backend::FP8
 *
 * The operation selector gemmul8::Func is used mainly by workspace-size
 * queries and internal dispatch logic.
 */
#pragma once

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cuComplex.h>
#endif
#if defined(__HIPCC__)
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
    #include <hipblaslt/hipblaslt.h>
    #include <hip/hip_complex.h>
#endif
#include <cstddef>
#include <vector>
#include <type_traits>

namespace gemmul8 {

//------------------------------
//  Low-precision backend used internally.
//  Choices:
//    - INT8-based emulation: gemmul8::Backend::INT8
//    - FP8-based emulation : gemmul8::Backend::FP8
//------------------------------
enum class Backend {
    INT8,
    FP8
};

//------------------------------
//  BLAS-like operation identifier.
//  This enum is primarily used by workspace-size queries.
//------------------------------
enum class Func {
    gemm,
    symm,
    syrk,
    syr2k,
    syrkx,
    trmm,
    hemm,
    herk,
    her2k,
    herkx,
    trsm,
    trtrmm
};

} // namespace gemmul8
