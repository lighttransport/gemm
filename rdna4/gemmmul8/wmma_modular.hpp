#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// The inputs are column-major K-by-M and K-by-N byte matrices. The output is
// column-major M-by-N. Strides are in elements. All pointers are device pointers.
// INT8 uses signed bytes and INT32 output; FP8 uses E4M3 bytes and FP32 output.
// The caller must arrange safe K chunks and reduce each output modulo its
// modulus between chunks. These conservative bounds assume FP8 limbs of at
// most 32 and a centered incoming residue of at most 65536.
namespace ozaki2_wmma {
enum class backend { int8, fp8 };
constexpr int max_k_int8 = 65536;
constexpr int max_k_fp8 = 15872;
// -1 selects the conservative default; 0..31 choose a tile variant.
void set_variant(int variant);

hipError_t product(backend type, hipStream_t stream, int m, int n, int k,
                   const uint8_t *a, size_t lda, size_t stride_a,
                   const uint8_t *b, size_t ldb, size_t stride_b,
                   void *c, size_t ldc, size_t stride_c,
                   int batches, bool accumulate);
}
