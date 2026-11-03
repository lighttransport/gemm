#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ozaki2 {
enum class backend { int8, fp8 };
enum class operation { normal, transpose };
enum class status { success, invalid_argument, insufficient_workspace, hip_error, backend_error };
struct context;
struct options {
    backend arithmetic = backend::int8;
    int moduli = 0; // 0 selects 15 for INT8 or 11 for FP8.
    bool fast_scaling = true;
    int wmma_variant = -2;  // -2 tuned auto, -1 conservative, 0..31 explicit.
    // Zero uses the unconstrained upstream workspace. A nonzero value enables
    // GEMMul8's memory-saving blocking with this maximum number of bytes.
    size_t max_workspace_bytes = 0;
};
struct gemm {
    int m, n, k;
    operation trans_a, trans_b;
    double alpha;
    const double *a; size_t lda;
    const double *b; size_t ldb;
    double beta;
    double *c; size_t ldc;
};
// All matrices use row-major storage and device pointers. A context must not be
// used concurrently by multiple host threads. Caller owns workspace and stream.
status create(context **out);
void destroy(context *ctx);
status workspace_size(context *ctx, const gemm &problem, const options &opts, size_t *bytes);
status dgemm(context *ctx, const gemm &problem, const options &opts,
             void *workspace, size_t workspace_bytes, hipStream_t stream = nullptr);
const char *status_string(status s);
}
