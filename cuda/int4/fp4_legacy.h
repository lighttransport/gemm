#ifndef FP4_LEGACY_BENCH_H
#define FP4_LEGACY_BENCH_H
#include "int4.h"

typedef struct {
    CUmodule modules[3];
    CUfunction functions[3];
} fp4_legacy_context;

extern const char *const fp4_legacy_names[3];
int fp4_legacy_init(fp4_legacy_context *ctx);
void fp4_legacy_destroy(fp4_legacy_context *ctx);
int fp4_legacy_supported(int4_matrix a, int4_matrix b, int variant);
CUresult fp4_legacy_gemm(fp4_legacy_context *ctx, int4_matrix a, int4_matrix b,
                         CUdeviceptr output, int variant, CUstream stream);
#endif
