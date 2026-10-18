/* Benchmark adapter: compile and launch the actual cuda/fp4 kernels. */
#include "fp4_legacy.h"
#include "../fp4/fp4_gemm_kernels.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const fp4_legacy_names[3] = {"fp4_ref", "fp4_sync", "fp4_async"};

void fp4_legacy_destroy(fp4_legacy_context *ctx) {
    for (int i = 0; i < 3; i++) if (ctx->modules[i]) cuModuleUnload(ctx->modules[i]);
    memset(ctx, 0, sizeof(*ctx));
}

int fp4_legacy_init(fp4_legacy_context *ctx) {
    const char *sources[] = {KSRC_NAIVE, KSRC_OPT, KSRC_CPA};
    const char *symbols[] = {"w4a4_gemm", "w4a4_gemm_opt", "w4a4_gemm_cpa"};
    const char *opts[] = {"--gpu-architecture=sm_120a"};
    memset(ctx, 0, sizeof(*ctx));
    if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS || !nvrtcGetCUBINSize || !nvrtcGetCUBIN) return -1;
    for (int i = 0; i < 3; i++) {
        nvrtcProgram p;
        if (nvrtcCreateProgram(&p, sources[i], symbols[i], 0, NULL, NULL) != NVRTC_SUCCESS) goto fail;
        nvrtcResult result = nvrtcCompileProgram(p, 1, opts);
        if (result != NVRTC_SUCCESS) {
            size_t n = 0; nvrtcGetProgramLogSize(p, &n);
            char *log = malloc(n + 1);
            if (log) { nvrtcGetProgramLog(p, log); log[n] = 0; fprintf(stderr, "%s: %s\n", symbols[i], log); free(log); }
            nvrtcDestroyProgram(&p); goto fail;
        }
        size_t n = 0;
        if (nvrtcGetCUBINSize(p, &n) != NVRTC_SUCCESS || !n) { nvrtcDestroyProgram(&p); goto fail; }
        char *blob = malloc(n);
        if (!blob) { nvrtcDestroyProgram(&p); goto fail; }
        result = nvrtcGetCUBIN(p, blob);
        nvrtcDestroyProgram(&p);
        CUresult loaded = result == NVRTC_SUCCESS ? cuModuleLoadData(&ctx->modules[i], blob) : CUDA_ERROR_INVALID_IMAGE;
        free(blob);
        if (loaded != CUDA_SUCCESS || cuModuleGetFunction(&ctx->functions[i], ctx->modules[i], symbols[i]) != CUDA_SUCCESS) goto fail;
    }
    return 0;
fail:
    fp4_legacy_destroy(ctx); return -1;
}

int fp4_legacy_supported(int4_matrix a, int4_matrix b, int variant) {
    return variant >= 0 && variant < 3 && a.rows > 0 && b.rows > 0 && a.k == b.k &&
        a.kp > 0 && a.kp == int4_padded_k(a.k) && a.kp == b.kp && !(b.rows & 1) &&
        (variant != 0 || (a.rows % 16 == 0 && b.rows % 8 == 0));
}

CUresult fp4_legacy_gemm(fp4_legacy_context *ctx, int4_matrix a, int4_matrix b,
                         CUdeviceptr output, int variant, CUstream stream) {
    if (!ctx || !fp4_legacy_supported(a, b, variant) || !ctx->functions[variant]) return CUDA_ERROR_INVALID_VALUE;
    unsigned bx, by, threads;
    if (variant == 0) {
        size_t blocks = ((size_t)(a.rows / 16) * (b.rows / 8) + 3) / 4;
        if (blocks > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
        bx = (unsigned)blocks; by = 1; threads = 128;
    } else {
        bx = (b.rows + 127u) / 128u; by = (a.rows + 63u) / 64u; threads = 256;
        if (by > 65535) return CUDA_ERROR_INVALID_VALUE;
    }
    void *args[] = {&a.codes, &b.codes, &a.scales, &b.scales, &output, &a.rows, &b.rows, &a.kp};
    return cuLaunchKernel(ctx->functions[variant], bx, by, 1, threads, 1, 1, 0, stream, args, NULL);
}
