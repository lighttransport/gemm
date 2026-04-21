/*
 * verify_mma_gemm.c — Microtest to check whether gemm_f16_f32 (MMA
 * m16n8k16) matches gemm_tiled_f16_f32 (non-MMA, shared-mem tiled) on
 * sm_120. Both are in cuda_kernels_common.h.
 *
 * Context: commit 21afa2f replaced MMA GEMM with the tiled variant for
 * DA3 because MMA produced wrong results. Commit a9b0df2 later added a
 * Blackwell sm_120 a1/a2 fragment swap. This test decides whether the
 * MMA kernel is correct today on sm_120 — if so, sam3 can wire it for
 * tensor-core speedups.
 */
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0)    f = sign;
    else if (exp==31) f = sign | 0x7f800000 | (mant << 13);
    else             f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

static void cpu_gemm(float *Y, const uint16_t *Wf16, const float *X, const float *b,
                     int n_out, int n_in, int n_tok) {
    for (int t = 0; t < n_tok; t++)
        for (int o = 0; o < n_out; o++) {
            float acc = b ? b[o] : 0.f;
            for (int k = 0; k < n_in; k++)
                acc += X[t*n_in + k] * f16_to_f32(Wf16[o*n_in + k]);
            Y[t*n_out + o] = acc;
        }
}

static void diff(const char *tag, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d;
    }
    fprintf(stderr, "  %-18s max_abs=%.6g mean_abs=%.6g\n", tag, mxd, mad/(double)n);
}

int main(void) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    /* Concat common src + closing brace for extern "C". */
    size_t n = strlen(cuda_kernels_common_src);
    char *src = (char*)malloc(n + 4);
    memcpy(src, cuda_kernels_common_src, n);
    memcpy(src + n, "\n}\n", 4);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, src, "verify_mma_gemm", 1, "mma_test") < 0) return 6;
    free(src);

    CUfunction fn_mma, fn_tiled;
    cuModuleGetFunction(&fn_mma,   mod, "gemm_f16_f32");
    cuModuleGetFunction(&fn_tiled, mod, "gemm_tiled_f16_f32");

    int test_sizes[][3] = {
        { 64, 64, 16 },
        { 256, 128, 32 },
        { 1024, 1024, 72 },   /* sam3 ViT dim */
        { 3072, 1024, 72 },   /* QKV projection */
        { 1024, 4096, 72 },   /* MLP fc2 */
    };
    int nT = sizeof(test_sizes)/sizeof(test_sizes[0]);

    for (int ti = 0; ti < nT; ti++) {
        int n_out = test_sizes[ti][0];
        int n_in  = test_sizes[ti][1];
        int n_tok = test_sizes[ti][2];
        size_t W_n = (size_t)n_out*n_in;
        size_t X_n = (size_t)n_tok*n_in;
        size_t Y_n = (size_t)n_tok*n_out;

        fprintf(stderr, "test[%d]: n_out=%d n_in=%d n_tok=%d\n", ti, n_out, n_in, n_tok);

        uint16_t *hW = (uint16_t*)malloc(W_n*2);
        float    *hX = (float*)malloc(X_n*4);
        float    *hB = (float*)malloc(n_out*4);
        float    *hY_cpu = (float*)malloc(Y_n*4);
        float    *hY_mma = (float*)malloc(Y_n*4);
        float    *hY_til = (float*)malloc(Y_n*4);

        unsigned seed = 12345u + ti;
        for (size_t i = 0; i < W_n; i++) { seed = seed*1103515245u+12345u;
            float f = ((int)(seed>>8)%2000 - 1000) / 1000.f; hW[i] = f32_to_f16(f*0.1f); }
        for (size_t i = 0; i < X_n; i++) { seed = seed*1103515245u+12345u;
            hX[i] = ((int)(seed>>8)%2000 - 1000) / 10000.f; }
        for (int i = 0; i < n_out; i++) { seed = seed*1103515245u+12345u;
            hB[i] = ((int)(seed>>8)%2000 - 1000) / 100000.f; }

        cpu_gemm(hY_cpu, hW, hX, hB, n_out, n_in, n_tok);

        CUdeviceptr dW, dX, dB, dY;
        cuMemAlloc(&dW, W_n*2); cuMemcpyHtoD(dW, hW, W_n*2);
        cuMemAlloc(&dX, X_n*4); cuMemcpyHtoD(dX, hX, X_n*4);
        cuMemAlloc(&dB, n_out*4); cuMemcpyHtoD(dB, hB, n_out*4);
        cuMemAlloc(&dY, Y_n*4);

        /* MMA GEMM: grid (ceil(n_out/256), ceil(n_tok/16)), block 128 threads (4 warps),
         * smem = 16*16*sizeof(float) = 1024 bytes */
        {
            void *args[] = {&dY, &dW, &dX, &dB, &n_out, &n_in, &n_tok};
            unsigned gx = (unsigned)((n_out+255)/256);
            unsigned gy = (unsigned)((n_tok+15)/16);
            cuMemsetD8(dY, 0, Y_n*4);
            CUresult r = cuLaunchKernel(fn_mma, gx, gy, 1, 128, 1, 1,
                                        16*16*sizeof(float), 0, args, NULL);
            cuCtxSynchronize();
            fprintf(stderr, "  mma launch=%d\n", (int)r);
            cuMemcpyDtoH(hY_mma, dY, Y_n*4);
        }
        /* Tiled GEMM: grid (ceil(n_out/64), ceil(n_tok/16)), block (16,16) */
        {
            void *args[] = {&dY, &dW, &dX, &dB, &n_out, &n_in, &n_tok};
            unsigned gx = (unsigned)((n_out+63)/64);
            unsigned gy = (unsigned)((n_tok+15)/16);
            cuMemsetD8(dY, 0, Y_n*4);
            CUresult r = cuLaunchKernel(fn_tiled, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL);
            cuCtxSynchronize();
            fprintf(stderr, "  tiled launch=%d\n", (int)r);
            cuMemcpyDtoH(hY_til, dY, Y_n*4);
        }

        diff("cpu vs tiled", hY_cpu, hY_til, Y_n);
        diff("cpu vs mma",   hY_cpu, hY_mma, Y_n);
        diff("tiled vs mma", hY_til, hY_mma, Y_n);

        cuMemFree(dW); cuMemFree(dX); cuMemFree(dB); cuMemFree(dY);
        free(hW); free(hX); free(hB); free(hY_cpu); free(hY_mma); free(hY_til);
    }

    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
