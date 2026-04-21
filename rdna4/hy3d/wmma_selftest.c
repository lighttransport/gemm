/* Minimal selftest: compare hy3d gemm_f16w_bf16a_wmma_t vs scalar gemm_tiled_f16_f32
 * on a synthetic small GEMM. */
#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_hy3d_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint16_t f32_to_f16(float f) {
    unsigned int x; memcpy(&x, &f, 4);
    unsigned int sign = (x >> 31) & 1;
    int exp = (int)((x >> 23) & 0xFF) - 127;
    unsigned int mant = x & 0x7FFFFF;
    if (exp >= 16) return (uint16_t)((sign << 15) | (0x1F << 10));
    if (exp <= -15) return (uint16_t)(sign << 15);
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    unsigned int sign = (h >> 15) & 1;
    int exp = ((h >> 10) & 0x1F);
    unsigned int mant = h & 0x3FF;
    unsigned int f;
    if (exp == 0) f = sign << 31;
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float out; memcpy(&out, &f, 4); return out;
}

int main(void) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) { fprintf(stderr, "rocewInit failed\n"); return 1; }
    hipSetDevice(0);

    int M = 128, K = 2048, N = 128;
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    uint16_t *W = (uint16_t *)malloc((size_t)N * K * sizeof(uint16_t));
    float *Wf = (float *)malloc((size_t)N * K * sizeof(float));
    float *Y_wmma = (float *)calloc((size_t)M * N, sizeof(float));
    float *Y_scal = (float *)calloc((size_t)M * N, sizeof(float));
    float *Y_ref  = (float *)calloc((size_t)M * N, sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) X[i] = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < N * K; i++) {
        float v = (float)(rand() % 200 - 100) / 100.0f;
        W[i] = f32_to_f16(v);
        Wf[i] = f16_to_f32(W[i]);  /* use the rounded F16 value as ref */
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += X[m*K + k] * Wf[n*K + k];
            Y_ref[m*N + n] = s;
        }

    /* Build kernel source identical to hip_hy3d_runner: common + hy3d specific */
    size_t n1 = strlen(hip_kernels_common_src), n2 = strlen(hip_hy3d_specific_kernels);
    char *src = (char *)malloc(n1 + n2 + 1);
    memcpy(src, hip_kernels_common_src, n1);
    memcpy(src + n1, hip_hy3d_specific_kernels, n2);
    src[n1 + n2] = '\0';

    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, src, "hy3d_selftest", 1, "selftest") < 0) return 1;

    hipFunction_t fn_wmma = NULL, fn_scal = NULL;
    hipModuleGetFunction(&fn_wmma, mod, "gemm_f16w_bf16a_wmma_t");
    hipModuleGetFunction(&fn_scal, mod, "gemm_tiled_f16_f32");
    fprintf(stderr, "fn_wmma=%p fn_scal=%p\n", (void*)fn_wmma, (void*)fn_scal);
    if (!fn_wmma || !fn_scal) return 1;

    void *dX, *dW, *dY1, *dY2;
    hipMalloc(&dX, (size_t)M*K*4);
    hipMalloc(&dW, (size_t)N*K*2);
    hipMalloc(&dY1, (size_t)M*N*4);
    hipMalloc(&dY2, (size_t)M*N*4);
    hipMemcpy(dX, X, (size_t)M*K*4, hipMemcpyHostToDevice);
    hipMemcpy(dW, W, (size_t)N*K*2, hipMemcpyHostToDevice);
    hipMemset(dY1, 0, (size_t)M*N*4);
    hipMemset(dY2, 0, (size_t)M*N*4);

    void *bias = NULL;
    int n_out = N, n_in = K, n_tok = M;
    void *args[] = {&dY1, &dW, &dX, &bias, &n_out, &n_in, &n_tok};
    hipError_t e1 = hipModuleLaunchKernel(fn_wmma, 1, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    hipDeviceSynchronize();
    fprintf(stderr, "wmma launch=%d\n", (int)e1);
    hipMemcpy(Y_wmma, dY1, (size_t)M*N*4, hipMemcpyDeviceToHost);

    void *args2[] = {&dY2, &dW, &dX, &bias, &n_out, &n_in, &n_tok};
    hipError_t e2 = hipModuleLaunchKernel(fn_scal, (N+63)/64, (M+15)/16, 1, 16, 16, 1, 0, 0, args2, NULL);
    hipDeviceSynchronize();
    fprintf(stderr, "scal launch=%d\n", (int)e2);
    hipMemcpy(Y_scal, dY2, (size_t)M*N*4, hipMemcpyDeviceToHost);

    float max_w = 0, max_s = 0, max_ws = 0;
    int wi = 0;
    for (int i = 0; i < M*N; i++) {
        float d1 = fabsf(Y_wmma[i] - Y_ref[i]);
        float d2 = fabsf(Y_scal[i] - Y_ref[i]);
        float d3 = fabsf(Y_wmma[i] - Y_scal[i]);
        if (d1 > max_w) { max_w = d1; wi = i; }
        if (d2 > max_s) max_s = d2;
        if (d3 > max_ws) max_ws = d3;
    }
    printf("Shape M=%d K=%d N=%d\n", M, K, N);
    printf("max |wmma - ref|   = %.6f\n", max_w);
    printf("max |scalar - ref| = %.6f\n", max_s);
    printf("max |wmma - scal|  = %.6f\n", max_ws);
    printf("worst@%d (m=%d n=%d): ref=%.6f wmma=%.6f scalar=%.6f\n",
           wi, wi/N, wi%N, Y_ref[wi], Y_wmma[wi], Y_scal[wi]);

    /* also print a 4x4 corner */
    printf("\nRef[0..3,0..3]:\n");
    for (int m = 0; m < 4; m++) { for (int n = 0; n < 4; n++) printf("%8.3f ", Y_ref[m*N+n]); printf("\n"); }
    printf("Wmma[0..3,0..3]:\n");
    for (int m = 0; m < 4; m++) { for (int n = 0; n < 4; n++) printf("%8.3f ", Y_wmma[m*N+n]); printf("\n"); }
    return 0;
}
