/* test_tf32_f32.c - validate cublasew f32 precision tiers (Phase 2).
 *
 * Compares pedantic / default-SGEMM / TF32 row-major NT GEMM against a CPU
 * f32 reference: TF32 should match to ~1e-3 relative error and run faster on
 * tensor cores; pedantic/default should match to ~1e-5. Also exercises the
 * CUBLASEW_ALLOW_TF32 gate via cublasew_set_tf32().
 *
 * Build: gcc -O3 -I.. -I../../common test_tf32_f32.c ../cuew.c ../cublasew.c \
 *            -ldl -lm -lpthread -o test_tf32_f32
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../cuew.h"
#include "../cublasew.h"

/* Row-major Y[m,n_out] = X[m,n_in] * W[n_out,n_in]^T */
static void cpu_ref(float *Y, const float *W, const float *X,
                    int m, int n_out, int n_in) {
    for (int r = 0; r < m; r++)
        for (int o = 0; o < n_out; o++) {
            double acc = 0.0;
            const float *xr = X + (size_t)r * n_in;
            const float *wr = W + (size_t)o * n_in;
            for (int k = 0; k < n_in; k++) acc += (double)xr[k] * (double)wr[k];
            Y[(size_t)r * n_out + o] = (float)acc;
        }
}

static void rel_err(const float *a, const float *ref, size_t n,
                    double *max_rel, double *mean_rel) {
    double mx = 0.0, sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)ref[i]);
        double den = fabs((double)ref[i]) + 1e-6;
        double r = d / den;
        if (r > mx) mx = r;
        sum += r;
    }
    *max_rel = mx;
    *mean_rel = sum / (double)n;
}

typedef int (*gemm_fn)(cublasew_context *, CUdeviceptr, CUdeviceptr, CUdeviceptr,
                       int, int, int);

static double time_gemm(gemm_fn fn, cublasew_context *bc, CUdeviceptr dY,
                        CUdeviceptr dW, CUdeviceptr dX, int m, int n_out,
                        int n_in, CUstream stream, int iters) {
    /* warmup */
    fn(bc, dY, dW, dX, m, n_out, n_in);
    cuStreamSynchronize(stream);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) fn(bc, dY, dW, dX, m, n_out, n_in);
    cuStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return secs / iters;
}

int main(void) {
    const int m = 512, n_out = 4096, n_in = 4096;
    const int iters = 50;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1; }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 1; }
    CUdevice dev; CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);
    char name[256]; cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s   shape m=%d n_out=%d n_in=%d\n", name, m, n_out, n_in);

    CUstream stream = NULL; cuStreamCreate(&stream, 0);
    cublasew_context *bc = NULL;
    if (cublasewCreate(&bc, stream) != 0) { fprintf(stderr, "cublasewCreate failed\n"); return 1; }

    size_t nW = (size_t)n_out * n_in, nX = (size_t)m * n_in, nY = (size_t)m * n_out;
    float *hW = malloc(nW * 4), *hX = malloc(nX * 4);
    float *hY = malloc(nY * 4), *hRef = malloc(nY * 4);
    srand(1234);
    for (size_t i = 0; i < nW; i++) hW[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < nX; i++) hX[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    cpu_ref(hRef, hW, hX, m, n_out, n_in);

    CUdeviceptr dW, dX, dY;
    cuMemAlloc(&dW, nW * 4); cuMemAlloc(&dX, nX * 4); cuMemAlloc(&dY, nY * 4);
    cuMemcpyHtoD(dW, hW, nW * 4); cuMemcpyHtoD(dX, hX, nX * 4);

    double mx, mean, secs, gflop = 2.0 * m * n_out * n_in * 1e-9;

    struct { const char *tag; gemm_fn fn; } tiers[] = {
        { "pedantic", cublasew_gemm_f32_pedantic_rowmajor_nt },
        { "default ", cublasew_gemm_f32_rowmajor_nt },
        { "tf32    ", cublasew_gemm_f32_tf32_rowmajor_nt },
    };
    printf("\n%-9s  %10s  %10s  %10s  %8s\n", "tier", "max_rel", "mean_rel", "ms/iter", "TFLOP/s");
    for (int t = 0; t < 3; t++) {
        if (tiers[t].fn(bc, dY, dW, dX, m, n_out, n_in) != 0) {
            printf("%-9s  FAILED (returned -1)\n", tiers[t].tag); continue; }
        cuStreamSynchronize(stream);
        cuMemcpyDtoH(hY, dY, nY * 4);
        rel_err(hY, hRef, nY, &mx, &mean);
        secs = time_gemm(tiers[t].fn, bc, dY, dW, dX, m, n_out, n_in, stream, iters);
        printf("%-9s  %10.2e  %10.2e  %10.4f  %8.2f\n",
               tiers[t].tag, mx, mean, secs * 1e3, gflop / secs);
    }

    /* Gate test: default path should switch to TF32 when the gate is on. */
    printf("\n--- gate (cublasew_set_tf32) ---\n");
    cublasew_set_tf32(bc, 1);
    cublasew_gemm_f32_rowmajor_nt(bc, dY, dW, dX, m, n_out, n_in);
    cuStreamSynchronize(stream);
    cuMemcpyDtoH(hY, dY, nY * 4);
    rel_err(hY, hRef, nY, &mx, &mean);
    printf("default+gate ON : max_rel=%.2e mean_rel=%.2e (expect ~tf32)\n", mx, mean);
    cublasew_set_tf32(bc, 0);
    cublasew_gemm_f32_rowmajor_nt(bc, dY, dW, dX, m, n_out, n_in);
    cuStreamSynchronize(stream);
    cuMemcpyDtoH(hY, dY, nY * 4);
    rel_err(hY, hRef, nY, &mx, &mean);
    printf("default+gate OFF: max_rel=%.2e mean_rel=%.2e (expect ~default)\n", mx, mean);

    cuMemFree(dW); cuMemFree(dX); cuMemFree(dY);
    free(hW); free(hX); free(hY); free(hRef);
    cublasewDestroy(bc);
    printf("\nOK\n");
    return 0;
}
