// Standalone correctness test for the int8 GEMM vs an fp32 reference.
// int8 has quantization error, so we check the RELATIVE error is small
// (a few %), and that the shape/norm are right.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <arm_sve.h>
#include "int8_gemm.h"

static uint64_t cntvct(void) { uint64_t t; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t)); return t; }

static float *xal64(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~(size_t)63); if (!p) exit(1); return (float *)p; }
static float *ref_gemm(const float *X, const float *W, int M, int K, int N) {
    float *C = (float *)malloc((size_t)M * N * 4);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)X[(size_t)m * K + k] * W[(size_t)n * K + k];
            C[(size_t)m * N + n] = (float)s;
        }
    return C;
}
static int test(const char *name, int M, int K, int N) {
    float *X = (float *)malloc((size_t)M * K * 4);
    float *W = (float *)malloc((size_t)N * K * 4);   // [N][K]
    srand(1);
    for (int i = 0; i < M * K; i++) X[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = ((rand() % 2000) / 1000.0f) - 1.0f;

    float *Cref = ref_gemm(X, W, M, K, N);

    // build BT [K][N] (BT[k][n] = W[n][k]) for take_int8_packed
    float *BT = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) BT[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *scale;
    int8_t *Bpack = take_int8_packed(&BT, K, N, &scale);
    float *Ci8 = (float *)malloc((size_t)M * N * 4);
    gemm_int8_BTP(M, K, N, X, K, Bpack, scale, Ci8, N);

    // compare
    double maxabs = 0, sumref = 0, sumerr = 0, re = 0;
    for (int i = 0; i < M * N; i++) {
        float d = Ci8[i] - Cref[i];
        if (fabsf(d) > maxabs) maxabs = fabsf(d);
        sumref += Cref[i] * Cref[i];
        sumerr += d * d;
        re += fabsf(d);
    }
    double rel = sqrt(sumerr / sumref);
    double mean_abs = re / (M * N);
    printf("%-10s M=%d K=%d N=%d : maxabs=%.4f  rel(L2)=%.4f  mean|err|=%.4f %s\n",
           name, M, K, N, maxabs, rel, mean_abs, rel <= 0.05 ? "OK" : "FAIL");
    free(X); free(W); free(Cref); free(Ci8); free(Bpack); free(scale);
    return (rel > 0.05) ? 1 : 0;   /* int8 quantization is a few % */
}
static int test16(const char *name, int M, int K, int N) {
    float *X = (float *)malloc((size_t)M * K * 4);
    float *W = (float *)malloc((size_t)N * K * 4);   // [N][K]
    srand(1);
    for (int i = 0; i < M * K; i++) X[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    float *Cref = ref_gemm(X, W, M, K, N);
    float *BT = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) BT[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *scale; int32_t *colsum;
    int8_t *Bpack = take_int16_packed(&BT, K, N, &scale, &colsum);
    float *Ci16 = (float *)malloc((size_t)M * N * 4);
    gemm_int16_BTP(M, K, N, X, K, Bpack, scale, colsum, Ci16, N);
    double maxabs = 0, sumref = 0, sumerr = 0, re = 0;
    for (int i = 0; i < M * N; i++) {
        float d = Ci16[i] - Cref[i];
        if (fabsf(d) > maxabs) maxabs = fabsf(d);
        sumref += Cref[i] * Cref[i]; sumerr += d * d; re += fabsf(d);
    }
    double rel16 = sqrt(sumref > 0 ? sumerr / sumref : 0);
    printf("i16 %-10s M=%d K=%d N=%d : maxabs=%.4f  rel(L2)=%.6f  mean|err|=%.5f %s\n",
           name, M, K, N, maxabs, rel16, re/(M*N), rel16 <= 1e-4 ? "OK" : "FAIL");
    free(X); free(W); free(Cref); free(Ci16); free(Bpack); free(scale); free(colsum);
    return (rel16 > 1e-4) ? 1 : 0;   /* int16 is near-exact */
}

// Time int8 vs int16 for the VLM shapes (48 threads), CNTVCT.
static void bench16(void) {
    int M = 96, K = 1024, N = 4096;
    float *X = (float *)malloc((size_t)M * K * 4);
    float *W = (float *)malloc((size_t)N * K * 4);
    srand(7);
    for (int i = 0; i < M * K; i++) X[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    float *BT = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) BT[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *Y = (float *)malloc((size_t)M * N * 4);
    float *s8; int8_t *b8 = take_int8_packed(&BT, K, N, &s8);
    float *BT2 = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) BT2[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *s16; int32_t *cs;
    int8_t *b16 = take_int16_packed(&BT2, K, N, &s16, &cs);
    const int iters = 20;
    // warmup
    for (int i = 0; i < 2; i++) { gemm_int8_BTP(M,K,N,X,K,b8,s8,Y,N); gemm_int16_BTP(M,K,N,X,K,b16,s16,cs,Y,N); }
    uint64_t t0 = cntvct();
    for (int i = 0; i < iters; i++) gemm_int8_BTP(M,K,N,X,K,b8,s8,Y,N);
    uint64_t t1 = cntvct();
    for (int i = 0; i < iters; i++) gemm_int16_BTP(M,K,N,X,K,b16,s16,cs,Y,N);
    uint64_t t2 = cntvct();
    double ms8 = (t1 - t0) / 1e5 / iters, ms16 = (t2 - t1) / 1e5 / iters;
    double flops = 2.0 * M * K * N;   // one GEMM's FLOPs
    printf("bench %dx%dx%d %d threads: int8 %.3f ms (%.0f GOPS) | int16 %.3f ms (%.0f GOPS) | int16/int8 = %.2fx\n",
           M,K,N, omp_get_max_threads(), ms8, flops/(ms8*1e-3)/1e9, ms16, flops/(ms16*1e-3)/1e9, ms16/ms8);
    free(X); free(W); free(BT); free(BT2); free(Y); free(b8); free(s8); free(b16); free(s16); free(cs);
}

/* Fused-path test: gemm_int8_BTP_fused must match gemm_int8_BTP exactly
 * (same quantize + pack + kernel math; only the C-buffer/dequant split differs). */
static int testfused(const char *name, int M, int K, int N) {
    float *X = (float *)malloc((size_t)M * K * 4);
    float *W = (float *)malloc((size_t)N * K * 4);   // [N][K]
    srand(1);
    for (int i = 0; i < M * K; i++) X[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = ((rand() % 2000) / 1000.0f) - 1.0f;

    float *BT = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) BT[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *scale;
    int8_t *Bpack = take_int8_packed(&BT, K, N, &scale);
    float *Ca = xal64((size_t)M * N * 4);
    float *Cb = xal64((size_t)M * N * 4);
    gemm_int8_BTP(M, K, N, X, K, Bpack, scale, Ca, N);
    gemm_int8_BTP_fused(M, K, N, X, K, Bpack, scale, Cb, N);

    int neq = 0;
    float maxdiff = 0;
    for (int i = 0; i < M * N; i++) {
        if (Ca[i] != Cb[i]) {
            neq++;
            float d = fabsf(Ca[i] - Cb[i]);
            if (d > maxdiff) maxdiff = d;
        }
    }
    printf("%-10s M=%d K=%d N=%d : fused vs non-fused %s (%d/%d differ, maxdiff=%.3g) %s\n",
           name, M, K, N, neq == 0 ? "BIT-IDENTICAL" : "MISMATCH", neq, M * N, maxdiff,
           neq == 0 ? "OK" : "FAIL");
    free(X); free(W); free(BT); free(Bpack); free(scale); free(Ca); free(Cb);
    return (neq != 0) ? 1 : 0;   /* fused must be bit-identical */
}

static void benchfused(void) {
    int M = 96, K = 1024, N = 4096;
    float *X = (float *)malloc((size_t)M * K * 4);
    float *W = (float *)malloc((size_t)N * K * 4);
    srand(7);
    for (int i = 0; i < M * K; i++) X[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < N * K; i++) W[i] = ((rand() % 2000) / 1000.0f) - 1.0f;
    float *BT = (float *)malloc((size_t)K * N * 4);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) BT[(size_t)k * N + n] = W[(size_t)n * K + k];
    float *s8; int8_t *b8 = take_int8_packed(&BT, K, N, &s8);
    float *Y = xal64((size_t)M * N * 4);
    const int iters = 20;
    for (int i = 0; i < 2; i++) { gemm_int8_BTP(M,K,N,X,K,b8,s8,Y,N); gemm_int8_BTP_fused(M,K,N,X,K,b8,s8,Y,N); }
    uint64_t t0 = cntvct();
    for (int i = 0; i < iters; i++) gemm_int8_BTP(M,K,N,X,K,b8,s8,Y,N);
    uint64_t t1 = cntvct();
    for (int i = 0; i < iters; i++) gemm_int8_BTP_fused(M,K,N,X,K,b8,s8,Y,N);
    uint64_t t2 = cntvct();
    double ms = (t1 - t0) / 1e5 / iters, mfs = (t2 - t1) / 1e5 / iters;
    double flops = 2.0 * M * K * N;
    printf("bench %dx%dx%d %d threads: int8 %.3f ms (%.0f GOPS) | int8-fused %.3f ms (%.0f GOPS) | fused/non-fused = %.2fx\n",
           M,K,N, omp_get_max_threads(), ms, flops/(ms*1e-3)/1e9, mfs, flops/(mfs*1e-3)/1e9, mfs/ms);
    free(X); free(W); free(BT); free(Y); free(b8); free(s8);
}

int main(int argc, char **argv) {
    int fails = 0;
    if (argc > 1 && !strcmp(argv[1], "fused")) {
        fails += testfused("ffn_up", 96, 1024, 4096);
        fails += testfused("qkv", 96, 1024, 3072);
        fails += testfused("odd_M", 49, 1024, 1024);
        if (argc > 2 && !strcmp(argv[2], "bench")) benchfused();
        return fails;
    }
    if (argc > 1 && argv[1][0] == '8') {
        fails += test("ffn_up", 96, 1024, 4096);
        fails += test("qkv", 96, 1024, 3072);
        fails += test("odd_M", 49, 1024, 1024);
        return fails;
    }
    fails += test16("ffn_up", 96, 1024, 4096);
    fails += test16("qkv", 96, 1024, 3072);
    fails += test16("odd_M", 49, 1024, 1024);
    if (argc > 1 && !strcmp(argv[1], "bench")) bench16();
    return fails;
}
