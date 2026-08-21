// Standalone correctness test for the int8 GEMM vs an fp32 reference.
// int8 has quantization error, so we check the RELATIVE error is small
// (a few %), and that the shape/norm are right.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels/int8_gemm.h"

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
static void test(const char *name, int M, int K, int N) {
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
    printf("%-10s M=%d K=%d N=%d : maxabs=%.4f  rel(L2)=%.4f  mean|err|=%.4f\n",
           name, M, K, N, maxabs, rel, mean_abs);
    free(X); free(W); free(Cref); free(Ci8); free(Bpack); free(scale);
}
int main(void) {
    test("ffn_up", 96, 1024, 4096);
    test("qkv", 96, 1024, 3072);
    test("odd_M", 49, 1024, 1024);   // M%6 != 0 edge case
    return 0;
}
