// Simple fused GEMM test
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    // Same as first fused test: M=64, K1=64, K2=96, N=48
    // E[64,48] = (A[64,64] @ B[64,96]) @ C[96,48]
    int M = 64;
    int K1 = 64;
    int K2 = 96;
    int N = 48;

    printf("Fused GEMM test: E[%d,%d] = (A[%d,%d] @ B[%d,%d]) @ C[%d,%d]\n",
           M, N, M, K1, K1, K2, K2, N);

    srand(42);

    float* A = (float*)aligned_alloc(64, (size_t)M * K1 * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K1 * K2 * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)K2 * N * sizeof(float));
    float* E = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    printf("Allocated matrices\n");

    // Initialize
    for (int i = 0; i < M * K1; i++) A[i] = (float)(rand() % 1000 - 500) / 500.0f;
    for (int i = 0; i < K1 * K2; i++) B[i] = (float)(rand() % 1000 - 500) / 500.0f;
    for (int i = 0; i < K2 * N; i++) C[i] = (float)(rand() % 1000 - 500) / 500.0f;
    memset(E, 0, (size_t)M * N * sizeof(float));

    printf("Calling fused_gemm_abc...\n");
    fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);

    printf("Result: E[0]=%f, E[last]=%f\n", E[0], E[M*N-1]);

    printf("Freeing...\n");
    free(A);
    free(B);
    free(C);
    free(E);

    printf("Done!\n");
    return 0;
}
