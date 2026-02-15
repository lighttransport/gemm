// MLP-like fused GEMM test
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // MLP config: M=512, K1=768, K2=3072, N=768
    int M = 512;
    int K1 = 768;
    int K2 = 3072;
    int N = 768;

    printf("Fused GEMM test: E[%d,%d] = (A[%d,%d] @ B[%d,%d]) @ C[%d,%d]\n",
           M, N, M, K1, K1, K2, K2, N);

    size_t A_size = (size_t)M * K1 * sizeof(float);
    size_t B_size = (size_t)K1 * K2 * sizeof(float);
    size_t C_size = (size_t)K2 * N * sizeof(float);
    size_t E_size = (size_t)M * N * sizeof(float);
    size_t D_size = (size_t)M * K2 * sizeof(float);  // Intermediate

    printf("Memory: A=%.1fMB B=%.1fMB C=%.1fMB D=%.1fMB E=%.1fMB\n",
           A_size/1e6, B_size/1e6, C_size/1e6, D_size/1e6, E_size/1e6);

    srand(42);

    float* A = (float*)aligned_alloc(64, A_size);
    float* B = (float*)aligned_alloc(64, B_size);
    float* C = (float*)aligned_alloc(64, C_size);
    float* E = (float*)aligned_alloc(64, E_size);

    if (!A || !B || !C || !E) {
        printf("Allocation failed!\n");
        return 1;
    }

    printf("Allocated matrices\n");

    // Initialize
    for (size_t i = 0; i < M * K1; i++) A[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    for (size_t i = 0; i < K1 * K2; i++) B[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    for (size_t i = 0; i < K2 * N; i++) C[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    memset(E, 0, E_size);

    printf("Calling fused_gemm_abc...\n");
    fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);

    printf("Result: E[0]=%f, E[last]=%f\n", E[0], E[M*N-1]);

    printf("Freeing...\n");
    free(A);
    printf("  A freed\n");
    free(B);
    printf("  B freed\n");
    free(C);
    printf("  C freed\n");
    free(E);
    printf("  E freed\n");

    printf("Done!\n");
    return 0;
}
