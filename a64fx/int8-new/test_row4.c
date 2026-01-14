#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gemm_driver.h"

int main() {
    const int M = 5;
    const int N = 64;
    const int K = 256;

    // Allocate matrices
    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C = (int32_t*)calloc(M * N, sizeof(int32_t));

    // Initialize with 1s
    for (int i = 0; i < M * K; i++) A[i] = 1;
    for (int i = 0; i < N * K; i++) B[i] = 1;

    // Compute with kernel
    gemm_5x4_driver(A, K, B, K, C, N, M, N, K);

    // Print all of row 4 (index 4)
    printf("Row 4 output (all 64 values, expected 256 for each):\n");
    for (int n = 0; n < N; n++) {
        int32_t val = C[4 * N + n];
        char status = (val == 256) ? ' ' : 'X';
        printf("  C[4,%2d] = %3d %c\n", n, val, status);
    }

    // Count errors
    int errors = 0;
    for (int n = 0; n < N; n++) {
        if (C[4 * N + n] != 256) errors++;
    }
    printf("\nTotal errors in row 4: %d / 64\n", errors);

    free(A);
    free(B);
    free(C);

    return 0;
}
