#include <stdio.h>
#include <stdint.h>
#include "gemm_driver.h"
#include "gemm_pack.h"

// Simple test: 5×64 × 64×256 with known values
int main() {
    const int M = 5;
    const int N = 64;
    const int K = 256;

    // Allocate matrices
    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C_ref = (int32_t*)calloc(M * N, sizeof(int32_t));
    int32_t* C_test = (int32_t*)calloc(M * N, sizeof(int32_t));

    // Initialize with simple pattern
    // A: all 1s
    for (int i = 0; i < M * K; i++) A[i] = 1;

    // B: all 1s
    for (int i = 0; i < N * K; i++) B[i] = 1;

    // Expected result: C[m][n] = sum of 256 (1*1) = 256
    printf("Test: A(5x256)=1, B(64x256)=1\n");
    printf("Expected: C[all] = 256\n\n");

    // Compute reference
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Compute with kernel
    gemm_5x4_driver(A, K, B, K, C_test, N, M, N, K);

    // Check results
    printf("Results (first row, first 10 elements):\n");
    printf("Ref:  ");
    for (int n = 0; n < 10; n++) printf("%5d ", C_ref[n]);
    printf("\n");

    printf("Test: ");
    for (int n = 0; n < 10; n++) printf("%5d ", C_test[n]);
    printf("\n\n");

    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (C_ref[m * N + n] != C_test[m * N + n]) {
                if (errors < 5) {
                    printf("ERROR [%d,%d]: ref=%d, test=%d\n",
                           m, n, C_ref[m * N + n], C_test[m * N + n]);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASS: All elements match!\n");
    } else {
        printf("FAIL: %d / %d elements wrong\n", errors, M * N);
    }

    free(A);
    free(B);
    free(C_ref);
    free(C_test);

    return (errors == 0) ? 0 : 1;
}
