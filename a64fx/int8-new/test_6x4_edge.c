#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gemm_driver.h"

// Naive reference
void gemm_ref(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    // Test the problematic case: M=100 (not divisible by 6)
    const int M = 100;
    const int N = 512;
    const int K = 256;

    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C_ref = (int32_t*)calloc(M * N, sizeof(int32_t));
    int32_t* C_test = (int32_t*)calloc(M * N, sizeof(int32_t));

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 21) - 10;
    for (int i = 0; i < N * K; i++) B[i] = (rand() % 21) - 10;

    printf("Testing 6x4 kernel with M=%d, N=%d, K=%d\n", M, N, K);
    printf("M is NOT divisible by 6 (testing edge case)\n\n");

    // Compute reference
    gemm_ref(A, B, C_ref, M, N, K);

    // Compute with 6x4 kernel
    gemm_6x4_driver(A, K, B, K, C_test, N, M, N, K);

    // Check results
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (C_ref[m * N + n] != C_test[m * N + n]) {
                if (errors < 10) {
                    printf("ERROR at [%d,%d]: ref=%d, test=%d, diff=%d\n",
                           m, n, C_ref[m * N + n], C_test[m * N + n],
                           C_ref[m * N + n] - C_test[m * N + n]);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASS: All %d elements match!\n", M * N);
    } else {
        printf("FAIL: %d / %d elements wrong\n", errors, M * N);
    }

    free(A);
    free(B);
    free(C_ref);
    free(C_test);

    return (errors == 0) ? 0 : 1;
}
