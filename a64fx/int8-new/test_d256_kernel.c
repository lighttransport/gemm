// test_d256_kernel.c - Simple test for D=256 6-row GEMM kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void kernel_ffn_6row_gemm_d256(const int8_t* A, const int8_t* B, int32_t* C);

int main() {
    printf("Testing 6-row GEMM kernel for D=256, D_ff=1024...\n");

    const int M = 6;
    const int K = 256;
    const int N = 1024;
    const int K_groups = K / 4;  // 64

    // Allocate memory
    int8_t* A = aligned_alloc(64, M * K);
    int8_t* B = aligned_alloc(64, K_groups * N * 4);  // Packed [64, 1024, 4]
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));

    if (!A || !B || !C) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Initialize with simple values
    for (int i = 0; i < M * K; i++) A[i] = 1;
    for (int i = 0; i < K_groups * N * 4; i++) B[i] = 1;
    memset(C, 0, M * N * sizeof(int32_t));

    printf("Running kernel...\n");
    kernel_ffn_6row_gemm_d256(A, B, C);

    printf("Checking results...\n");
    // Each element should be K = 256 (sum of K products of 1*1)
    int32_t expected = K;
    int errors = 0;
    for (int m = 0; m < M && errors < 10; m++) {
        for (int n = 0; n < N && errors < 10; n++) {
            int32_t val = C[m * N + n];
            if (val != expected) {
                printf("Error at C[%d,%d]: got %d, expected %d\n", m, n, val, expected);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASSED: All %d elements correct!\n", M * N);
    } else {
        printf("FAILED: %d+ errors found\n", errors);
    }

    // Print a few sample values
    printf("\nSample values:\n");
    printf("C[0,0]=%d C[0,1]=%d C[0,100]=%d\n", C[0], C[1], C[100]);
    printf("C[5,0]=%d C[5,1]=%d C[5,100]=%d\n", C[5*N], C[5*N+1], C[5*N+100]);

    free(A);
    free(B);
    free(C);

    return 0;
}
