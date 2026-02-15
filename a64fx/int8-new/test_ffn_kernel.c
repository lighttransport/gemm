// test_ffn_kernel.c - Simple test for the 6-row GEMM kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void kernel_ffn_6row_gemm_d512(const int8_t* A, const int8_t* B, int32_t* C);

int main() {
    printf("Testing 6-row GEMM kernel for D=512, D_ff=2048...\n");

    const int M = 6;
    const int K = 512;
    const int N = 2048;
    const int K_groups = K / 4;  // 128

    // Allocate memory
    int8_t* A = aligned_alloc(64, M * K);
    int8_t* B = aligned_alloc(64, K_groups * N * 4);  // Packed [128, 2048, 4]
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
    kernel_ffn_6row_gemm_d512(A, B, C);

    printf("Checking results...\n");
    // Each element should be M*K = 6*512 = 3072 (since all inputs are 1)
    // Actually: sum over K dimension = K = 512 (per output element)
    int32_t expected = K;  // Each element is sum of K products of 1*1
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
