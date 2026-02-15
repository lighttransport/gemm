// test_correctness_large.c - Correctness test for larger sizes
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

// Reference GEMM
static void gemm_ref(const int8_t* A, int lda, const int8_t* B, int ldb,
                     int32_t* C, int ldc, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

int test_size(int M, int N, int K) {
    printf("Testing M=%d, N=%d, K=%d... ", M, N, K);
    fflush(stdout);

    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C_ref = malloc((size_t)M * N * sizeof(int32_t));
    int32_t* C_ultra = malloc((size_t)M * N * sizeof(int32_t));
    int32_t* C_pipe = malloc((size_t)M * N * sizeof(int32_t));

    if (!A || !B || !C_ref || !C_ultra || !C_pipe) {
        printf("ALLOC FAILED\n");
        return 1;
    }

    // Use simple pattern for easy verification
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (rand() % 5) - 2;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (rand() % 5) - 2;
    memset(C_ref, 0, (size_t)M * N * sizeof(int32_t));
    memset(C_ultra, 0, (size_t)M * N * sizeof(int32_t));
    memset(C_pipe, 0, (size_t)M * N * sizeof(int32_t));

    // Compute reference (only for small sizes)
    if (M * N <= 256 * 256) {
        gemm_ref(A, K, B, K, C_ref, N, M, N, K);
    }

    // Compute optimized versions
    gemm_opt_driver(A, K, B, K, C_ultra, N, M, N, K);
    gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);

    // Compare
    int ultra_errors = 0, pipe_errors = 0, ultra_vs_pipe = 0;

    if (M * N <= 256 * 256) {
        // Compare both to reference
        for (size_t i = 0; i < (size_t)M * N; i++) {
            if (C_ref[i] != C_ultra[i]) ultra_errors++;
            if (C_ref[i] != C_pipe[i]) pipe_errors++;
        }
    }

    // Compare ultra vs pipe (should match even for large sizes)
    for (size_t i = 0; i < (size_t)M * N; i++) {
        if (C_ultra[i] != C_pipe[i]) ultra_vs_pipe++;
    }

    // Print first few values to verify results look reasonable
    printf("\n  First 4 C values: ultra=[%d,%d,%d,%d] pipe=[%d,%d,%d,%d]\n",
           C_ultra[0], C_ultra[1], C_ultra[2], C_ultra[3],
           C_pipe[0], C_pipe[1], C_pipe[2], C_pipe[3]);

    int pass = 1;
    if (M * N <= 256 * 256) {
        if (ultra_errors > 0) {
            printf("  Ultra vs ref: FAIL (%d errors)\n", ultra_errors);
            pass = 0;
        } else {
            printf("  Ultra vs ref: PASS\n");
        }
        if (pipe_errors > 0) {
            printf("  Pipe vs ref: FAIL (%d errors)\n", pipe_errors);
            pass = 0;
        } else {
            printf("  Pipe vs ref: PASS\n");
        }
    }

    if (ultra_vs_pipe > 0) {
        printf("  Ultra vs pipe: MISMATCH (%d differences)\n", ultra_vs_pipe);
        pass = 0;
    } else {
        printf("  Ultra vs pipe: MATCH\n");
    }

    free(A); free(B); free(C_ref); free(C_ultra); free(C_pipe);
    return pass ? 0 : 1;
}

int main() {
    printf("=== Correctness Test for Large Sizes ===\n\n");

    int failures = 0;

    // Small sizes (with reference check)
    failures += test_size(6, 64, 256);
    failures += test_size(12, 128, 256);
    failures += test_size(24, 192, 256);
    failures += test_size(48, 256, 256);
    failures += test_size(96, 256, 256);
    failures += test_size(192, 256, 256);

    // Larger sizes (compare ultra vs pipe only)
    failures += test_size(384, 384, 256);
    failures += test_size(768, 768, 256);
    failures += test_size(1536, 1536, 256);

    printf("\n=== Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d tests FAILED\n", failures);
    }

    return failures;
}
