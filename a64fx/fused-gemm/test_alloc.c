// Test allocation behavior
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Same allocation pattern as benchmark
    int M = 64;
    int K = 64;
    int N = 48;

    printf("Testing allocation pattern\n");

    // Benchmark allocations
    size_t size1 = (size_t)M * K * sizeof(float);  // 16384
    size_t size2 = (size_t)K * N * sizeof(float);  // 12288
    size_t size3 = (size_t)M * N * sizeof(float);  // 12288
    size_t size4 = (size_t)M * N * sizeof(float);  // 12288

    printf("Sizes: %zu, %zu, %zu, %zu\n", size1, size2, size3, size4);

    float* A = (float*)aligned_alloc(64, size1);
    float* B = (float*)aligned_alloc(64, size2);
    float* C = (float*)aligned_alloc(64, size3);
    float* C_ref = (float*)aligned_alloc(64, size4);

    printf("Allocated: A=%p, B=%p, C=%p, C_ref=%p\n", A, B, C, C_ref);

    // Check alignment
    printf("Aligned: A=%d, B=%d, C=%d, C_ref=%d\n",
           ((size_t)A % 64) == 0,
           ((size_t)B % 64) == 0,
           ((size_t)C % 64) == 0,
           ((size_t)C_ref % 64) == 0);

    // Touch memory
    memset(A, 0, size1);
    memset(B, 0, size2);
    memset(C, 0, size3);
    memset(C_ref, 0, size4);

    printf("Memory touched\n");

    // Free in same order as benchmark
    free(A);
    printf("Freed A\n");
    free(B);
    printf("Freed B\n");
    free(C);
    printf("Freed C\n");
    free(C_ref);
    printf("Freed C_ref\n");

    printf("Done\n");
    return 0;
}
