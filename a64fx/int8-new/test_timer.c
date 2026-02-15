// Simple test to verify timer and kernels work
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void kernel_ffn_6row_gemm_d256(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

// Get cycle counter
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main(void) {
    printf("Testing timer and kernel...\n\n");

    // Test timer
    uint64_t t1 = get_cycles();
    printf("Timer read 1: %llu\n", (unsigned long long)t1);

    for (volatile int i = 0; i < 1000000; i++);  // Busy loop

    uint64_t t2 = get_cycles();
    printf("Timer read 2: %llu\n", (unsigned long long)t2);
    printf("Elapsed: %llu cycles\n", (unsigned long long)(t2 - t1));

    if (t2 <= t1) {
        printf("ERROR: Timer not working!\n");
        return 1;
    }

    // Allocate matrices
    const int M = 6, K = 256, N = 1024;
    int8_t* A = (int8_t*)calloc(M * K, sizeof(int8_t));
    int8_t* B = (int8_t*)calloc(K * N, sizeof(int8_t));
    int32_t* C = (int32_t*)calloc(M * N, sizeof(int32_t));

    // Initialize with small values
    for (int i = 0; i < M * K; i++) A[i] = 1;
    for (int i = 0; i < K * N; i++) B[i] = 1;

    printf("\nTesting D=256 kernel...\n");
    uint64_t start = get_cycles();
    kernel_ffn_6row_gemm_d256(A, B, C, M, K, N);
    uint64_t end = get_cycles();

    printf("Kernel elapsed: %llu timer cycles\n", (unsigned long long)(end - start));
    printf("Kernel elapsed: %llu CPU cycles (×20)\n", (unsigned long long)((end - start) * 20));

    // Check results
    int nonzero = 0;
    for (int i = 0; i < M * N && i < 10; i++) {
        if (C[i] != 0) nonzero++;
        printf("C[%d] = %d\n", i, C[i]);
    }
    printf("Nonzero results: %d / %d\n", nonzero, 10);

    free(A);
    free(B);
    free(C);
    return 0;
}
