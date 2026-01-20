#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void kernel_ffn_6row_gemm_d256(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d256_prefetch(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main(void) {
    const int M = 6, K = 256, N = 1024;
    
    int8_t* A = (int8_t*)calloc(M * K, sizeof(int8_t));
    int8_t* B = (int8_t*)calloc(K * N, sizeof(int8_t));
    int32_t* C = (int32_t*)calloc(M * N, sizeof(int32_t));
    
    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (i % 100) - 50;
    for (int i = 0; i < K * N; i++) B[i] = (i % 100) - 50;
    
    printf("D=256 Baseline:\n");
    for (int i = 0; i < 5; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        uint64_t start = get_cycles();
        kernel_ffn_6row_gemm_d256(A, B, C, M, K, N);
        uint64_t end = get_cycles();
        printf("  Iter %d: %llu cycles\n", i, (unsigned long long)(end - start));
    }
    
    printf("\nD=256 Prefetch:\n");
    for (int i = 0; i < 5; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        uint64_t start = get_cycles();
        gemm_6row_int8_d256_prefetch(A, B, C, M, K, N);
        uint64_t end = get_cycles();
        printf("  Iter %d: %llu cycles\n", i, (unsigned long long)(end - start));
    }
    
    free(A);
    free(B);
    free(C);
    return 0;
}
