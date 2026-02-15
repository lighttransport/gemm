#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void kernel_ffn_6row_gemm_d512(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d512_prefetch(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d512_ktile(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main(void) {
    const int M = 6, K = 512, N = 2048;
    
    int8_t* A = (int8_t*)calloc(M * K, sizeof(int8_t));
    int8_t* B = (int8_t*)calloc(K * N, sizeof(int8_t));
    int32_t* C = (int32_t*)calloc(M * N, sizeof(int32_t));
    
    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (i % 100) - 50;
    for (int i = 0; i < K * N; i++) B[i] = (i % 100) - 50;
    
    printf("D=512 Baseline:\n");
    uint64_t baseline_total = 0;
    for (int i = 0; i < 20; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        uint64_t start = get_cycles();
        kernel_ffn_6row_gemm_d512(A, B, C, M, K, N);
        uint64_t end = get_cycles();
        uint64_t elapsed = end - start;
        baseline_total += elapsed;
        if (i < 5) printf("  Iter %d: %llu cycles\n", i, (unsigned long long)elapsed);
    }
    printf("  Average (20 iters): %.1f cycles\n", baseline_total / 20.0);
    
    printf("\nD=512 Prefetch:\n");
    uint64_t prefetch_total = 0;
    for (int i = 0; i < 20; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        uint64_t start = get_cycles();
        gemm_6row_int8_d512_prefetch(A, B, C, M, K, N);
        uint64_t end = get_cycles();
        uint64_t elapsed = end - start;
        prefetch_total += elapsed;
        if (i < 5) printf("  Iter %d: %llu cycles\n", i, (unsigned long long)elapsed);
    }
    printf("  Average (20 iters): %.1f cycles\n", prefetch_total / 20.0);
    
    printf("\nD=512 K-Tiling:\n");
    uint64_t ktile_total = 0;
    for (int i = 0; i < 20; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        uint64_t start = get_cycles();
        gemm_6row_int8_d512_ktile(A, B, C, M, K, N);
        uint64_t end = get_cycles();
        uint64_t elapsed = end - start;
        ktile_total += elapsed;
        if (i < 5) printf("  Iter %d: %llu cycles\n", i, (unsigned long long)elapsed);
    }
    printf("  Average (20 iters): %.1f cycles\n", ktile_total / 20.0);
    
    printf("\n=== Summary ===\n");
    printf("Baseline:  %.1f cycles\n", baseline_total / 20.0);
    printf("Prefetch:  %.1f cycles (%.2fx speedup)\n", 
           prefetch_total / 20.0, (double)baseline_total / prefetch_total);
    printf("K-Tiling:  %.1f cycles (%.2fx speedup)\n",
           ktile_total / 20.0, (double)baseline_total / ktile_total);
    
    free(A);
    free(B);
    free(C);
    return 0;
}
