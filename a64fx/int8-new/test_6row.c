// test_6row.c - Simple test for 6-row kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define D512 512

static inline uint64_t rdtsc(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, CNTFRQ_EL0" : "=r"(f));
    return f;
}

extern void kernel_fused_d512_5row(const int8_t*, const int8_t*, const int8_t*, int32_t*);
extern void kernel_fused_d512_6row(const int8_t*, const int8_t*, const int8_t*, int32_t*);

int main() {
    printf("Testing 6-row kernel...\n");

    uint64_t freq = get_timer_freq();
    printf("Timer freq: %lu\n", freq);

    int D = D512;
    int L = 4096;
    int M = 30;  // divisible by 5 and 6
    int N_chunks = L / 64;
    int K_chunk_size = 128 * 64 * 4;
    int V_chunk_size = 8 * 16 * 64 * 4;

    printf("Allocating memory...\n");
    int8_t* Q = aligned_alloc(64, M * D);
    int8_t* K_int = aligned_alloc(64, N_chunks * K_chunk_size);
    int8_t* V_t = aligned_alloc(64, N_chunks * V_chunk_size);
    int32_t* O = aligned_alloc(64, M * D * sizeof(int32_t));

    if (!Q || !K_int || !V_t || !O) {
        printf("Allocation failed!\n");
        return 1;
    }

    printf("Initializing...\n");
    memset(Q, 1, M * D);
    memset(K_int, 1, N_chunks * K_chunk_size);
    memset(V_t, 1, N_chunks * V_chunk_size);
    memset(O, 0, M * D * sizeof(int32_t));

    printf("Testing 5-row kernel...\n");
    uint64_t t0 = rdtsc();
    kernel_fused_d512_5row(Q, K_int, V_t, O);
    uint64_t t1 = rdtsc();
    printf("5-row: %lu ticks\n", t1 - t0);

    printf("Testing 6-row kernel...\n");
    memset(O, 0, M * D * sizeof(int32_t));
    uint64_t t2 = rdtsc();
    kernel_fused_d512_6row(Q, K_int, V_t, O);
    uint64_t t3 = rdtsc();
    printf("6-row: %lu ticks\n", t3 - t2);

    // Verify some output
    printf("O[0]=%d, O[100]=%d, O[500]=%d\n", O[0], O[100], O[500]);

    // Full benchmark
    printf("\nFull benchmark (M=%d, L=%d):\n", M, L);
    int M_tiles_5 = M / 5;
    int M_tiles_6 = M / 6;

    memset(O, 0, M * D * sizeof(int32_t));
    t0 = rdtsc();
    for (int mt = 0; mt < M_tiles_5; mt++) {
        for (int nc = 0; nc < N_chunks; nc++) {
            kernel_fused_d512_5row(Q + mt * 5 * D, K_int + nc * K_chunk_size,
                                    V_t + nc * V_chunk_size, O + mt * 5 * D);
        }
    }
    t1 = rdtsc();
    double sec_5 = (double)(t1 - t0) / freq;
    double flops = 2.0 * 2.0 * M * L * D;
    printf("5-row: %lu ticks = %.4f sec, %.1f GFLOPS\n",
           t1 - t0, sec_5, flops / sec_5 / 1e9);

    memset(O, 0, M * D * sizeof(int32_t));
    t2 = rdtsc();
    for (int mt = 0; mt < M_tiles_6; mt++) {
        for (int nc = 0; nc < N_chunks; nc++) {
            kernel_fused_d512_6row(Q + mt * 6 * D, K_int + nc * K_chunk_size,
                                    V_t + nc * V_chunk_size, O + mt * 6 * D);
        }
    }
    t3 = rdtsc();
    double sec_6 = (double)(t3 - t2) / freq;
    printf("6-row: %lu ticks = %.4f sec, %.1f GFLOPS\n",
           t3 - t2, sec_6, flops / sec_6 / 1e9);

    free(Q); free(K_int); free(V_t); free(O);
    printf("Done.\n");
    return 0;
}
