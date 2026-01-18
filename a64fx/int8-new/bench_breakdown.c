// bench_breakdown.c - Time breakdown for SVE fused attention

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "softmax_sve.h"

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

extern void kernel_qkt_6x4_2x(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);
extern void kernel_pv_int8_opt(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);

static void pack_P_opt(const int8_t* P, int8_t* Pp, int K) {
    int K_groups = K / 4;
    for (int kg = 0; kg < K_groups; kg++) {
        for (int m = 0; m < 6; m++) {
            for (int k = 0; k < 4; k++) {
                Pp[kg * 24 + m * 4 + k] = P[m * K + kg * 4 + k];
            }
        }
    }
}

int main() {
    printf("================================================================\n");
    printf("Time Breakdown for SVE Fused Attention\n");
    printf("================================================================\n\n");
    
    uint64_t timer_freq = get_timer_freq();
    
    int M = 384, N = 2048, D = 64;
    int M_tiles = M / 6;
    int N_chunks = N / 64;
    
    // Allocate (simplified - no actual packing)
    int32_t* S = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    int8_t* P = aligned_alloc(64, 6 * 64);
    int8_t* Pp = aligned_alloc(64, 16 * 24);
    int32_t* O = aligned_alloc(64, 6 * D * sizeof(int32_t));
    float* O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));
    int8_t* Q = aligned_alloc(64, 6 * D);
    int8_t* K = aligned_alloc(64, 64 * D);
    int8_t* V = aligned_alloc(64, 64 * D);
    
    float scale = 1.0f / sqrtf(D);
    softmax_state_t state[6];
    float max_exp[6];
    
    int iters = 10;
    uint64_t t_qkt = 0, t_softmax = 0, t_pack = 0, t_pv = 0, t_accum = 0, t_memset = 0;
    
    for (int iter = 0; iter < iters; iter++) {
        for (int mt = 0; mt < M_tiles; mt++) {
            softmax_state_init(state, 6);
            memset(O_fp32, 0, 6 * D * sizeof(float));
            
            for (int nc = 0; nc < N_chunks; nc++) {
                // Memset S
                uint64_t t0 = rdtsc();
                memset(S, 0, 6 * 64 * sizeof(int32_t));
                uint64_t t1 = rdtsc();
                t_memset += (t1 - t0);
                
                // Q@K^T
                t0 = rdtsc();
                kernel_qkt_6x4_2x(Q, K, S, D, 64 * sizeof(int32_t));
                __asm__ volatile("" ::: "memory");
                t1 = rdtsc();
                t_qkt += (t1 - t0);
                
                // Softmax
                t0 = rdtsc();
                softmax_tile_sve(S, scale, state, O_fp32, D, P, max_exp);
                __asm__ volatile("" ::: "memory");
                t1 = rdtsc();
                t_softmax += (t1 - t0);
                
                // Pack P
                t0 = rdtsc();
                pack_P_opt(P, Pp, 64);
                t1 = rdtsc();
                t_pack += (t1 - t0);
                
                // Memset O
                t0 = rdtsc();
                memset(O, 0, 6 * D * sizeof(int32_t));
                t1 = rdtsc();
                t_memset += (t1 - t0);
                
                // P@V
                t0 = rdtsc();
                kernel_pv_int8_opt(Pp, V, O, D, D * sizeof(int32_t));
                __asm__ volatile("" ::: "memory");
                t1 = rdtsc();
                t_pv += (t1 - t0);
                
                // Accumulate
                t0 = rdtsc();
                for (int r = 0; r < 6; r++) {
                    float dequant = max_exp[r] / 127.0f;
                    for (int d = 0; d < D; d++) {
                        O_fp32[r * D + d] += (float)O[r * D + d] * dequant;
                    }
                }
                t1 = rdtsc();
                t_accum += (t1 - t0);
            }
        }
    }
    
    double total = (double)(t_qkt + t_softmax + t_pack + t_pv + t_accum + t_memset);
    double scale_to_cyc = 2.0e9 / timer_freq / iters;
    
    printf("M=%d, N=%d, D=%d\n\n", M, N, D);
    printf("%-15s %12s %8s\n", "Operation", "Cycles", "Percent");
    printf("%-15s %12s %8s\n", "---------------", "------------", "--------");
    printf("%-15s %12.0f %7.1f%%\n", "Q@K^T kernel", t_qkt * scale_to_cyc, t_qkt / total * 100);
    printf("%-15s %12.0f %7.1f%%\n", "SVE Softmax", t_softmax * scale_to_cyc, t_softmax / total * 100);
    printf("%-15s %12.0f %7.1f%%\n", "Pack P", t_pack * scale_to_cyc, t_pack / total * 100);
    printf("%-15s %12.0f %7.1f%%\n", "P@V kernel", t_pv * scale_to_cyc, t_pv / total * 100);
    printf("%-15s %12.0f %7.1f%%\n", "FP32 Accum", t_accum * scale_to_cyc, t_accum / total * 100);
    printf("%-15s %12.0f %7.1f%%\n", "Memset", t_memset * scale_to_cyc, t_memset / total * 100);
    printf("%-15s %12s %8s\n", "---------------", "------------", "--------");
    printf("%-15s %12.0f %7.1f%%\n", "TOTAL", total * scale_to_cyc, 100.0);
    
    printf("\n");
    printf("SDOT kernels: %.1f%% of total\n", (t_qkt + t_pv) / total * 100);
    
    free(S); free(P); free(Pp); free(O); free(O_fp32);
    free(Q); free(K); free(V);
    
    return 0;
}
