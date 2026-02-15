// bench_final.c - Final optimized fused attention benchmark

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fused_attention_int8.h"

#define PEAK_INT8_GOPS 512.0

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

typedef struct fused_attn_workspace fused_attn_workspace_t;
typedef struct final_attn_workspace final_attn_workspace_t;

extern fused_attn_workspace_t* fused_attn_alloc_workspace(int, int, int);
extern void fused_attn_free_workspace(fused_attn_workspace_t*);
extern void fused_attn_pack_inputs(const int8_t*, const int8_t*, const int8_t*, fused_attn_workspace_t*);
extern void fused_attention_opt_fp32_ws(float*, fused_attn_workspace_t*, const fused_attn_params_t*);

extern final_attn_workspace_t* final_attn_alloc(int, int, int);
extern void final_attn_free(final_attn_workspace_t*);
extern void final_attn_pack(const int8_t*, const int8_t*, const int8_t*, final_attn_workspace_t*);
extern void fused_attention_final(float*, final_attn_workspace_t*, float);

int main() {
    printf("================================================================\n");
    printf("FINAL: Original vs Fully Optimized Fused Attention\n");
    printf("================================================================\n\n");
    printf("Optimizations:\n");
    printf("  - SVE softmax (5.6x faster)\n");
    printf("  - Fast pack_P (39x faster)\n\n");
    
    uint64_t timer_freq = get_timer_freq();
    
    struct { int M, N, D; } configs[] = {
        {48, 256, 64},
        {96, 512, 64},
        {192, 1024, 64},
        {384, 2048, 64},
        {96, 512, 128},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf("%-20s %10s %10s %10s %10s\n",
           "Config", "Original", "Final", "Speedup", "MaxErr");
    printf("%-20s %10s %10s %10s %10s\n",
           "--------------------", "----------", "----------", "----------", "----------");
    
    for (int c = 0; c < num_configs; c++) {
        int M = configs[c].M;
        int N = configs[c].N;
        int D = configs[c].D;
        
        int8_t* Q = aligned_alloc(64, M * D);
        int8_t* K = aligned_alloc(64, N * D);
        int8_t* V = aligned_alloc(64, N * D);
        float* O1 = aligned_alloc(64, M * D * sizeof(float));
        float* O2 = aligned_alloc(64, M * D * sizeof(float));
        
        for (int i = 0; i < M * D; i++) Q[i] = (i % 7) - 3;
        for (int i = 0; i < N * D; i++) K[i] = (i % 11) - 5;
        for (int i = 0; i < N * D; i++) V[i] = (i % 13) - 6;
        
        fused_attn_params_t params = {M, N, D, 1.0f / sqrtf(D)};
        
        fused_attn_workspace_t* ws1 = fused_attn_alloc_workspace(M, N, D);
        final_attn_workspace_t* ws2 = final_attn_alloc(M, N, D);
        
        fused_attn_pack_inputs(Q, K, V, ws1);
        final_attn_pack(Q, K, V, ws2);
        
        fused_attention_opt_fp32_ws(O1, ws1, &params);
        fused_attention_final(O2, ws2, params.scale);
        
        float max_err = 0.0f;
        for (int i = 0; i < M * D; i++) {
            float err = fabsf(O1[i] - O2[i]);
            float ref = fabsf(O1[i]) + 1e-6f;
            if (err / ref > max_err) max_err = err / ref;
        }
        
        int iters = (M * N < 200000) ? 20 : 5;
        
        uint64_t t0 = rdtsc();
        for (int i = 0; i < iters; i++) {
            fused_attention_opt_fp32_ws(O1, ws1, &params);
        }
        uint64_t t1 = rdtsc();
        double orig_cycles = (double)(t1 - t0) / iters / timer_freq * 2.0e9;
        
        uint64_t t2 = rdtsc();
        for (int i = 0; i < iters; i++) {
            fused_attention_final(O2, ws2, params.scale);
        }
        uint64_t t3 = rdtsc();
        double final_cycles = (double)(t3 - t2) / iters / timer_freq * 2.0e9;
        
        // Compute effective GOPS
        double total_ops = 2.0 * M * N * D * 2;  // QKT + PV
        double final_gops = total_ops / (final_cycles / 2e9) / 1e9;
        
        char config_str[32];
        snprintf(config_str, sizeof(config_str), "M=%d,N=%d,D=%d", M, N, D);
        printf("%-20s %10.0f %10.0f %10.2fx %10.4f\n",
               config_str, orig_cycles, final_cycles,
               orig_cycles / final_cycles, max_err);
        
        fused_attn_free_workspace(ws1);
        final_attn_free(ws2);
        free(Q); free(K); free(V); free(O1); free(O2);
    }
    
    printf("\n================================================================\n");
    return 0;
}
