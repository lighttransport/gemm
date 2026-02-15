// bench_softmax.c - Compare scalar vs SVE softmax performance

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

// Original scalar softmax (from fused_attention_opt.c)
void softmax_scalar(
    const int32_t* scores,  // [6][64]
    float scale,
    float* row_max,         // [6] running max
    float* row_sum,         // [6] running sum
    float* O_fp32,          // [6][D] O accumulator
    int D,
    float* softmax_buf,     // [6][64] temp
    int8_t* P_tile,         // [6][64] output
    float* max_exp)         // [6] output for dequant
{
    for (int r = 0; r < 6; r++) {
        // Find chunk max
        float chunk_max = -1e30f;
        for (int i = 0; i < 64; i++) {
            float val = (float)scores[r * 64 + i] * scale;
            if (val > chunk_max) chunk_max = val;
        }

        // Update running max
        float old_max = row_max[r];
        float new_max = (chunk_max > old_max) ? chunk_max : old_max;

        // Rescale O
        float rescale = expf(old_max - new_max);
        for (int d = 0; d < D; d++) {
            O_fp32[r * D + d] *= rescale;
        }

        // Compute exp and find max_exp
        float chunk_sum = 0.0f;
        float chunk_max_exp = 0.0f;
        for (int i = 0; i < 64; i++) {
            float val = (float)scores[r * 64 + i] * scale;
            float exp_val = expf(val - new_max);
            softmax_buf[r * 64 + i] = exp_val;
            chunk_sum += exp_val;
            if (exp_val > chunk_max_exp) chunk_max_exp = exp_val;
        }

        row_sum[r] = row_sum[r] * rescale + chunk_sum;
        row_max[r] = new_max;
        max_exp[r] = chunk_max_exp;

        // Quantize
        if (chunk_max_exp > 0) {
            float inv_max = 127.0f / chunk_max_exp;
            for (int i = 0; i < 64; i++) {
                int32_t p = (int32_t)(softmax_buf[r * 64 + i] * inv_max + 0.5f);
                if (p > 127) p = 127;
                P_tile[r * 64 + i] = (int8_t)p;
            }
        } else {
            for (int i = 0; i < 64; i++) {
                P_tile[r * 64 + i] = 0;
            }
        }
    }
}

int main() {
    printf("================================================================\n");
    printf("Softmax Performance: Scalar vs SVE\n");
    printf("================================================================\n\n");

    uint64_t timer_freq = get_timer_freq();
    int D = 64;
    
    // Allocate
    int32_t* scores = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    float* O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));
    float* softmax_buf = aligned_alloc(64, 6 * 64 * sizeof(float));
    int8_t* P_tile1 = aligned_alloc(64, 6 * 64);
    int8_t* P_tile2 = aligned_alloc(64, 6 * 64);
    
    float scale = 1.0f / sqrtf(D);
    
    // Initialize scores with realistic values
    for (int i = 0; i < 6 * 64; i++) {
        scores[i] = (i % 200) - 100;  // Range [-100, 100]
    }
    
    // Scalar softmax state
    float row_max_s[6], row_sum_s[6], max_exp_s[6];
    
    // SVE softmax state
    softmax_state_t state_v[6];
    float max_exp_v[6];
    
    // Warmup
    for (int r = 0; r < 6; r++) {
        row_max_s[r] = -1e30f; row_sum_s[r] = 0;
    }
    memset(O_fp32, 0, 6 * D * sizeof(float));
    softmax_scalar(scores, scale, row_max_s, row_sum_s, O_fp32, D, 
                   softmax_buf, P_tile1, max_exp_s);
    
    softmax_state_init(state_v, 6);
    memset(O_fp32, 0, 6 * D * sizeof(float));
    softmax_tile_sve(scores, scale, state_v, O_fp32, D, P_tile2, max_exp_v);
    
    // Verify accuracy
    int mismatch = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < 6 * 64; i++) {
        int diff = abs((int)P_tile1[i] - (int)P_tile2[i]);
        if (diff > 1) mismatch++;
        if (diff > max_diff) max_diff = diff;
    }
    printf("Accuracy: %d mismatches (>1 diff), max diff = %.0f\n", mismatch, max_diff);
    
    // Benchmark scalar
    int iters = 10000;
    for (int r = 0; r < 6; r++) {
        row_max_s[r] = -1e30f; row_sum_s[r] = 0;
    }
    memset(O_fp32, 0, 6 * D * sizeof(float));
    
    uint64_t t0 = rdtsc();
    for (int i = 0; i < iters; i++) {
        for (int r = 0; r < 6; r++) {
            row_max_s[r] = -1e30f; row_sum_s[r] = 0;
        }
        softmax_scalar(scores, scale, row_max_s, row_sum_s, O_fp32, D,
                       softmax_buf, P_tile1, max_exp_s);
    }
    uint64_t t1 = rdtsc();
    double scalar_cycles = (double)(t1 - t0) / iters / timer_freq * 2.0e9;
    
    // Benchmark SVE
    softmax_state_init(state_v, 6);
    memset(O_fp32, 0, 6 * D * sizeof(float));
    
    uint64_t t2 = rdtsc();
    for (int i = 0; i < iters; i++) {
        softmax_state_init(state_v, 6);
        softmax_tile_sve(scores, scale, state_v, O_fp32, D, P_tile2, max_exp_v);
    }
    uint64_t t3 = rdtsc();
    double sve_cycles = (double)(t3 - t2) / iters / timer_freq * 2.0e9;
    
    printf("\nPerformance (6x64 tile):\n");
    printf("  Scalar: %.0f cycles\n", scalar_cycles);
    printf("  SVE:    %.0f cycles\n", sve_cycles);
    printf("  Speedup: %.2fx\n", scalar_cycles / sve_cycles);
    
    // Test with different D values
    printf("\nWith O rescaling (D dimension):\n");
    for (int test_D = 64; test_D <= 256; test_D *= 2) {
        float* O_test = aligned_alloc(64, 6 * test_D * sizeof(float));
        
        // Scalar with O rescaling
        for (int r = 0; r < 6; r++) {
            row_max_s[r] = -1e30f; row_sum_s[r] = 0;
        }
        memset(O_test, 1.0f, 6 * test_D * sizeof(float));  // Non-zero O
        
        uint64_t ts0 = rdtsc();
        for (int i = 0; i < iters; i++) {
            for (int r = 0; r < 6; r++) {
                row_max_s[r] = 0; row_sum_s[r] = 1;  // Simulate mid-attention
            }
            softmax_scalar(scores, scale, row_max_s, row_sum_s, O_test, test_D,
                           softmax_buf, P_tile1, max_exp_s);
        }
        uint64_t ts1 = rdtsc();
        double s_cyc = (double)(ts1 - ts0) / iters / timer_freq * 2.0e9;
        
        // SVE
        memset(O_test, 1.0f, 6 * test_D * sizeof(float));
        
        uint64_t tv0 = rdtsc();
        for (int i = 0; i < iters; i++) {
            for (int r = 0; r < 6; r++) {
                state_v[r].max = 0; state_v[r].sum = 1;
            }
            softmax_tile_sve(scores, scale, state_v, O_test, test_D, P_tile2, max_exp_v);
        }
        uint64_t tv1 = rdtsc();
        double v_cyc = (double)(tv1 - tv0) / iters / timer_freq * 2.0e9;
        
        printf("  D=%d: Scalar=%.0f, SVE=%.0f, Speedup=%.2fx\n",
               test_D, s_cyc, v_cyc, s_cyc / v_cyc);
        
        free(O_test);
    }
    
    free(scores); free(O_fp32); free(softmax_buf);
    free(P_tile1); free(P_tile2);
    
    return 0;
}
