// bench_fused_attn.c
// Benchmark fused INT8 attention: O = softmax(Q @ K^T) @ V

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fused_attention_int8.h"

#define PEAK_INT8_GOPS 512.0

static inline uint64_t rdtsc(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (f));
    return f;
}

// Reference attention (FP32)
void reference_attention(
    const int8_t* Q, const int8_t* K, const int8_t* V,
    float* O, int M, int N, int D, float scale
) {
    float* S = malloc(M * N * sizeof(float));
    float* P = malloc(M * N * sizeof(float));

    // Q @ K^T
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += (float)Q[m * D + d] * (float)K[n * D + d];
            }
            S[m * N + n] = sum * scale;
        }
    }

    // Softmax per row
    for (int m = 0; m < M; m++) {
        float max_val = S[m * N];
        for (int n = 1; n < N; n++) {
            if (S[m * N + n] > max_val) max_val = S[m * N + n];
        }
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            P[m * N + n] = expf(S[m * N + n] - max_val);
            sum += P[m * N + n];
        }
        for (int n = 0; n < N; n++) {
            P[m * N + n] /= sum;
        }
    }

    // P @ V
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += P[m * N + n] * (float)V[n * D + d];
            }
            O[m * D + d] = sum;
        }
    }

    free(S);
    free(P);
}

int main() {
    uint64_t timer_freq = get_timer_freq();

    printf("================================================================\n");
    printf("Fused INT8 Attention Benchmark\n");
    printf("================================================================\n\n");

    printf("Computing: O = softmax(Q @ K^T / sqrt(D)) @ V\n");
    printf("Using: INT8 SDOT for Q@K^T and P@V, FP32 for softmax\n");
    printf("Tile size: 6 query rows × 64 key chunk\n\n");

    // Test configurations
    int configs[][3] = {
        // M, N, D
        {6, 64, 64},      // Minimal: 1 M-tile, 1 N-chunk
        {6, 128, 64},     // 1 M-tile, 2 N-chunks
        {12, 64, 64},     // 2 M-tiles, 1 N-chunk
        {48, 256, 64},    // Small attention
        {96, 512, 64},    // Medium attention
        {192, 1024, 64},  // Large attention
        {384, 2048, 128}, // Very large
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < num_configs; c++) {
        int M = configs[c][0];
        int N = configs[c][1];
        int D = configs[c][2];

        // Must be divisible by tile sizes
        if (M % 6 != 0 || N % 64 != 0 || D % 4 != 0) {
            printf("Skipping M=%d, N=%d, D=%d (not aligned)\n", M, N, D);
            continue;
        }

        printf("=== M=%d, N=%d, D=%d ===\n", M, N, D);

        // Allocate matrices
        int8_t* Q = aligned_alloc(64, M * D);
        int8_t* K = aligned_alloc(64, N * D);
        int8_t* V = aligned_alloc(64, N * D);
        float* O_ref = aligned_alloc(64, M * D * sizeof(float));
        float* O_test = aligned_alloc(64, M * D * sizeof(float));

        // Packed versions
        int8_t* Qp = aligned_alloc(64, (M/6) * (D/4) * 6 * 4);
        int8_t* Kp = aligned_alloc(64, (N/64) * (D/4) * 64 * 4);
        int8_t* Vp = aligned_alloc(64, (N/4) * 4 * D);

        // Initialize with small random values
        for (int i = 0; i < M * D; i++) Q[i] = (rand() % 16) - 8;
        for (int i = 0; i < N * D; i++) K[i] = (rand() % 16) - 8;
        for (int i = 0; i < N * D; i++) V[i] = (rand() % 16) - 8;

        // Pack matrices
        pack_Q_fused(Q, Qp, M, D);
        pack_K_fused(K, Kp, N, D);
        pack_V_fused(V, Vp, N, D);

        // Setup parameters
        fused_attn_params_t params = {
            .M = M,
            .N = N,
            .D = D,
            .scale = 1.0f / sqrtf((float)D),
            .qk_scale = 1.0f,      // Assume Q,K are already scaled
            .p_scale = 127.0f,     // Scale softmax to INT8 range
            .v_scale = 1.0f,
            .o_scale = 1.0f
        };

        // Reference computation
        reference_attention(Q, K, V, O_ref, M, N, D, params.scale);

        // Warmup
        for (int t = 0; t < 10; t++) {
            fused_attention_int8_fp32(Qp, Kp, Vp, O_test, &params);
        }

        // Benchmark
        int trials = (M * N < 10000) ? 10000 : 1000;
        uint64_t start = rdtsc();
        for (int t = 0; t < trials; t++) {
            fused_attention_int8_fp32(Qp, Kp, Vp, O_test, &params);
        }
        uint64_t end = rdtsc();

        double timer_ticks = (double)(end - start);
        double time_s = timer_ticks / timer_freq;
        double cycles = time_s * 2.0e9 / trials;

        // Total INT8 ops: Q@K^T + P@V
        // Q@K^T: 2 * M * N * D (M queries × N keys × D dimension)
        // P@V:   2 * M * N * D (M queries × N keys × D dimension)
        double qkt_ops = 2.0 * M * N * D;
        double pv_ops = 2.0 * M * N * D;
        double total_ops = qkt_ops + pv_ops;

        double gops = (total_ops * trials) / time_s / 1e9;
        double eff = gops / PEAK_INT8_GOPS * 100.0;

        // Check accuracy (sample a few outputs)
        float max_err = 0.0f;
        float avg_err = 0.0f;
        for (int i = 0; i < M * D; i++) {
            float err = fabsf(O_test[i] - O_ref[i]);
            float rel = err / (fabsf(O_ref[i]) + 1e-6f);
            if (rel > max_err) max_err = rel;
            avg_err += rel;
        }
        avg_err /= (M * D);

        printf("  Cycles: %.1f  GOPS: %.1f  Eff: %.1f%%\n", cycles, gops, eff);
        printf("  Q@K^T ops: %.1fM  P@V ops: %.1fM  Total: %.1fM\n",
               qkt_ops/1e6, pv_ops/1e6, total_ops/1e6);
        printf("  Max rel err: %.4f  Avg rel err: %.4f\n\n", max_err, avg_err);

        free(Q); free(K); free(V);
        free(O_ref); free(O_test);
        free(Qp); free(Kp); free(Vp);
    }

    printf("================================================================\n");
    printf("Notes:\n");
    printf("  - Efficiency is relative to peak INT8 SDOT (512 GOPS)\n");
    printf("  - Actual throughput limited by softmax overhead\n");
    printf("  - Q@K^T achieves ~93%%, P@V achieves ~75%% on their own\n");
    printf("  - Combined efficiency depends on softmax/packing overhead\n");
    printf("================================================================\n");

    return 0;
}
