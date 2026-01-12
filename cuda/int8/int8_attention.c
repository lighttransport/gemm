/*
 * INT8 Flash Attention Implementation
 *
 * Implements Flash Attention algorithm with INT8 quantized Q, K, V
 * Uses integer approximations for softmax (exp function)
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
 * https://arxiv.org/abs/2205.14135
 *
 * Key differences from FP implementation:
 * - Q, K, V are INT8, accumulation in INT32
 * - Softmax uses integer exp approximation
 * - Online softmax with running max and sum
 * - Output can be INT8 (with stochastic rounding) or INT32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "int8_types.h"
#include "int8_approx.h"

/* Default dimensions */
static int g_batch = 1;
static int g_heads = 4;
static int g_seq_len = 64;
static int g_head_dim = 64;
static int g_block_size = 32;  /* Flash attention block size */

/* Scaling factor for attention scores (1/sqrt(d_k)) in Q8.8 */
static int16_t g_scale_q8 = 32;  /* 1/8 ≈ 0.125 for d=64, sqrt(64)=8 */

/*
 * Reference FP32 Attention for verification
 */
static void attention_ref_fp32(
    const float* Q,  /* [seq_len, head_dim] */
    const float* K,  /* [seq_len, head_dim] */
    const float* V,  /* [seq_len, head_dim] */
    float* O,        /* [seq_len, head_dim] */
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Allocate attention scores */
    float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    float* probs = (float*)malloc(seq_len * seq_len * sizeof(float));

    /* Compute Q @ K^T */
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            scores[i * seq_len + j] = sum * scale;
        }
    }

    /* Softmax per row */
    for (int i = 0; i < seq_len; i++) {
        float max_val = scores[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            if (scores[i * seq_len + j] > max_val) {
                max_val = scores[i * seq_len + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            probs[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
            sum += probs[i * seq_len + j];
        }

        for (int j = 0; j < seq_len; j++) {
            probs[i * seq_len + j] /= sum;
        }
    }

    /* Compute probs @ V */
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += probs[i * seq_len + j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = sum;
        }
    }

    free(scores);
    free(probs);
}

/*
 * INT8 Attention (Standard, non-Flash)
 * For comparison and verification
 */
static void attention_int8_standard(
    const int8_t* Q,   /* [seq_len, head_dim] */
    const int8_t* K,   /* [seq_len, head_dim] */
    const int8_t* V,   /* [seq_len, head_dim] */
    int32_t* O,        /* [seq_len, head_dim] output in INT32 */
    int seq_len,
    int head_dim,
    int16_t scale_q8   /* 1/sqrt(d_k) in Q8.8 */
) {
    /* Allocate scores in Q16.16 format */
    int32_t* scores = (int32_t*)malloc(seq_len * seq_len * sizeof(int32_t));

    /* Compute Q @ K^T with INT8 inputs, INT32 accumulation */
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int32_t sum = 0;
            for (int k = 0; k < head_dim; k++) {
                sum += (int32_t)Q[i * head_dim + k] * (int32_t)K[j * head_dim + k];
            }
            /* Apply scale: score * scale_q8 >> 8, then convert to Q16.16 */
            scores[i * seq_len + j] = ((int64_t)sum * scale_q8) << 8;
        }
    }

    /* Integer softmax per row */
    for (int i = 0; i < seq_len; i++) {
        int_softmax_q16(&scores[i * seq_len], seq_len);
    }

    /* Compute probs @ V with INT32 accumulation */
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            int64_t sum = 0;
            for (int j = 0; j < seq_len; j++) {
                /* probs are Q16.16, V is INT8 */
                sum += (int64_t)scores[i * seq_len + j] * (int64_t)V[j * head_dim + d];
            }
            /* Result is Q16.16 * INT8 = Q16.24, shift to INT32 */
            O[i * head_dim + d] = (int32_t)(sum >> 16);
        }
    }

    free(scores);
}

/*
 * INT8 Flash Attention
 *
 * Tiled attention with online softmax
 * Reduces memory from O(N²) to O(N) by processing in blocks
 */
static void flash_attention_int8(
    const int8_t* Q,   /* [seq_len, head_dim] */
    const int8_t* K,   /* [seq_len, head_dim] */
    const int8_t* V,   /* [seq_len, head_dim] */
    int32_t* O,        /* [seq_len, head_dim] output in INT32 */
    int seq_len,
    int head_dim,
    int block_size,
    int16_t scale_q8   /* 1/sqrt(d_k) in Q8.8 */
) {
    int num_blocks = (seq_len + block_size - 1) / block_size;

    /* Allocate per-row accumulators */
    int64_t* row_sum = (int64_t*)calloc(seq_len, sizeof(int64_t));     /* Running sum of exp */
    int32_t* row_max = (int32_t*)malloc(seq_len * sizeof(int32_t));    /* Running max */
    int64_t* O_acc = (int64_t*)calloc(seq_len * head_dim, sizeof(int64_t));  /* Accumulator */

    /* Initialize row_max to minimum value */
    for (int i = 0; i < seq_len; i++) {
        row_max[i] = INT32_MIN;
    }

    /* Block buffer for scores */
    int32_t* block_scores = (int32_t*)malloc(block_size * block_size * sizeof(int32_t));
    uint32_t* block_exp = (uint32_t*)malloc(block_size * block_size * sizeof(uint32_t));

    /* Process K, V in blocks */
    for (int j_block = 0; j_block < num_blocks; j_block++) {
        int j_start = j_block * block_size;
        int j_end = (j_start + block_size < seq_len) ? j_start + block_size : seq_len;
        int j_len = j_end - j_start;

        /* For each query row */
        for (int i = 0; i < seq_len; i++) {
            /* Compute Q[i] @ K[j_start:j_end]^T */
            int32_t local_max = row_max[i];

            for (int j = 0; j < j_len; j++) {
                int32_t score = 0;
                for (int k = 0; k < head_dim; k++) {
                    score += (int32_t)Q[i * head_dim + k] *
                             (int32_t)K[(j_start + j) * head_dim + k];
                }
                /* Apply scale and convert to Q16.16 */
                score = ((int64_t)score * scale_q8) << 8;
                block_scores[j] = score;

                if (score > local_max) local_max = score;
            }

            /* Compute exp(score - max) for this block */
            uint64_t block_sum = 0;
            for (int j = 0; j < j_len; j++) {
                int32_t shifted = block_scores[j] - local_max;
                block_exp[j] = int_exp_fast_q16(shifted);
                block_sum += block_exp[j];
            }

            /* Online softmax update */
            if (local_max > row_max[i]) {
                /* New max found, rescale previous accumulator */
                int32_t max_diff = row_max[i] - local_max;
                uint32_t rescale = int_exp_fast_q16(max_diff);

                /* Rescale O_acc and row_sum */
                for (int d = 0; d < head_dim; d++) {
                    O_acc[i * head_dim + d] = (O_acc[i * head_dim + d] * rescale) >> 16;
                }
                row_sum[i] = (row_sum[i] * rescale) >> 16;
                row_max[i] = local_max;
            }

            /* Add contribution from this block: O += exp(score) @ V[block] */
            for (int j = 0; j < j_len; j++) {
                uint32_t exp_val = block_exp[j];
                for (int d = 0; d < head_dim; d++) {
                    O_acc[i * head_dim + d] += (int64_t)exp_val *
                                               (int64_t)V[(j_start + j) * head_dim + d];
                }
            }
            row_sum[i] += block_sum;
        }
    }

    /* Final normalization: O = O_acc / row_sum */
    for (int i = 0; i < seq_len; i++) {
        int64_t sum = row_sum[i];
        if (sum == 0) sum = 1;  /* Prevent division by zero */

        for (int d = 0; d < head_dim; d++) {
            /* O_acc is in Q16.16 * INT8 = high precision, normalize */
            O[i * head_dim + d] = (int32_t)((O_acc[i * head_dim + d] << 8) / sum);
        }
    }

    free(row_sum);
    free(row_max);
    free(O_acc);
    free(block_scores);
    free(block_exp);
}

/*
 * Multi-Head INT8 Flash Attention
 */
static void mha_flash_attention_int8(
    const int8_t* Q,   /* [batch, heads, seq_len, head_dim] */
    const int8_t* K,   /* [batch, heads, seq_len, head_dim] */
    const int8_t* V,   /* [batch, heads, seq_len, head_dim] */
    int32_t* O,        /* [batch, heads, seq_len, head_dim] */
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    int block_size
) {
    int16_t scale_q8 = (int16_t)(256.0f / sqrtf((float)head_dim));

    size_t head_size = seq_len * head_dim;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            size_t offset = (b * heads + h) * head_size;

            flash_attention_int8(
                Q + offset,
                K + offset,
                V + offset,
                O + offset,
                seq_len,
                head_dim,
                block_size,
                scale_q8
            );
        }
    }
}

/*
 * Verification and Testing
 */
static float compute_mse(const int32_t* a, const float* b, int n, float scale) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        float a_fp = (float)a[i] * scale;
        double diff = a_fp - b[i];
        mse += diff * diff;
    }
    return (float)(mse / n);
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --batch N      Batch size (default: %d)\n", g_batch);
    printf("  --heads N      Number of attention heads (default: %d)\n", g_heads);
    printf("  --seq N        Sequence length (default: %d)\n", g_seq_len);
    printf("  --dim N        Head dimension (default: %d)\n", g_head_dim);
    printf("  --block N      Flash attention block size (default: %d)\n", g_block_size);
    printf("  -h, --help     Show this help\n");
}

int main(int argc, char** argv) {
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            g_batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            g_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            g_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            g_head_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            g_block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("INT8 Flash Attention\n");
    printf("====================\n");
    printf("Batch: %d, Heads: %d, Seq: %d, Dim: %d, Block: %d\n",
           g_batch, g_heads, g_seq_len, g_head_dim, g_block_size);

    /* For single-head testing */
    int seq_len = g_seq_len;
    int head_dim = g_head_dim;
    size_t size = seq_len * head_dim;

    /* Allocate memory */
    int8_t* Q_int8 = (int8_t*)malloc(size);
    int8_t* K_int8 = (int8_t*)malloc(size);
    int8_t* V_int8 = (int8_t*)malloc(size);
    int32_t* O_int8_std = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t* O_int8_flash = (int32_t*)malloc(size * sizeof(int32_t));

    float* Q_fp32 = (float*)malloc(size * sizeof(float));
    float* K_fp32 = (float*)malloc(size * sizeof(float));
    float* V_fp32 = (float*)malloc(size * sizeof(float));
    float* O_fp32 = (float*)malloc(size * sizeof(float));

    /* Initialize with random data */
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);

    for (size_t i = 0; i < size; i++) {
        int8_t val = xoro_random_s8_range(&rng, 50);
        Q_int8[i] = val;
        K_int8[i] = xoro_random_s8_range(&rng, 50);
        V_int8[i] = xoro_random_s8_range(&rng, 50);

        Q_fp32[i] = (float)Q_int8[i] / 50.0f;
        K_fp32[i] = (float)K_int8[i] / 50.0f;
        V_fp32[i] = (float)V_int8[i] / 50.0f;
    }

    int16_t scale_q8 = (int16_t)(256.0f / sqrtf((float)head_dim));
    printf("Scale (1/sqrt(%d)): %.4f (Q8.8: %d)\n",
           head_dim, 1.0f / sqrtf((float)head_dim), scale_q8);

    /* Run FP32 reference */
    printf("\n--- FP32 Reference ---\n");
    clock_t start = clock();
    attention_ref_fp32(Q_fp32, K_fp32, V_fp32, O_fp32, seq_len, head_dim);
    clock_t end = clock();
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    /* Run INT8 standard attention */
    printf("\n--- INT8 Standard Attention ---\n");
    start = clock();
    attention_int8_standard(Q_int8, K_int8, V_int8, O_int8_std, seq_len, head_dim, scale_q8);
    end = clock();
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    /* Run INT8 Flash Attention */
    printf("\n--- INT8 Flash Attention (block=%d) ---\n", g_block_size);
    start = clock();
    flash_attention_int8(Q_int8, K_int8, V_int8, O_int8_flash, seq_len, head_dim,
                         g_block_size, scale_q8);
    end = clock();
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    /* Compare results */
    printf("\n--- Comparison ---\n");

    /* INT8 standard vs FP32 */
    float scale_out = 50.0f / 256.0f;  /* Approximate output scale */
    float mse_std = compute_mse(O_int8_std, O_fp32, (int)size, scale_out);
    printf("INT8 Standard vs FP32: MSE = %.6f\n", mse_std);

    /* INT8 Flash vs FP32 */
    float mse_flash = compute_mse(O_int8_flash, O_fp32, (int)size, scale_out);
    printf("INT8 Flash vs FP32: MSE = %.6f\n", mse_flash);

    /* INT8 Flash vs INT8 Standard */
    int32_t max_diff = 0;
    for (size_t i = 0; i < size; i++) {
        int32_t diff = abs(O_int8_flash[i] - O_int8_std[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("INT8 Flash vs INT8 Standard: Max diff = %d\n", max_diff);

    /* Print sample output */
    printf("\nSample output (first 8 values):\n");
    printf("  FP32 ref:    ");
    for (int i = 0; i < 8; i++) printf("%.3f ", O_fp32[i]);
    printf("\n");
    printf("  INT8 std:    ");
    for (int i = 0; i < 8; i++) printf("%d ", O_int8_std[i]);
    printf("\n");
    printf("  INT8 flash:  ");
    for (int i = 0; i < 8; i++) printf("%d ", O_int8_flash[i]);
    printf("\n");

    /* Test integer approximation functions */
    printf("\n--- Integer Approximation Tests ---\n");

    printf("exp() approximation:\n");
    for (float x = -4.0f; x <= 0.5f; x += 0.5f) {
        int16_t x_q8 = float_to_q8(x);
        uint16_t exp_q8 = int_exp_q8(x_q8);
        float exp_approx = q8_to_float((int16_t)exp_q8);
        float exp_exact = expf(x);
        printf("  exp(%.1f): exact=%.4f, approx=%.4f, err=%.4f\n",
               x, exp_exact, exp_approx, fabsf(exp_exact - exp_approx));
    }

    printf("\ntanh() approximation:\n");
    for (float x = -2.0f; x <= 2.0f; x += 0.5f) {
        int16_t x_q8 = float_to_q8(x);
        int16_t tanh_q8 = int_tanh_q8(x_q8);
        float tanh_approx = q8_to_float(tanh_q8);
        float tanh_exact = tanhf(x);
        printf("  tanh(%.1f): exact=%.4f, approx=%.4f, err=%.4f\n",
               x, tanh_exact, tanh_approx, fabsf(tanh_exact - tanh_approx));
    }

    printf("\ngelu() approximation:\n");
    for (float x = -2.0f; x <= 2.0f; x += 0.5f) {
        int16_t x_q8 = float_to_q8(x);
        int16_t gelu_q8 = int_gelu_q8(x_q8);
        float gelu_approx = q8_to_float(gelu_q8);
        /* Exact GELU */
        float gelu_exact = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        printf("  gelu(%.1f): exact=%.4f, approx=%.4f, err=%.4f\n",
               x, gelu_exact, gelu_approx, fabsf(gelu_exact - gelu_approx));
    }

    /* Cleanup */
    free(Q_int8); free(K_int8); free(V_int8);
    free(O_int8_std); free(O_int8_flash);
    free(Q_fp32); free(K_fp32); free(V_fp32); free(O_fp32);

    printf("\nDone.\n");
    return 0;
}
