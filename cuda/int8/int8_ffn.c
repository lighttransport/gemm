/*
 * INT8 Feed-Forward Network (FFN) Implementation
 *
 * Implements transformer FFN block with INT8 quantization:
 *   FFN(x) = Linear2(Activation(Linear1(x)))
 *
 * Supported activation functions (integer approximations):
 * - GELU: Gaussian Error Linear Unit
 * - SiLU: Sigmoid Linear Unit (Swish)
 * - ReLU: Rectified Linear Unit
 *
 * Linear layers: INT8 weights, INT8 input, INT32 accumulation
 * Activation: Applied on Q8.8 fixed-point representation
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
static int g_seq_len = 64;
static int g_hidden_dim = 256;
static int g_ffn_dim = 1024;   /* Usually 4x hidden_dim */
static int g_activation = 0;   /* 0=GELU, 1=SiLU, 2=ReLU */

/*
 * Activation function enum
 */
typedef enum {
    ACT_GELU = 0,
    ACT_SILU = 1,
    ACT_RELU = 2,
    ACT_GELU_FAST = 3,
    ACT_SILU_FAST = 4
} activation_t;

static const char* activation_names[] = {
    "GELU", "SiLU", "ReLU", "GELU_fast", "SiLU_fast"
};

/*
 * INT8 Linear Layer
 *
 * Computes: output = input @ weight^T + bias
 * - Input: [batch, seq_len, in_dim] INT8
 * - Weight: [out_dim, in_dim] INT8
 * - Bias: [out_dim] INT32
 * - Output: [batch, seq_len, out_dim] INT32
 */
static void linear_int8(
    const int8_t* input,    /* [batch * seq_len, in_dim] */
    const int8_t* weight,   /* [out_dim, in_dim] */
    const int32_t* bias,    /* [out_dim] or NULL */
    int32_t* output,        /* [batch * seq_len, out_dim] */
    int batch_seq,          /* batch * seq_len */
    int in_dim,
    int out_dim
) {
    for (int i = 0; i < batch_seq; i++) {
        for (int j = 0; j < out_dim; j++) {
            int32_t acc = 0;
            for (int k = 0; k < in_dim; k++) {
                acc += (int32_t)input[i * in_dim + k] *
                       (int32_t)weight[j * in_dim + k];
            }
            if (bias) acc += bias[j];
            output[i * out_dim + j] = acc;
        }
    }
}

/*
 * Apply activation function (INT32 -> INT8)
 *
 * Input is INT32 from linear layer, needs scaling to Q8.8 for activation
 * Output is INT8 for next layer
 */
static void apply_activation_int8(
    const int32_t* input,   /* [n] INT32 from linear */
    int8_t* output,         /* [n] INT8 for next layer */
    int n,
    activation_t act,
    float input_scale,      /* Scale to convert INT32 to Q8.8 */
    float output_scale,     /* Scale to convert activation result to INT8 */
    xoroshiro128plus_t* rng /* For stochastic rounding, NULL for nearest */
) {
    for (int i = 0; i < n; i++) {
        /* Convert INT32 to Q8.8 */
        int32_t x_q8 = (int32_t)(input[i] * input_scale);

        /* Clamp to int16 range */
        if (x_q8 > 32767) x_q8 = 32767;
        if (x_q8 < -32768) x_q8 = -32768;

        /* Apply activation */
        int16_t y_q8;
        switch (act) {
            case ACT_GELU:
                y_q8 = int_gelu_q8((int16_t)x_q8);
                break;
            case ACT_GELU_FAST:
                y_q8 = int_gelu_fast_q8((int16_t)x_q8);
                break;
            case ACT_SILU:
                y_q8 = int_silu_q8((int16_t)x_q8);
                break;
            case ACT_SILU_FAST:
                y_q8 = int_silu_fast_q8((int16_t)x_q8);
                break;
            case ACT_RELU:
            default:
                y_q8 = int_relu_q8((int16_t)x_q8);
                break;
        }

        /* Convert Q8.8 to INT8 output */
        float y_fp = q8_to_float(y_q8);
        int32_t y_int = (int32_t)(y_fp * output_scale);

        /* Stochastic rounding or clamp */
        if (rng) {
            float y_exact = y_fp * output_scale;
            y_int = stochastic_round_f32(y_exact, rng);
        }

        output[i] = clamp_s8(y_int);
    }
}

/*
 * INT8 FFN Block
 *
 * FFN(x) = Linear2(Activation(Linear1(x)))
 *
 * Standard transformer FFN with:
 * - Linear1: hidden_dim -> ffn_dim (4x expansion)
 * - Activation: GELU/SiLU/ReLU
 * - Linear2: ffn_dim -> hidden_dim (projection back)
 */
static void ffn_int8(
    const int8_t* input,      /* [batch_seq, hidden_dim] */
    const int8_t* weight1,    /* [ffn_dim, hidden_dim] */
    const int32_t* bias1,     /* [ffn_dim] */
    const int8_t* weight2,    /* [hidden_dim, ffn_dim] */
    const int32_t* bias2,     /* [hidden_dim] */
    int8_t* output,           /* [batch_seq, hidden_dim] */
    int batch_seq,
    int hidden_dim,
    int ffn_dim,
    activation_t act,
    float scale1,             /* Input scale for activation */
    float scale2,             /* Output scale from activation */
    xoroshiro128plus_t* rng
) {
    /* Allocate intermediate buffers */
    int32_t* hidden1 = (int32_t*)malloc(batch_seq * ffn_dim * sizeof(int32_t));
    int8_t* activated = (int8_t*)malloc(batch_seq * ffn_dim);
    int32_t* hidden2 = (int32_t*)malloc(batch_seq * hidden_dim * sizeof(int32_t));

    /* Linear1: hidden_dim -> ffn_dim */
    linear_int8(input, weight1, bias1, hidden1, batch_seq, hidden_dim, ffn_dim);

    /* Apply activation with quantization */
    apply_activation_int8(hidden1, activated, batch_seq * ffn_dim, act,
                          scale1, scale2, rng);

    /* Linear2: ffn_dim -> hidden_dim */
    linear_int8(activated, weight2, bias2, hidden2, batch_seq, ffn_dim, hidden_dim);

    /* Final quantization to INT8 output */
    for (int i = 0; i < batch_seq * hidden_dim; i++) {
        float val = (float)hidden2[i] * scale1;
        int32_t out;
        if (rng) {
            out = stochastic_round_f32(val * scale2, rng);
        } else {
            out = (int32_t)(val * scale2);
        }
        output[i] = clamp_s8(out);
    }

    free(hidden1);
    free(activated);
    free(hidden2);
}

/*
 * Gated FFN (GLU variant)
 *
 * GatedFFN(x) = Linear2(Activation(Linear1a(x)) * Linear1b(x))
 *
 * Used in LLaMA, PaLM, etc.
 */
static void gated_ffn_int8(
    const int8_t* input,      /* [batch_seq, hidden_dim] */
    const int8_t* weight1a,   /* [ffn_dim, hidden_dim] - gate */
    const int8_t* weight1b,   /* [ffn_dim, hidden_dim] - up */
    const int8_t* weight2,    /* [hidden_dim, ffn_dim] - down */
    int8_t* output,           /* [batch_seq, hidden_dim] */
    int batch_seq,
    int hidden_dim,
    int ffn_dim,
    activation_t act,
    float scale1,
    float scale2,
    xoroshiro128plus_t* rng
) {
    /* Allocate buffers */
    int32_t* gate = (int32_t*)malloc(batch_seq * ffn_dim * sizeof(int32_t));
    int32_t* up = (int32_t*)malloc(batch_seq * ffn_dim * sizeof(int32_t));
    int8_t* gated = (int8_t*)malloc(batch_seq * ffn_dim);
    int32_t* hidden2 = (int32_t*)malloc(batch_seq * hidden_dim * sizeof(int32_t));

    /* Gate projection */
    linear_int8(input, weight1a, NULL, gate, batch_seq, hidden_dim, ffn_dim);

    /* Up projection */
    linear_int8(input, weight1b, NULL, up, batch_seq, hidden_dim, ffn_dim);

    /* Apply activation to gate and multiply with up */
    for (int i = 0; i < batch_seq * ffn_dim; i++) {
        /* Convert gate to Q8.8 and apply activation */
        int32_t g_q8 = (int32_t)(gate[i] * scale1);
        if (g_q8 > 32767) g_q8 = 32767;
        if (g_q8 < -32768) g_q8 = -32768;

        int16_t g_act;
        switch (act) {
            case ACT_SILU:
            case ACT_SILU_FAST:
                g_act = int_silu_q8((int16_t)g_q8);
                break;
            case ACT_GELU:
            case ACT_GELU_FAST:
                g_act = int_gelu_q8((int16_t)g_q8);
                break;
            default:
                g_act = int_relu_q8((int16_t)g_q8);
                break;
        }

        /* Convert up to Q8.8 */
        int32_t u_q8 = (int32_t)(up[i] * scale1);
        if (u_q8 > 32767) u_q8 = 32767;
        if (u_q8 < -32768) u_q8 = -32768;

        /* Multiply gate * up in Q8.8 -> Q16.16, then back to INT8 */
        int32_t prod = ((int32_t)g_act * u_q8) >> Q8_FRAC_BITS;
        float prod_fp = q8_to_float((int16_t)(prod >> 8));  /* Adjust scaling */

        int32_t out;
        if (rng) {
            out = stochastic_round_f32(prod_fp * scale2, rng);
        } else {
            out = (int32_t)(prod_fp * scale2);
        }
        gated[i] = clamp_s8(out);
    }

    /* Down projection */
    linear_int8(gated, weight2, NULL, hidden2, batch_seq, ffn_dim, hidden_dim);

    /* Final quantization */
    for (int i = 0; i < batch_seq * hidden_dim; i++) {
        float val = (float)hidden2[i] * scale1;
        int32_t out;
        if (rng) {
            out = stochastic_round_f32(val * scale2, rng);
        } else {
            out = (int32_t)(val * scale2);
        }
        output[i] = clamp_s8(out);
    }

    free(gate);
    free(up);
    free(gated);
    free(hidden2);
}

/*
 * Reference FP32 FFN for comparison
 */
static void ffn_ref_fp32(
    const float* input,
    const float* weight1,
    const float* bias1,
    const float* weight2,
    const float* bias2,
    float* output,
    int batch_seq,
    int hidden_dim,
    int ffn_dim,
    activation_t act
) {
    float* hidden1 = (float*)malloc(batch_seq * ffn_dim * sizeof(float));
    float* activated = (float*)malloc(batch_seq * ffn_dim * sizeof(float));

    /* Linear1 */
    for (int i = 0; i < batch_seq; i++) {
        for (int j = 0; j < ffn_dim; j++) {
            float sum = bias1 ? bias1[j] : 0.0f;
            for (int k = 0; k < hidden_dim; k++) {
                sum += input[i * hidden_dim + k] * weight1[j * hidden_dim + k];
            }
            hidden1[i * ffn_dim + j] = sum;
        }
    }

    /* Activation */
    for (int i = 0; i < batch_seq * ffn_dim; i++) {
        float x = hidden1[i];
        float y;
        switch (act) {
            case ACT_GELU:
            case ACT_GELU_FAST:
                y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                break;
            case ACT_SILU:
            case ACT_SILU_FAST:
                y = x / (1.0f + expf(-x));
                break;
            case ACT_RELU:
            default:
                y = x > 0 ? x : 0;
                break;
        }
        activated[i] = y;
    }

    /* Linear2 */
    for (int i = 0; i < batch_seq; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            float sum = bias2 ? bias2[j] : 0.0f;
            for (int k = 0; k < ffn_dim; k++) {
                sum += activated[i * ffn_dim + k] * weight2[j * ffn_dim + k];
            }
            output[i * hidden_dim + j] = sum;
        }
    }

    free(hidden1);
    free(activated);
}

/*
 * Benchmark activation functions
 */
static void benchmark_activations(void) {
    printf("\n=== Activation Function Benchmark ===\n");

    const int N = 1000000;
    int16_t* input = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* output = (int16_t*)malloc(N * sizeof(int16_t));

    xoroshiro128plus_t rng;
    xoro_seed(&rng, 123);

    for (int i = 0; i < N; i++) {
        input[i] = (int16_t)((xoro_next(&rng) % 1024) - 512);  /* Q8.8 range [-2, 2] */
    }

    clock_t start, end;

    /* GELU */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_gelu_q8(input[i]);
    }
    end = clock();
    printf("GELU:      %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* GELU fast */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_gelu_fast_q8(input[i]);
    }
    end = clock();
    printf("GELU_fast: %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* SiLU */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_silu_q8(input[i]);
    }
    end = clock();
    printf("SiLU:      %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* SiLU fast */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_silu_fast_q8(input[i]);
    }
    end = clock();
    printf("SiLU_fast: %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* tanh */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_tanh_q8(input[i]);
    }
    end = clock();
    printf("tanh:      %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* sigmoid */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = (int16_t)int_sigmoid_q8(input[i]);
    }
    end = clock();
    printf("sigmoid:   %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    /* ReLU */
    start = clock();
    for (int i = 0; i < N; i++) {
        output[i] = int_relu_q8(input[i]);
    }
    end = clock();
    printf("ReLU:      %.2f M ops/sec\n",
           N / ((double)(end - start) / CLOCKS_PER_SEC) / 1e6);

    free(input);
    free(output);
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --batch N      Batch size (default: %d)\n", g_batch);
    printf("  --seq N        Sequence length (default: %d)\n", g_seq_len);
    printf("  --hidden N     Hidden dimension (default: %d)\n", g_hidden_dim);
    printf("  --ffn N        FFN dimension (default: %d)\n", g_ffn_dim);
    printf("  --act TYPE     Activation: gelu, silu, relu (default: gelu)\n");
    printf("  --bench        Run activation benchmark\n");
    printf("  -h, --help     Show this help\n");
}

int main(int argc, char** argv) {
    int run_bench = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            g_batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            g_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            g_hidden_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ffn") == 0 && i + 1 < argc) {
            g_ffn_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--act") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "gelu") == 0) g_activation = ACT_GELU;
            else if (strcmp(argv[i], "silu") == 0) g_activation = ACT_SILU;
            else if (strcmp(argv[i], "relu") == 0) g_activation = ACT_RELU;
        } else if (strcmp(argv[i], "--bench") == 0) {
            run_bench = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("INT8 Feed-Forward Network\n");
    printf("=========================\n");
    printf("Batch: %d, Seq: %d, Hidden: %d, FFN: %d\n",
           g_batch, g_seq_len, g_hidden_dim, g_ffn_dim);
    printf("Activation: %s\n", activation_names[g_activation]);

    if (run_bench) {
        benchmark_activations();
        return 0;
    }

    int batch_seq = g_batch * g_seq_len;
    int hidden_dim = g_hidden_dim;
    int ffn_dim = g_ffn_dim;

    /* Allocate memory */
    int8_t* input_int8 = (int8_t*)malloc(batch_seq * hidden_dim);
    int8_t* weight1_int8 = (int8_t*)malloc(ffn_dim * hidden_dim);
    int8_t* weight2_int8 = (int8_t*)malloc(hidden_dim * ffn_dim);
    int32_t* bias1_int32 = (int32_t*)calloc(ffn_dim, sizeof(int32_t));
    int32_t* bias2_int32 = (int32_t*)calloc(hidden_dim, sizeof(int32_t));
    int8_t* output_int8 = (int8_t*)malloc(batch_seq * hidden_dim);

    float* input_fp32 = (float*)malloc(batch_seq * hidden_dim * sizeof(float));
    float* weight1_fp32 = (float*)malloc(ffn_dim * hidden_dim * sizeof(float));
    float* weight2_fp32 = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    float* bias1_fp32 = (float*)calloc(ffn_dim, sizeof(float));
    float* bias2_fp32 = (float*)calloc(hidden_dim, sizeof(float));
    float* output_fp32 = (float*)malloc(batch_seq * hidden_dim * sizeof(float));

    /* Initialize with random data */
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);

    float input_scale = 1.0f / 64.0f;
    float weight_scale = 1.0f / 64.0f;

    for (int i = 0; i < batch_seq * hidden_dim; i++) {
        input_int8[i] = xoro_random_s8_range(&rng, 64);
        input_fp32[i] = (float)input_int8[i] * input_scale;
    }

    for (int i = 0; i < ffn_dim * hidden_dim; i++) {
        weight1_int8[i] = xoro_random_s8_range(&rng, 32);
        weight1_fp32[i] = (float)weight1_int8[i] * weight_scale;
    }

    for (int i = 0; i < hidden_dim * ffn_dim; i++) {
        weight2_int8[i] = xoro_random_s8_range(&rng, 32);
        weight2_fp32[i] = (float)weight2_int8[i] * weight_scale;
    }

    /* Run FP32 reference */
    printf("\n--- FP32 Reference FFN ---\n");
    clock_t start = clock();
    ffn_ref_fp32(input_fp32, weight1_fp32, bias1_fp32, weight2_fp32, bias2_fp32,
                 output_fp32, batch_seq, hidden_dim, ffn_dim, (activation_t)g_activation);
    clock_t end = clock();
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    /* Run INT8 FFN */
    printf("\n--- INT8 FFN ---\n");

    /* Scales for quantization */
    float scale1 = 1.0f / (hidden_dim * 64.0f);  /* After linear1 */
    float scale2 = 64.0f;  /* To INT8 */

    start = clock();
    ffn_int8(input_int8, weight1_int8, bias1_int32, weight2_int8, bias2_int32,
             output_int8, batch_seq, hidden_dim, ffn_dim,
             (activation_t)g_activation, scale1, scale2, NULL);
    end = clock();
    printf("Time: %.3f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    /* Compare results */
    printf("\n--- Comparison ---\n");

    float output_scale = 1.0f / 64.0f;
    double mse = 0.0;
    float max_diff = 0.0f;

    for (int i = 0; i < batch_seq * hidden_dim; i++) {
        float int8_val = (float)output_int8[i] * output_scale;
        float diff = fabsf(int8_val - output_fp32[i]);
        mse += diff * diff;
        if (diff > max_diff) max_diff = diff;
    }
    mse /= (batch_seq * hidden_dim);

    printf("MSE: %.6f\n", mse);
    printf("Max diff: %.6f\n", max_diff);

    /* Print sample output */
    printf("\nSample output (first 8 values):\n");
    printf("  FP32:  ");
    for (int i = 0; i < 8; i++) printf("%.4f ", output_fp32[i]);
    printf("\n");
    printf("  INT8:  ");
    for (int i = 0; i < 8; i++) printf("%.4f ", (float)output_int8[i] * output_scale);
    printf("\n");

    /* Test activation function accuracy */
    printf("\n--- Activation Accuracy Test ---\n");
    printf("Testing %s activation:\n", activation_names[g_activation]);

    double act_mse = 0.0;
    int test_n = 1000;
    for (int i = 0; i < test_n; i++) {
        float x = ((float)i / test_n - 0.5f) * 4.0f;  /* Range [-2, 2] */
        int16_t x_q8 = float_to_q8(x);

        int16_t y_q8;
        float y_exact;

        switch (g_activation) {
            case ACT_GELU:
                y_q8 = int_gelu_q8(x_q8);
                y_exact = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                break;
            case ACT_SILU:
                y_q8 = int_silu_q8(x_q8);
                y_exact = x / (1.0f + expf(-x));
                break;
            case ACT_RELU:
            default:
                y_q8 = int_relu_q8(x_q8);
                y_exact = x > 0 ? x : 0;
                break;
        }

        float y_approx = q8_to_float(y_q8);
        float diff = y_approx - y_exact;
        act_mse += diff * diff;
    }
    act_mse /= test_n;
    printf("Activation MSE (vs exact): %.6f\n", act_mse);

    /* Cleanup */
    free(input_int8); free(weight1_int8); free(weight2_int8);
    free(bias1_int32); free(bias2_int32); free(output_int8);
    free(input_fp32); free(weight1_fp32); free(weight2_fp32);
    free(bias1_fp32); free(bias2_fp32); free(output_fp32);

    printf("\nDone.\n");
    return 0;
}
