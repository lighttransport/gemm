/* Compare fused kernels against independently dequantized weights and Q8 input. */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"

int main(void) {
    unsigned char weights[16 * sizeof(block_q6_K)];
    float input[4096], decoded[4096];
    glm5_iq_q8_block xq[16];
    unsigned rng = 7;
    double worst = 0.0;
    for (int type = GLM53F_GGML_Q4_K; type <= GLM53F_GGML_Q6_K; ++type) {
        for (int trial = 0; trial < 100; ++trial) {
            int n = trial % 2 ? 4096 : 256;
            size_t bytes = dequant_row_size(type, n);
            for (size_t i = 0; i < bytes; ++i) {
                rng = rng * 1664525u + 1013904223u;
                weights[i] = rng >> 24;
            }
            for (int i = 0; i < n; ++i) {
                rng = rng * 1664525u + 1013904223u;
                input[i] = ((int)(rng >> 16) - 32768) / 32768.0f;
            }
            for (int b = 0; b < n / 256; ++b) {
                if (type == GLM53F_GGML_Q4_K) {
                    block_q4_K *w = (block_q4_K *)weights;
                    w[b].d = ggml_fp32_to_fp16(0.01f);
                    w[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else if (type == GLM53F_GGML_Q5_K) {
                    block_q5_K *w = (block_q5_K *)weights;
                    w[b].d = ggml_fp32_to_fp16(0.01f);
                    w[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else ((block_q6_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
            }
            glm5_iq_quant_q8(xq, input, n);
            if (type == GLM53F_GGML_Q4_K) dequantize_row_q4_K(weights, decoded, n);
            else if (type == GLM53F_GGML_Q5_K) dequantize_row_q5_K(weights, decoded, n);
            else dequantize_row_q6_K(weights, decoded, n);
            double ref = 0.0, magnitude = 0.0;
            for (int i = 0; i < n; ++i) {
                double term = (double)decoded[i] * xq[i / 256].d * xq[i / 256].q[i % 256];
                ref += term;
                magnitude += fabs(term);
            }
            float got = iq_row(type, weights, xq, n / 256);
            double error = fabs(got - ref) / (1.0 + magnitude);
            if (!isfinite(got) || error > 2e-6) {
                fprintf(stderr, "FAIL type=%d n=%d ref=%.9g got=%.9g error=%g\n", type, n, ref, got, error);
                return 1;
            }
            if (error > worst) worst = error;
        }
    }
    printf("PASS K-quants cases=300 worst_normalized_error=%g\n", worst);
    if (getenv("GLM53F_KQUANT_BENCH")) {
        enum { ROWS = 8192, COLS = 4096, REPS = 20 };
        float *out = malloc(ROWS * sizeof(float));
        if (!out) return 2;
        for (int type = GLM53F_GGML_Q4_K; type <= GLM53F_GGML_Q5_K; ++type) {
            size_t rb = dequant_row_size(type, COLS);
            unsigned char *matrix = malloc(ROWS * rb);
            if (!matrix) return 2;
            /* A repeated finite block exercises the real row stride and HBM
             * footprint without allowing the compiler to fold the dot. */
            memset(weights, 0x55, sizeof(weights));
            for (int b = 0; b < COLS / 256; ++b) {
                if (type == GLM53F_GGML_Q4_K) {
                    ((block_q4_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
                    ((block_q4_K *)weights)[b].dmin = ggml_fp32_to_fp16(0.005f);
                } else {
                    ((block_q5_K *)weights)[b].d = ggml_fp32_to_fp16(0.01f);
                    ((block_q5_K *)weights)[b].dmin = ggml_fp32_to_fp16(0.005f);
                }
            }
#pragma omp parallel for schedule(static)
            for (int r = 0; r < ROWS; ++r) memcpy(matrix + r * rb, weights, rb);
            double start = omp_get_wtime();
            for (int rep = 0; rep < REPS; ++rep) {
#pragma omp parallel for schedule(static)
                for (int r = 0; r < ROWS; ++r)
                    out[r] = iq_row(type, matrix + r * rb, xq, COLS / 256);
            }
            double seconds = omp_get_wtime() - start;
            printf("BENCH type=%d Gweights_s=%.3f check=%.9g\n", type,
                   (double)ROWS * COLS * REPS / seconds / 1e9, out[ROWS-1]);
            free(matrix);
        }
        free(out);
    }
    return 0;
}
