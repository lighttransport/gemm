#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

int main(void)
{
    enum { ROWS = 48, COLS = 5120 };
    transformer_model model = {0};
    model.n_threads = 48; /* No live pool: exercise the serial fallback. */
    size_t source_bytes = (size_t)ROWS * COLS / 32 * sizeof(block_q8_0);
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t mapping_bytes = (source_bytes + page - 1) & ~(page - 1);
    block_q8_0 *source = mmap(NULL, mapping_bytes, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *expected = malloc((size_t)ROWS * COLS * sizeof(float));
    float x[COLS], got[ROWS];
    if (source == MAP_FAILED || !expected) return 1;
    for (int k = 0; k < COLS; k++) x[k] = (float)(k % 37 - 18) / 31.0f;
    for (int r = 0; r < ROWS; r++) for (int b = 0; b < COLS / 32; b++) {
        block_q8_0 *q = source + r * (COLS / 32) + b;
        q->d = (uint16_t)(0x1400 + ((r + b) % 8) * 37);
        float d = ggml_fp16_to_fp32(q->d);
        for (int k = 0; k < 32; k++) {
            q->qs[k] = (int8_t)((r * 7 + b * 13 + k) % 255 - 127);
            expected[(size_t)r * COLS + b * 32 + k] = d * q->qs[k];
        }
    }
    qtensor mat = {0};
    mat.data = source; mat.type = GGML_TYPE_Q8_0; mat.n_rows = ROWS; mat.n_cols = COLS;
    size_t bytes = tf_materialize_q8_small_f32(&model, &mat);
    if (bytes != (size_t)ROWS * COLS * sizeof(float) || mat.type != GGML_TYPE_F32 ||
        memcmp(mat.data, expected, bytes)) return 1;
    tf_matvec_qtensor_rows(got, &mat, x, 0, ROWS);
    for (int r = 0; r < ROWS; r++) {
        double ref = 0, sum_abs = 0;
        for (int k = 0; k < COLS; k++) {
            double v = (double)expected[(size_t)r * COLS + k] * x[k];
            ref += v; sum_abs += fabs(v);
        }
        if (!isfinite(got[r]) || fabs(got[r] - ref) > sum_abs * 2e-6 + 1e-6) return 1;
    }
    puts("Small Q8->FP32 PASS: exact weights, 48x5120 direct row-dot oracle");
    for (int i = 0; i < model.decode_owned_count; i++) free(model.decode_owned[i]);
    free(model.decode_owned); free(expected); munmap(source, mapping_bytes);

    model.is_hybrid = 1; model.n_layers = 1; model.ssm_dt_rank = 48; model.ssm_d_state = 128;
    model.recurrent_state = calloc(1, sizeof(float *));
    size_t n = (size_t)48 * 128 * 128;
    if (!model.recurrent_state) return 1;
    model.recurrent_state[0] = malloc(n * sizeof(float));
    if (!model.recurrent_state[0]) return 1;
    for (size_t i = 0; i < n; i++) model.recurrent_state[0][i] = (float)((i * 17) % 10007) * 0.001f;
    if (tf_ssm_state_bind_cmg(&model) || !model.recurrent_state_mapped[0]) return 1;
    for (size_t i = 0; i < n; i++)
        if (model.recurrent_state[0][i] != (float)((i * 17) % 10007) * 0.001f) return 1;
    for (int h = 0; h < 48; h++) {
        int node = -1;
        if (syscall(SYS_get_mempolicy, &node, NULL, 0,
                    model.recurrent_state[0] + (size_t)h * 128 * 128, 3) || node != 4 + h / 12) {
            fprintf(stderr, "state placement mismatch head=%d node=%d\n", h, node);
            return 1;
        }
    }
    tf_free_recurrent_state(&model, 0);
    if (model.recurrent_state[0] || model.recurrent_state_mapped[0]) return 1;
    free(model.recurrent_state); free(model.recurrent_state_mapped);
    puts("SSM state PASS: exact copy, all 48 heads on owner nodes, mmap cleanup");
    return 0;
}
