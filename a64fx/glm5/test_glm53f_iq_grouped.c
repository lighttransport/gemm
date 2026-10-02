/* Grouped native expert verification must retain scalar arithmetic per token. */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"

static unsigned rng = 12345;
static unsigned next(void) { return rng = rng * 1664525u + 1013904223u; }
static uint8_t *matrix(int type, int rows, int columns) {
    size_t rb = glm53f_iq_row_size(type, columns), bytes = rb * rows;
    uint8_t *w = malloc(bytes);
    if (!w) return NULL;
    for (size_t i = 0; i < bytes; ++i) w[i] = next() >> 24;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < columns / 256; ++b) {
            uint8_t *p = w + r * rb + b * rb / (columns / 256);
            if (type == GLM53F_GGML_Q4_K) {
                ((block_q4_K *)p)->d = ggml_fp32_to_fp16(0.0001f);
                ((block_q4_K *)p)->dmin = ggml_fp32_to_fp16(0.00005f);
            } else if (type == GLM53F_GGML_Q5_K) {
                ((block_q5_K *)p)->d = ggml_fp32_to_fp16(0.0001f);
                ((block_q5_K *)p)->dmin = ggml_fp32_to_fp16(0.00005f);
            } else ((block_q6_K *)p)->d = ggml_fp32_to_fp16(0.0001f);
        }
    return w;
}
static int same(const float *a, const float *b, int count, const char *label) {
    for (int i = 0; i < count; ++i) {
        uint32_t aa, bb;
        memcpy(&aa, a + i, 4); memcpy(&bb, b + i, 4);
        if (aa != bb || (aa & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "FAIL %s i=%d scalar=%a grouped=%a bits=%08x/%08x\n",
                    label, i, a[i], b[i], aa, bb);
            return 1;
        }
    }
    return 0;
}
static int rows_pair(void) {
    iqf_src_block q[2][16]; iqf_act a[2][16];
    float input[4096], ref[2][71], pair[2][71];
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < 4096; ++i) input[i] = ((int)(next() >> 16) - 32768) / 32768.f;
        glm5_iq_quant_q8((glm5_iq_q8_block *)q[t], input, 4096);
        iqf_prepare(a[t], q[t], 16);
    }
    iqf_init();
    for (int type = 12; type <= 13; ++type)
        for (int blocks = 1; blocks <= 16; blocks *= 2) {
            uint8_t *w = matrix(type, 71, blocks * 256);
            if (!w) return 1;
            size_t rb = glm53f_iq_row_size(type, blocks * 256);
            for (int rows = 1; rows <= 71; ++rows) {
                for (int t = 0; t < 2; ++t) iqf_rows(ref[t], w, rb, rows, a[t], blocks, type == 13);
                iqf_rows_pair(pair[0], pair[1], w, rb, rows, a[0], a[1], blocks, type == 13);
                if (same(ref[0], pair[0], rows, "pair0") || same(ref[1], pair[1], rows, "pair1")) return 1;
            }
            free(w);
        }
    return 0;
}
struct persistent_expert_call {
    float *out, *gu, *act;
    const glm53f_iq_part *parts;
    const float *weights, *input;
    int count, rc;
};
static void persistent_expert_controller(void *context) {
    struct persistent_expert_call *a = context;
    a->rc = glm53f_iq_expert_weighted(a->out, a->parts, a->weights, a->count, a->input, a->gu, a->act);
}
int main(void) {
    enum { H = 4096, STRIDE = 8, EXPERTS = 4 };
    glm53f_iq_part experts[EXPERTS], parts[4 * STRIDE];
    float weights[4 * STRIDE], x[4 * H], ref[4 * H], out[4 * H];
    float *gu = malloc(9 * 1024 * sizeof(float)), *act = malloc(9 * 512 * sizeof(float));
    glm53f_iq_batch_scratch *scratch = glm53f_iq_batch_scratch_create();
    int counts[4], cases = 0;
    if (!gu || !act || !scratch || rows_pair()) return 1;
    for (int e = 0; e < EXPERTS; ++e) {
        int inter = e & 1 ? 512 : 256, gt = e == 3 ? 14 : e == 1 ? 13 : 12, dt = e == 2 ? 12 : 13;
        experts[e] = (glm53f_iq_part){matrix(gt, 2 * inter, H), matrix(dt, H, inter), gt, dt, inter};
        if (!experts[e].gate_up || !experts[e].down) return 1;
    }
    for (int t = 0; t < 4; ++t)
        for (int i = 0; i < H; ++i) x[t * H + i] = ((int)(next() >> 16) - 32768) / 32768.f;
    for (int tokens = 1; tokens <= 4; ++tokens)
        for (int pattern = 0; pattern < 5; ++pattern) {
            for (int t = 0; t < tokens; ++t) {
                counts[t] = pattern == 0 ? 0 : pattern == 4 ? 8 : 1 + t % 3;
                for (int k = 0; k < counts[t]; ++k) {
                    int e = pattern == 1 ? 0 : pattern == 2 ? k % 3 : (t + k) % EXPERTS;
                    parts[t * STRIDE + k] = experts[e];
                    weights[t * STRIDE + k] = 0.1f * (k + 1);
                }
                if (!counts[t]) memset(ref + t * H, 0, H * sizeof(float));
                else if (glm53f_iq_expert_weighted(ref + t * H, parts + t * STRIDE,
                            weights + t * STRIDE, counts[t], x + t * H, gu, act)) return 1;
            }
            if (glm53f_iq_expert_weighted_batch(out, parts, weights, counts, STRIDE, x, tokens, scratch) ||
                same(ref, out, tokens * H, "expert")) return 1;
            if (glm53f_team_run) {
                for (int t = 0; t < tokens; ++t) {
                    if (!counts[t]) continue;
                    struct persistent_expert_call call = {out + t * H, gu, act,
                        parts + t * STRIDE, weights + t * STRIDE, x + t * H, counts[t], 0};
                    glm53f_team_run(persistent_expert_controller, &call);
                    if (call.rc || same(ref + t * H, out + t * H, H, "persistent-expert")) return 1;
                }
            }
            ++cases;
        }
    counts[0] = 9;
    if (!glm53f_iq_expert_weighted_batch(out, parts, weights, counts, STRIDE, x, 1, scratch)) return 1;
    for (int e = 0; e < EXPERTS; ++e) { free((void *)experts[e].gate_up); free((void *)experts[e].down); }
    glm53f_iq_batch_scratch_free(scratch); free(gu); free(act);
    printf("GLM53F_IQ_GROUPED cases=%d row_pairs=710 iq_fast=%s threads=%d bitwise=PASS\n", cases,
           getenv("GLM53F_IQ_FAST") ? getenv("GLM53F_IQ_FAST") : "1", omp_get_max_threads());
    return 0;
}
