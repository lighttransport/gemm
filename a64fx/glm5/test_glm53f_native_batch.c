/* Batched native projections must preserve the scalar decode contract. */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"

static unsigned rng = 123;
static unsigned next(void) { rng = rng * 1664525u + 1013904223u; return rng; }

int main(void) {
    enum { ROWS = 37, TOKENS = 32, MAXC = 4096 };
    const int columns[] = {32, 96, 128, 640, 768, 1024, 4096};
    const int batches[] = {1, 2, 3, 4, 5, 7, 16, 31, 32};
    float *x = malloc((size_t)TOKENS * MAXC * sizeof(float));
    float ref[TOKENS * ROWS], out[TOKENS * ROWS], other[TOKENS * ROWS];
    block_q8_0 *w = malloc((size_t)ROWS * MAXC / 32 * sizeof(*w));
    int cases = 0;
    if (!x || !w) return 1;
    for (size_t shape = 0; shape < sizeof(columns) / sizeof(columns[0]); ++shape) {
        int n = columns[shape];
        for (int i = 0; i < ROWS * n / 32; ++i) {
            w[i].d = ggml_fp32_to_fp16((1 + (next() >> 28)) * 0.001f);
            for (int j = 0; j < 32; ++j) w[i].qs[j] = (int8_t)(next() >> 24);
        }
        for (int t = 0; t < TOKENS; ++t)
            for (int i = 0; i < n; ++i)
                x[(size_t)t * n + i] = t == 0 ? 0.0f :
                    ((int)(next() >> 16) - 32768) / 4096.0f;
        for (int t = 0; t < TOKENS; ++t)
            if (glm53f_iq_matvec(ref + t * ROWS, (const uint8_t *)w,
                    GLM53F_GGML_Q8_0, ROWS, n, x + (size_t)t * n)) return 1;
        uint8_t *packed = NULL;
        int type;
        if (glm53f_native_repack(GLM53F_GGML_Q8_0, (const uint8_t *)w,
                ROWS, n, &packed, &type)) return 1;
        glm53f_native_matrix m[2] = {
            {out, packed ? packed : (const uint8_t *)w, type, ROWS, n},
            {other, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, n}};
        size_t stride = glm53f_native_act_bytes(n) + 256;
        uint8_t *act = NULL;
        if (posix_memalign((void **)&act, 256, TOKENS * stride)) return 1;
        for (int t = 0; t < TOKENS; ++t)
            if (glm53f_native_act_prepare(act + t * stride,
                    x + (size_t)t * n, n, 0, 1)) return 1;
        for (int t = 0; t < TOKENS; ++t) {
            m[0].output = out + (size_t)t * ROWS;
            if (glm53f_native_matvec_prepared_n(m, 1, act + (size_t)t * stride)) return 1;
        }
        m[0].output = out;
        if (memcmp(out, ref, sizeof(ref))) return 1;
        /* Separate activation pointers exercise fused head projections. */
        for (int t = 1; t < TOKENS; ++t) {
            const void *activation[2] = {act, act + (size_t)t * stride};
            int bad = 0;
#pragma omp parallel reduction(|:bad)
            bad |= glm53f_native_matvec_multi_team(m, 2, activation) != 0;
            if (bad || memcmp(out, ref, ROWS * sizeof(float)) ||
                memcmp(other, ref + (size_t)t * ROWS, ROWS * sizeof(float))) return 1;
        }
        for (size_t b = 0; b < sizeof(batches) / sizeof(batches[0]); ++b) {
            int tokens = batches[b], bad = 0;
            if (glm53f_native_matvec_batch(m, 2, x, tokens) ||
                memcmp(out, ref, (size_t)tokens * ROWS * sizeof(float)) ||
                memcmp(other, ref, (size_t)tokens * ROWS * sizeof(float))) {
                fprintf(stderr, "FAIL native batch columns=%d tokens=%d\n", n, tokens);
                return 1;
            }
            /* Exercise the orphaned worksharing API and padded activations. */
            memset(out, 0x55, sizeof(out));
#pragma omp parallel reduction(|:bad)
            bad |= glm53f_native_matvec_batch_team(m, 2, act, stride, tokens) != 0;
            if (bad || memcmp(out, ref, (size_t)tokens * ROWS * sizeof(float))) {
                fprintf(stderr, "FAIL native batch team columns=%d tokens=%d\n", n, tokens);
                return 1;
            }
            /* The homogeneous repacked path can use the eight-position tile. */
            if (glm53f_native_matvec_batch(m, 1, x, tokens) ||
                memcmp(out, ref, (size_t)tokens * ROWS * sizeof(float))) return 1;
#pragma omp parallel reduction(|:bad)
            bad |= glm53f_native_matvec_batch_team(m, 1, act, stride, tokens) != 0;
            if (bad || memcmp(out, ref, (size_t)tokens * ROWS * sizeof(float))) return 1;
            cases += 2;
        }
        free(act);
        free(packed);
    }
    /* K-quant fallback and mixed activation contracts in the same batch. */
    const size_t rb = glm53f_iq_row_size(GLM53F_GGML_Q4_K, MAXC);
    block_q4_K *wk = malloc(ROWS * rb);
    if (!wk) return 1;
    for (size_t i = 0; i < ROWS * rb; ++i) ((uint8_t *)wk)[i] = next() >> 24;
    for (int i = 0; i < ROWS * MAXC / 256; ++i) {
        wk[i].d = ggml_fp32_to_fp16(0.01f);
        wk[i].dmin = ggml_fp32_to_fp16(0.005f);
    }
    glm53f_native_matrix mix[2] = {
        {out,(const uint8_t *)wk,GLM53F_GGML_Q4_K,ROWS,MAXC},
        {other,(const uint8_t *)w,GLM53F_GGML_Q8_0,ROWS,MAXC}};
    if (glm53f_native_matvec_batch(mix, 2, x, TOKENS) ||
        memcmp(other, ref, sizeof(ref))) return 1;
    for (int t = 0; t < TOKENS; ++t)
        if (glm53f_iq_matvec(ref + t * ROWS, (const uint8_t *)wk,
                GLM53F_GGML_Q4_K, ROWS, MAXC, x + (size_t)t * MAXC)) return 1;
    if (memcmp(out, ref, sizeof(ref))) return 1;
    if (glm53f_native_matvec_batch(mix, 0, x, 4) != -1 ||
        glm53f_native_matvec_batch(mix, 2, x, 0) != -1 ||
        glm53f_native_matvec_batch(mix, 2, NULL, 4) != -1) return 1;
    mix[0].columns = 96;
    mix[1].columns = 96;
    if (glm53f_native_matvec_batch(mix, 2, x, 4) != -1) return 1;
    printf("PASS native batch cases=%d rows=%d tokens=1..32 mixed=BIT_EXACT\n", cases, ROWS);
    free(wk); free(w); free(x);
    return 0;
}
