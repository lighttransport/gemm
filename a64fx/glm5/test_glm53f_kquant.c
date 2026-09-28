/* Compare fused kernels against independently dequantized weights and Q8 input. */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"

/* Independent scalar Q8_0 reference: llama.cpp's ARM quantizer rounds the
 * scaled activation to nearest-even and stores an fp16 block scale. */
static void ref_quant_q8_0(block_q8_0 *q, const float *x, int n) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[32 * b + j]));
        float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        q[b].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < 32; ++j) q[b].qs[j] = (int8_t)nearbyintf(x[32 * b + j] * id);
    }
}

static double ref_q8_0_dot(const block_q8_0 *w, const block_q8_0 *x, int n,
                           double *magnitude) {
    double sum = 0.0;
    *magnitude = 0.0;
    for (int b = 0; b < n / 32; ++b)
        for (int j = 0; j < 32; ++j) {
            double term = (double)ggml_fp16_to_fp32(w[b].d) * w[b].qs[j] *
                          ggml_fp16_to_fp32(x[b].d) * x[b].qs[j];
            sum += term;
            *magnitude += fabs(term);
        }
    return sum;
}

static int test_q8_0_native(unsigned *rng) {
    enum { ROWS = 37, MAXC = 4096 };
    static const int shapes[] = {32, 96, 128, 640, 768, 1024, 4096};
    block_q8_0 *w = malloc((size_t)ROWS * (MAXC / 32) * sizeof(block_q8_0));
    unsigned char *wk = malloc((size_t)ROWS * dequant_row_size(GLM53F_GGML_Q4_K, MAXC));
    block_q8_0 xr[MAXC / 32];
    float x[MAXC], y0[ROWS], y1[ROWS], y2[ROWS], yk[ROWS];
    double worst = 0.0;
    int cases = 0, repacked = 0;
    if (!w || !wk) return 1;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        int n = shapes[s], nb = n / 32;
        for (int trial = 0; trial < 6; ++trial) {
            for (int i = 0; i < ROWS * nb; ++i) {
                *rng = *rng * 1664525u + 1013904223u;
                w[i].d = ggml_fp32_to_fp16(1e-3f * (1 + (*rng >> 28)));
                for (int j = 0; j < 32; ++j) {
                    *rng = *rng * 1664525u + 1013904223u;
                    w[i].qs[j] = (int8_t)((int)(*rng >> 24) - 128);
                    if (w[i].qs[j] == -128) w[i].qs[j] = -127;
                }
            }
            for (int i = 0; i < n; ++i) {
                *rng = *rng * 1664525u + 1013904223u;
                x[i] = ((int)(*rng >> 16) - 32768) / 3276.8f * (trial + 1);
            }
            ref_quant_q8_0(xr, x, n);
            /* Single, paired and three-way entry points must agree exactly
             * with each other and with the scalar reference within fp32. */
            glm53f_native_matrix m[3] = {
                {y0, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, n},
                {y1, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, n},
                {y2, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, n}};
            if (glm53f_native_matvec_n(m, 3, x)) {
                fprintf(stderr, "FAIL q8_0 matvec_n n=%d\n", n);
                return 1;
            }
            float single[ROWS];
            if (glm53f_iq_matvec(single, (const uint8_t *)w, GLM53F_GGML_Q8_0,
                                 ROWS, n, x) ||
                glm53f_iq_matvec_2(y1, (const uint8_t *)w, GLM53F_GGML_Q8_0,
                                   y2, (const uint8_t *)w, GLM53F_GGML_Q8_0,
                                   ROWS, n, x)) {
                fprintf(stderr, "FAIL q8_0 matvec n=%d\n", n);
                return 1;
            }
            for (int r = 0; r < ROWS; ++r) {
                double mag, ref = ref_q8_0_dot(w + (size_t)r * nb, xr, n, &mag);
                double error = fabs(y0[r] - ref) / (1.0 + mag);
                if (!isfinite(y0[r]) || error > 2e-6 || y0[r] != single[r] ||
                    y0[r] != y1[r] || y0[r] != y2[r]) {
                    fprintf(stderr, "FAIL q8_0 n=%d row=%d ref=%.9g got=%.9g single=%.9g pair=%.9g/%.9g error=%g\n",
                            n, r, ref, y0[r], single[r], y1[r], y2[r], error);
                    return 1;
                }
                if (error > worst) worst = error;
            }
            /* Team variant inside a caller-owned parallel region. */
            size_t ab = glm53f_native_act_bytes(n);
            void *act = NULL;
            int bad = 0;
            if (!ab || posix_memalign(&act, 256, ab)) return 1;
#pragma omp parallel reduction(|:bad)
            {
#pragma omp single
                bad |= glm53f_native_act_prepare(act, x, n, 0, 1) != 0;
                bad |= glm53f_native_matvec_team(m, 1, act) != 0;
            }
            free(act);
            if (bad) { fprintf(stderr, "FAIL q8_0 team n=%d\n", n); return 1; }
            for (int r = 0; r < ROWS; ++r)
                if (y0[r] != single[r]) {
                    fprintf(stderr, "FAIL q8_0 team mismatch n=%d row=%d\n", n, r);
                    return 1;
                }
            /* Repacked layout must be bit-identical to the GGUF block kernel,
             * through the single-matrix API and the 4-row team kernel with a
             * ragged final group (37 rows). */
            if (n % 64 == 0) {
                uint8_t *rp = NULL;
                int rtype = 0;
                if (glm53f_native_repack(GLM53F_GGML_Q8_0, (const uint8_t *)w,
                                         ROWS, n, &rp, &rtype) || !rp ||
                    rtype != GLM53F_NATIVE_Q8_0R ||
                    glm53f_iq_matvec(y1, rp, rtype, ROWS, n, x)) {
                    fprintf(stderr, "FAIL q8_0r setup n=%d\n", n);
                    return 1;
                }
                glm53f_native_matrix rm[2] = {
                    {y2, rp, rtype, ROWS, n},
                    {yk, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, n}};
                if (glm53f_native_matvec_n(rm, 2, x)) return 1;
                for (int r = 0; r < ROWS; ++r)
                    if (y1[r] != single[r] || y2[r] != single[r] || yk[r] != single[r]) {
                        fprintf(stderr, "FAIL q8_0r mismatch n=%d row=%d %.9g %.9g %.9g\n",
                                n, r, single[r], y1[r], y2[r]);
                        return 1;
                    }
                free(rp);
                ++repacked;
            }
            ++cases;
        }
    }
    /* Mixed Q8_0 + Q4_K pair at 4096 columns uses both activation formats. */
    {
        size_t rb = dequant_row_size(GLM53F_GGML_Q4_K, MAXC);
        for (size_t i = 0; i < (size_t)ROWS * rb; ++i) {
            *rng = *rng * 1664525u + 1013904223u;
            wk[i] = *rng >> 24;
        }
        for (int r = 0; r < ROWS; ++r)
            for (int b = 0; b < MAXC / 256; ++b) {
                block_q4_K *blk = (block_q4_K *)(wk + (size_t)r * rb) + b;
                blk->d = ggml_fp32_to_fp16(0.01f);
                blk->dmin = ggml_fp32_to_fp16(0.005f);
            }
        float ref0[ROWS], refk[ROWS];
        if (glm53f_iq_matvec(ref0, (const uint8_t *)w, GLM53F_GGML_Q8_0, ROWS, MAXC, x) ||
            glm53f_iq_matvec(refk, wk, GLM53F_GGML_Q4_K, ROWS, MAXC, x) ||
            glm53f_iq_matvec_2(y0, (const uint8_t *)w, GLM53F_GGML_Q8_0,
                               yk, wk, GLM53F_GGML_Q4_K, ROWS, MAXC, x)) {
            fprintf(stderr, "FAIL mixed pair setup\n");
            return 1;
        }
        for (int r = 0; r < ROWS; ++r)
            if (y0[r] != ref0[r] || yk[r] != refk[r]) {
                fprintf(stderr, "FAIL mixed pair row=%d\n", r);
                return 1;
            }
    }
    printf("PASS Q8_0 native cases=%d repacked_bit_exact=%d worst_normalized_error=%g mixed_pair=BIT_EXACT\n",
           cases, repacked, worst);
    if (getenv("GLM53F_KQUANT_BENCH")) {
        enum { BR = 8192, BC = 4096, REPS = 200 };
        size_t rb = dequant_row_size(GLM53F_GGML_Q8_0, BC);
        unsigned char *matrix = malloc((size_t)BR * rb);
        float *out = malloc(BR * sizeof(float));
        float *reference = malloc(BR * sizeof(float));
        if (!matrix || !out || !reference) return 1;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < BR; ++r) memcpy(matrix + (size_t)r * rb, w, rb);
        glm53f_native_matrix bm = {out, matrix, GLM53F_GGML_Q8_0, BR, BC};
        double start = omp_get_wtime();
        for (int rep = 0; rep < REPS; ++rep)
            if (glm53f_native_matvec_n(&bm, 1, x)) return 1;
        double seconds = omp_get_wtime() - start;
        printf("BENCH type=Q8_0 Gweights_s=%.3f GB_s=%.3f check=%.9g\n",
               (double)BR * BC * REPS / seconds / 1e9,
               (double)BR * rb * REPS / seconds / 1e9, out[BR - 1]);
        uint8_t *rp = NULL;
        int rtype = 0;
        if (glm53f_native_repack(GLM53F_GGML_Q8_0, matrix, BR, BC, &rp, &rtype) || !rp)
            return 1;
        memcpy(reference, out, BR * sizeof(float));
        glm53f_native_matrix rm = {out, rp, rtype, BR, BC};
        size_t rrb = glm53f_native_row_size(rtype, BC);
        start = omp_get_wtime();
        for (int rep = 0; rep < REPS; ++rep)
            if (glm53f_native_matvec_n(&rm, 1, x)) return 1;
        seconds = omp_get_wtime() - start;
        double worst_error = 0.0;
        for (int r = 0; r < BR; ++r) {
            double error = fabs((double)out[r] - reference[r]) /
                           (1.0 + fabs((double)reference[r]));
            if (!isfinite(out[r]) || error > 2e-5) {
                fprintf(stderr, "FAIL q8 bench panel row=%d error=%g\n", r, error);
                return 1;
            }
            if (error > worst_error) worst_error = error;
        }
        printf("BENCH type=%s Gweights_s=%.3f GB_s=%.3f check=%.9g worst_normalized_error=%.9g\n",
               rtype == GLM53F_NATIVE_Q8_0R16 ? "Q8_0R16" : "Q8_0R",
               (double)BR * BC * REPS / seconds / 1e9,
               (double)BR * rrb * REPS / seconds / 1e9, out[BR - 1],
               worst_error);
        free(rp);
        free(reference);
        free(out);
        free(matrix);
    }
    free(wk);
    free(w);
    return 0;
}

static int test_q8_0_panel(unsigned *rng) {
    enum { ROWS = 32, COLUMNS = 128, TOKENS = 5 };
    block_q8_0 w[ROWS * COLUMNS / 32];
    float x[TOKENS * COLUMNS], baseline[TOKENS * ROWS], panel_y[TOKENS * ROWS];
    for (size_t i = 0; i < sizeof(w) / sizeof(w[0]); ++i) {
        *rng = *rng * 1664525u + 1013904223u;
        w[i].d = ggml_fp32_to_fp16(0.001f * (1 + (*rng >> 28)));
        for (int j = 0; j < 32; ++j) {
            *rng = *rng * 1664525u + 1013904223u;
            w[i].qs[j] = (int8_t)((int)(*rng >> 24) - 127);
        }
    }
    for (int i = 0; i < TOKENS * COLUMNS; ++i) {
        *rng = *rng * 1664525u + 1013904223u;
        x[i] = ((int)(*rng >> 16) - 32768) / 4096.0f;
    }
    glm53f_native_matrix base = {baseline, (const uint8_t *)w,
                                  GLM53F_GGML_Q8_0, ROWS, COLUMNS};
    if (glm53f_native_matvec_batch(&base, 1, x, TOKENS)) return 1;
    const char *prior_panel = getenv("GLM53F_NATIVE_Q8_PANEL");
    char *saved_panel = prior_panel ? strdup(prior_panel) : NULL;
    if (prior_panel && !saved_panel) return 1;
    if (setenv("GLM53F_NATIVE_Q8_PANEL", "1", 1)) return 1;
    uint8_t *packed = NULL;
    int type = 0;
    int rc = glm53f_native_repack(GLM53F_GGML_Q8_0, (const uint8_t *)w,
                                  ROWS, COLUMNS, &packed, &type);
    uint8_t *rowwise = NULL;
    int row_type = 0;
    int row_rc = glm53f_native_repack_rowwise(GLM53F_GGML_Q8_0,
        (const uint8_t *)w, ROWS, COLUMNS, &rowwise, &row_type);
    if (saved_panel) {
        setenv("GLM53F_NATIVE_Q8_PANEL", saved_panel, 1);
        free(saved_panel);
    } else unsetenv("GLM53F_NATIVE_Q8_PANEL");
    if (rc || !packed || type != GLM53F_NATIVE_Q8_0R16 ||
        row_rc || !rowwise || row_type != GLM53F_NATIVE_Q8_0R) return 1;
    glm53f_native_matrix rm = {panel_y, rowwise, row_type, ROWS, COLUMNS};
    if (glm53f_native_matvec_batch(&rm, 1, x, TOKENS)) return 1;
    for (int i = 0; i < TOKENS * ROWS; ++i)
        if (panel_y[i] != baseline[i]) return 1;
    free(rowwise);
    glm53f_native_matrix pm = {panel_y, packed, type, ROWS, COLUMNS};
    if (glm53f_native_matvec_batch(&pm, 1, x, TOKENS)) return 1;
    double worst = 0;
    for (int i = 0; i < TOKENS * ROWS; ++i) {
        double err = fabs((double)panel_y[i] - baseline[i]) /
                     (1.0 + fabs((double)baseline[i]));
        if (!isfinite(panel_y[i]) || err > 2e-5) {
            fprintf(stderr, "FAIL q8 panel index=%d baseline=%g panel=%g error=%g\n",
                    i, baseline[i], panel_y[i], err);
            free(packed);
            return 1;
        }
        if (err > worst) worst = err;
    }
    for (int t = 0; t < TOKENS; ++t) {
        pm.output = panel_y + (size_t)t * ROWS;
        if (glm53f_native_matvec_n(&pm, 1, x + (size_t)t * COLUMNS)) return 1;
    }
    for (int i = 0; i < TOKENS * ROWS; ++i)
        if (!isfinite(panel_y[i]) ||
            fabs((double)panel_y[i] - baseline[i]) /
                (1.0 + fabs((double)baseline[i])) > 2e-5) return 1;
    free(packed);
    printf("PASS Q8_0R16 panel rows=%d tokens=%d worst_normalized_error=%g\n",
           ROWS, TOKENS, worst);
    return 0;
}

int main(void) {
    unsigned char weights[16 * sizeof(block_q6_K)];
    float input[4096], decoded[4096];
    glm5_iq_q8_block xq[16];
    unsigned rng = 7;
    if (test_q8_0_panel(&rng)) return 1;
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
    if (test_q8_0_native(&rng)) return 1;
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
