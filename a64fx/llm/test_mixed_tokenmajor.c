/* Validate mixed-format prefill with a small subset of real GGUF weight rows.
 * Metadata-only open + bounded pread: never loads the complete model.
 * Build with the same flags as test_iq3_tokenmajor.c; pass a GSQ GGUF path.
 */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#ifndef TF_TEST_TRANSFORMER_HEADER
#define TF_TEST_TRANSFORMER_HEADER "../../common/transformer.h"
#endif
#include TF_TEST_TRANSFORMER_HEADER

static int validate(const qtensor *mat, int K, int N, int nt) {
    const int rows = 13, xs = K + 11, ys = rows + 7;
    float *x = malloc((size_t)N * xs * sizeof(float));
    float *y = malloc((size_t)N * ys * sizeof(float));
    float *w = malloc((size_t)K * sizeof(float));
    if (!x || !y || !w) exit(2);
    int bad = 0;
    double worst = 0;
#ifdef A64FX_MIXED_IQ_DECODE_H
    tf_mixed_q8_block *qref = N == 1 && tf_mixed_iq_q8_enabled && tf_mixed_iq_expandable(mat->type) &&
        tf_mixed_iq_supported(mat->type, K) ? malloc((size_t)(K/256) * sizeof(*qref)) : NULL;
#endif
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < N * xs; i++)
            x[i] = pass == 2 ? 0 : ((i * 37 + pass * 53) % 255 - 127) / 127.f;
        for (int i = 0; i < N * ys; i++) y[i] = 12345.f;
#ifdef A64FX_MIXED_IQ_DECODE_H
        if (qref) tf_mixed_quant_q8(qref, x, K);
#endif
        if (N == 1) {
            tf_matvec_qtensor_rows(y, mat, x, 0, 5);
            tf_matvec_qtensor_rows(y, mat, x, 5, 5);
            tf_matvec_qtensor_rows(y, mat, x, 5, rows);
        }
        else tf_gemm_f16_mt_tokenmajor(y, mat, x, rows, N, ys, xs, nt);
        for (int r = 0; r < rows; r++) {
            dequant_row(mat->type, (const char *)mat->data + r * tf_row_bytes(mat->type, K), w, K);
            for (int t = 0; t < N; t++) {
                double ref = 0, magnitude = 0;
                for (int k = 0; k < K; k++) {
                    double xv = x[(size_t)t * xs + k];
#ifdef A64FX_MIXED_IQ_DECODE_H
                    if (qref) xv = (double)qref[k/256].d * qref[k/256].q[k%256];
#endif
                    double v = (double)w[k] * xv;
                    ref += v; magnitude += fabs(v);
                }
                float got = y[(size_t)t * ys + r];
                double err = fabs(got - ref) / fmax(1., magnitude);
                if (err > worst) worst = err;
                if (!isfinite(got) || err > 2e-6) bad++;
            }
        }
        for (int t = 0; t < N; t++)
            for (int r = rows; r < ys; r++)
                if (y[(size_t)t * ys + r] != 12345.f) bad++;
    }
    printf("type=%u K=%d N=%d nt=%d max_scaled_error=%.3g %s\n",
           mat->type, K, N, nt, worst, bad ? "FAIL" : "PASS");
    free(x); free(y); free(w);
#ifdef A64FX_MIXED_IQ_DECODE_H
    free(qref);
#endif
    return bad;
}

static void benchmark(const qtensor *mat) {
    int rows = mat->n_rows, K = mat->n_cols, N = 96;
    float *x = malloc((size_t)N * K * sizeof(float));
    float *y = malloc((size_t)N * rows * sizeof(float));
    if (!x || !y) exit(2);
    for (int i = 0; i < N * K; i++) x[i] = (i % 255 - 127) / 127.f;
    tf_gemm_f16_mt_tokenmajor(y, mat, x, rows, N, rows, K, 48);
    double start = omp_get_wtime();
    for (int rep = 0; rep < 3; rep++)
        tf_gemm_f16_mt_tokenmajor(y, mat, x, rows, N, rows, K, 48);
    double elapsed = (omp_get_wtime() - start) / 3;
    printf("BENCH type=%u M=%d K=%d N=%d ms=%.3f GFLOP/s=%.2f checksum=%g\n",
           mat->type, rows, K, N, elapsed * 1000,
           2. * rows * K * N / elapsed / 1e9, y[0]);
    {
        start = omp_get_wtime();
        for (int rep = 0; rep < 10; rep++) {
            #pragma omp parallel num_threads(48)
            {
                int tid = omp_get_thread_num(), team = omp_get_num_threads();
                tf_matvec_qtensor_rows(y, mat, x, rows * tid / team, rows * (tid + 1) / team);
            }
        }
        printf("MATVEC type=%u M=%d K=%d ms=%.3f checksum=%g\n",
               mat->type, rows, K, (omp_get_wtime() - start) * 100, y[0]);
    }
    free(x); free(y);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) { fprintf(stderr, "usage: %s GSQ.gguf [max_rows] [q8]\n", argv[0]); return 2; }
#ifdef A64FX_MIXED_IQ_DECODE_H
    if (argc == 4) {
        if (strcmp(argv[3], "q8")) return 2;
        transformer_set_mixed_iq_q8(1);
        puts("Q8 activation mode: reference uses independently dequantized weights and reconstructed Q8 inputs");
    }
#endif
    int row_limit = argc > 2 ? atoi(argv[2]) : 1024;
    if (row_limit < 13) return 2;
    gguf_context *g = gguf_open(argv[1], 3);
    int fd = open(argv[1], O_RDONLY);
    if (!g || fd < 0) return 2;
    int seen[64] = {0}, bad = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_info *t = &g->tensors[i];
        if (t->type >= 64 || seen[t->type] || t->type == GGML_TYPE_IQ3_XXS ||
            t->n_dims != 2 || t->dims[1] < 13 || t->dims[0] < 256) continue;
        seen[t->type] = 1;
        int K = (int)t->dims[0], rows = t->dims[1] < (uint64_t)row_limit ? t->dims[1] : row_limit;
        size_t bytes = rows * tf_row_bytes(t->type, K);
        void *data = malloc(bytes);
        if (!data || pread(fd, data, bytes, g->data_offset + t->offset) != (ssize_t)bytes) return 2;
        qtensor mat = {.data=data, .type=t->type, .n_rows=rows, .n_cols=K};
        bad += validate(&mat, K, 1, 1);
        bad += validate(&mat, K, 3, 1);
        bad += validate(&mat, K, 9, 4);
        if (!bad) benchmark(&mat);
#ifdef A64FX_MIXED_IQ_DECODE_H
        if (tf_mixed_iq_supported(mat.type, K) && tf_mixed_iq_expandable(mat.type)) {
            size_t nb = K/256;
            tf_mixed_weight_block *cache = malloc((size_t)rows*nb*sizeof(*cache));
            if (!cache) return 2;
            int packing_bad = 0;
            #pragma omp parallel for num_threads(48) schedule(static) reduction(+:packing_bad)
            for (int r = 0; r < rows; r++)
                packing_bad += tf_mixed_expand_row(cache+(size_t)r*nb,
                    (const uint8_t *)data + (size_t)r*tf_row_bytes(mat.type,K), mat.type,K) != 0;
            printf("CACHE type=%u reconstructed weight rows=%d %s\n", mat.type, rows, packing_bad ? "FAIL" : "PASS");
            bad += packing_bad;
            if (!packing_bad) {
                mat.mixed_iq_cache = cache;
                bad += validate(&mat, K, 1, 1);
                if (!bad) benchmark(&mat);
                mat.mixed_iq_cache = NULL;
            }
            free(cache);
        }
#endif
        free(data);
        fflush(stdout);
    }
    /* Exercise masked vector tails as well as the model's aligned dimensions. */
    float data[13 * 37];
    for (int i = 0; i < 13 * 37; i++) data[i] = (i % 19 - 9) / 19.f;
    qtensor tail = {.data=data, .type=GGML_TYPE_F32, .n_rows=13, .n_cols=37};
    bad += validate(&tail, 37, 9, 4);
    close(fd); gguf_close(g);
    return bad ? 1 : 0;
}
