/* A64FX IQ3 token-major validation against independently dequantized weights.
 * fccpx -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *   -D_GNU_SOURCE -ffunction-sections -fdata-sections test_iq3_tokenmajor.c \
 *   -Wl,--gc-sections -lm -lpthread -o build/test_iq3_tokenmajor
 */
#include <omp.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

static uint32_t rng = 1;
static uint32_t next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static block_iq3_xxs *weights(int rows, int K) {
    size_t blocks = (size_t)rows * (K / 256);
    block_iq3_xxs *w = malloc(blocks * sizeof(*w));
    if (!w) exit(2);
    for (size_t i = 0; i < blocks; i++) {
        w[i].d = ggml_fp32_to_fp16(0.001f * (1 + next_u32() % 8));
        for (int j = 0; j < 96; j++) w[i].qs[j] = (uint8_t)next_u32();
    }
    return w;
}

static int check(int N, int nt, int padded) {
    const int K = 768, rows = 13, xs = K + padded, ys = rows + 7;
    block_iq3_xxs *w = weights(rows, K);
    qtensor mat = {.data=w, .type=GGML_TYPE_IQ3_XXS, .n_rows=rows, .n_cols=K};
    float *x = malloc((size_t)N * xs * sizeof(float));
    float *y = malloc((size_t)N * ys * sizeof(float));
    if (!x || !y) exit(2);
    for (int i = 0; i < N * xs; i++) x[i] = ((int)(next_u32() % 255) - 127) / 127.f;
    float worst = 0;
    int bad = 0;
    for (int pass = 0; pass < 3; pass++) {
        /* Mutate an element that the old eight-sample cache never checked. */
        if (pass == 1) x[37] += 0.25f;
        if (pass == 2) memset(x, 0, (size_t)N * xs * sizeof(float));
        for (int i = 0; i < N * ys; i++) y[i] = 12345.f;
        tf_gemm_f16_mt_tokenmajor(y, &mat, x, rows, N, ys, xs, nt);
        for (int t = 0; t < N; t++) {
            tf_iq3_q8_block q[3];
            float dw[K];
            tf_iq3_quant_q8(q, x + (size_t)t * xs, K);
            for (int r = 0; r < rows; r++) {
                dequantize_row_iq3_xxs(w + r * (K / 256), dw, K);
                double ref = 0, magnitude = 0;
                for (int k = 0; k < K; k++) {
                    double v = (double)dw[k] * q[k / 256].q[k % 256] * q[k / 256].d;
                    ref += v; magnitude += fabs(v);
                }
                float got = y[(size_t)t * ys + r];
                float err = fabs((double)got - ref) / fmax(1., magnitude);
                if (err > worst) worst = err;
                if (!isfinite(got) || err > 2e-6f) bad++;
            }
            for (int r = rows; r < ys; r++) if (y[(size_t)t * ys + r] != 12345.f) bad++;
        }
    }
    printf("N=%d nt=%d padding=%d max_scaled_error=%.3g %s\n", N, nt, padded, worst, bad ? "FAIL" : "PASS");
    free(w); free(x); free(y);
    return bad;
}

static void bench(int nt) {
    const int rows = 1024, K = 5120, N = 32, reps = 3;
    block_iq3_xxs *w = weights(rows, K);
    qtensor mat = {.data=w, .type=GGML_TYPE_IQ3_XXS, .n_rows=rows, .n_cols=K};
    float *x = malloc((size_t)N * K * 4), *y = malloc((size_t)N * rows * 4);
    tf_iq3_q8_block *q = malloc((size_t)N * (K / 256) * sizeof(*q));
    if (!x || !y || !q) exit(2);
    for (int i = 0; i < N * K; i++) x[i] = ((int)(next_u32() % 255) - 127) / 127.f;
    for (int t = 0; t < N; t++) tf_iq3_quant_q8(q + t * (K / 256), x + (size_t)t * K, K);
    tf_gemm_f16_mt_tokenmajor(y, &mat, x, rows, N, rows, K, nt);
    double start = omp_get_wtime();
    for (int rep = 0; rep < reps; rep++)
        tf_gemm_f16_mt_tokenmajor(y, &mat, x, rows, N, rows, K, nt);
    double batch = (omp_get_wtime() - start) / reps;
    start = omp_get_wtime();
    for (int rep = 0; rep < reps; rep++) {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < N; t++)
                y[(size_t)t * rows + r] = tf_iq3_xxs_dot_sve(
                    w + r * (K / 256), q + t * (K / 256), K / 256);
    }
    double separate = (omp_get_wtime() - start) / reps;
    printf("BENCH nt=%d M=%d K=%d N=%d batch=%.3fms separate=%.3fms speedup=%.2fx effective=%.2fGop/s checksum=%g\n",
           nt, rows, K, N, batch * 1000, separate * 1000, separate / batch,
           2. * rows * K * N / batch / 1e9, y[0]);
    free(w); free(x); free(y); free(q);
}

typedef struct { int cpu, count; } affinity_result;
static void *read_affinity(void *arg) {
    affinity_result *r = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    r->count = sched_getaffinity(0, sizeof(set), &set) ? -1 : CPU_COUNT(&set);
    r->cpu = sched_getcpu();
    return NULL;
}

static int check_pool_affinity(void) {
    transformer_model m = {0};
    m.n_threads = 48;
    setenv("NUMA_INTERLEAVE", "1", 1);
    int bad = 0;
    for (int pass = 0; pass < 2; pass++) {
        /* Simulate VLM prefill pinning the primary before pool recreation. */
        #pragma omp parallel num_threads(48)
        { __asm__ volatile("" ::: "memory"); }
        cpu_set_t before, after;
        sched_getaffinity(0, sizeof(before), &before);
        affinity_result results[48];
        tf_pool_start(&m);
        tf_pool_dispatch(&m, read_affinity, results, sizeof(results[0]));
        sched_getaffinity(0, sizeof(after), &after);
        if (!CPU_EQUAL(&before, &after)) bad++;
        for (int t = 0; t < 48; t++)
            if (results[t].cpu != 12 + t || results[t].count != 1) bad++;
        tf_pool_shutdown(&m);
    }
    printf("NUMA_INTERLEAVE pool restart: 48 distinct pinned cores + caller affinity restored %s\n",
           bad ? "FAIL" : "PASS");
    return bad;
}

int main(void) {
    if (svcntb() != 64) { fprintf(stderr, "SVE-512 required\n"); return 2; }
    int bad = 0;
    const int ns[] = {2, 3, 4, 5, 8, 33};
    for (int i = 0; i < 6; i++) {
        bad += check(ns[i], 1, 0);
        bad += check(ns[i], 4, 11);
    }
    if (!bad) { bench(1); bench(48); }
    bad += check_pool_affinity();
    return bad ? 1 : 0;
}
